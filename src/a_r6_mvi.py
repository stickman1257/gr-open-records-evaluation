"""Imputation with a common linear trend."""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
from scipy import sparse, stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC, H_BASES, C_BASES
import s04_analysis as A

LABEL = "post hoc (added 11/16 after the final run)"
HC, DC = "보건", "어린이집"
SAMPLES = {"health_centre": (HC,), "pooled": (HC, DC)}
G7ROW = {"health_centre": "imputation_health_only", "pooled": "imputation_use"}
FIT_LO = 24
BINS = [(a, a + 2, "pre") for a in range(-24, -4, 3)] + [(-3, -1, "ant"), (1, 3, "post"), (4, 6, "post"), (7, 9, "post"), (10, 12, "post")]
EPS = 1e-6
pct = lambda a: float(np.exp(a) - 1) if a is not None and np.isfinite(a) else None
_mi = lambda v: (np.asarray(v, dtype=np.int64) // 100) * 12 + np.asarray(v, dtype=np.int64) % 100


class MVI:
    """weather=None -> plain two-way imputation (dense solver). weather=(hb, cb) -> building-specific HDD / CDD slopes."""
    def __init__(self, S, weather=None, leads=False):
        self.S = S; D = S["D"]; self.pi, self.ki = S["pi"], S["ki"]; self.nP, self.nK = len(S["pids"]), len(S["tks"])
        self.y = D.y.to_numpy(float); self.post, self.ant, self.isc = S["post"], S["ant"], S["isc"]
        self.F0 = ~self.post & ~self.ant; N = len(self.y); r = np.arange(N); self.rel = D.rel.to_numpy(); self.N = N
        cols = [sparse.csr_matrix((np.ones(N), (r, self.pi)), shape=(N, self.nP)), sparse.csr_matrix((np.ones(N), (r, self.ki)), shape=(N, self.nK))]
        self.pen_idx = {}; self.mean_idx = {}; self.weather = weather; p = self.nP + self.nK
        self.uses = sorted(set(S["use_p"])); ui = np.array([self.uses.index(u) for u in S["use_p"]]); self.ui = ui
        if weather is not None:
            hb, cb = weather
            H = (D[f"HDD{hb}"] / D.days / 10).to_numpy(float); C = (D[f"CDD{cb}"] / D.days / 10).to_numpy(float)
            if not (np.isfinite(H).all() and np.isfinite(C).all()): raise ValueError("missing degree days")
            for nm, v in (("H", H), ("C", C)):
                cols.append(sparse.csr_matrix((v, (r, ui[self.pi])), shape=(N, len(self.uses))))
                self.mean_idx[nm] = np.arange(p, p + len(self.uses)); p += len(self.uses)
                cols.append(sparse.csr_matrix((v, (r, self.pi)), shape=(N, self.nP))); self.pen_idx[nm] = np.arange(p, p + self.nP); p += self.nP
        self.n_lead = 0; self.lead_names = []
        if leads:
            tr = ~self.isc & self.F0
            edges = [(-9, -7), (-6, -4)] if leads == "two" else [(a, min(a + 2, -4)) for a in range(-FIT_LO, -3, 3)][1:]
            L = np.zeros((N, len(edges)))
            for j, (a, b) in enumerate(edges): L[tr & (self.rel >= a) & (self.rel <= b), j] = 1
            keep = L.sum(0) > 0; cols.append(sparse.csr_matrix(L[:, keep])); self.n_lead = int(keep.sum())
            self.lead_names = [f"lead_{a}_{b}" for (a, b), k in zip(edges, keep) if k]
        self.X = sparse.hstack(cols).tocsr(); self.p = self.X.shape[1]

    def _solve(self, wr, F, pen, drop=None):
        XF = self.X[F]; wf = wr[F]
        G = (XF.T @ XF.multiply(wf[:, None])).toarray(); rhs = XF.T @ (wf * self.y[F])
        Gp = G + np.diag(pen); act = np.diag(Gp) > 0
        if drop is not None: act[drop] = False
        lam, V = np.linalg.eigh(Gp[np.ix_(act, act)]); pos = lam > 1e-10 * lam.max()
        beta = np.zeros(self.p); beta[act] = V[:, pos] @ ((V[:, pos].T @ rhs[act]) / lam[pos])
        return beta, act, lam, V, pos

    def variance_components(self, w=None):
        """Stage 1 (negligible penalty) -> sigma^2 and DerSimonian-Laird tau^2 per use type and regressor."""
        w = np.ones(self.nP) if w is None else np.asarray(w, float); wr = w[self.pi]; F = self.F0 & (wr > 0)
        pen = np.zeros(self.p)
        for ix in self.pen_idx.values(): pen[ix] = EPS
        drop = np.concatenate(list(self.mean_idx.values())) if self.mean_idx else None
        beta, act, lam, V, pos = self._solve(wr, F, pen, drop)
        res = self.y[F] - self.X[F] @ beta; rank = int(pos.sum()); s2 = float((wr[F] * res ** 2).sum() / max(wr[F].sum() - rank, 1))
        ai = np.cumsum(act) - 1; vc = dict(sigma2=s2, tau2={}, n_slopes={}, rss=float((wr[F] * res ** 2).sum()), rank=rank, n_obs=int(F.sum()))
        for nm, ix in self.pen_idx.items():
            v = np.full(self.nP, np.inf); a = act[ix]
            v[a] = s2 * ((V[ai[ix[a]]][:, pos] ** 2) @ (1 / lam[pos])); d = beta[ix]
            for k, u in enumerate(self.uses):
                m = (self.ui == k) & np.isfinite(v) & (v < s2 / (10 * EPS)) & (w > 0)
                if m.sum() < 3: vc["tau2"][f"{nm}|{u}"] = 0.0; vc["n_slopes"][f"{nm}|{u}"] = int(m.sum()); continue
                ww = 1 / v[m]; mu = (ww * d[m]).sum() / ww.sum(); Q = (ww * (d[m] - mu) ** 2).sum()
                vc["tau2"][f"{nm}|{u}"] = float(max(0.0, (Q - (m.sum() - 1)) / (ww.sum() - (ww ** 2).sum() / ww.sum())))
                vc["n_slopes"][f"{nm}|{u}"] = int(m.sum())
        return vc

    def penalty(self, vc, mode="eb"):
        pen = np.zeros(self.p)
        for nm, ix in self.pen_idx.items():
            for k, u in enumerate(self.uses):
                t2 = vc["tau2"][f"{nm}|{u}"] if vc else 0.0
                pen[ix[self.ui == k]] = EPS if mode == "none" else (1e8 if (mode == "common" or t2 <= 0) else vc["sigma2"] / t2)
        return pen

    def fit(self, w=None, vc=None, mode="eb", min_ctrl=1):
        w = np.ones(self.nP) if w is None else np.asarray(w, float); wr = w[self.pi]; F = self.F0 & (wr > 0)
        pen = self.penalty(vc, mode) if self.pen_idx else np.zeros(self.p)
        beta, act, lam, V, pos = self._solve(wr, F, pen)
        pred = self.X @ beta
        bad = np.asarray(abs(self.X[:, ~act]).sum(1)).ravel() > 0 if (~act).any() else np.zeros(self.N, bool)
        Nn = V[:, ~pos]
        if Nn.shape[1]: bad |= np.abs(self.X[:, act] @ Nn).max(1) > 1e-6
        nctrl = np.bincount(self.ki[F & self.isc], minlength=self.nK); sup = (nctrl >= min_ctrl)[self.ki]
        res = self.y - pred; ok = sup & ~bad & np.isfinite(res) & (wr > 0)
        return dict(res=res, ok=ok, wr=wr, F=F, beta=beta, pen=pen, null_dim=int((~pos).sum()))

    def pretrend(self, vc=None, mode="eb", two=False):
        """Model + lead bins under the same penalty; cluster-robust (building) sandwich; Wald / q against F(q, G-1)."""
        M = MVI(self.S, self.weather, leads="two" if two else True); q = M.n_lead
        if q == 0: return dict(error="no lead bin")
        F = M.F0; XF = M.X[F]; pen = M.penalty(vc, mode) if M.pen_idx else np.zeros(M.p)
        G0 = (XF.T @ XF).toarray(); Gp = G0 + np.diag(pen); lam, V = np.linalg.eigh(Gp); pos = lam > 1e-10 * lam.max()
        Ainv = (V[:, pos] / lam[pos]) @ V[:, pos].T; beta = Ainv @ (XF.T @ M.y[F]); e = M.y[F] - XF @ beta
        g = M.pi[F]; Gn = int(np.unique(g).size); N = int(F.sum())
        edf = float((Ainv * G0).sum()); k_par = edf - Gn
        Sc = sparse.csr_matrix((np.ones(F.sum()), (g, np.arange(F.sum()))), shape=(M.nP, int(F.sum()))) @ XF.multiply(e[:, None])
        Sc = np.asarray(Sc.todense()); AL = Ainv[M.p - q:]
        Vb = AL @ (Sc.T @ Sc) @ AL.T * (Gn / (Gn - 1)) * ((N - 1) / (N - k_par)); b = beta[M.p - q:]
        W = float(b @ np.linalg.solve(Vb, b)); se = np.sqrt(np.diag(Vb))
        return dict(reference="all earlier fit months (s-24..s-10)" if two else "lead_-24_-22", effective_parameters=edf, terms=M.lead_names, coef=[float(v) for v in b], se=[float(v) for v in se], wald=W, df=[q, Gn - 1],
                    p_joint=float(stats.f.sf(W / q, q, Gn - 1)), n_obs=int(F.sum()), n_buildings=Gn,
                    last_two_leads=dict(terms=M.lead_names[-2:], coef=[float(v) for v in b[-2:]], se=[float(v) for v in se[-2:]]),
                    test="Wald / q against F(q, G-1); penalised estimating equations, sandwich clustered by building x G/(G-1) x (N-1)/(N-k), "
                         "k = effective parameters (trace of the hat matrix) - building effects")

    def bin_masks(self, f):
        out = []
        for a, b, kind in BINS:
            inb = ~self.isc & (self.rel >= a) & (self.rel <= b)
            out.append(inb & (self.post if kind == "post" else self.ant if kind == "ant" else (f["F"] & ~self.post & ~self.ant)))
        return out


def choose_base(S):
    """Pair (HDD base, CDD base) of the degree-day baseline grid with the smallest stage-1 residual sum of squares on the fit sample."""
    rows = []
    for hb in H_BASES:
        for cb in C_BASES:
            if cb < hb: continue
            try: vc = MVI(S, (hb, cb)).variance_components(); rows.append(dict(hb=hb, cb=cb, rss=vc["rss"], sigma2=vc["sigma2"]))
            except Exception as ex: rows.append(dict(hb=hb, cb=cb, rss=np.inf, error=str(ex)[:80]))
    best = min(rows, key=lambda r: r["rss"])
    return (best["hb"], best["cb"]), rows


def estimate(S, M, nboot, vc=None, mode="eb", profile=False, seed=0):
    """Point estimate, percentile bootstrap (variance components held fixed), optional event-bin profile."""
    f = M.fit(vc=vc, mode=mode); a = S["att"](f)[0]; m = M.post & f["ok"]
    masks0 = M.bin_masks(f) if profile else None; est_b = [S["att"](f, mm)[0] for mm in masks0] if profile else None
    rng = np.random.default_rng(seed); bs, bb = [], []
    for _ in range(nboot):
        fb = M.fit(A.boot_weights(S, rng), vc=vc, mode=mode); bs.append(S["att"](fb)[0])
        if profile: bb.append([S["att"](fb, mm)[0] for mm in M.bin_masks(fb)])
    bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan))
    tb = pd.DataFrame({"p": M.pi[m], "r": f["res"][m]}).groupby("p").r.mean()
    r = dict(att_log=float(a), ci_log_lo=float(lo), ci_log_hi=float(hi), pct=pct(a), pct_lo=pct(lo), pct_hi=pct(hi),
             ci_excludes_zero=bool((lo > 0) or (hi < 0)), boot_sd_log=float(bs.std(ddof=1)) if len(bs) > 1 else None,
             n_treated=int(pd.unique(M.pi[m]).size), n_control=int(pd.unique(M.pi[f["F"] & M.isc]).size),
             support_rows=float(m.sum() / max(int(M.post.sum()), 1)), nboot=int(nboot), nboot_ok=int(len(bs)),
             anticipation_gap_log=float(S["att"](f, M.ant)[1]) if M.ant.any() else None,
             median_bld_log=float(tb.median()), median_bld_pct=pct(tb.median()), sd_bld_log=float(tb.std()),
             resid_sd_fit=float(np.std(f["res"][f["F"]])))
    prof = None
    if profile:
        bb = np.array(bb, float).reshape(-1, len(BINS)); prof = []
        for j, ((a_, b_, kind), mm) in enumerate(zip(BINS, masks0)):
            ok = mm & f["ok"]; d = bb[:, j]; d = d[np.isfinite(d)]
            l_, h_ = (np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan))
            prof.append(dict(term=f"ev_{a_}_{b_}", kind={"pre": "pre_in_sample", "ant": "anticipation", "post": "post"}[kind], est_log=float(est_b[j]),
                             lo_log=float(l_), hi_log=float(h_), est_pct=100 * pct(est_b[j]) if np.isfinite(est_b[j]) else None,
                             lo=100 * pct(l_) if np.isfinite(l_) else None, hi=100 * pct(h_) if np.isfinite(h_) else None,
                             n_treated=int(pd.unique(M.pi[ok]).size), n_rows=int(ok.sum())))
    return r, prof


def synthetic(seed=3, n_t=150, n_c=150, tau=-0.15):
    """Staggered treated buildings with steeper heating slopes than the controls, a warming trend (post winters milder), province weather,"""
    rng = np.random.default_rng(seed); months = [y * 100 + m for y in range(2020, 2026) for m in range(1, 13)][:70]
    provs = list("ABCDE"); rowsW = []
    for k, p in enumerate(provs):
        for j, ym in enumerate(months):
            mo = ym % 100; T = 12 + 2 * k - 13 * np.cos(2 * np.pi * (mo - 1) / 12) + 0.12 * j + rng.normal(0, 1.2)
            rowsW.append(dict(sido=p, ym=ym, T=T))
    W = pd.DataFrame(rowsW)
    for b in H_BASES: W[f"HDD{b}"] = np.clip(b - W["T"], 0, None) * 30
    for b in C_BASES: W[f"CDD{b}"] = np.clip(W["T"] - b, 0, None) * 30
    W["days"] = 30; Bs, Es = [], []
    lam = {ym: rng.normal(0, 0.05) for ym in months}
    for i in range(n_t + n_c):
        tr = i < n_t; pid = f"s{i}"; prov = provs[rng.integers(5)]
        bH = max(0.0, rng.normal(0.60 if tr else 0.15, 0.12)); bC = max(0.0, rng.normal(0.15, 0.08)); a = rng.normal(0, 0.4)
        if tr: si = rng.integers(26, 44); s = months[si]; e = months[si + rng.integers(3, 8)]
        else: s = e = 202512
        w = W[W.sido == prov]
        y = a + bH * w.HDD18 / 300 + bC * w.CDD24 / 300 + w.ym.map(lam) + rng.normal(0, 0.10, len(w)) + np.where(tr & (w.ym > e), tau, 0.0)
        d = w.assign(pid=pid, carrier="elec", kwh=np.exp(y) * 200, gr_gfa=200.0, start_ym=s, end_ym=e, role="treated" if tr else "control")
        d["phase"] = np.where(d.ym < s, "pre", np.where(d.ym > e, "post", "constr"))
        if not tr: d = d[d.ym < 202509]
        Es.append(d); Bs.append(dict(pid=pid, link_grade="A_cand", use_class="U", role="treated" if tr else "control", start_ym=s, end_ym=e, sido=prov))
    B = pd.DataFrame(Bs); E = pd.concat(Es, ignore_index=True)
    S = A.imputation_setup(B, E, "elec", uses=("U",), ctrl_roles=("control",), fit_lo=FIT_LO, min_win=0)
    M0 = MVI(S); f0 = M0.fit(); plain = S["att"](f0)[0]
    out = dict(true_effect_log=tau, plain_imputation_log=float(plain), n_treated=int((~S["ctrl_p"]).sum()), n_control=int(S["ctrl_p"].sum()),
               design="treated mean heating slope 0.60 against 0.15 for controls (per 10 HDD18 per day), warming 1.44 C per year, noise sd 0.10")
    base, grid = choose_base(S); M = MVI(S, base); vc = M.variance_components()
    for mode in ["eb", "none", "common"]:
        f = M.fit(vc=vc, mode=mode); out[f"mvi_{mode}_log"] = float(S["att"](f)[0])
    out.update(base_chosen=list(base), base_true=[18, 24], sigma2=vc["sigma2"], tau2=vc["tau2"], true_sigma2=0.01, true_tau2_H=0.12 ** 2, true_tau2_C=0.08 ** 2,
               mvi_error_log=out["mvi_eb_log"] - tau, plain_error_log=out["plain_imputation_log"] - tau)
    out["ok"] = bool(abs(out["mvi_error_log"]) < 0.02 and abs(out["plain_error_log"]) > 0.04)
    return out


def main():
    NBH = int(os.environ.get("IMP_NBOOT_HEAD", "1000")); NBP = int(os.environ.get("R6_NBOOT_POOLED", "200")); t0 = time.time()
    out = dict(label=LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), definition=__doc__,
               panel_sha256=hashlib.sha256((PROC / "energy_monthly.csv").read_bytes()).hexdigest(), samples={})
    out["synthetic"] = synthetic(); print("synthetic:", json.dumps({k: v for k, v in out["synthetic"].items() if k not in ("design",)}, default=float), flush=True)
    B, E = A.load()
    g7 = {r.get("spec"): r for r in json.loads((RES / "G7_imputation.json").read_text(encoding="utf-8")) if isinstance(r, dict) and r.get("carrier") == "elec"}
    Sp = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, fit_lo=FIT_LO)); base, grid = choose_base(Sp)
    out["base_temperatures"] = dict(chosen=dict(hdd_base=base[0], cdd_base=base[1]), rule="smallest stage-1 residual sum of squares on the untreated observations of the pooled sample",
                                    grid=grid, degree_day_baseline_grid=dict(H_BASES=H_BASES, C_BASES=C_BASES, note="common.fit_dd_model chooses a pair per building by BIC"))
    print("base pair:", base, flush=True)
    (RES / "a_r6_mvi_base.json").write_text(json.dumps(dict(hb=base[0], cb=base[1], label=LABEL)), encoding="utf-8")
    att_rows, ev_rows = [], []
    for s, uses in SAMPLES.items():
        nb = NBH if s == "health_centre" else NBP
        S = Sp if s == "pooled" else A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=uses, fit_lo=FIT_LO))
        M = MVI(S, base); vc = M.variance_components(); res = {}
        nfit = np.bincount(M.pi[M.F0], minlength=M.nP)
        info = dict(variance_components=vc, n_buildings=M.nP, n_treated=int((~S["ctrl_p"]).sum()), n_control=int(S["ctrl_p"].sum()),
                    treated_fit_months_median=float(np.median(nfit[~S["ctrl_p"]])), control_fit_months_median=float(np.median(nfit[S["ctrl_p"]])))
        D = S["D"]; shr = {}
        for nm, col in (("H", f"HDD{base[0]}"), ("C", f"CDD{base[1]}")):
            v = (D[col] / D.days / 10).to_numpy(float); F = M.F0
            ss = pd.Series(v[F]).groupby(M.pi[F]).agg(lambda x: ((x - x.mean()) ** 2).sum()).reindex(range(M.nP)).fillna(0).to_numpy()
            t2 = np.array([vc["tau2"][f"{nm}|{u}"] for u in S["use_p"]]); wgt = np.where(t2 > 0, ss / (ss + vc["sigma2"] / np.where(t2 > 0, t2, 1.0)), 0.0)
            shr[nm] = dict(own_slope_weight_median_treated=float(np.median(wgt[~S["ctrl_p"]])), own_slope_weight_median_control=float(np.median(wgt[S["ctrl_p"]])),
                           own_slope_weight_p10_treated=float(np.percentile(wgt[~S["ctrl_p"]], 10)), own_slope_weight_p90_treated=float(np.percentile(wgt[~S["ctrl_p"]], 90)))
        f = M.fit(vc=vc); ub = {}
        for nm, off in (("H", M.nP + M.nK), ("C", M.nP + M.nK + len(M.uses) + M.nP)):
            for k, u in enumerate(M.uses):
                d = f["beta"][M.pen_idx[nm]][M.ui == k]; tr = ~S["ctrl_p"][M.ui == k]
                ub[f"{nm}|{u}"] = dict(use_mean_slope=float(f["beta"][off + k]), deviation_sd=float(d.std()), mean_deviation_treated=float(d[tr].mean()) if tr.any() else None,
                                       mean_deviation_control=float(d[~tr].mean()) if (~tr).any() else None)
        info.update(shrinkage=shr, slopes=ub, slope_units="log points per 10 degree days per day")
        for name, (MM, v, mode, prof) in {"mvi": (M, vc, "eb", True), "imp_s24": (MVI(S), None, "eb", True), "mvi_no_shrink": (M, vc, "none", False),
                                          "mvi_common_slopes": (M, vc, "common", False)}.items():
            r, p_ = estimate(S, MM, nb, vc=v, mode=mode, profile=prof); pt_own = MM.pretrend(vc=v, mode=mode); pt2 = MM.pretrend(vc=v, mode=mode, two=True)
            pt = pt_own; src = "a_r6_mvi.MVI.pretrend (s04 bin definition; pyfixest cannot fit the penalised slopes)"
            if name == "imp_s24":
                fs = S["fit"](); pt = A.imputation_pretrend(dict(S, window=FIT_LO), fs, B); pt2 = A.imputation_pretrend(S, fs, B)
                src = "s04_analysis.imputation_pretrend with window = 24 (pyfixest)"; r["pretrend_p_own_sandwich_check"] = pt_own.get("p_joint")
            r.update(sample=s, variant=name, label=LABEL, primary=bool(name == "mvi"), pretrend_p=pt.get("p_joint"), pretrend_terms=pt.get("terms"),
                     pretrend_coef=pt.get("coef"), pretrend_se=pt.get("se"), pretrend_df=pt.get("df"), fit_window="s-24..s-4",
                     pretrend_definition="s04 imputation_pretrend definition: 3-month bins from -24 to -4, earliest bin (-24..-22) = reference, joint Wald on the six leads",
                     pretrend_source=src, pretrend_p_two_leads_r5=pt2.get("p_joint"), pretrend_two_leads_coef=pt2.get("coef"))
            res[name] = dict(r, pretrend=pt, pretrend_two_leads_r5=pt2); att_rows.append(r)
            if p_:
                for q in p_: ev_rows.append(dict(use=s, carrier="elec", spec=name, label=LABEL, **q, n_control=r["n_control"], pretrend_p=r["pretrend_p"],
                                                 note="pre bins are in-sample residuals (centred on zero by construction; shape only)" if q["kind"] == "pre_in_sample" else "imputed gaps"))
            print(f"  {s} {name}: ATT {100 * r['pct']:+.1f}% ({100 * r['pct_lo']:+.1f}, {100 * r['pct_hi']:+.1f}); T {r['n_treated']}, C {r['n_control']}; "
                  f"pre-trend p {r['pretrend_p']:.4f} (two leads, R5: {r['pretrend_p_two_leads_r5']:.4f}); boot sd {r['boot_sd_log']:.4f}; fit resid sd {r['resid_sd_fit']:.4f}; {time.time() - t0:.0f} s", flush=True)
        h = g7[G7ROW[s]]
        res["headline_G7"] = dict(pct=h["pct"], pct_ci=h["pct_ci"], att_log=h["att_log"], n_treated=h["n_treated"], n_control=h["n_control"], pretrend_p=h["pretrend_p"], fit_window="s-12..s-4")
        out["samples"][s] = dict(uses=list(uses), info=info, rows=res)
        print("  variance components:", json.dumps(vc, ensure_ascii=False, default=float), "\n  shrinkage:", json.dumps(shr, default=float), "\n  slopes:", json.dumps(ub, ensure_ascii=False, default=float), flush=True)
    pd.DataFrame(att_rows).to_csv(RES / "a_r6_mvi_att.csv", index=False); pd.DataFrame(ev_rows).to_csv(RES / "a_r6_mvi_event.csv", index=False)
    out["runtime_s"] = round(time.time() - t0, 1)
    (RES / "a_r6_mvi.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("written results/a_r6_mvi.json, a_r6_mvi_att.csv, a_r6_mvi_event.csv;", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
