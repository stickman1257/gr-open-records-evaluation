"""Pre-trend tests on imputation residuals."""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
from scipy import sparse, stats
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC
import s04_analysis as A

LABEL = "post hoc (added 11/16 after the final run)"
HC, DC = "보건", "어린이집"
SAMPLES = {"health_centre": (HC,), "pooled": (HC, DC)}
G7ROW = {"health_centre": "imputation_health_only", "pooled": "imputation_use"}
BINS = [(-12, -10, "pre"), (-9, -7, "pre"), (-6, -4, "pre"), (-3, -1, "ant"), (1, 3, "post"), (4, 6, "post"), (7, 9, "post"), (10, 12, "post")]
MBARS = [0.5, 1.0, 1.5, 2.0]
MIN_FIT = 9
pct = lambda a: float(np.exp(a) - 1) if a is not None and np.isfinite(a) else None
_mi = lambda v: (np.asarray(v, dtype=np.int64) // 100) * 12 + np.asarray(v, dtype=np.int64) % 100


class Model:
    def __init__(self, S, trend=None, leads=False):
        """trend: None or bool array over buildings (True = building-specific linear trend). leads: add the lead dummies -9..-7, -6..-4."""
        self.S = S; D = S["D"]; self.pi, self.ki = S["pi"], S["ki"]; self.nP, self.nK = len(S["pids"]), len(S["tks"])
        self.y = D.y.to_numpy(float); self.post, self.ant, self.isc = S["post"], S["ant"], S["isc"]
        self.F0 = ~self.post & ~self.ant; N = len(self.y); r = np.arange(N); self.rel = D.rel.to_numpy()
        m = _mi(D.ym.to_numpy()).astype(float)
        self.nfit = np.bincount(self.pi[self.F0], minlength=self.nP)
        cols = [sparse.csr_matrix((np.ones(N), (r, self.pi)), shape=(N, self.nP)), sparse.csr_matrix((np.ones(N), (r, self.ki)), shape=(N, self.nK))]
        self.n_trend = 0
        if trend is not None:
            trend = np.asarray(trend, bool); tb = np.bincount(self.pi[self.F0], m[self.F0], self.nP) / np.maximum(self.nfit, 1)
            tix = np.full(self.nP, -1); tix[trend] = np.arange(trend.sum()); k = trend[self.pi]
            cols.append(sparse.csr_matrix(((m[k] - tb[self.pi[k]]) / 12.0, (r[k], tix[self.pi[k]])), shape=(N, int(trend.sum()))))
            self.n_trend = int(trend.sum())
        self.n_lead = 0
        if leads:
            L = np.zeros((N, 2)); tr = ~self.isc & self.F0
            L[tr & (self.rel >= -9) & (self.rel <= -7), 0] = 1; L[tr & (self.rel >= -6) & (self.rel <= -4), 1] = 1
            cols.append(sparse.csr_matrix(L)); self.n_lead = 2
        self.X = sparse.hstack(cols).tocsr(); self.p = self.X.shape[1]

    def fit(self, w=None, min_ctrl=1):
        w = np.ones(self.nP) if w is None else np.asarray(w, float)
        wr = w[self.pi]; F = self.F0 & (wr > 0); XF = self.X[F]; wf = wr[F]
        G = (XF.T @ XF.multiply(wf[:, None])).toarray(); rhs = XF.T @ (wf * self.y[F])
        act = np.diag(G) > 0; Ga = G[np.ix_(act, act)]
        lam, V = np.linalg.eigh(Ga); pos = lam > 1e-10 * lam.max()
        ba = V[:, pos] @ ((V[:, pos].T @ rhs[act]) / lam[pos])
        beta = np.zeros(self.p); beta[act] = ba
        pred = self.X @ beta
        bad = np.asarray(abs(self.X[:, ~act]).sum(1)).ravel() > 0 if (~act).any() else np.zeros(len(self.y), bool)
        Nn = V[:, ~pos]
        if Nn.shape[1]: bad |= np.abs(self.X[:, act] @ Nn).max(1) > 1e-6
        nctrl = np.bincount(self.ki[F & self.isc], minlength=self.nK); sup = (nctrl >= min_ctrl)[self.ki]
        res = self.y - pred
        ok = sup & ~bad & np.isfinite(res) & (wr > 0)
        return dict(res=res, ok=ok, wr=wr, F=F, nctrl=nctrl, beta=beta, lead=beta[self.p - self.n_lead:] if self.n_lead else None,
                    null_dim=int((~pos).sum()))

    def pretrend(self, f):
        """Lead bins -9..-7 and -6..-4 (reference -12..-10) added to the model on the fit sample; building-clustered Wald test."""
        F = f["F"]; XF = self.X[F][:, :self.p - self.n_lead]; N = int(F.sum())
        tr = (~self.isc)[F]; rel = self.rel[F]
        L = np.c_[(tr & (rel >= -9) & (rel <= -7)), (tr & (rel >= -6) & (rel <= -4))].astype(float)
        G = (XF.T @ XF).toarray(); lam, V = np.linalg.eigh(G); pos = lam > 1e-10 * lam.max(); rank = int(pos.sum())
        Z = np.c_[self.y[F], L]; XtZ = XF.T @ Z
        R = Z - XF @ (V[:, pos] @ ((V[:, pos].T @ XtZ) / lam[pos][:, None]))
        ry, RL = R[:, 0], R[:, 1:]
        keep = (RL ** 2).sum(0) > 1e-8
        if not keep.any(): return dict(error="lead terms are absorbed by the model")
        RL = RL[:, keep]; q = int(keep.sum())
        bread = np.linalg.inv(RL.T @ RL); b = bread @ (RL.T @ ry); e = ry - RL @ b
        g = self.pi[F]; sc = np.zeros((self.nP, q)); np.add.at(sc, g, RL * e[:, None]); Gn = int(np.unique(g).size)
        k_par = q + rank - Gn
        V_ = bread @ (sc.T @ sc) @ bread * (Gn / (Gn - 1)) * ((N - 1) / (N - k_par))
        W = float(b @ np.linalg.solve(V_, b)); se = np.sqrt(np.diag(V_))
        names = [n for n, kk in zip(["lead_-9_-7", "lead_-6_-4"], keep) if kk]
        return dict(reference="lead_-12_-10", terms=names, coef=[float(v) for v in b], se=[float(v) for v in se], wald=W, df=[q, Gn - 1],
                    p_joint=float(stats.f.sf(W / q, q, Gn - 1)), n_obs=N, n_buildings=Gn,
                    test="Wald / q against F(q, G-1); CRV1 by building, factor G/(G-1) x (N-1)/(N-k), k = leads + model rank - building effects")


def bin_masks(M, f):
    out = []
    for a, b, kind in BINS:
        inb = ~M.isc & (M.rel >= a) & (M.rel <= b)
        out.append(inb & (M.post if kind == "post" else M.ant if kind == "ant" else (f["F"] & ~M.post & ~M.ant)))
    return out


def run_att(S, M, nboot, name, sample, extra=None):
    f = M.fit(); a = S["att"](f)[0]; m = M.post & f["ok"]
    rng = np.random.default_rng(0); bs = []
    for _ in range(nboot):
        fb = M.fit(A.boot_weights(S, rng)); bs.append(S["att"](fb)[0])
    bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan))
    pt = M.pretrend(f)
    if S.get("_fit_lo", 12) != 12 and "reference" in pt: pt["reference"] = f"all earlier fit months (s-{S['_fit_lo']}..s-10)"
    gap = S["att"](f, M.ant)[1] if M.ant.any() else np.nan
    tb = pd.DataFrame({"p": M.pi[m], "r": f["res"][m]}).groupby("p").r.mean()
    r = dict(sample=sample, variant=name, label=LABEL, att_log=float(a), ci_log_lo=float(lo), ci_log_hi=float(hi), pct=pct(a), pct_lo=pct(lo), pct_hi=pct(hi),
             ci_excludes_zero=bool((lo > 0) or (hi < 0)), boot_sd_log=float(bs.std(ddof=1)) if len(bs) > 1 else None,
             n_treated=int(pd.unique(M.pi[m]).size), n_control=int(pd.unique(M.pi[f["F"] & M.isc]).size),
             n_treated_in_sample=int(pd.unique(M.pi[~M.isc]).size), support_rows=float(m.sum() / max(int(M.post.sum()), 1)),
             n_trend_terms=M.n_trend, n_buildings=M.nP, n_buildings_without_trend=int(M.nP - M.n_trend) if M.n_trend else None,
             null_dim=f["null_dim"], nboot=int(nboot), nboot_ok=int(len(bs)),
             pretrend_p=pt.get("p_joint"), pretrend_coef=pt.get("coef"), pretrend_se=pt.get("se"), pretrend_terms=pt.get("terms"),
             pretrend_error=pt.get("error"), pretrend_reference=pt.get("reference"), anticipation_gap_log=float(gap), median_bld_log=float(tb.median()), sd_bld_log=float(tb.std()),
             fit_window=f"s-{S.get('_fit_lo', 12)}..s-{S['anticip'] + 1}")
    if extra: r.update(extra)
    print(f"  {sample} {name}: ATT {100 * r['pct']:+.1f}% ({100 * r['pct_lo']:+.1f}, {100 * r['pct_hi']:+.1f}); T {r['n_treated']}, C {r['n_control']}; "
          f"trend terms {M.n_trend}; pre-trend p {r['pretrend_p']}; draws {len(bs)}", flush=True)
    return r, pt


def rm_lp(Mbar, maxd, steps, l):
    """Largest l'delta_post over Delta^RM by linear programming (delta at chain positions 1..max(steps); position 0 = reference)."""
    n = int(max(steps)); c = np.zeros(n)
    for lj, sj in zip(l, steps): c[int(sj) - 1] -= lj
    Aub, bub = [], []
    for t in range(n):
        r = np.zeros(n); r[t] = 1
        if t > 0: r[t - 1] = -1
        Aub += [r, -r]; bub += [Mbar * maxd, Mbar * maxd]
    o = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), bounds=[(None, None)] * n, method="highs")
    return float(-o.fun)


def rm_vectors(Mbar, c):
    """b = (b_P1, b_P2, b_A, b_Q1..b_Q4) re-based on P3. Returns the 4 vectors of the upper end and of the lower end."""
    lp = np.r_[0, 0, 0, np.full(4, 0.25)]; D1 = np.r_[-1, 1, 0, 0, 0, 0, 0.0]; D2 = np.r_[0, -1, 0, 0, 0, 0, 0.0]
    up = [lp + Mbar * c * s * D for D in (D1, D2) for s in (1, -1)]; lo = [lp - Mbar * c * s * D for D in (D1, D2) for s in (1, -1)]
    return np.array(up), np.array(lo)


def rm_interval(b, Sig, Mbar, c, alpha=0.05, nsim=200000, seed=1):
    up, lo = rm_vectors(Mbar, c); rng = np.random.default_rng(seed)
    def side(Av, upper):
        se = np.sqrt(np.einsum("ki,ij,kj->k", Av, Sig, Av)); V = Av @ Sig @ Av.T; C = V / np.outer(se, se)
        lam, U = np.linalg.eigh(C); Lh = U * np.sqrt(np.clip(lam, 0, None))
        Z = rng.standard_normal((nsim, len(se))) @ Lh.T
        q = float(np.quantile((-Z).max(1) if upper else Z.max(1), 1 - alpha / 2))
        return (float((Av @ b + q * se).max()), q) if upper else (float((Av @ b - q * se).min()), q)
    u, qu = side(up, True); l_, ql = side(lo, False)
    return dict(set_lo=float((lo @ b).min()), set_hi=float((up @ b).max()), ci_lo=l_, ci_hi=u, q_upper=qu, q_lower=ql)


def rm_analysis(b, draws, steps, tag):
    """b: point (7,), draws: (nb, 7), both re-based on P3. steps: steps from P3 to Q1..Q4."""
    l = np.full(4, 0.25); c = float(l @ np.asarray(steps, float)); Sig = np.cov(draws, rowvar=False)
    D1, D2 = b[1] - b[0], -b[1]; maxd = max(abs(D1), abs(D2)); theta = float(b[3:].mean())
    rows = []
    for Mb in [0.0] + MBARS:
        r = rm_interval(b, Sig, Mb, c)
        lpv = rm_lp(Mb, maxd, steps, l) if Mb > 0 else 0.0
        assert abs(lpv - Mb * maxd * c) < 1e-9, "LP and closed form differ"
        assert abs(r["set_hi"] - (theta + lpv)) < 1e-9 and abs(r["set_lo"] - (theta - lpv)) < 1e-9
        rows.append(dict(Mbar=Mb, **r, ci_excludes_zero=bool(r["ci_hi"] < 0 or r["ci_lo"] > 0), set_excludes_zero=bool(r["set_hi"] < 0 or r["set_lo"] > 0),
                         pct_set=[pct(r["set_lo"]), pct(r["set_hi"])], pct_ci=[pct(r["ci_lo"]), pct(r["ci_hi"])], worst_case_bias_log=float(lpv)))
    grid = np.round(np.arange(0, 3.0001, 0.01), 2); bd = None
    for Mb in grid:
        r = rm_interval(b, Sig, float(Mb), c, nsim=50000)
        if r["ci_hi"] < 0 or r["ci_lo"] > 0: bd = float(Mb)
        else: break
    se_theta = float(np.sqrt(np.r_[0, 0, 0, l] @ Sig @ np.r_[0, 0, 0, l]))
    return dict(chain_input=tag, steps_P3_to_Q=[int(s) for s in steps], c=c, theta_rebased_log=theta, theta_rebased_pct=pct(theta), se_theta_log=se_theta,
                b_rebased=dict(zip(["P1", "P2", "A", "Q1", "Q2", "Q3", "Q4"], [float(v) for v in b])),
                se_b=dict(zip(["P1", "P2", "A", "Q1", "Q2", "Q3", "Q4"], [float(v) for v in np.sqrt(np.diag(Sig))])),
                pre_differences=dict(D1_P2_minus_P1=float(D1), D2_P3_minus_P2=float(D2), max_abs=float(maxd)),
                breakdown_Mbar_ci=bd if bd is not None else None,
                breakdown_note="largest Mbar on a 0.01 grid (0..3) whose 95% interval excludes zero; null = the interval includes zero already at Mbar = 0",
                breakdown_Mbar_identified_set=float(abs(theta) / (c * maxd)) if maxd > 0 else None, by_Mbar=rows, n_draws=int(len(draws)))


def synthetic_checks():
    """Known answer: b_pre = (0.03, 0.01), reference 0 -> D = (-0.02, -0.01), max 0.02; post = -0.10 in every bin; consecutive bins"""
    b = np.r_[0.03, 0.01, -0.05, -0.10, -0.10, -0.10, -0.10]; steps = [2, 3, 4, 5]; l = np.full(4, 0.25)
    lpv = rm_lp(1.0, 0.02, steps, l); up, lo = rm_vectors(1.0, 3.5)
    out = dict(known_answer=dict(expected_set=[-0.17, -0.03], lp_bias=lpv, set=[float((lo @ b).min()), float((up @ b).max())],
                                 expected_breakdown_identified_set=0.10 / (0.02 * 3.5),
                                 ok=bool(abs(lpv - 0.07) < 1e-9 and abs((up @ b).max() + 0.03) < 1e-9 and abs((lo @ b).min() + 0.17) < 1e-9)))
    lpg = rm_lp(1.0, 0.02, [5, 6, 7, 8], l)
    out["known_answer_gap"] = dict(expected_bias=0.02 * 6.5, lp_bias=lpg, ok=bool(abs(lpg - 0.13) < 1e-9))
    rng = np.random.default_rng(7); Av = rng.standard_normal((7, 7)); Sig = Av @ Av.T / 7 * 0.03 ** 2
    cov = {}
    for nm, bpre in [("clear_pre_trend", (0.06, 0.02)), ("no_pre_trend_kink", (0.0, 0.0)), ("two_equal_differences", (0.04, 0.02))]:
        maxd = max(abs(bpre[1] - bpre[0]), abs(bpre[1])); tau = -0.10
        for edge, sgn in [("upper", -1), ("lower", +1)]:
            btrue = np.r_[bpre[0], bpre[1], sgn * 1 * maxd * 1, tau + sgn * maxd * np.array(steps)]
            hit = 0; R = 400
            for i in range(R):
                bh = rng.multivariate_normal(btrue, Sig); r = rm_interval(bh, Sig, 1.0, 3.5, nsim=20000, seed=i)
                hit += (r["ci_lo"] <= tau <= r["ci_hi"])
            cov[f"{nm}_{edge}_edge"] = hit / R
    out["coverage_Mbar1"] = dict(replications=400, nominal=0.95, coverage=cov, ok=bool(min(cov.values()) >= 0.93),
                                 note="truth at the edge of the identified set; coverage >= 0.95 expected (conservative), Monte Carlo s.e. about 0.011")
    return out


def profile_draws(S, M, nboot, lsqr=True):
    """Bin estimates (cohort-weighted, as a_imp_event.py) and bootstrap draws. lsqr=True: the s04 solver (exact reproduction of a_imp_event);"""
    fit = (lambda w=None: S["fit"](w, tol=1e-9) if w is not None else S["fit"]()) if lsqr else (lambda w=None: M.fit(w))
    f = fit(); est = [S["att"](f, m)[0] for m in bin_masks(M, f)]; lead0 = None if lsqr else f["lead"]
    rng = np.random.default_rng(0); bs, ld = [], []
    for _ in range(nboot):
        fb = fit(A.boot_weights(S, rng)); bs.append([S["att"](fb, m)[0] for m in bin_masks(M, fb)])
        if not lsqr: ld.append(fb["lead"])
    return np.array(est, float), np.array(bs, float), lead0, (np.array(ld, float) if ld else None)


def main():
    NBH = int(os.environ.get("IMP_NBOOT_HEAD", "1000")); NBP = int(os.environ.get("R5_NBOOT_POOLED", "200")); t0 = time.time()
    B, E = A.load()
    g7 = {r.get("spec"): r for r in json.loads((RES / "G7_imputation.json").read_text(encoding="utf-8")) if isinstance(r, dict) and r.get("carrier") == "elec"}
    ev = pd.read_csv(RES / "a_imp_event.csv"); ev = ev[ev.carrier == "elec"]
    out = dict(label=LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), definition=__doc__,
               panel_sha256=hashlib.sha256((PROC / "energy_monthly.csv").read_bytes()).hexdigest(),
               decision_rule="(a) primary variant excludes zero AND breakdown Mbar (primary chain and input) >= 1 -> 'holds under violations as large as "
                             "the pre-trend'; otherwise 'sensitive to parallel trends' (PLAN.md 11/16 09:55, recorded before these computations)",
               package_decision="own implementation: honestdid 0.1.1 requires torch + cvxpy (not installed); diff-diff 4.0.0 treats the largest pre-period "
                                "difference as known in its Delta^RM interval. Neither was installed into the project environment.",
               synthetic_checks=synthetic_checks(), samples={})
    print("synthetic checks:", json.dumps({k: v.get("ok") for k, v in out["synthetic_checks"].items()}), out["synthetic_checks"]["coverage_Mbar1"]["coverage"], flush=True)
    att_rows, rm_rows = [], []
    for s, uses in SAMPLES.items():
        nb = NBH if s == "health_centre" else NBP; kw = dict(A.PRIMARY_IMP, uses=uses); res = {}
        print(f"== {s} ({nb} draws for (a)) ==", flush=True)
        S0 = A.imputation_setup(B, E, "elec", **kw); M0 = Model(S0)
        f_s04 = S0["fit"](); f_d = M0.fit(); a_s04 = S0["att"](f_s04)[0]; a_d = S0["att"](f_d)[0]
        chk = dict(att_s04=float(a_s04), att_dense=float(a_d), abs_diff=float(abs(a_s04 - a_d)), ok_masks_identical=bool((f_s04["ok"] == f_d["ok"]).all()),
                   G7_att_log=g7[G7ROW[s]]["att_log"], G7_pretrend_p=g7[G7ROW[s]]["pretrend_p"])
        print("  solver check:", chk, flush=True)
        nfit = M0.nfit; pids = S0["pids"]; few = set(pids[nfit < MIN_FIT]); isc_p = S0["ctrl_p"]
        counts = dict(n_buildings=int(M0.nP), n_treated=int((~isc_p).sum()), n_control=int(isc_p.sum()), treated_lt9_fit_months=int(((nfit < MIN_FIT) & ~isc_p).sum()),
                      control_lt9_fit_months=int(((nfit < MIN_FIT) & isc_p).sum()), treated_fit_months=pd.Series(nfit[~isc_p]).value_counts().sort_index().to_dict(),
                      control_fit_months_median=float(np.median(nfit[isc_p])))
        r, _ = run_att(S0, M0, nb, "no_trend_headline", s); res["no_trend_headline"] = r; chk["pretrend_p_dense"] = r["pretrend_p"]
        S1 = A.imputation_setup(B, E, "elec", **dict(kw, drop=few)); M1 = Model(S1, trend=np.ones(len(S1["pids"]), bool))
        assert (M1.nfit >= MIN_FIT).all()
        r, pt = run_att(S1, M1, nb, "trend_ge9_only", s, dict(primary=True)); res["trend_ge9_only"] = r; res["trend_ge9_only"]["pretrend_detail"] = pt
        r, _ = run_att(S1, Model(S1), nb, "no_trend_same_sample", s); res["no_trend_same_sample"] = r
        r, _ = run_att(S0, Model(S0, trend=nfit >= MIN_FIT), nb, "trend_ge9_others_fe", s); res["trend_ge9_others_fe"] = r
        try:
            S2 = A.imputation_setup(B, E, "elec", **dict(kw, fit_lo=24)); S2["_fit_lo"] = 24; M2 = Model(S2)
            few2 = set(S2["pids"][M2.nfit < MIN_FIT])
            if few2: S2 = A.imputation_setup(B, E, "elec", **dict(kw, fit_lo=24, drop=few2)); S2["_fit_lo"] = 24
            M2t = Model(S2, trend=np.ones(len(S2["pids"]), bool))
            ex = dict(extra=True, treated_fit_months_median=float(np.median(M2t.nfit[~S2["ctrl_p"]])))
            r, _ = run_att(S2, M2t, nb, "trend_fit_s24", s, ex); res["trend_fit_s24"] = r
            r, _ = run_att(S2, Model(S2), nb, "no_trend_fit_s24", s, dict(extra=True)); res["no_trend_fit_s24"] = r
        except Exception as ex_:
            res["trend_fit_s24"] = dict(error=f"{type(ex_).__name__}: {ex_}"[:200]); print("  fit_lo=24 variant failed:", res["trend_fit_s24"], flush=True)
        att_rows += [v for v in res.values() if "att_log" in v]
        idx = lambda v: (int(v) // 100) * 12 + int(v) % 100
        f0 = S0["fit"](); T0 = set(pids[pd.unique(S0["pi"][S0["post"] & f0["ok"]])]); Th = S0["T"][S0["T"].pid.isin(T0)]
        cm = float(np.median([idx(b_) - idx(a_) + 1 for a_, b_ in zip(Th.start_ym, Th.end_ym)])); g = int(round(cm / 3))
        est, bs, _, _ = profile_draws(S0, M0, NBH, lsqr=True)
        evs = ev[ev.use == s].set_index("term"); terms = [f"ev_{a}_{b}" for a, b, _ in BINS]
        repro = dict(max_abs_diff_est_log=float(np.abs(est - evs.est_log.reindex(terms).to_numpy()).max()),
                     max_abs_diff_ci_log=float(max(np.abs(np.percentile(bs, 2.5, axis=0) - evs.lo_log.reindex(terms).to_numpy()).max(),
                                                   np.abs(np.percentile(bs, 97.5, axis=0) - evs.hi_log.reindex(terms).to_numpy()).max())))
        print("  a_imp_event.csv reproduced:", repro, flush=True)
        okd = np.isfinite(bs).all(1); reb = lambda v: np.delete(v - v[..., [2]], 2, axis=-1)
        b1, d1 = reb(est), reb(bs[okd])
        ML_ = Model(S0, leads=True); est2, bs2, lead0, ld = profile_draws(S0, ML_, NBH, lsqr=False)
        full2 = np.r_[0.0, lead0, est2[3:]]; fulld = np.c_[np.zeros(len(ld)), ld, bs2[:, 3:]]; ok2 = np.isfinite(fulld).all(1)
        b2, d2 = reb(full2), reb(fulld[ok2])
        rm = {}
        for inp, (bb, dd) in {"profile_bins": (b1, d1), "lead_coefficients": (b2, d2)}.items():
            for ch, steps in {"bins_consecutive": [2, 3, 4, 5], "with_construction_gap": [2 + g + j for j in range(4)]}.items():
                r = rm_analysis(bb, dd, steps, f"{inp}|{ch}"); r.update(input=inp, chain=ch, primary=bool(inp == "profile_bins" and ch == "bins_consecutive"))
                rm[f"{inp}|{ch}"] = r
                for q in r["by_Mbar"]:
                    rm_rows.append(dict(sample=s, input=inp, chain=ch, primary=r["primary"], label=LABEL, Mbar=q["Mbar"], theta_log=r["theta_rebased_log"],
                                        set_lo_log=q["set_lo"], set_hi_log=q["set_hi"], ci_lo_log=q["ci_lo"], ci_hi_log=q["ci_hi"],
                                        set_lo_pct=q["pct_set"][0], set_hi_pct=q["pct_set"][1], ci_lo_pct=q["pct_ci"][0], ci_hi_pct=q["pct_ci"][1],
                                        ci_excludes_zero=q["ci_excludes_zero"], breakdown_Mbar_ci=r["breakdown_Mbar_ci"],
                                        breakdown_Mbar_identified_set=r["breakdown_Mbar_identified_set"], max_pre_diff_log=r["pre_differences"]["max_abs"],
                                        c=r["c"], n_draws=r["n_draws"]))
                print(f"  RM {inp} | {ch}: theta {100 * r['theta_rebased_pct']:+.1f}%, max pre diff {r['pre_differences']['max_abs']:.4f}, c {r['c']}, breakdown (CI) "
                      f"{r['breakdown_Mbar_ci']}, (set) {r['breakdown_Mbar_identified_set']:.2f}; " +
                      "; ".join(f"M{q['Mbar']}: [{100 * q['pct_ci'][0]:+.1f}, {100 * q['pct_ci'][1]:+.1f}]" for q in r["by_Mbar"]), flush=True)
        prim_a = res["trend_ge9_only"]; prim_b = rm["profile_bins|bins_consecutive"]; bd = prim_b["breakdown_Mbar_ci"]
        holds = bool(prim_a["ci_excludes_zero"] and bd is not None and bd >= 1.0)
        verdict = ("the conclusion holds under violations as large as the pre-trend" if holds else "the conclusion is sensitive to parallel trends")
        det = (f"(a) building-trend imputation {100 * prim_a['pct']:+.1f}% ({100 * prim_a['pct_lo']:+.1f}, {100 * prim_a['pct_hi']:+.1f}), "
               f"{'excludes' if prim_a['ci_excludes_zero'] else 'includes'} zero; (b) breakdown Mbar = {bd if bd is not None else 'none (interval includes zero at Mbar = 0)'} "
               f"({'>= 1' if (bd is not None and bd >= 1) else '< 1'})")
        out["samples"][s] = dict(uses=list(uses), solver_check=chk, fit_month_counts=counts, a_trend_imputation=res,
                                 b_relative_magnitudes=dict(median_construction_months=cm, gap_periods=g, reproduction_of_a_imp_event=repro,
                                                            lead_coefficients_point=[float(v) for v in lead0], results=rm),
                                 verdict=verdict, verdict_detail=det, verdict_inputs=dict(a_ci_excludes_zero=prim_a["ci_excludes_zero"], breakdown_Mbar=bd))
        print(f"  VERDICT {s}: {verdict} | {det}; {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(att_rows).drop(columns=["pretrend_detail"], errors="ignore").to_csv(RES / "a_r5_pretrend_att.csv", index=False)
    pd.DataFrame(rm_rows).to_csv(RES / "a_r5_pretrend_rm.csv", index=False)
    out["runtime_s"] = round(time.time() - t0, 1)
    (RES / "a_r5_pretrend.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("written results/a_r5_pretrend.json, a_r5_pretrend_att.csv, a_r5_pretrend_rm.csv;", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
