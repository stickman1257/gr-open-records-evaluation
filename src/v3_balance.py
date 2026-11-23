"""Entropy-balancing reweighting of comparison buildings."""
import os, sys, json, time, hashlib, warnings, multiprocessing as mp
import numpy as np, pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix, csr_matrix, hstack as sp_hstack
from scipy.sparse.linalg import lsqr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC
import s04_analysis as A

LABEL = "post hoc robustness (verification round V, after the final run)"
HC, DC = "보건", "어린이집"
SAMPLES = {"health_centre": (HC,), "pooled": (HC, DC)}
CR = ("control", "aux_2023")
COVS = ["built", "log_area", "zone_c1", "zone_south", "elec_mean", "season", "slope"]
COV_LABEL = {"built": "Built year", "log_area": "Log floor area", "zone_c1": "Coldest climate zone", "zone_south": "Southern climate zone",
             "elec_mean": "Baseline electricity", "season": "Winter/shoulder ratio", "slope": "Baseline slope"}
SETS = {"B1": [0, 1, 2, 3, 4, 5], "B2": [0, 1, 2, 3, 4, 5, 6]}
SEASON_COL = 5
BINS = [(-12, -10, "pre"), (-9, -7, "pre"), (-6, -4, "pre"), (-3, -1, "ant"), (1, 3, "post"), (4, 6, "post"), (7, 9, "post"), (10, 12, "post")]
NBOOT = int(os.environ.get("V3_NBOOT", "1000")); NPROC = int(os.environ.get("V3_NPROC", "10"))
MIN_BASE = 9; ESS_MIN = 20.0; SHARE_MAX = 0.10
_mi = lambda v: (np.asarray(v, dtype=np.int64) // 100) * 12 + np.asarray(v, dtype=np.int64) % 100
pct = lambda a: float(np.exp(a) - 1) if a is not None and np.isfinite(a) else None
kish = lambda w: float(w.sum() ** 2 / (w ** 2).sum()) if (w > 0).any() else 0.0


def ebal(X, q, target, tol=1e-8, maxit=200):
    """min sum w log(w / q) s.t. sum w x = target, sum w = 1. Newton on the dual. Returns w or None (infeasible / not converged)."""
    if len(X) == 0 or not np.isfinite(target).all(): return None
    mu = q @ X; sd = np.sqrt(q @ (X - mu) ** 2); keep = sd > 1e-12
    if (np.abs(mu[~keep] - target[~keep]) > 1e-9).any(): return None
    if not keep.any(): return q.copy()
    Z = (X[:, keep] - target[keep]) / sd[keep]; lam = np.zeros(Z.shape[1])
    def obj(l):
        e = Z @ l; m = e.max(); return m + np.log(q @ np.exp(e - m))
    for _ in range(maxit):
        e = Z @ lam; e -= e.max(); w = q * np.exp(e); w /= w.sum(); g = Z.T @ w
        if np.abs(g).max() < tol: return w
        H = (Z * w[:, None]).T @ Z - np.outer(g, g)
        try: step = np.linalg.solve(H + 1e-12 * np.eye(len(g)), g)
        except np.linalg.LinAlgError: return None
        f0 = obj(lam); t = 1.0
        while t > 1e-10 and not obj(lam - t * step) <= f0 - 1e-4 * t * (g @ step): t /= 2
        lam = lam - t * step
        if not np.isfinite(lam).all() or np.abs(lam).max() > 1e3: return None
    return None


def build_covariates(B, E, S):
    """Per type: treated covariates (own window) and control covariates per cohort (cohort window). Arrays index S['pids']."""
    Ec = A.carrier_frame(E, "elec"); Ec = Ec[Ec.pid.isin(set(S["pids"]))]
    Bi = B.drop_duplicates("pid").set_index("pid").reindex(S["pids"])
    pos = {p: i for i, p in enumerate(S["pids"])}; nP = len(S["pids"])
    m0 = int(_mi(Ec.ym).min()); nT = int(_mi(Ec.ym).max()) - m0 + 1
    M = np.full((nP, nT), np.nan); M[Ec.pid.map(pos).to_numpy(), _mi(Ec.ym) - m0] = (Ec.kwh / Ec.gr_gfa).to_numpy()
    start = _mi(Bi.start_ym.astype(int).to_numpy()) - m0; ctrl = S["ctrl_p"]; use = S["use_p"]
    cols = np.arange(nT)
    for i in range(nP): M[i, cols >= (start[i] - 3 if ctrl[i] else start[i])] = np.nan
    moy = (cols + m0 - 1) % 12 + 1
    prof = {}
    for u in np.unique(use):
        L = np.log(M[ctrl & (use == u)]); prof[u] = np.array([np.nanmean(L[:, moy == k]) if np.isfinite(L[:, moy == k]).any() else 0.0 for k in range(1, 13)])
    static = np.c_[pd.to_numeric(Bi.built_year.astype(str).str[:4], errors="coerce").to_numpy(float), np.log(Bi.area_m2.to_numpy(float)),
                   (Bi.zone == "중부1_춘천").to_numpy(float), (Bi.zone == "남부_부산").to_numpy(float)]

    def window(rows, lo, u):
        """rows: building indices; lo: first window month (index, scalar or array). -> n_obs, mean, season, slope."""
        lo = np.broadcast_to(np.asarray(lo), (len(rows),)); out = np.full((len(rows), 4), np.nan)
        for r, (i, l) in enumerate(zip(rows, lo)):
            ix = np.arange(l, l + 12); ok = (ix >= 0) & (ix < nT); v = np.full(12, np.nan); v[ok] = M[i, ix[ok]]
            mo = (ix + m0 - 1) % 12 + 1; n = int(np.isfinite(v).sum()); out[r, 0] = n
            if n < MIN_BASE: continue
            out[r, 1] = np.nanmean(v); wi, sh = v[np.isin(mo, [12, 1, 2])], v[np.isin(mo, [4, 5, 10])]
            if np.isfinite(wi).any() and np.isfinite(sh).any(): out[r, 2] = np.nanmean(wi) / np.nanmean(sh)
            k = np.isfinite(v); z = np.log(v[k]) - prof[u][mo[k] - 1]; t = np.arange(12)[k] / 12.0
            out[r, 3] = float(((t - t.mean()) * (z - z.mean())).sum() / ((t - t.mean()) ** 2).sum())
        return out

    T = S["T"].copy(); T["cohort"] = ((T.end_ym // 100) * 10 + ((T.end_ym % 100) > 6)).astype(int)
    data = {}
    for u in np.unique(use):
        Tu = T[T.use_class == u]; ti = np.array([pos[p] for p in Tu.pid]); ci = np.flatnonzero(ctrl & (use == u))
        wt = window(ti, start[ti] - 15, u); Xt = np.c_[static[ti], wt[:, 1:]]
        cohorts = sorted(Tu.cohort.unique()); Xc, elig, cov_rows = {}, {}, []
        for c in cohorts:
            smin = int(_mi(int(Tu[Tu.cohort == c].start_ym.min())) - m0); wc = window(ci, smin - 15, u); X = np.c_[static[ci], wc[:, 1:]]
            Xc[c] = X; elig[c] = (wc[:, 0] >= MIN_BASE) & np.isfinite(X).all(1)
            wtc = window(ti[Tu.cohort.to_numpy() == c], smin - 15, u)
            cov_rows.append(dict(use=u, cohort=int(c), n_treated=int((Tu.cohort == c).sum()), earliest_start=int(Tu[Tu.cohort == c].start_ym.min()),
                                 latest_start=int(Tu[Tu.cohort == c].start_ym.max()),
                                 treated_ge9_in_cohort_calendar_window=int((wtc[:, 0] >= MIN_BASE).sum()),
                                 treated_ge9_in_own_window=int((wt[Tu.cohort.to_numpy() == c, 0] >= MIN_BASE).sum()),
                                 controls_in_type=int(len(ci)), controls_ge9=int((wc[:, 0] >= MIN_BASE).sum()), controls_eligible=int(elig[c].sum())))
        data[u] = dict(ti=ti, ci=ci, Xt=Xt, cohT=Tu.cohort.to_numpy(), cohorts=cohorts, Xc=Xc, elig=elig, coverage=cov_rows)
    return data


def cohort_weights(d, c, mT, mC, cols):
    """Weights over d['ci'] (sum 1) for cohort c; multiplicities mT (treated of the type), mC (controls of the type)."""
    sel = (d["cohT"] == c) & (mT > 0)
    if not sel.any(): return None
    Xt, mt = d["Xt"][sel], mT[sel]
    tgt = np.array([np.average(Xt[np.isfinite(Xt[:, k]), k], weights=mt[np.isfinite(Xt[:, k])]) if np.isfinite(Xt[:, k]).any() else np.nan for k in range(Xt.shape[1])])
    el = d["elig"][c] & (mC > 0); w = np.zeros(len(mC))
    if not el.any(): return w, "no_eligible_control", tgt
    q = mC[el] / mC[el].sum(); X = d["Xc"][c][el]
    for stage, cc in [("full", cols), ("no_season", [k for k in cols if k != SEASON_COL])]:
        r = ebal(X[:, cc], q, tgt[cc])
        if r is not None: w[el] = r; return w, stage, tgt
    w[el] = q; return w, "infeasible_uniform", tgt


def unit_weights(S, data, m, cols):
    """Building weights for the imputation fit (len = buildings) and the cohort weights. m = bootstrap multiplicities (ones = sample)."""
    w = np.asarray(m, float).copy(); cw = {}
    if cols is None: return w, cw
    for u, d in data.items():
        mT, mC = m[d["ti"]], m[d["ci"]]; W = np.zeros(len(mC)); ntot = mT.sum()
        for c in d["cohorts"]:
            r = cohort_weights(d, c, mT, mC, cols)
            if r is None: continue
            cw[(u, c)] = r; W += mT[d["cohT"] == c].sum() / ntot * r[0]
        w[d["ci"]] = W * mC[W > 0].sum()
    return w, cw


def bin_masks(S, f):
    D = S["D"]; rel = D.rel.to_numpy(); out = []
    for a, b, kind in BINS:
        inb = ~S["isc"] & (rel >= a) & (rel <= b)
        out.append(inb & (S["post"] if kind == "post" else S["ant"] if kind == "ant" else (f["F"] & ~S["post"] & ~S["ant"])))
    return out


def imp_stats(S, w, tol=None):
    f = S["fit"](w) if tol is None else S["fit"](w, tol=tol)
    return f, np.r_[S["att"](f)[0], [S["att"](f, mk)[0] for mk in bin_masks(S, f)]]


def fe_fit_w(Dw, x, w):
    """A.fe_fit with weights (pyfixest, CRV1 by building, same small-sample convention)."""
    import pyfixest as pf
    nm = [f"x{i}" for i in range(len(x))]
    d = pd.DataFrame({"y": Dw["y"].to_numpy(float), "unit_fe": Dw["unit_fe"].to_numpy(), "time_fe": Dw["time_fe"].to_numpy(), "cl": Dw["pid"].astype(str).to_numpy(), "w": np.asarray(w, float)})
    for n_, c in zip(nm, x): d[n_] = Dw[c].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pf.feols("y ~ " + " + ".join(nm) + " | unit_fe + time_fe", data=d, vcov={"CRV1": "cl"}, weights="w",
                     demeaner=pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100000))
    names = list(m._coefnames); k = len(x); b = np.full(k, np.nan); V = np.full((k, k), np.nan); p = [nm.index(n_) for n_ in names]
    b[p] = m.coef().reindex(names).to_numpy(float); V[np.ix_(p, p)] = np.asarray(m._vcov); G = int(d.cl.nunique())
    return dict(b=b, cov=V, G=G, nobs=int(m._N), crit=float(stats.t.ppf(0.975, G - 1)))


def pretrend_w(S, f):
    """s04 imputation_pretrend on the weighted fit sample (weights = f['wr'])."""
    D = S["D"][f["F"]].copy(); D["w"] = f["wr"][f["F"]]; x = []
    for a, b in [(-9, -7), (-6, -4)]:
        c = f"lead_{a}_{b}"; D[c] = ((D.ctrl == 0) & (D.rel >= a) & (D.rel <= b)).astype(int); x.append(c)
    D = D[D.groupby("pid").y.transform("size") > 1]; D = D.assign(unit_fe=D.pid, time_fe=D.tk).reset_index(drop=True)
    F = fe_fit_w(D, x, D.w); b, V = F["b"], F["cov"]; q = len(b); W = float(b @ np.linalg.solve(V, b))
    return dict(terms=x, coef=b.tolist(), cov=V.tolist(), se=np.sqrt(np.diag(V)).tolist(), wald=W, df=[q, F["G"] - 1],
                p_joint=float(stats.f.sf(W / q, q, F["G"] - 1)), n_obs=F["nobs"], n_buildings=F["G"])


def stacked_prepare(B, E, uses, event):
    kw = dict(uses=uses, ctrl_roles=CR, fe="use", ctrl_trim=3, anticip=0 if event else 3)
    Sg, x = A.stacked_did(B, E, "elec", event=event, return_sample=True, **kw)
    S0, _ = A.stacked_did(B, E, "elec", event=event, return_sample=True, support=False, **kw)
    Sg = Sg.reset_index(drop=True); stacks = sorted(Sg["stack"].unique())
    if not event:
        x = []
        for c in stacks:
            col = f"post_c{int(c)}"; Sg[col] = ((Sg.post == 1) & (Sg["stack"] == c)).astype(int); x.append(col)
    pids, pi = np.unique(Sg.pid.to_numpy(), return_inverse=True)
    return dict(S=Sg, x=x, X=Sg[x].to_numpy(float), y=Sg.y.to_numpy(float), ui=pd.factorize(Sg.unit_fe)[0], ti=pd.factorize(Sg.time_fe)[0],
                treat=Sg.treat.to_numpy() == 1, post=Sg.post.to_numpy() == 1, stack=Sg["stack"].to_numpy().astype(int), stacks=[int(c) for c in stacks],
                pids=pids, pi=pi, n_treat_rows0=int((S0.treat == 1).sum()), event=event)


def stack_row_weights(P, Simp, data, m_imp, cols):
    """Row weights of the stacked sample: multiplicity x (1 for treated rows; cohort weight scaled to mean 1 for control rows of the stack)."""
    pos = {p: i for i, p in enumerate(Simp["pids"])}; ip = np.array([pos.get(p, -1) for p in P["pids"]])
    assert (ip >= 0).all(), "stacked building outside the imputation frame"
    m = m_imp[ip]; w = m[P["pi"]].astype(float)
    if cols is None: return w, {}
    cw = {}; fac = np.zeros((len(P["pids"]), len(P["stacks"])))
    inv = {int(g): k for k, g in enumerate(Simp["pids"])} if False else None
    loc = {int(i): k for k, i in enumerate(ip)}
    for u, d in data.items():
        mT, mC = m_imp[d["ti"]], m_imp[d["ci"]]
        for c in d["cohorts"]:
            if int(c) not in P["stacks"]: continue
            r = cohort_weights(d, c, mT, mC, cols)
            if r is None: continue
            cw[(u, c)] = r; n_pos = mC[r[0] > 0].sum(); j = P["stacks"].index(int(c))
            for k, gi in enumerate(d["ci"]):
                if gi in loc and mC[k] > 0: fac[loc[gi], j] = r[0][k] * n_pos / mC[k]
    sj = np.searchsorted(P["stacks"], P["stack"]); wc = fac[P["pi"], sj]
    return np.where(P["treat"], w, w * wc), cw


def stacked_fit(P, w):
    """Weighted two-way FE by sparse LSQR after the support rule. -> coefficient vector, kept-row mask."""
    keep = w > 0; nK = P["ti"].max() + 1
    has_c = np.bincount(P["ti"][keep & ~P["treat"]], minlength=nK) > 0
    keep &= ~P["treat"] | has_c[P["ti"]]
    n = int(keep.sum()); sw = np.sqrt(w[keep]); r = np.arange(n); k = P["X"].shape[1]
    ui = pd.factorize(P["ui"][keep])[0]; ti = pd.factorize(P["ti"][keep])[0]
    Fe = coo_matrix((np.r_[sw, sw], (np.r_[r, r], np.r_[ui, ui.max() + 1 + ti])), shape=(n, ui.max() + ti.max() + 2)).tocsr()
    Xd = P["X"][keep] * sw[:, None]; yv = P["y"][keep]; mu = np.average(yv, weights=w[keep])
    Mx = sp_hstack([csr_matrix(Xd), Fe]).tocsc(); nrm = np.sqrt(np.asarray(Mx.multiply(Mx).sum(0)).ravel()); nrm[nrm == 0] = 1
    sol = lsqr(Mx @ csr_matrix((1 / nrm, (np.arange(len(nrm)), np.arange(len(nrm))))), sw * (yv - mu), atol=1e-13, btol=1e-13, iter_lim=40000)[0] / nrm
    b = sol[:k]; b[np.asarray(np.abs(Xd).sum(0)).ravel() == 0] = np.nan
    return b, keep


def stacked_att(P, w):
    b, keep = stacked_fit(P, w); m = keep & P["post"]
    mult = np.zeros(len(P["pids"])); mult[P["pi"][m]] = w[m]
    nb = np.array([mult[np.unique(P["pi"][m & (P["stack"] == c)])].sum() for c in P["stacks"]]); ok = np.isfinite(b) & (nb > 0)
    return float((b[ok] * nb[ok]).sum() / nb[ok].sum()), b, keep


def stacked_pyfixest(P, w, keep):
    Sg = P["S"][keep].copy(); return fe_fit_w(Sg, P["x"], w[keep])


G = {}

def _work_imp(idx):
    S, data = G["S"], G["data"]; out = []
    for d in idx:
        m = G["mult"][d]; row = []
        for name, cols in G["specs"]:
            w, cw = unit_weights(S, data, m, cols); st = imp_stats(S, w, tol=1e-9)[1]
            row.append(np.r_[st, sum(1 for v in cw.values() if v[1] == "no_season"), sum(1 for v in cw.values() if v[1] not in ("full", "no_season"))])
        out.append(row)
    return np.array(out)

def _work_stk(idx):
    S, data, Pa, Pe = G["S"], G["data"], G["Pa"], G["Pe"]; out = []
    for d in idx:
        m = G["mult"][d]; row = []
        for name, cols in G["specs_stk"]:
            wa, _ = stack_row_weights(Pa, S, data, m, cols); we, _ = stack_row_weights(Pe, S, data, m, cols)
            row.append(np.r_[stacked_att(Pa, wa)[0], stacked_fit(Pe, we)[0]])
        out.append(row)
    return np.array(out)

def run_pool(fn, n):
    chunks = np.array_split(np.arange(n), max(1, min(NPROC * 4, n)))
    with mp.get_context("fork").Pool(NPROC) as p: res = p.map(fn, [c for c in chunks if len(c)])
    return np.concatenate(res, 0)


def smd_table(S, data, cws, sample):
    """Mixture SMD before / after for every covariate. Denominator sqrt((var_T + var_C) / 2), var_C of the unweighted eligible controls."""
    rows = []
    for u, d in data.items():
        nT = len(d["ti"]); pi_c = {c: (d["cohT"] == c).mean() for c in d["cohorts"]}
        for k, cov in enumerate(COVS):
            xt = d["Xt"][:, k]; xt = xt[np.isfinite(xt)]; mt, vt = xt.mean(), xt.var(ddof=1)
            mc0 = sum(pi_c[c] * d["Xc"][c][d["elig"][c], k].mean() for c in d["cohorts"] if d["elig"][c].any())
            ex2 = sum(pi_c[c] * (d["Xc"][c][d["elig"][c], k] ** 2).mean() for c in d["cohorts"] if d["elig"][c].any()); vc = ex2 - mc0 ** 2
            den = np.sqrt((vt + vc) / 2); r = dict(sample=sample, use=u, covariate=cov, label=COV_LABEL[cov], n_T=int(len(xt)), mean_T=float(mt), mean_C_before=float(mc0),
                                                   smd_before=float((mt - mc0) / den) if den > 0 else np.nan)
            for name, cw in cws.items():
                mc = sum(pi_c[c] * (cw[(u, c)][0] @ np.nan_to_num(d["Xc"][c][:, k])) for c in d["cohorts"])
                r[f"mean_C_{name}"] = float(mc); r[f"smd_{name}"] = float((mt - mc) / den) if den > 0 else np.nan
            rows.append(r)
    return rows


def cohort_table(data, cws, sample):
    rows = []
    for u, d in data.items():
        for c in d["cohorts"]:
            r = dict(sample=sample, use=u, cohort=int(c), n_treated=int((d["cohT"] == c).sum()), controls_eligible=int(d["elig"][c].sum()))
            for name, cw in cws.items():
                w, stage, _ = cw[(u, c)]; e = kish(w); sh = float(w.max()) if w.sum() > 0 else np.nan
                r.update({f"{name}_stage": stage, f"{name}_ess": e, f"{name}_max_share": sh, f"{name}_fragile": bool(e < ESS_MIN or sh > SHARE_MAX or stage != "full")})
            rows.append(r)
    return rows


def overlap(data, sample):
    """Logistic propensity score on the B1 covariates, stacked cohort rows (treated once; eligible controls once per cohort), by type."""
    import statsmodels.api as sm
    rows = []
    for u, d in data.items():
        k = SETS["B1"]; okT = np.isfinite(d["Xt"][:, k]).all(1); Xs = [d["Xt"][okT][:, k]] + [d["Xc"][c][d["elig"][c]][:, k] for c in d["cohorts"]]
        y = np.r_[np.ones(int(okT.sum())), np.zeros(sum(len(x) for x in Xs[1:]))]; X = np.vstack(Xs); X = (X - X.mean(0)) / X.std(0)
        r = dict(sample=sample, use=u, n_treated=int(okT.sum()), n_control_rows=int((y == 0).sum()))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); fit = sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=200)
            ps = fit.predict(sm.add_constant(X)); pt, pc = ps[y == 1], ps[y == 0]
            from sklearn.metrics import roc_auc_score
            r.update(auc=float(roc_auc_score(y, ps)), ps_T_min=float(pt.min()), ps_T_median=float(np.median(pt)), ps_T_max=float(pt.max()), ps_C_min=float(pc.min()),
                     ps_C_median=float(np.median(pc)), ps_C_max=float(pc.max()), share_T_above_C_max=float((pt > pc.max()).mean()),
                     share_T_above_C_p99=float((pt > np.percentile(pc, 99)).mean()), share_T_above_C_p95=float((pt > np.percentile(pc, 95)).mean()),
                     share_C_below_T_min=float((pc < pt.min()).mean()), converged=bool(fit.mle_retvals.get("converged", True)))
        except Exception as ex:
            r.update(error=f"{type(ex).__name__}: {ex}"[:160])
        rows.append(r)
    return rows


def main():
    t0 = time.time(); B, E = A.load()
    g7 = {r.get("spec"): r for r in json.loads((RES / "G7_imputation.json").read_text(encoding="utf-8")) if isinstance(r, dict) and r.get("carrier") == "elec"}
    out = dict(label=LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), definition=__doc__, nboot=NBOOT,
               panel_sha256=hashlib.sha256((PROC / "energy_monthly.csv").read_bytes()).hexdigest(),
               estimand="ATT on the original treated buildings (no treated building is trimmed; controls without a baseline window drop out)", samples={})
    T_rows, smd_rows, coh_rows, ov_rows, ev_rows, npz = [], [], [], [], [], {}
    for s, uses in SAMPLES.items():
        print(f"== {s} ==", flush=True); res = {}
        S = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=uses)); data = build_covariates(B, E, S); nP = len(S["pids"]); one = np.ones(nP)
        res["window_coverage"] = [r for d in data.values() for r in d["coverage"]]
        print(pd.DataFrame(res["window_coverage"]).to_string(index=False), flush=True)
        res["treated_missing_covariate"] = {u: {c: int((~np.isfinite(d["Xt"][:, k])).sum()) for k, c in enumerate(COVS)} for u, d in data.items()}
        W, CW = {"Original": one.copy()}, {}
        for name, cols in SETS.items(): W[name], CW[name] = unit_weights(S, data, one, cols)
        smd_rows += smd_table(S, data, CW, s); coh_rows += cohort_table(data, CW, s); ov_rows += overlap(data, s)
        rng = np.random.default_rng(0); mult = np.array([A.boot_weights(S, rng) for _ in range(NBOOT)])
        specs = [("Original", None), ("B1", SETS["B1"]), ("B2", SETS["B2"])]
        G.update(S=S, data=data, mult=mult, specs=specs); t1 = time.time(); bs = run_pool(_work_imp, NBOOT)
        print(f"  imputation bootstrap {NBOOT} draws x 3: {time.time() - t1:.0f} s", flush=True)
        pt_lead = {}
        for j, (name, cols) in enumerate(specs):
            w = W[name]; f, st = imp_stats(S, w); m = S["post"] & f["ok"]; ctrl_w = w[S["ctrl_p"]]; pos_c = ctrl_w[ctrl_w > 0]
            pt = pretrend_w(S, f); pt_lead[name] = pt; d = bs[:, j, 0]; d = d[np.isfinite(d)]; lo, hi = np.percentile(d, [2.5, 97.5])
            frag = [r for r in coh_rows if r["sample"] == s and name in SETS and r[f"{name}_fragile"]]
            row = dict(sample=s, estimator="imputation", weights=name, label=LABEL if name != "Original" else "reproduction of the headline row",
                       att_log=float(st[0]), pct=pct(st[0]), pct_lo=pct(lo), pct_hi=pct(hi), ci_log_lo=float(lo), ci_log_hi=float(hi), ci_method=f"percentile bootstrap, {len(d)} draws, weights re-estimated in every draw",
                       n_treated=int(pd.unique(S["pi"][m]).size), n_control=int(pd.unique(S["pi"][f["F"] & S["isc"]]).size),
                       pretrend_p=pt["p_joint"], pretrend_note="not informative: the baseline slope is balanced" if name == "B2" else "",
                       ess=kish(pos_c), ess_min_cohort=min([r[f"{name}_ess"] for r in coh_rows if r["sample"] == s], default=np.nan) if name in SETS else np.nan,
                       support_share=float(m.sum() / max(int(S["post"].sum()), 1)), w_max=float(pos_c.max()), w_p99=float(np.percentile(pos_c, 99)),
                       w_max_share=float(pos_c.max() / pos_c.sum()), n_cohorts_fragile=len(frag) if name in SETS else 0,
                       n_cohorts=len([r for r in coh_rows if r["sample"] == s]), boot_draws_season_dropped=int((bs[:, j, 9] > 0).sum()), boot_draws_infeasible_uniform=int((bs[:, j, 10] > 0).sum()), nboot_ok=int(len(d)))
            T_rows.append(row)
            for (a, b, kind), e, k in zip(BINS, st[1:], range(1, 9)):
                dd = bs[:, j, k]; dd = dd[np.isfinite(dd)]
                ev_rows.append(dict(sample=s, estimator="imputation", weights=name, term=f"ev_{a}_{b}", kind={"pre": "pre_in_sample", "ant": "anticipation", "post": "post"}[kind],
                                    est_log=float(e), lo_log=float(np.percentile(dd, 2.5)), hi_log=float(np.percentile(dd, 97.5)), reference="none (in-sample residuals)"))
            npz[f"{s}|imp|{name}|est"] = st[1:]; npz[f"{s}|imp|{name}|draws"] = bs[:, j, 1:9]
            npz[f"{s}|lead|{name}|coef"] = np.array(pt["coef"]); npz[f"{s}|lead|{name}|cov"] = np.array(pt["cov"]); npz[f"{s}|lead|{name}|G"] = np.array(pt["n_buildings"])
            print(f"  imputation {name}: {100 * row['pct']:+.1f}% ({100 * row['pct_lo']:+.1f}, {100 * row['pct_hi']:+.1f}); T/C {row['n_treated']}/{row['n_control']}; pre-trend p {row['pretrend_p']:.4f}; "
                  f"ESS {row['ess']:.1f}; support {row['support_share']:.3f}; draws with seasonality dropped {row['boot_draws_season_dropped']}, with a uniform cohort {row['boot_draws_infeasible_uniform']}", flush=True)
        g = g7[{"health_centre": "imputation_health_only", "pooled": "imputation_use"}[s]]; o = T_rows[-3]
        res["reproduction_headline"] = dict(att_log_diff=abs(o["att_log"] - g["att_log"]), ci_log_diff=float(max(abs(o["ci_log_lo"] - g["ci_log"][0]), abs(o["ci_log_hi"] - g["ci_log"][1]))) if s == "health_centre" else None,
                                            pretrend_p_diff=abs(o["pretrend_p"] - g["pretrend_p"]), s04_pretrend_p=g["pretrend_p"])
        print("  reproduction of G7:", res["reproduction_headline"], flush=True)
        Pa, Pe = stacked_prepare(B, E, uses, False), stacked_prepare(B, E, uses, True); specs_stk = [("Original", None), ("B1", SETS["B1"])]
        G.update(Pa=Pa, Pe=Pe, specs_stk=specs_stk); t1 = time.time(); bk = run_pool(_work_stk, NBOOT)
        print(f"  stacked bootstrap {NBOOT} draws x 2 x 2: {time.time() - t1:.0f} s", flush=True)
        for j, (name, cols) in enumerate(specs_stk):
            wa, cwa = stack_row_weights(Pa, S, data, one, cols); a, b, keep = stacked_att(Pa, wa)
            Fp = stacked_pyfixest(Pa, wa, keep); mk = keep & Pa["post"]
            nb = np.array([np.unique(Pa["pi"][mk & (Pa["stack"] == c)]).size for c in Pa["stacks"]], float); wv = nb / nb.sum()
            a_pf = float(wv @ Fp["b"]); se_pf = float(np.sqrt(wv @ Fp["cov"] @ wv))
            ess_st = []
            for c in Pa["stacks"]:
                rr = keep & ~Pa["treat"] & (Pa["stack"] == c); u_w = pd.Series(wa[rr]).groupby(Pa["pi"][rr]).first().to_numpy(); ess_st.append(kish(u_w))
            we, _ = stack_row_weights(Pe, S, data, one, cols); be, keep_e = stacked_fit(Pe, we); Fe_ = stacked_pyfixest(Pe, we, keep_e)
            ix = [Pe["x"].index(c) for c in ("ev_-12_-10", "ev_-9_-7")]; bb = Fe_["b"][ix]; Vv = Fe_["cov"][np.ix_(ix, ix)]
            d = bk[:, j, 0]; d = d[np.isfinite(d)]; lo, hi = np.percentile(d, [2.5, 97.5])
            imp_row = [r for r in T_rows if r["sample"] == s and r["estimator"] == "imputation" and r["weights"] == name][0]
            row = dict(sample=s, estimator="stacked", weights=name, label=LABEL if name != "Original" else "reproduction of the stacked row",
                       att_log=float(a), pct=pct(a), pct_lo=pct(lo), pct_hi=pct(hi), ci_log_lo=float(lo), ci_log_hi=float(hi), ci_method=f"percentile bootstrap, {len(d)} draws, weights re-estimated in every draw",
                       pct_lo_analytic=pct(a_pf - Fp["crit"] * se_pf), pct_hi_analytic=pct(a_pf + Fp["crit"] * se_pf), att_log_pyfixest=a_pf,
                       n_treated=int(np.unique(Pa["pi"][keep & Pa["treat"]]).size), n_control=int(np.unique(Pa["pi"][keep & ~Pa["treat"]]).size),
                       pretrend_p=imp_row["pretrend_p"], pretrend_note="s04 definition on the imputation fit sample with the building weights of the same row",
                       pretrend_event_study_wald_p=float(stats.chi2.sf(float(bb @ np.linalg.solve(Vv, bb)), 2)),
                       ess=float(np.sum(ess_st)), ess_min_stack=float(np.min(ess_st)), ess_min_cohort=imp_row["ess_min_cohort"], support_share=float((keep & Pa["treat"]).sum() / Pa["n_treat_rows0"]),
                       n_cohorts_fragile=imp_row["n_cohorts_fragile"], n_cohorts=len(Pa["stacks"]), nboot_ok=int(len(d)))
            T_rows.append(row)
            dE = bk[:, j, 1:]; okd = np.isfinite(dE).all(1)
            for k, term in enumerate(Pe["x"]):
                ev_rows.append(dict(sample=s, estimator="stacked", weights=name, term=term, kind="lead" if term.startswith("ev_-") and term != "ev_-3_-1" else ("anticipation" if term == "ev_-3_-1" else "post"),
                                    est_log=float(be[k]), lo_log=float(np.percentile(dE[okd, k], 2.5)), hi_log=float(np.percentile(dE[okd, k], 97.5)), reference="ev_-6_-4"))
            npz[f"{s}|stk|{name}|terms"] = np.array(Pe["x"]); npz[f"{s}|stk|{name}|coef"] = be; npz[f"{s}|stk|{name}|coef_pyfixest"] = Fe_["b"]
            npz[f"{s}|stk|{name}|cov_analytic"] = Fe_["cov"]; npz[f"{s}|stk|{name}|draws"] = dE[okd]
            print(f"  stacked {name}: {100 * row['pct']:+.1f}% boot ({100 * row['pct_lo']:+.1f}, {100 * row['pct_hi']:+.1f}), analytic ({100 * row['pct_lo_analytic']:+.1f}, {100 * row['pct_hi_analytic']:+.1f}); "
                  f"T/C {row['n_treated']}/{row['n_control']}; LSQR - pyfixest {a - a_pf:.1e}; ES lead Wald p {row['pretrend_event_study_wald_p']:.3f}; min stack ESS {row['ess_min_stack']:.1f}; "
                  f"support {row['support_share']:.3f}; ES max |LSQR - pyfixest| {np.nanmax(np.abs(be - Fe_['b'])):.1e}", flush=True)
        out["samples"][s] = res
    pd.DataFrame(T_rows).to_csv(RES / "v3_table.csv", index=False); pd.DataFrame(smd_rows).to_csv(RES / "v3_smd.csv", index=False)
    pd.DataFrame(coh_rows).to_csv(RES / "v3_cohorts.csv", index=False); pd.DataFrame(ov_rows).to_csv(RES / "v3_overlap.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(RES / "v3_event_profiles.csv", index=False); np.savez(RES / "v3_inputs_for_v4.npz", **npz)
    out["runtime_s"] = round(time.time() - t0, 1); out["table"] = T_rows
    (RES / "v3_balance.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("written results/v3_*;", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
