"""Reweighting comparison buildings on pre-works trajectories."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, json, time, multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
import v3_balance as V3

NB = int(os.environ.get("R3_NBOOT", "500")); NPROC = int(os.environ.get("R3_NPROC", "10")); _mi = V3._mi
COLS_FULL = [0, 1, 2, 3, 4, 5, 6]; COLS_B1 = [0, 1, 2, 3, 4, 5]
_G = {}


def covariates(B, E, S):
    Ec = A.carrier_frame(E, "elec"); Ec = Ec[Ec.pid.isin(set(S["pids"]))]
    Bi = B.drop_duplicates("pid").set_index("pid").reindex(S["pids"]); pos = {p: i for i, p in enumerate(S["pids"])}; nP = len(S["pids"])
    m0 = int(_mi(Ec.ym).min()); nT = int(_mi(Ec.ym).max()) - m0 + 1
    M = np.full((nP, nT), np.nan); M[Ec.pid.map(pos).to_numpy(), _mi(Ec.ym) - m0] = (Ec.kwh / Ec.gr_gfa).to_numpy()
    start = _mi(Bi.start_ym.astype(int).to_numpy()) - m0; ctrl = S["ctrl_p"]; cols = np.arange(nT)
    for i in range(nP): M[i, cols >= (start[i] - 3 if ctrl[i] else start[i])] = np.nan
    static = np.c_[pd.to_numeric(Bi.built_year.astype(str).str[:4], errors="coerce").to_numpy(float), np.log(Bi.area_m2.to_numpy(float)),
                   (Bi.zone == "중부1_춘천").to_numpy(float), (Bi.zone == "남부_부산").to_numpy(float)]

    def win(rows, lo, n, need):
        lo = np.broadcast_to(np.asarray(lo), (len(rows),)); mean = np.full(len(rows), np.nan); seas = np.full(len(rows), np.nan)
        for r, (i, l) in enumerate(zip(rows, lo)):
            ix = np.arange(l, l + n); ok = (ix >= 0) & (ix < nT); v = np.full(n, np.nan); v[ok] = M[i, ix[ok]]; mo = (ix + m0 - 1) % 12 + 1
            if np.isfinite(v).sum() < need: continue
            mean[r] = np.nanmean(v); wi, sh = v[np.isin(mo, [12, 1, 2])], v[np.isin(mo, [4, 5, 10])]
            if np.isfinite(wi).any() and np.isfinite(sh).any(): seas[r] = np.nanmean(wi) / np.nanmean(sh)
        return mean, seas

    T = S["T"].copy(); T["cohort"] = ((T.end_ym // 100) * 10 + ((T.end_ym % 100) > 6)).astype(int); u = "보건"
    Tu = T[T.use_class == u]; ti = np.array([pos[p] for p in Tu.pid]); ci = np.flatnonzero(ctrl & (S["use_p"] == u))
    lm, _ = win(ti, start[ti] - 12, 9, 7); _, ls = win(ti, start[ti] - 15, 12, 9); em, _ = win(ti, start[ti] - 24, 12, 9)
    Xt = np.c_[static[ti], lm, ls, em]; Xc, elig, cov = {}, {}, []
    for c in sorted(Tu.cohort.unique()):
        smin = int(_mi(int(Tu[Tu.cohort == c].start_ym.min())) - m0)
        clm, _ = win(ci, smin - 12, 9, 7); _, cls = win(ci, smin - 15, 12, 9); cem, _ = win(ci, smin - 24, 12, 9)
        X = np.c_[static[ci], clm, cls, cem]; Xc[c] = X; elig[c] = np.isfinite(X[:, COLS_B1]).all(1)
        tsel = Tu.cohort.to_numpy() == c
        cov.append(dict(cohort=int(c), n_treated=int(tsel.sum()), treated_with_early=int(np.isfinite(em[tsel]).sum()), controls_b1=int(elig[c].sum()),
                        controls_with_early=int((elig[c] & np.isfinite(cem)).sum())))
    return dict(ti=ti, ci=ci, Xt=Xt, cohT=Tu.cohort.to_numpy(), cohorts=sorted(Tu.cohort.unique()), Xc=Xc, elig=elig, coverage=cov)


def weights(S, d, m, early=True):
    w = np.asarray(m, float).copy(); mT, mC = m[d["ti"]], m[d["ci"]]; W = np.zeros(len(mC)); ntot = mT.sum(); flags = {}
    for c in d["cohorts"]:
        sel = (d["cohT"] == c) & (mT > 0)
        if not sel.any(): continue
        Xt, mt = d["Xt"][sel], mT[sel]; cols = COLS_FULL if early else COLS_B1
        if early:
            ce = d["elig"][c] & np.isfinite(d["Xc"][c][:, 6]) & (mC > 0)
            if ce.sum() < 20 or np.isfinite(Xt[:, 6]).mean() <= 0.5: cols = COLS_B1; flags[c] = "early_dropped"
        tgt = np.array([np.average(Xt[np.isfinite(Xt[:, k]), k], weights=mt[np.isfinite(Xt[:, k])]) if np.isfinite(Xt[:, k]).any() else np.nan for k in range(Xt.shape[1])])
        el = d["elig"][c] & (mC > 0) & np.isfinite(d["Xc"][c][:, cols]).all(1); wc = np.zeros(len(mC))
        if el.any():
            q = mC[el] / mC[el].sum(); X = d["Xc"][c][el]; r = None
            for stage, cc in (("full", cols), ("no_season", [k for k in cols if k != 5])):
                r = V3.ebal(X[:, cc], q, tgt[cc])
                if r is not None: break
            if r is None: r = q; flags[c] = flags.get(c, "") + "|uniform"
            elif stage != "full": flags[c] = flags.get(c, "") + "|no_season"
            wc[el] = r
        W += mT[d["cohT"] == c].sum() / ntot * wc
    w[d["ci"]] = W * mC[W > 0].sum(); return w, flags


def _work(seed):
    S, d = _G["S"], _G["d"]; rng = np.random.default_rng([seed, 23]); m = A.boot_weights(S, rng)
    out = []
    for early in (True, False):
        w, _ = weights(S, d, m, early); out.append(S["att"](S["fit"](w, tol=1e-9))[0])
    return out


def main():
    t0 = time.time(); B, E = A.load()
    S = A.imputation_setup(B, E, "elec", uses=("보건",), fe="use", ctrl_roles=("control", "aux_2023"), ctrl_trim=3, anticip=3)
    d = covariates(B, E, S); one = np.ones(len(S["pids"])); res = dict(label="post hoc", coverage=d["coverage"])
    for tag, early in (("trajectory_B3", True), ("B1_on_v3", False)):
        w, fl = weights(S, d, one, early); f = S["fit"](w); a = S["att"](f)[0]; pw = w[d["ci"]]; pw = pw[pw > 0]
        res[tag] = dict(att_log=float(a), pct=float(np.exp(a) - 1), flags={str(k): v for k, v in fl.items()}, ess=float(pw.sum() ** 2 / (pw ** 2).sum()), n_pos_controls=int(len(pw)),
                        w_max_share=float(pw.max() / pw.sum()))
        Xt = d["Xt"]; res[tag]["treated_means"] = [float(np.nanmean(Xt[:, k])) for k in range(Xt.shape[1])]
    _G.update(S=S, d=d)
    with mp.get_context("fork").Pool(NPROC) as p: dr = np.array(p.map(_work, range(NB), chunksize=5), float)
    for j, tag in enumerate(("trajectory_B3", "B1_on_v3")):
        v = dr[:, j]; v = v[np.isfinite(v)]; res[tag]["pct_ci"] = [float(np.exp(np.percentile(v, 2.5)) - 1), float(np.exp(np.percentile(v, 97.5)) - 1)]; res[tag]["nboot"] = int(len(v))
    res["covariates"] = ["built year", "log floor area", "coldest zone", "southern zone", "mean s-12..s-4", "seasonality s-15..s-4", "mean s-24..s-13"]
    res["runtime_min"] = round((time.time() - t0) / 60, 1); (RES / "r3_balance.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
