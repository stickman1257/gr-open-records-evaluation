"""Benchmark on non-retrofitted buildings (500 replications) and fall towards the average."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, json, time, multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
import s06_benchmark as BM
from s03_energy import ym_add, ym_range

SEED = int(json.loads((RES / "B2_benchmark_meta.json").read_text())["seed"]); USES = ("보건", "어린이집")
REPS = int(os.environ.get("R3_BENCH_REPS", "500")); MR_REPS = int(os.environ.get("R3_MR_REPS", "200")); NPROC = int(os.environ.get("R3_NPROC", "10"))
mi = lambda v: (np.asarray(v).astype(int) // 100) * 12 + np.asarray(v).astype(int) % 100
_G = {}


def imp_point(B2, E2, bld=False):
    S = A.imputation_setup(B2, E2, "elec", uses=None, fe="use", ctrl_roles=("control",), ctrl_trim=3, anticip=3, min_win=0)
    if S is None: return np.nan, None
    f = S["fit"](); a = S["att"](f)[0]
    return a, (S["building_tau"](f) if bld else None)


def did_point(B2, E2, **kw):
    try: return BM.est_did(B2, E2, "elec", **kw)["est_log"]
    except Exception: return np.nan


def one_rep(rep):
    B, Bp, e, pools = _G["B"], _G["Bp"], _G["e"], _G["pools"]
    rng = np.random.default_rng([SEED, rep]); pseudo, n_drawn = BM.draw(Bp, e, pools, rng)
    if not pseudo: return [], []
    P = BM.make_panel(e, pseudo); act0 = P[P.pphase == "post"].groupby("pid").kwh.sum(); rows = []
    base = dict(rep=rep, n_pseudo=len(pseudo))
    for name, fn in (("naive", BM.cf_naive), ("dd", BM.cf_dd)):
        try: r = BM.agg_from_cf(act0, fn(P), np.random.default_rng([SEED, rep, 7])); rows.append(dict(base, estimator=name, est_log=r["est_log"] if r else np.nan))
        except Exception: rows.append(dict(base, estimator=name, est_log=np.nan))
    B2, E2 = BM.did_inputs(Bp, P, 0.0)
    rows.append(dict(base, estimator="did", est_log=did_point(B2, E2, ctrl_trim=3)))
    rows.append(dict(base, estimator="did_usefe", est_log=did_point(B2, E2, fe="use", ctrl_trim=3, anticip=3)))
    a, bt = imp_point(B2, E2, bld=True); rows.append(dict(base, estimator="imp", est_log=a))
    gaps = []
    if bt is not None:
        real = Bp.set_index("pid").start_ym; use = Bp.set_index("pid").use_class; dev = _G["dev"]
        for pid, r in bt.iterrows():
            s, en = pseudo[pid]; d = dev[(dev.pid == pid) & (dev.ym >= ym_add(int(s), -12)) & (dev.ym <= ym_add(int(s), -1))].d
            gaps.append(dict(rep=rep, pid=pid, use=use[pid], tau=float(r.tau_mean), months_post_end_to_real_start=int(mi(real[pid]) - mi(ym_add(int(en), 12))),
                             pseudo_start=int(s), real_start=int(real[pid]), baseline_dev=float(d.mean()) if len(d) else np.nan))
    return rows, gaps


def mr_rep(rep):
    Bh, eh, pool = _G["Bh"], _G["eh"], _G["pool_h"]
    rng = np.random.default_rng([SEED, rep, 99]); have = eh.groupby("pid").ym.apply(set).to_dict()
    lvl = eh.assign(ly=np.log(eh.kwh / eh.gr_gfa)); out = {}
    for p in np.sort(Bh.pid.values):
        s, en = (int(x) for x in pool[rng.integers(len(pool))])
        if (len(set(ym_range(ym_add(s, -12), ym_add(s, -1))) & have[p]) >= A.MIN_WIN and len(set(ym_range(ym_add(en, 1), ym_add(en, 12))) & have[p]) >= A.MIN_WIN):
            out[p] = (s, en)
    if len(out) < 9: return []
    base = {p: float(lvl[(lvl.pid == p) & (lvl.ym >= ym_add(s, -12)) & (lvl.ym <= ym_add(s, -1))].ly.mean()) for p, (s, en) in out.items()}
    ser = pd.Series(base).sort_values(ascending=False); k = len(ser) // 3
    sel = {"high_use_third": list(ser.index[:k]), "random_third": list(rng.choice(np.array(sorted(out)), k, replace=False))}
    rows = []
    for tag, pids in sel.items():
        pseudo = {p: out[p] for p in pids}; P = BM.make_panel(eh, pseudo); B2, E2 = BM.did_inputs(Bh, P, 0.0)
        rest = ser.drop(pids)
        rows.append(dict(rep=rep, selection=tag, estimator="imp", est_log=imp_point(B2, E2)[0], n_pseudo=len(pids), n_feasible=len(out),
                         baseline_ratio=float(np.exp(ser[pids].mean() - rest.mean()))))
        rows.append(dict(rep=rep, selection=tag, estimator="did_usefe", est_log=did_point(B2, E2, fe="use", ctrl_trim=3, anticip=3), n_pseudo=len(pids), n_feasible=len(out),
                         baseline_ratio=float(np.exp(ser[pids].mean() - rest.mean()))))
    return rows


def summ(d, keys):
    out = []
    for k, g in d.groupby(keys):
        v = g.est_log.dropna().to_numpy(); n = len(v); m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(n))
        out.append(dict(zip(keys if isinstance(keys, list) else [keys], k if isinstance(k, tuple) else (k,)), reps=n, bias_log=m, mc_se=se, mc_lo=m - 1.96 * se, mc_hi=m + 1.96 * se,
                        bias_pct=float(np.exp(m) - 1), mc_lo_pct=float(np.exp(m - 1.96 * se) - 1), mc_hi_pct=float(np.exp(m + 1.96 * se) - 1),
                        bias_over_mc_se=abs(m) / se, rmse_log=float(np.sqrt((v ** 2).mean())), sd_log=float(v.std(ddof=1))))
    return pd.DataFrame(out)


def main():
    t0 = time.time(); B, E = A.load()
    Bp, e = BM.population(B, E, "elec", USES); pools = BM.date_pools(B, USES, "use")
    dv = e.assign(ly=np.log(e.kwh / e.gr_gfa)); dv = dv.merge(Bp[["pid", "use_class"]], on="pid")
    dv["d"] = dv.ly - dv.groupby(["use_class", "ym"]).ly.transform("mean"); dev = dv[["pid", "ym", "d", "use_class"]]
    _G.update(B=B, Bp=Bp, e=e, pools=pools, dev=dev)
    S0 = A.imputation_setup(B, E, "elec", uses=("보건",), fe="use", ctrl_roles=("control", "aux_2023"), ctrl_trim=3, anticip=3)
    f0 = S0["fit"](); head = set(S0["pids"][pd.unique(S0["pi"][S0["post"] & f0["ok"]])])
    Et = A.carrier_frame(E, "elec"); Et = Et[Et.pid.isin(head) & (Et.ym < Et.start_ym) & (mi(Et.ym) >= mi(Et.start_ym) - 12)].copy()
    Et["ly"] = np.log(Et.kwh / Et.gr_gfa); pm = dv[dv.use_class == "보건"].groupby("ym").ly.mean()
    Et["d"] = Et.ly - Et.ym.map(pm); dT = float(Et.groupby("pid").d.mean().mean())
    Bh = Bp[Bp.use_class == "보건"].copy(); eh = e[e.pid.isin(Bh.pid)].copy(); _G.update(Bh=Bh, eh=eh, pool_h=pools["보건"])
    with mp.get_context("fork").Pool(NPROC) as p:
        res = p.map(one_rep, range(REPS), chunksize=2)
        mr = p.map(mr_rep, range(MR_REPS), chunksize=2)
    R = pd.DataFrame([r for x, _ in res for r in x]); Gp = pd.DataFrame([g for _, x in res for g in x]); M = pd.DataFrame([r for x in mr for r in x])
    R.to_csv(RES / "r3_benchmark_reps.csv", index=False); Gp.to_csv(RES / "r3_benchmark_bldgap.csv", index=False); M.to_csv(RES / "r3_meanrev_reps.csv", index=False)
    Sm = pd.concat([summ(R, "estimator").assign(subset=f"reps 0..{REPS - 1}"), summ(R[R.rep < 100], "estimator").assign(subset="reps 0..99")])
    Sm.to_csv(RES / "r3_benchmark_summary.csv", index=False)
    Mm = summ(M, ["selection", "estimator"]); Mm["mean_baseline_ratio"] = Mm.apply(lambda r: float(M[(M.selection == r.selection) & (M.estimator == r.estimator)].baseline_ratio.mean()), axis=1)
    Mm.to_csv(RES / "r3_meanrev_summary.csv", index=False)
    pv = M.pivot_table(index=["rep", "estimator"], columns="selection", values="est_log").dropna(); pv["diff"] = pv.high_use_third - pv.random_third
    md = pv.groupby("estimator")["diff"].agg(["mean", "std", "count"]); md["mc_se"] = md["std"] / np.sqrt(md["count"])
    Gp["dist_bin"] = pd.cut(Gp.months_post_end_to_real_start, [-1, 6, 12, 24, 60], labels=["0-6", "7-12", "13-24", "25+"])
    diag = Gp.groupby(["use", "dist_bin"], observed=True).tau.agg(["mean", "count"]).reset_index()
    import statsmodels.formula.api as smf
    fit = smf.ols("tau ~ months_post_end_to_real_start + C(use)", data=Gp).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(Gp.pid)[0]})
    gh = Gp[(Gp.use == "보건") & Gp.baseline_dev.notna()]
    rv = smf.ols("tau ~ baseline_dev", data=gh).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(gh.pid)[0]})
    rho = float(rv.params["baseline_dev"]); rho_se = float(rv.bse["baseline_dev"])
    reversion = dict(rho=rho, rho_se=rho_se, intercept=float(rv.params["Intercept"]), n=int(rv.nobs), treated_mean_dev=dT,
                     implied_reversion_log=rho * dT, implied_reversion_se=abs(dT) * rho_se, implied_pct=float(np.exp(rho * dT) - 1),
                     note="building-level pseudo gap of random pseudo-treated health posts regressed on their baseline deviation from contemporaneous peers; "
                          "implied = rho x mean deviation of the retrofitted health posts (added after the debug run, before the v3 run)")
    out = dict(label="post hoc (Energy Policy review R3)", seed=SEED, reps=REPS, mr_reps=MR_REPS, summary=Sm.to_dict("records"), meanrev=Mm.to_dict("records"),
               meanrev_paired_diff={k: dict(mean_log=float(v["mean"]), mc_se=float(v["mc_se"]), reps=int(v["count"]), pct=float(np.exp(v["mean"]) - 1)) for k, v in md.iterrows()},
               bias_diagnosis=dict(by_distance=diag.to_dict("records"), slope_per_month=float(fit.params["months_post_end_to_real_start"]),
                                   slope_se=float(fit.bse["months_post_end_to_real_start"]), slope_p=float(fit.pvalues["months_post_end_to_real_start"]), n=int(fit.nobs)),
               mean_reversion_calibration=reversion, runtime_min=round((time.time() - t0) / 60, 1))
    (RES / "r3_benchmark.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(Sm.to_string()); print(Mm.to_string()); print(md); print(diag.to_string()); print(reversion); print("done", out["runtime_min"], "min")


if __name__ == "__main__":
    main()
