"""Benchmark table from the stored replications (Fig. 4)."""
import json
from pathlib import Path
import numpy as np, pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
ORDER = ["naive", "dd", "gb", "did", "did_usefe", "imp", "mvi", "sdid", "imp_s24", "mvi_w12"]
LAB = {"naive": "Before-after", "dd": "Degree-day baseline", "gb": "Gradient boosting", "did": "Stacked DiD, province x month",
       "did_usefe": "Stacked DiD, type x month", "imp": "Imputation (headline estimator)", "mvi": "M&V-informed imputation", "sdid": "Synthetic DiD",
       "imp_s24": "Imputation, s-24..s-4 (auxiliary)", "mvi_w12": "MVI, s-12..s-4 (auxiliary)"}
CAND = ["naive", "dd", "gb", "did", "did_usefe", "imp", "mvi", "sdid"]


def main():
    B = pd.read_csv(RES / "B1_benchmark_reps.csv"); B = B[(B.test == "random") & (B.carrier == "elec")]
    R = pd.read_csv(RES / "a_r6_benchmark_reps.csv")
    keep = ["rep", "tau", "estimator", "status", "est_log", "lo_log", "hi_log", "n_used", "n_pseudo"]
    A = pd.concat([B[B.estimator.isin(ORDER)][keep + ["ci_method"]], R[R.estimator.isin(ORDER)][keep]], ignore_index=True)
    A = A[A.status == "ok"]; A["truth"] = np.log1p(A.tau); A["err"] = A.est_log - A.truth
    A["cover"] = (A.lo_log <= A.truth) & (A.truth <= A.hi_log); A["rej"] = (A.lo_log > 0) | (A.hi_log < 0)
    rng = np.random.default_rng(0); rows = []; E0 = {}
    for e in ORDER:
        d0 = A[(A.estimator == e) & np.isclose(A.tau, 0)].set_index("rep"); n = len(d0); err = d0.err.to_numpy(); E0[e] = d0.err
        i = rng.integers(0, n, (2000, n)); rb = np.sqrt((err[i] ** 2).mean(1))
        pw = {t: float(A[(A.estimator == e) & np.isclose(A.tau, t)].rej.mean()) for t in (-0.05, -0.10)}
        bias = err.mean(); mc = err.std(ddof=1) / np.sqrt(n)
        rows.append(dict(estimator=e, label=LAB[e], candidate=e in CAND, n_reps=n, mean_n_pseudo=float(d0.n_pseudo.mean()), mean_n_used=float(d0.n_used.mean()),
                         bias_log=bias, bias_pct=100 * (np.exp(bias) - 1), mc_se=mc, bias_over_mc_se=bias / mc, rmse_log=float(np.sqrt((err ** 2).mean())), rmse_mc_se=float(rb.std(ddof=1)),
                         coverage95=float(d0.cover.mean()), mean_ci_width_log=float((d0.hi_log - d0.lo_log).mean()), size_tau0=float(d0.rej.mean()),
                         power_minus5=pw[-0.05], power_minus10=pw[-0.10],
                         passes_screen=bool(abs(bias) < 2 * mc and d0.cover.mean() >= 0.90),
                         interval="bootstrap over pseudo-treated buildings, counterfactual fixed" if e in ("naive", "dd", "gb") else
                                  "building-clustered, analytic" if e.startswith("did") else "building bootstrap, full refit (200 draws)"))
    T = pd.DataFrame(rows); T.to_csv(RES / "v6_benchmark_table.csv", index=False)
    C = {}
    npz = A[np.isclose(A.tau, 0)].pivot(index="rep", columns="estimator", values="n_pseudo")
    C["1_same_splits"] = dict(n_pseudo_identical_across_estimators=bool((npz.nunique(axis=1) == 1).all()), reps=int(len(npz)),
                              mean_n_used={e: float(A[(A.estimator == e) & np.isclose(A.tau, 0)].n_used.mean()) for e in ORDER},
                              reps_where_n_used_lt_n_pseudo={e: int((A[(A.estimator == e) & np.isclose(A.tau, 0)].n_used < A[(A.estimator == e) & np.isclose(A.tau, 0)].n_pseudo).sum()) for e in ORDER})
    sh = {}
    for e in ORDER:
        d = A[A.estimator == e].pivot(index="rep", columns="tau", values=["est_log", "lo_log", "hi_log"])
        sh[e] = {str(t): float(max(((d[c][t] - d[c][0.0]) - np.log1p(t)).abs().max() for c in ["est_log", "lo_log", "hi_log"])) for t in (-0.05, -0.10)}
    C["4_injection_shift_max_abs_deviation"] = sh
    C["4_note"] = "a_r6 rows for tau < 0 were written as the tau = 0 row plus log(1 + tau) (not re-estimated); B1 rows were estimated at each tau"
    el = [e for e in CAND if bool(T.set_index("estimator").passes_screen[e])]
    X = pd.DataFrame({e: E0[e] for e in el}).dropna(); n = len(X); i = rng.integers(0, n, (2000, n)); gaps = {}
    for a in el:
        for b in el:
            if a >= b: continue
            xa, xb = X[a].to_numpy(), X[b].to_numpy(); d = np.sqrt((xa ** 2).mean()) - np.sqrt((xb ** 2).mean())
            db = np.sqrt((xa[i] ** 2).mean(1)) - np.sqrt((xb[i] ** 2).mean(1))
            gaps[f"{a} - {b}"] = dict(rmse_gap=float(d), paired_boot_se=float(db.std(ddof=1)), ci95=[float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))])
    C["6_rmse"] = dict(passing_candidates=el, rmse_min=float(T[T.estimator.isin(el)].rmse_log.min()), rmse_max=float(T[T.estimator.isin(el)].rmse_log.max()),
                       rmse_mc_se_of_passing=[float(v) for v in T[T.estimator.isin(el)].rmse_mc_se], paired_gaps=gaps,
                       error_correlation_between_passing={f"{a}~{b}": float(X[a].corr(X[b])) for a in el for b in el if a < b})
    sd = T.set_index("estimator")
    C["6_sdid"] = dict(bias_over_mc_se=float(sd.bias_over_mc_se["sdid"]), rmse=float(sd.rmse_log["sdid"]),
                       gap_to_best=float(sd.rmse_log["sdid"] - T[T.estimator.isin(el)].rmse_log.min()))
    (RES / "v6_checks.json").write_text(json.dumps(C, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(T[["estimator", "n_reps", "mean_n_used", "bias_pct", "mc_se", "bias_over_mc_se", "rmse_log", "rmse_mc_se", "coverage95", "size_tau0", "power_minus5", "power_minus10", "passes_screen"]].round(4).to_string(index=False))
    print(json.dumps(C, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
