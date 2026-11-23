"""Own-history methods applied to the retrofitted health posts."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
from s03_energy import ym_add
import s04_analysis as A
import s05_ml as ML
import s06_benchmark as BM

SERIES = [("health_centre", dict(uses=("보건",))), ("daycare", dict(uses=("어린이집",))), ("pooled", {})]
COLS = ["use", "carrier", "estimator", "label", "pre_window", "status", "change_pct", "lo", "hi", "est_log", "lo_log", "hi_log", "agg_ratio_change_pct",
        "log_effect", "n_buildings", "n_control_pool_gb", "ci_method", "note"]
CI = "percentile bootstrap over treated buildings, counterfactual held fixed (interval over buildings only)"


def naive_matched(P):
    """Per building: (actual post sum, pre-year sum) over calendar months observed in both the post window and [s-12, s-1]."""
    pre = P[(P.pphase == "pre") & (P.ym >= P.ps.map(lambda x: ym_add(int(x), -12) if pd.notna(x) else 0))]
    post = P[P.pphase == "post"]; a, c = {}, {}
    for pid, q in post.groupby("pid"):
        r = pre[pre.pid == pid]; cal = set(q.ym % 100) & set(r.ym % 100)
        if not cal: continue
        a[pid] = q[(q.ym % 100).isin(cal)].kwh.sum(); c[pid] = r[(r.ym % 100).isin(cal)].kwh.sum()
    return pd.Series(a, dtype=float), pd.Series(c, dtype=float)


def one(B, E, use, kw, carrier="elec"):
    base = dict(use=use, carrier=carrier, ci_method=CI)
    S = A.imputation_setup(B, E, carrier, **dict(A.PRIMARY_IMP, **kw))
    if S is None: return [dict(base, estimator="(all)", status="no treated buildings")]
    f = S["fit"](); D = S["D"]
    tp = set(S["pids"][pd.unique(S["pi"][S["post"] & f["ok"]])]); cp = set(S["pids"][pd.unique(S["pi"][f["F"] & S["isc"]])])
    Bi = B.drop_duplicates("pid").set_index("pid")
    e = ML.prep(E[E.pid.isin(tp | cp)], carrier)
    s_ = e.pid.map(Bi.start_ym).astype(int)
    e = e[np.where(e.pid.isin(tp), True, e.ym < s_.map(lambda v: ym_add(int(v), -3)))].reset_index(drop=True)
    dates = {p: (int(Bi.start_ym[p]), int(Bi.end_ym[p])) for p in tp}
    rows = []
    for pw, cut in [("s-12..s-4", 3), ("s-12..s-1", 0)]:
        ee = e
        if cut:
            st = ee.pid.map({p: v[0] for p, v in dates.items()})
            ee = ee[~(st.notna() & (ee.ym >= st.map(lambda v: ym_add(int(v), -cut) if pd.notna(v) else 0)) & (ee.ym < st.fillna(0)))].reset_index(drop=True)
        P = BM.make_panel(ee, dates)
        act = P[P.pphase == "post"].groupby("pid").kwh.sum()
        ests = []
        try:
            a, c = naive_matched(P); ests.append(("naive", a, c, np.nan))
        except Exception as ex: rows.append(dict(base, estimator="naive", pre_window=pw, status=f"{type(ex).__name__}: {ex}"[:160]))
        for name, fn in [("dd", lambda: BM.cf_dd(P)), ("gb", lambda: BM.cf_gb(P))]:
            try:
                cf = fn(); ests.append((name, act, cf, cf.attrs.get("smear", np.nan) if name == "gb" else np.nan))
            except Exception as ex: rows.append(dict(base, estimator=name, pre_window=pw, status=f"{type(ex).__name__}: {ex}"[:160]))
        for name, a, c, sm in ests:
            r = BM.agg_from_cf(a, c, np.random.default_rng([20261113, len(rows)]))
            if r is None: rows.append(dict(base, estimator=name, pre_window=pw, status="no building with a counterfactual")); continue
            rows.append(dict(base, estimator=name, label=BM.LABELS.get(name, name), pre_window=pw, status="ok", change_pct=-100 * r["est"],
                             lo=-100 * r["hi"], hi=-100 * r["lo"], est_log=r["est_log"], lo_log=r["lo_log"], hi_log=r["hi_log"],
                             agg_ratio_change_pct=-100 * r["est_agg_ratio"], log_effect=r["log_effect"], n_buildings=r["n_used"],
                             n_control_pool_gb=len(cp) if name == "gb" else np.nan,
                             note="descriptive baseline on the treated buildings of the imputation estimate; not an attributable change"))
    return rows


def main():
    B, E = A.load(); rows = []
    for use, kw in SERIES:
        try: rows += one(B, E, use, kw)
        except Exception as ex: rows.append(dict(use=use, carrier="elec", estimator="(all)", status=f"{type(ex).__name__}: {ex}"[:200]))
    R = pd.DataFrame(rows).reindex(columns=COLS); R.to_csv(RES / "a_baselines_real.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"generated {time.strftime('%Y-%m-%d %H:%M')}")
    print(R[["use", "estimator", "pre_window", "status", "change_pct", "lo", "hi", "agg_ratio_change_pct", "n_buildings"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
