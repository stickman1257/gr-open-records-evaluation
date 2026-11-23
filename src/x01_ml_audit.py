"""Counterfactual predictors on untreated buildings (helpers for x08)."""
import os, sys, time, argparse, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("XGB_DEVICE", "cpu")
from common import RES, fit_dd_model, predict_dd
from s03_energy import ym_add, ym_range
import s04_analysis as A
import s05_ml as ML
import s06_benchmark as BM
import xgboost as xgb

warnings.filterwarnings("ignore")
USES = ["보건", "어린이집"]
HOLD = 3
GB_ROUNDS = 600


def months_since(ym): return (np.asarray(ym) // 100 - 2020) * 12 + (np.asarray(ym) % 100 - 1)


def half(ym): return (int(ym) // 100) * 10 + (1 if int(ym) % 100 <= 6 else 2)


def gap_naive(P, target):
    """log actual - log(same calendar month one year earlier, own meter)."""
    key = P.set_index(["pid", "ym"]).y
    prev = [key.get((p, ym_add(int(m), -12)), np.nan) for p, m in zip(target.pid, target.ym)]
    return target.y.to_numpy(float) - np.asarray(prev, float)


def gap_dd(P_train, target):
    out = np.full(len(target), np.nan); fits = {}
    for pid, d in P_train.groupby("pid"):
        m = fit_dd_model(d.sort_values("ym").tail(24))
        if m is None: continue
        fits[pid] = m
    for pid, idx in target.groupby("pid").groups.items():
        m = fits.get(pid)
        if m is None: continue
        t = target.loc[idx]; pr = np.asarray(predict_dd(m, t), float)
        ok = pr > 0
        out[target.index.get_indexer(idx[ok])] = np.log(t.kwh.to_numpy(float)[ok] / pr[ok])
    return out, fits


def gap_gb(train, target, drop=()):
    pred = ML._fit(train, target, GB_ROUNDS, tuple(drop))
    return target.y.to_numpy(float) - np.asarray(pred, float)


def gap_gb_did(P, pseudo_post_mask, rounds=GB_ROUNDS):
    """E2. One fit per pseudo-start half-year cohort, trained on rows strictly before the cohort's earliest start."""
    T = P[pseudo_post_mask]
    coh = T.ps.map(half)
    est_rows = []
    for h, Tc in T.groupby(coh):
        s0 = int(P[P.pphase != "ctrl"].loc[lambda d: d.ps.map(half) == h].ps.min())
        train = P[(P.ym < s0) & (P.pphase.isin(["ctrl", "pre"]))]
        cal = set(Tc.ym)
        C = P[(P.pphase == "ctrl") & P.ym.isin(cal) & (P.ym >= s0)]
        C = C[C.pid.isin(train.pid.unique())]
        Tc = Tc[Tc.pid.isin(train.pid.unique())]
        if C.empty or Tc.empty: continue
        both = pd.concat([Tc, C])
        g = both.y.to_numpy(float) - np.asarray(ML._fit(train, both, rounds), float)
        gT, gC = g[:len(Tc)], g[len(Tc):]
        cm = pd.Series(gC).groupby([C.use_class.to_numpy(), C.ym.to_numpy()]).mean()
        corr = np.array([cm.get((u, m), np.nan) for u, m in zip(Tc.use_class, Tc.ym)])
        est_rows.append(pd.DataFrame({"pid": Tc.pid.to_numpy(), "gap": gT - corr}))
    if not est_rows: return np.nan, 0
    D = pd.concat(est_rows).dropna()
    return float(D.gap.mean()), int(D.pid.nunique())


def imp_point(Bp, P):
    B2, E2 = BM.did_inputs(Bp, P, 0.0)
    r = A.imputation_did(B2, E2, "elec", uses=None, ctrl_roles=("control",), nboot=0, pretrend=False, min_win=0)
    return float(r["att_log"])


def realised_drift(P):
    C = P[P.pphase == "ctrl"].copy()
    C["yd"] = C.y - C.groupby("pid").y.transform("mean")
    cm = C.groupby(["use_class", "ym"]).yd.mean()
    out = []
    for pid, d in P[P.pphase != "ctrl"].groupby("pid"):
        u = d.use_class.iloc[0]; s = int(d.ps.iloc[0])
        pre = [cm.get((u, m), np.nan) for m in ym_range(ym_add(s, -12), ym_add(s, -1))]
        post = [cm.get((u, m), np.nan) for m in d[d.pphase == "post"].ym]
        out.append(np.nanmean(post) - np.nanmean(pre))
    return float(np.nanmean(out))


def one_rep(Bp, e, pools, seed, rep, backtest=True, which=("naive", "dd", "gb", "gb_noenc", "gb_not", "gb_noctrl", "gb_did", "imp")):
    rng = np.random.default_rng([seed, rep])
    pseudo, _ = BM.draw(Bp, e, pools, rng)
    if not pseudo: return [], []
    P = BM.make_panel(e, pseudo)
    P = P.merge(Bp[["pid", "use_class"]], on="pid", how="left") if "use_class" not in P else P
    P = P.reset_index(drop=True)
    post = P.pphase == "post"
    rows, dd_rows = [], []
    base = dict(rep=rep, n_pseudo=len(pseudo))
    drift = realised_drift(P)

    def rec(name, bias, t, n=None, bt=np.nan):
        rows.append(dict(base, estimator=name, bias_log=bias, backtest_rmse_log=bt, drift_log=drift, n_used=n, runtime_s=t))

    hold = (P.pphase == "pre") & (P.ym >= P.ps.map(lambda x: ym_add(int(x), -HOLD) if pd.notna(x) else 0))
    Ptr_bt = P[~hold & ~post]
    rmse = lambda g: float(np.sqrt(np.nanmean(np.square(g)))) if np.isfinite(g).any() else np.nan
    T = P[post]
    if "naive" in which:
        t0 = time.time()
        r = BM.agg_from_cf(T.groupby("pid").kwh.sum(), BM.cf_naive(P), np.random.default_rng([seed, rep, 7]))
        bt = rmse(gap_naive(P, P[hold])) if backtest else np.nan
        rec("naive", r["est_log"], time.time() - t0, r["n_used"], bt)
    if "dd" in which:
        t0 = time.time(); g, fits = gap_dd(P[P.pphase == "pre"], T)
        by = pd.Series(g, index=T.index).groupby(T.pid).mean()
        r = BM.agg_from_cf(T.groupby("pid").kwh.sum(), BM.cf_dd(P), np.random.default_rng([seed, rep, 7]))
        bt = rmse(gap_dd(P[(P.pphase == "pre") & ~hold], P[hold])[0]) if backtest else np.nan
        rec("dd", r["est_log"], time.time() - t0, r["n_used"], bt)
        for pid, m in fits.items():
            dd_rows.append(dict(rep=rep, pid=pid, cvrmse=m["cvrmse"], nmbe=m["nmbe"], n=m["n"], gap_log=by.get(pid, np.nan)))
    for name, drop, ph in [("gb", (), ("ctrl", "pre")), ("gb_noenc", tuple(BM.ENC_COLS), ("ctrl", "pre")),
                           ("gb_not", ("t",), ("ctrl", "pre")), ("gb_noctrl", (), ("pre",))]:
        if name not in which: continue
        t0 = time.time(); g = gap_gb(P[P.pphase.isin(list(ph))], T, drop)
        bt = rmse(gap_gb(Ptr_bt[Ptr_bt.pphase.isin(list(ph))], P[hold], drop)) if backtest else np.nan
        rec(name, float(np.nanmean(g)), time.time() - t0, int(T.pid.nunique()), bt)
    if "gb_did" in which:
        t0 = time.time(); b, n = gap_gb_did(P, post)
        rec("gb_did", b, time.time() - t0, n, np.nan)
    if "imp" in which:
        t0 = time.time()
        try: rec("imp", imp_point(Bp, P), time.time() - t0, len(pseudo))
        except Exception as ex: print("imp failed", rep, ex, flush=True)
    return rows, dd_rows


def add_drift(e, delta):
    """Known common drift: delta log points per year, relative to 2020-01, added to every building-month."""
    e = e.copy(); sh = delta * months_since(e.ym) / 12.0
    e["y"] = e.y + sh; e["kwh"] = e.kwh * np.exp(sh)
    return e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50); ap.add_argument("--drift-reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261113); ap.add_argument("--part", default="main,drift")
    a = ap.parse_args(); t_all = time.time()
    B, E = A.load()
    Bp, e = BM.population(B, E, "elec", USES)
    e = e.merge(Bp[["pid", "use_class"]], on="pid", how="left")
    pools = BM.date_pools(B, USES, "use")
    print(f"population {len(Bp)} {Bp.use_class.value_counts().to_dict()}", flush=True)
    parts = a.part.split(",")
    if "main" in parts:
        R, D = [], []
        for rep in range(a.reps):
            r, d = one_rep(Bp, e, pools, a.seed, rep); R += r; D += d
            pd.DataFrame(R).to_csv(RES / "x01_reps.csv", index=False); pd.DataFrame(D).to_csv(RES / "x01_dd_ashrae.csv", index=False)
            print(f"rep {rep + 1}/{a.reps} {time.time() - t_all:.0f}s", flush=True)
    if "drift" in parts:
        R = []
        for delta in (-0.06, -0.03, 0.0, 0.03, 0.06):
            ed = add_drift(e, delta)
            for rep in range(a.drift_reps):
                r, _ = one_rep(Bp, ed, pools, a.seed, rep, backtest=False, which=("naive", "dd", "gb", "gb_did", "imp"))
                R += [dict(x, delta=delta) for x in r]
                pd.DataFrame(R).to_csv(RES / os.environ.get("X01_DRIFT_OUT", "x01_drift_reps.csv"), index=False)
            print(f"delta {delta} done {time.time() - t_all:.0f}s", flush=True)
    print(f"total {time.time() - t_all:.0f}s")
