"""Benchmark split by facility type and calibration (helpers for x08)."""
import os, sys, time, argparse, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("XGB_DEVICE", "cpu")
from common import RES, fit_dd_model, predict_dd
from s03_energy import ym_add, ym_range
import s04_analysis as A
import s05_ml as ML
import s06_benchmark as BM
import x01_ml_audit as X
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
USES = X.USES
TYPE = {"보건": "health", "어린이집": "daycare"}


def gap_gb_cut(P, post_mask, correct):
    """Time-split GB per completion half-year (symmetric encodings: every building's encodings come from months before the cut)."""
    T = P[post_mask]; out = []
    for h, Tc in T.groupby(T.ps.map(X.half)):
        s0 = int(P[P.pphase != "ctrl"].loc[lambda d: d.ps.map(X.half) == h].ps.min())
        train = P[(P.ym < s0) & (P.pphase.isin(["ctrl", "pre"]))]
        Tc = Tc[Tc.pid.isin(train.pid.unique())]
        C = P[(P.pphase == "ctrl") & P.ym.isin(set(Tc.ym)) & (P.ym >= s0)]; C = C[C.pid.isin(train.pid.unique())]
        if Tc.empty or (correct and C.empty): continue
        both = pd.concat([Tc, C]); g = both.y.to_numpy(float) - np.asarray(ML._fit(train, both, X.GB_ROUNDS), float)
        gT = g[:len(Tc)]
        if correct:
            cm = pd.Series(g[len(Tc):]).groupby([C.use_class.to_numpy(), C.ym.to_numpy()]).mean()
            gT = gT - np.array([cm.get((u, m), np.nan) for u, m in zip(Tc.use_class, Tc.ym)])
        out.append(pd.DataFrame({"pid": Tc.pid.to_numpy(), "gap": gT}))
    return pd.concat(out).dropna() if out else pd.DataFrame(columns=["pid", "gap"])


def by_building_from_rows(Tpid, gaps):
    return pd.Series(gaps, index=Tpid.index).groupby(Tpid.to_numpy()).mean()


def imp_by_building(Bp, P):
    B2, E2 = BM.did_inputs(Bp, P, 0.0)
    r, Pt = A.imputation_did(B2, E2, "elec", uses=None, ctrl_roles=("control",), nboot=0, pretrend=False, min_win=0, return_tau=True)
    return float(r["att_log"]), Pt.groupby("pid").tau.mean()


def estimates(Bp, P, which, seed, rep):
    """Per predictor: (overall estimate on the predictor's own aggregation, per-building gap series or None)."""
    post = P.pphase == "post"; T = P[post]; out = {}
    if "naive" in which:
        cf = BM.cf_naive(P); a = T.groupby("pid").kwh.sum(); d = pd.concat([a.rename("a"), cf.rename("c")], axis=1).dropna()
        d = d[(d.a > 0) & (d.c > 0)]; s = np.log(d.a / d.c); out["naive"] = (float(s.mean()), s)
    if "dd" in which:
        cf = BM.cf_dd(P); a = T.groupby("pid").kwh.sum(); d = pd.concat([a.rename("a"), cf.rename("c")], axis=1).dropna()
        d = d[(d.a > 0) & (d.c > 0)]; s = np.log(d.a / d.c); out["dd"] = (float(s.mean()), s)
    for name, drop, ph in [("gb", (), ("ctrl", "pre")), ("gb_noenc", tuple(BM.ENC_COLS), ("ctrl", "pre")),
                           ("gb_not", ("t",), ("ctrl", "pre")), ("gb_noctrl", (), ("pre",))]:
        if name not in which: continue
        g = X.gap_gb(P[P.pphase.isin(list(ph))], T, drop); s = by_building_from_rows(T.pid, g)
        out[name] = (float(np.nanmean(g)), s)
    for name, corr in [("gb_did", True), ("gb_cut", False)]:
        if name not in which: continue
        D = gap_gb_cut(P, post, corr)
        out[name] = (float(D.gap.mean()) if len(D) else np.nan, D.groupby("pid").gap.mean() if len(D) else None)
    if "imp" in which:
        try:
            att, tau = imp_by_building(Bp, P)
            out["imp"] = (att, tau)
        except Exception as ex: print("imp failed", rep, ex, flush=True)
    return out


def rows_for(base, Bp, P, which, seed, rep):
    est = estimates(Bp, P, which, seed, rep)
    um = Bp.set_index("pid").use_class; rows = []
    for k, (e, s) in est.items():
        r = dict(base, estimator=k, est_log=e)
        if s is not None and len(s):
            for u, lab in TYPE.items():
                v = s[s.index.map(um) == u]; r[f"est_log_{lab}"] = float(v.mean()) if len(v) else np.nan; r[f"n_{lab}"] = int(len(v))
        rows.append(r)
    return rows


def drift_by_type(P):
    C = P[P.pphase == "ctrl"].copy(); C["yd"] = C.y - C.groupby("pid").y.transform("mean")
    cm = C.groupby(["use_class", "ym"]).yd.mean(); out = {}
    for u, lab in TYPE.items():
        v = []
        for pid, d in P[(P.pphase != "ctrl") & (P.use_class == u)].groupby("pid"):
            s = int(d.ps.iloc[0])
            pre = [cm.get((u, m), np.nan) for m in ym_range(ym_add(s, -12), ym_add(s, -1))]
            post = [cm.get((u, m), np.nan) for m in d[d.pphase == "post"].ym]
            v.append(np.nanmean(post) - np.nanmean(pre))
        out[f"drift_{lab}"] = float(np.nanmean(v)) if v else np.nan
    return out


def backtest_panel(P):
    """Out-of-time backtest at the deployment horizon: pseudo start s' = s-12, post = s-12..s-1, pre = months before s-12;"""
    t = P.pphase != "ctrl"; s = P.ps
    Q = P[~t | (P.ym < s)].copy()
    tt = Q.pphase != "ctrl"; s2 = Q.ps.map(lambda v: ym_add(int(v), -12) if pd.notna(v) else np.nan)
    Q.loc[tt, "pphase"] = np.where(Q.loc[tt, "ym"] >= s2[tt], "post", "pre")
    Q.loc[tt, "pe"] = Q.loc[tt, "ps"].map(lambda v: ym_add(int(v), -13)); Q.loc[tt, "ps"] = s2[tt]
    npre = Q[tt & (Q.pphase == "pre")].groupby("pid").size(); npost = Q[tt & (Q.pphase == "post")].groupby("pid").size()
    ok = set(npre[npre >= 9].index) & set(npost[npost >= A.MIN_WIN].index)
    drop = tt & ~Q.pid.isin(ok)
    Q.loc[drop & True, "pphase"] = "ctrl_drop"
    Q = Q[Q.pphase != "ctrl_drop"]
    Q.loc[Q.pphase == "ctrl", ["ps", "pe"]] = np.nan
    return Q, len(ok)


def ashrae_rows(P, rep):
    post = P[P.pphase == "post"]; rows = []
    for pid, d in P[P.pphase == "pre"].groupby("pid"):
        m = fit_dd_model(d.sort_values("ym").tail(24))
        if m is None: continue
        q = post[post.pid == pid]; pr = float(predict_dd(m, q).sum()); a = float(q.kwh.sum())
        if pr > 0 and a > 0: rows.append(dict(rep=rep, pid=pid, cvrmse=m["cvrmse"], nmbe=m["nmbe"], n=m["n"], gap_log=np.log(a / pr)))
    return rows


def propensity(B, E, Bp, e):
    """Odds of being a real treated building (grade A, same types) from built year, log floor area, coldest zone, pre intensity."""
    def feats(Bs, Es):
        X_ = pd.DataFrame(index=Bs.pid.values)
        X_["built"] = pd.to_numeric(Bs.built_year.astype(str).str[:4], errors="coerce").values
        X_["log_area"] = np.log(Bs.area_m2.values); X_["zone_c1"] = (Bs.zone.values == "중부1_춘천").astype(float)
        X_["health"] = (Bs.use_class.values == "보건").astype(float)
        X_["y_pre"] = Es.groupby("pid").y.mean().reindex(X_.index).values
        return X_
    T = B[(B.role == "treated") & (B.link_grade == "A_cand") & B.use_class.isin(USES)]
    ET = ML.prep(E[E.pid.isin(T.pid) & (E.phase == "pre")], "elec")
    XT, XC = feats(T, ET), feats(Bp, e)
    Xa = pd.concat([XT, XC]); Xa = Xa.fillna(Xa.median()); ya = np.r_[np.ones(len(XT)), np.zeros(len(XC))]
    mu, sd = Xa.mean(), Xa.std().replace(0, 1)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xa - mu) / sd, ya)
    pc = lr.predict_proba((Xa.iloc[len(XT):] - mu) / sd)[:, 1]
    return pd.Series(pc / (1 - pc), index=XC.index), XT, XC.fillna(Xa.median())


def draw_weighted(Bp, e, pools, rng, w):
    have = e.groupby("pid").ym.apply(set).to_dict(); out = {}
    for u, g in Bp.groupby("use_class"):
        pids = np.sort(g.pid.values); pool = pools.get(u)
        if len(pids) < 2 or pool is None or len(pool) == 0: continue
        k = max(1, int(round(BM.FRAC * len(pids)))); p = w.reindex(pids).fillna(w.median()).to_numpy(float); p = p / p.sum()
        for q in rng.choice(pids, k, replace=False, p=p):
            s, en = (int(x) for x in pool[rng.integers(len(pool))])
            if (len(set(ym_range(ym_add(s, -12), ym_add(s, -1))) & have[q]) >= A.MIN_WIN and
                    len(set(ym_range(ym_add(en, 1), ym_add(en, 12))) & have[q]) >= A.MIN_WIN): out[q] = (s, en)
    return out


def smd(a, b):
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)) if len(a) > 1 and len(b) > 1 else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="main"); ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--diff-reps", type=int, default=30); ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args(); t0 = time.time(); seed = a.seed
    B, E = A.load(); Bp, e = BM.population(B, E, "elec", USES)
    e = e.merge(Bp[["pid", "use_class"]], on="pid", how="left"); pools = BM.date_pools(B, USES, "use")
    ALL = ("naive", "dd", "gb", "gb_noenc", "gb_not", "gb_noctrl", "gb_did", "gb_cut", "imp")
    KEY = ("naive", "dd", "gb", "gb_did", "gb_cut", "imp")
    for part in a.part.split(","):
        R, AS = [], []
        if part == "main":
            for rep in range(a.reps):
                pseudo, _ = BM.draw(Bp, e, pools, np.random.default_rng([seed, rep]))
                P = BM.make_panel(e, pseudo).reset_index(drop=True)
                base = dict(design="random", rep=rep, n_pseudo=len(pseudo), **drift_by_type(P), drift_log=X.realised_drift(P))
                R += rows_for(base, Bp, P, ALL, seed, rep)
                Q, nq = backtest_panel(P)
                R += rows_for(dict(design="backtest_h12", rep=rep, n_pseudo=nq), Bp, Q, ("naive", "dd", "gb", "gb_did", "imp"), seed, rep)
                AS += ashrae_rows(P, rep)
                pd.DataFrame(R).to_csv(RES / "x03_main.csv", index=False); pd.DataFrame(AS).to_csv(RES / "x03_ashrae.csv", index=False)
                if (rep + 1) % 5 == 0: print(f"main rep {rep + 1} {time.time() - t0:.0f}s", flush=True)
        elif part == "btx":
            for rep in range(a.reps):
                pseudo, _ = BM.draw(Bp, e, pools, np.random.default_rng([seed, rep]))
                Q, nq = backtest_panel(BM.make_panel(e, pseudo).reset_index(drop=True))
                R += rows_for(dict(design="backtest_h12", rep=rep, n_pseudo=nq), Bp, Q, ("gb_noenc", "gb_not", "gb_noctrl", "gb_cut"), seed, rep)
                pd.DataFrame(R).to_csv(RES / "x03_btx.csv", index=False)
                if (rep + 1) % 10 == 0: print(f"btx rep {rep + 1} {time.time() - t0:.0f}s", flush=True)
        elif part == "prop":
            w, XT, XC = propensity(B, E, Bp, e); bal = []
            for rep in range(a.reps):
                pseudo = draw_weighted(Bp, e, pools, np.random.default_rng([seed, rep, 11]), w)
                P = BM.make_panel(e, pseudo).reset_index(drop=True)
                pt = XC.loc[list(pseudo)]; cc = XC.drop(index=list(pseudo))
                bal.append(dict(rep=rep, **{f"smd_{c}": smd(pt[c], cc[c]) for c in XC.columns}))
                R += rows_for(dict(design="propensity", rep=rep, n_pseudo=len(pseudo), drift_log=X.realised_drift(P)), Bp, P, KEY, seed, rep)
                pd.DataFrame(R).to_csv(RES / "x03_prop.csv", index=False); pd.DataFrame(bal).to_csv(RES / "x03_prop_balance.csv", index=False)
                if (rep + 1) % 10 == 0: print(f"prop rep {rep + 1} {time.time() - t0:.0f}s", flush=True)
            ref = dict(rep=-1, **{f"smd_{c}": smd(XT[c].fillna(XT[c].median()), XC[c]) for c in XC.columns})
            pd.DataFrame(bal + [ref]).to_csv(RES / "x03_prop_balance.csv", index=False)
        elif part == "cohort":
            Bc, ec = BM.population(B, E, "elec", USES); ec = ec.merge(Bc[["pid", "use_class"]], on="pid", how="left")
            T = B[(B.role == "treated") & B.start_ym.notna() & B.end_ym.notna()]
            idx = lambda v: (int(v) // 100) * 12 + int(v) % 100
            gap = {u: int(np.median([idx(b) - idx(a_) + 1 for a_, b in zip(g.start_ym, g.end_ym)])) for u, g in T.groupby("use_class")}
            have = ec.groupby("pid").ym.apply(set).to_dict(); pseudo = {}
            for b in Bc[Bc.role == "aux_2023"].itertuples():
                pe = ym_add(int(b.start_ym), -16); h = have.get(b.pid, set())
                if len(set(ym_range(ym_add(pe, 1), ym_add(pe, 12))) & h) < A.MIN_WIN: continue
                for g_ in range(gap.get(b.use_class, 0), -1, -1):
                    ps = ym_add(pe, -g_ + 1)
                    if len(set(ym_range(ym_add(ps, -12), ym_add(ps, -1))) & h) >= A.MIN_WIN: pseudo[b.pid] = (ps, pe); break
            Bc = Bc[(Bc.role == "control") | Bc.pid.isin(list(pseudo))]; ec = ec[ec.pid.isin(Bc.pid)]
            P = BM.make_panel(ec, pseudo).reset_index(drop=True)
            R += rows_for(dict(design="selection_cohort", rep=0, n_pseudo=len(pseudo), n_control=int((Bc.role == "control").sum()),
                               drift_log=X.realised_drift(P)), Bc, P, ALL, seed, 0)
            pd.DataFrame(R).to_csv(RES / "x03_cohort.csv", index=False)
        elif part == "diff":
            for delta in (-0.06, 0.0, 0.06):
                for rep in range(a.diff_reps):
                    pseudo, _ = BM.draw(Bp, e, pools, np.random.default_rng([seed, rep]))
                    P = BM.make_panel(e, pseudo).reset_index(drop=True)
                    t = P.pphase != "ctrl"; sh = delta * X.months_since(P.ym) / 12.0 * t
                    P["y"] = P.y + sh; P["kwh"] = P.kwh * np.exp(sh)
                    R += rows_for(dict(design="diff_drift", delta=delta, rep=rep, n_pseudo=len(pseudo)), Bp, P, ("naive", "dd", "gb", "gb_did", "imp"), seed, rep)
                    pd.DataFrame(R).to_csv(RES / "x03_diff.csv", index=False)
                print(f"diff {delta} {time.time() - t0:.0f}s", flush=True)
        elif part == "calendar":
            C = e[["pid", "ym", "y", "use_class"]].copy(); prev = C.assign(ym=C.ym.map(lambda v: ym_add(int(v), 12)))
            C = C.merge(prev.rename(columns={"y": "y_prev"})[["pid", "ym", "y_prev"]], on=["pid", "ym"])
            C["yoy"] = C.y - C.y_prev; C["year"] = C.ym // 100
            g = C.groupby(["use_class", "year"]).agg(yoy_log=("yoy", "mean"), n_bld=("pid", "nunique"), n_pairs=("yoy", "size")).reset_index()
            g.to_csv(RES / "x03_calendar.csv", index=False); print(g.round(4).to_string())
    print(f"total {time.time() - t0:.0f}s")
