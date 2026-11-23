"""Step 5: gradient-boosting counterfactual and degree-day baselines."""
import os, sys, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, fit_dd_model, predict_dd
from s03_energy import ym_add
from s04_analysis import load, MIN_WIN, complete_windows

DEVICE = os.environ.get("XGB_DEVICE", "cuda")
from common import COVID_MONTHS as COVID
PARAMS = dict(objective="reg:squarederror", eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, tree_method="hist", device=DEVICE, seed=0)

def months_since(ym): return (ym // 100 - 2020) * 12 + (ym % 100 - 1)

def features(df, enc, loo=False):
    """loo=True for training rows: building encodings exclude the row's own target (no target leakage)."""
    X = pd.DataFrame(index=df.index)
    m = df.ym % 100
    X["sin_m"] = np.sin(2 * np.pi * m / 12); X["cos_m"] = np.cos(2 * np.pi * m / 12)
    for c in ["HDD14", "HDD18", "CDD20", "CDD24", "tmean"]: X[c] = df[c] / df.days
    X["t"] = months_since(df.ym); X["covid"] = df.ym.isin(COVID).astype(int)
    X["log_gfa"] = np.log(df.gr_gfa)
    X["sido"] = df.sido.astype("category").cat.codes
    e = enc.reindex(df.pid).set_index(df.index)
    for c in enc.columns: X[c] = e[c]
    if loo:
        g = df.groupby("pid").y; n = g.transform("size"); sm = g.transform("sum")
        X["enc_mean"] = np.where(n > 1, (sm - df.y) / (n - 1).clip(lower=1), np.nan)
        for col, mask in [("enc_winter", (df.ym % 100) <= 2), ("enc_summer", (df.ym % 100).isin([7, 8]))]:
            gs = df.y.where(mask); cs = gs.groupby(df.pid).transform("count"); ss = gs.groupby(df.pid).transform("sum")
            own = mask.astype(int); m_loo = (ss - df.y.where(mask, 0)) / (cs - own).replace(0, np.nan)
            X[col] = m_loo - X["enc_mean"]
    return X

def encodings(train):
    g = train.groupby("pid")
    enc = pd.DataFrame({"enc_mean": g.y.mean(), "enc_std": g.y.std(),
                        "enc_winter": train[train.ym % 100 <= 2].groupby("pid").y.mean() - g.y.mean(),
                        "enc_summer": train[(train.ym % 100).isin([7, 8])].groupby("pid").y.mean() - g.y.mean()})
    return enc

SMEAR_FOLDS = int(os.environ.get("GB_SMEAR_FOLDS", "0"))
LAST = {"smear": 1.0}
ENC_INFO = dict(building_encodings=["enc_mean", "enc_std", "enc_winter", "enc_summer"], n_building_encodings=4,
                leave_one_out_on_training_rows=["enc_mean", "enc_winter", "enc_summer"], n_leave_one_out=3,
                not_leave_one_out=["enc_std"])

def _fit(train, test, rounds, drop=()):
    enc = encodings(train)
    Xtr, Xte = features(train, enc, loo=True), features(test, enc)
    if drop: Xtr, Xte = Xtr.drop(columns=list(drop)), Xte.drop(columns=list(drop))
    b = xgb.train(PARAMS, xgb.DMatrix(Xtr, label=train.y), rounds)
    return b.predict(xgb.DMatrix(Xte))

def smear_factor(train, rounds=600, drop=(), folds=None, seed=0):
    """Duan smearing factor mean(exp(y - yhat)) from out-of-fold residuals of the training rows."""
    folds = SMEAR_FOLDS if folds is None else folds
    if folds < 2 or len(train) < 10 * folds: return 1.0
    fold = np.random.default_rng(seed).integers(0, folds, len(train)); res = []
    for k in range(folds):
        tr, ho = train[fold != k], train[fold == k]
        ho = ho[ho.pid.isin(tr.pid.unique())]
        if len(ho) and len(tr): res.append(ho.y.to_numpy() - _fit(tr, ho, rounds, drop))
    if not res: return 1.0
    r = np.concatenate(res); r = r[np.isfinite(r)]
    return float(np.mean(np.exp(r))) if len(r) else 1.0

def fit_predict(train, test, rounds=600, drop=(), smear=None):
    """Log prediction including the log smearing factor, so that exp(.) is the retransformed level (LAST['smear'] = factor used)."""
    pred = _fit(train, test, rounds, drop)
    LAST["smear"] = smear_factor(train, rounds, drop) if smear is None else float(smear)
    return pred + np.log(LAST["smear"])

def prep(E, carrier):
    e = E[(E.carrier == carrier) & (E.kwh > 0)].copy()
    e["y"] = np.log(e.kwh / e.gr_gfa / e.days)
    return e

def savings_from_pred(test, pred):
    t = test.assign(pred_kwh=np.exp(pred) * test.gr_gfa * test.days)
    g = t.groupby("pid").agg(actual=("kwh", "sum"), pred=("pred_kwh", "sum"), n=("kwh", "size"))
    g["sav"] = 1 - g.actual / g.pred
    return g

def placebo_controls(e, B):
    ctrl = e[(e.role == "control") & (e.phase == "pre")]; last = ctrl.groupby("pid").ym.transform("max")
    hold = ctrl[ctrl.ym > last.map(lambda x: ym_add(int(x), -12))]
    train = pd.concat([e[(e.role == "treated") & (e.phase == "pre")], ctrl.drop(hold.index)])
    keep = hold.pid.isin(train.pid.unique())
    hold = hold[keep]; s_ml = savings_from_pred(hold, fit_predict(train, hold))
    rows = []
    for pid, h in hold.groupby("pid"):
        tr = ctrl[(ctrl.pid == pid) & ~ctrl.index.isin(hold.index)]
        m = fit_dd_model(tr.tail(24))
        dd = 1 - h.kwh.sum() / predict_dd(m, h).sum() if m else np.nan
        rows.append(dict(pid=pid, sav_dd=dd, cvrmse=m["cvrmse"] if m else np.nan))
    out = s_ml.join(pd.DataFrame(rows).set_index("pid"))
    return out[out.n >= MIN_WIN]

def placebo_treated_pre(e, B):
    T = B[B.role == "treated"].set_index("pid")
    t = e[(e.role == "treated") & (e.phase == "pre")].copy(); s = t.pid.map(T.start_ym)
    hold = t[(t.ym >= s.map(lambda x: ym_add(int(x), -12)))]
    early = t.drop(hold.index)
    train = pd.concat([early, e[(e.role == "control") & (e.phase == "pre")]])
    hold = hold[hold.pid.isin(early.pid.unique())]
    if hold.empty: return pd.DataFrame()
    s_ml = savings_from_pred(hold, fit_predict(train, hold))
    rows = []
    for pid, h in hold.groupby("pid"):
        m = fit_dd_model(early[early.pid == pid])
        rows.append(dict(pid=pid, sav_dd=1 - h.kwh.sum() / predict_dd(m, h).sum() if m else np.nan, cvrmse=m["cvrmse"] if m else np.nan))
    out = s_ml.join(pd.DataFrame(rows).set_index("pid"))
    return out[out.n >= MIN_WIN]


def placebo_pseudo_treated(e, B, frac=0.25, reps=4, seed=0):
    """P3: pseudo-treat a random 25% of controls with pseudo completion dates drawn from the treated distribution;"""
    rng = np.random.default_rng(seed)
    T = B[B.role == "treated"]; ends = T.end_ym.dropna().astype(int).values
    ctrl = e[(e.role == "control") & (e.phase == "pre")]; pids = ctrl.pid.unique(); out = []
    for r in range(reps):
        pick = rng.choice(pids, max(1, int(frac * len(pids))), replace=False)
        hold_idx = []
        for p in pick:
            d = ctrl[ctrl.pid == p]; pe = int(rng.choice(ends)); win = [ym_add(pe, k) for k in range(1, 13)]
            h = d[d.ym.isin(win)]
            if len(h) >= MIN_WIN and (d.ym <= pe).sum() >= MIN_WIN: hold_idx += list(h.index)
        if not hold_idx: continue
        hold = e.loc[hold_idx]; train = e.drop(hold_idx)
        s = savings_from_pred(hold, fit_predict(train, hold)); s = s[s.n >= MIN_WIN]
        rows = []
        for pid, h in hold.groupby("pid"):
            tr = ctrl[(ctrl.pid == pid) & (ctrl.ym < h.ym.min())].tail(24)
            m = fit_dd_model(tr)
            rows.append(dict(pid=pid, sav_dd=1 - h.kwh.sum() / predict_dd(m, h).sum() if m else np.nan))
        out.append(s.join(pd.DataFrame(rows).set_index("pid")).assign(rep=r))
    return pd.concat(out) if out else pd.DataFrame()

def treated_effect(e, B, nboot=200):
    T = B[B.role == "treated"].set_index("pid")
    post = e[(e.role == "treated") & (e.phase == "post")].copy()
    post = post[post.ym <= post.pid.map(T.end_ym).map(lambda x: ym_add(int(x), 12))]
    train = pd.concat([e[(e.role == "treated") & (e.phase == "pre")], e[(e.role == "control") & (e.phase == "pre")]])
    post = post[post.pid.isin(train.pid.unique()) & post.pid.isin(complete_windows(e, B))]
    s = savings_from_pred(post, fit_predict(train, post)); s = s[s.n >= MIN_WIN]; sm0 = LAST["smear"]
    rng = np.random.default_rng(0); pids = train.pid.unique(); agg = []
    for b in range(nboot):
        pick = rng.choice(pids, len(pids), replace=True)
        tb = pd.concat([train[train.pid == p].assign(pid=f"{p}#{k}") for k, p in enumerate(pick)])
        tp = post[post.pid.isin(set(pick))]
        if tp.empty: continue
        first = {p: f"{p}#{k}" for k, p in reversed(list(enumerate(pick)))}
        tp = tp.assign(pid=tp.pid.map(first))
        sb = savings_from_pred(tp, fit_predict(tb, tp, smear=sm0)); sb = sb[sb.n >= MIN_WIN]
        mult = pd.Series(pick).value_counts()
        w = sb.index.str.split("#").str[0].map(mult).fillna(0).values
        agg.append(1 - (sb.actual.values * w).sum() / (sb.pred.values * w).sum())
    return s, np.array(agg)

if __name__ == "__main__":
    B, E = load(); summary = {}; outs, plac = [], []
    for car in ["elec", "gas"]:
        e = prep(E, car)
        if e[e.role == "treated"].pid.nunique() < 10: continue
        p1 = placebo_controls(e, B).assign(test="P1_control_time_extrap", carrier=car)
        p2 = placebo_treated_pre(e, B).assign(test="P2_treated_pre", carrier=car)
        p3 = placebo_pseudo_treated(e, B).assign(test="P3_pseudo_treated", carrier=car)
        plac += [p1, p2, p3]
        s, boot = treated_effect(e, B, nboot=int(os.environ.get("NBOOT", "200")))
        outs.append(s.assign(carrier=car))
        summary[car] = dict(n_treated=int(len(s)), agg_saving=float(1 - s.actual.sum() / s.pred.sum()),
                            median_saving=float(s.sav.median()), boot_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) else None,
                            placebo={t: dict(n=int(len(p)), ml_median=float(p.sav.median()), ml_mae=float(p.sav.abs().median()),
                                             dd_median=float(p.sav_dd.median()), dd_mae=float(p.sav_dd.abs().median()))
                                     for t, p in [("P1_control_time_extrap", p1), ("P2_treated_pre", p2), ("P3_pseudo_treated", p3)] if len(p)})
    if outs: pd.concat(outs).to_csv(RES / "H1_ml_building.csv")
    if plac: pd.concat(plac).to_csv(RES / "H2_ml_placebo.csv")
    (RES / "H0_ml_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1)); print(json.dumps(summary, indent=1, ensure_ascii=False))
