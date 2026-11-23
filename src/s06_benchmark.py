"""Step 6: pseudo-treatment benchmark of the estimators."""
import os, sys, time, argparse, inspect, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RES, fit_dd_model, predict_dd
from s03_energy import ym_add, ym_range
import s04_analysis as A
import s05_ml as ML
import xgboost as xgb

warnings.filterwarnings("ignore")
TAUS = (0.0, -0.10, -0.05)
POP_TRIM = 3
FRAC = 0.30
NBOOT = 500
IMP_NBOOT = 200
ENC_COLS = ["enc_mean", "enc_std", "enc_winter", "enc_summer"]
MIN_BOOT_N = 2
USE_EN = {"보건": "health", "어린이집": "daycare", "경로당": "senior", "기타": "other"}
TESTS = {"random": "Random assignment among not-yet-treated", "selection_cohort": "Selection-cohort placebo (2023 vs 2024-25 selections)",
         "treated_pre_period": "Treated pre-period placebo (fake start at s-12)"}
SMEAR_SENS_FOLDS = 3
LABELS = {"naive": "Naive pre–post", "dd": "Degree-day baseline", "gb": "Gradient boosting", "gb_smear_level": "GB, level with smearing (sensitivity)",
          "gb_noenc": "GB, no building encodings", "gb_not": "GB, no time feature",
          "gb_noctrl": "GB, no control rows", "gb_nocovid": "GB, no pandemic flag",
          "did": "Stacked DiD (province×month)", "did_usefe": "Stacked DiD (use×month)", "imp": "Imputation DiD"}


def population(B, E, carrier, uses):
    Bp = B[B.role.isin(["control", "aux_2023"]) & (B.link_grade == "A_cand") & B.use_class.isin(uses)]
    Ep = E[E.pid.isin(Bp.pid) & (E.phase == "pre")]
    Ep = Ep[Ep.ym < Ep.start_ym.map(lambda v: ym_add(int(v), -POP_TRIM))]
    e = ML.prep(Ep, carrier)
    Bp = Bp[Bp.pid.isin(e.pid.unique())].copy()
    return Bp, e.reset_index(drop=True)


def date_pools(B, uses, mode):
    T = B[(B.role == "treated") & B.start_ym.notna() & B.end_ym.notna()]
    T = T[T.end_ym >= T.start_ym]
    allp = T[["start_ym", "end_ym"]].astype(int).values
    if mode == "all": return {u: allp for u in uses}
    return {u: T[T.use_class == u][["start_ym", "end_ym"]].astype(int).values for u in uses}


def draw(Bp, e, pools, rng, feasible_dates=False):
    """Returns dict pid -> (s, e) of feasible pseudo-treated buildings, and the number drawn."""
    have = e.groupby("pid").ym.apply(set).to_dict()
    out, n_drawn = {}, 0
    for u, g in Bp.groupby("use_class"):
        pids = np.sort(g.pid.values); pool = pools.get(u)
        if len(pids) < 2 or pool is None or len(pool) == 0: continue
        k = max(1, int(round(FRAC * len(pids))))
        for p in rng.choice(pids, k, replace=False):
            n_drawn += 1
            ok = lambda s, en: (len(set(ym_range(ym_add(s, -12), ym_add(s, -1))) & have[p]) >= A.MIN_WIN and
                                len(set(ym_range(ym_add(en, 1), ym_add(en, 12))) & have[p]) >= A.MIN_WIN)
            cand = pool
            if feasible_dates:
                cand = np.array([q for q in pool if ok(int(q[0]), int(q[1]))])
                if len(cand) == 0: continue
            s, en = (int(x) for x in cand[rng.integers(len(cand))])
            if ok(s, en): out[p] = (s, en)
    return out, n_drawn


def make_panel(e, pseudo):
    """Apply the pseudo dates: drop pseudo-treated months in [s, e] and after e+12; mark pseudo phases."""
    s = e.pid.map({p: v[0] for p, v in pseudo.items()}); en = e.pid.map({p: v[1] for p, v in pseudo.items()})
    is_t = s.notna()
    ph = np.where(~is_t, "ctrl", np.where(e.ym < s, "pre", np.where((e.ym > en) & (e.ym <= en.map(
        lambda x: ym_add(int(x), 12) if pd.notna(x) else 0)), "post", "drop")))
    P = e.assign(ps=s, pe=en, pphase=ph)
    return P[P.pphase != "drop"].copy()


def cf_naive(P):
    """Sum of the pre-year months [ps-12, ps-1] whose calendar month also occurs among the building's pseudo-post months."""
    pre = P[(P.pphase == "pre") & (P.ym >= P.ps.map(lambda x: ym_add(int(x), -12) if pd.notna(x) else 0))]
    post = P[P.pphase == "post"]
    cal = set(zip(post.pid, post.ym % 100))
    keep = [(p, m % 100) in cal for p, m in zip(pre.pid, pre.ym)]
    return pre[keep].groupby("pid").kwh.sum()


def cf_dd(P):
    out = {}
    post = P[P.pphase == "post"]
    for pid, d in P[P.pphase == "pre"].groupby("pid"):
        m = fit_dd_model(d.sort_values("ym").tail(24))
        if m is None: continue
        out[pid] = float(predict_dd(m, post[post.pid == pid]).sum())
    return pd.Series(out, dtype=float)


def gb_predict(train, test, drop=()):
    """drop == () -> the project's s05_ml.fit_predict unchanged; otherwise the same model without some features."""
    return ML.fit_predict(train, test, drop=tuple(drop))


def cf_gb(P, drop=(), phases=("ctrl", "pre"), smear_level=False):
    """Per-building counterfactual post sum (level, plain exp) with attrs['loggap'] = per-building sum and count of the monthly log gaps"""
    train = P[P.pphase.isin(list(phases))]; test = P[P.pphase == "post"]
    pred = ML._fit(train, test, 600, tuple(drop))
    lvl = np.exp(pred) * test.gr_gfa * test.days
    if smear_level:
        sm = ML.smear_factor(train, 600, tuple(drop), folds=SMEAR_SENS_FOLDS)
        out = (lvl * sm).groupby(test.pid).sum(); out.attrs["smear"] = sm
        return out
    out = lvl.groupby(test.pid).sum()
    gap = pd.Series(test.y.to_numpy(float) - np.asarray(pred, float), index=test.index)
    out.attrs["loggap"] = pd.DataFrame({"gs": gap.groupby(test.pid).sum(), "n": gap.groupby(test.pid).size()})
    out.attrs["smear"] = 1.0
    return out


def agg_from_cf(actual, cf, rng, shift=0.0):
    """Mean log effect (primary), its bootstrap CI over buildings (counterfactual fixed), and the same numbers as 1 - exp(.)."""
    d = pd.concat([actual.rename("a"), cf.rename("c")], axis=1).dropna()
    d = d[(d.a > 0) & (d.c > 0)]
    if d.empty: return None
    a, c = d.a.values, d.c.values
    lg = cf.attrs.get("loggap") if hasattr(cf, "attrs") else None
    lo = hi = np.nan
    if lg is not None:
        g = lg.reindex(d.index); gs, n = g.gs.to_numpy(float) + shift * g.n.to_numpy(float), g.n.to_numpy(float)
        elog = float(gs.sum() / n.sum()); scale = "mean over building-months of log actual - log predicted"
        if len(d) >= MIN_BOOT_N:
            i = rng.integers(0, len(d), (NBOOT, len(d))); lo, hi = np.percentile(gs[i].sum(1) / n[i].sum(1), [2.5, 97.5])
    else:
        r = np.log(a / c); elog = float(r.mean()); scale = "mean over buildings of log(actual sum / counterfactual sum)"
        if len(d) >= MIN_BOOT_N:
            i = rng.integers(0, len(d), (NBOOT, len(d))); lo, hi = np.percentile(r[i].mean(1), [2.5, 97.5])
    return dict(est=float(1 - np.exp(elog)), est_log=elog, lo=float(1 - np.exp(hi)), hi=float(1 - np.exp(lo)), lo_log=float(lo), hi_log=float(hi),
                est_agg_ratio=float(1 - a.sum() / c.sum()), log_effect=scale, n_used=len(d))


def did_inputs(Bp, P, tau):
    """B/E copies in the format of s04_analysis: pseudo-treated -> role 'treated' with pseudo dates, rest 'control'."""
    pseudo = P[P.pphase != "ctrl"].groupby("pid")[["ps", "pe"]].first()
    B2 = Bp.copy(); B2["role"] = np.where(B2.pid.isin(pseudo.index), "treated", "control")
    B2["start_ym"] = B2.pid.map(pseudo.ps).fillna(B2.start_ym); B2["end_ym"] = B2.pid.map(pseudo.pe).fillna(B2.end_ym)
    E2 = P.copy(); t = E2.pphase != "ctrl"
    E2["role"] = np.where(t, "treated", "control")
    E2["start_ym"] = np.where(t, E2.ps, E2.start_ym); E2["end_ym"] = np.where(t, E2.pe, E2.end_ym)
    E2["phase"] = np.where(E2.pphase == "post", "post", "pre")
    E2.loc[E2.pphase == "post", "kwh"] *= (1 + tau)
    return B2, E2.drop(columns=["ps", "pe", "pphase", "y"])


def est_did(B2, E2, carrier, **kw):
    r = A.stacked_did(B2, E2, carrier, min_win=0, **kw)
    if r is None: raise RuntimeError("stacked_did returned None (no cohort with >= 3 pseudo-treated)")
    r = r[r.term == "post"].iloc[0]
    if not np.isfinite(r.coef) or not np.isfinite(r.se) or r.se <= 0: raise RuntimeError("stacked_did: coef/se not finite")
    cw, sw = r.get("coef_cohort_w", np.nan), r.get("se_cohort_w", np.nan)
    if np.isfinite(cw) and np.isfinite(sw) and sw > 0:
        return dict(est=float(1 - np.exp(cw)), est_log=float(cw), lo=float(1 - np.exp(r.hi_cohort_w)), hi=float(1 - np.exp(r.lo_cohort_w)),
                    lo_log=float(r.lo_cohort_w), hi_log=float(r.hi_cohort_w), n_used=int(r.n_treated))
    return dict(est=float(1 - np.exp(r.coef)), est_log=float(r.coef), lo=float(1 - np.exp(r.hi)), hi=float(1 - np.exp(r.lo)),
                lo_log=float(r.lo), hi_log=float(r.hi), n_used=int(r.n_treated))


def est_imp(B2, E2, carrier, nboot=None):
    """s04_analysis.imputation_did (Borusyak et al.): returns att_log and a building-cluster refit-bootstrap CI."""
    r = A.imputation_did(B2, E2, carrier, uses=None, ctrl_roles=("control",), nboot=IMP_NBOOT if nboot is None else nboot, pretrend=False, min_win=0)
    coef = r["att_log"]; lo, hi = r["ci_log"]
    if not np.isfinite(coef): raise RuntimeError("imputation_did: att_log not finite (no supported treated post row)")
    return dict(est=float(1 - np.exp(coef)), est_log=float(coef), lo=float(1 - np.exp(hi)), hi=float(1 - np.exp(lo)),
                lo_log=float(lo), hi_log=float(hi), n_used=int(r["n_treated"]))


def evaluate(base, Bp, P, carrier, seed, rep, has_imp=True):
    rows = []
    act0 = P[P.pphase == "post"].groupby("pid").kwh.sum()
    cfs = {}
    for name, fn in [("naive", lambda: cf_naive(P)), ("dd", lambda: cf_dd(P)), ("gb", lambda: cf_gb(P)),
                     ("gb_noenc", lambda: cf_gb(P, ENC_COLS)), ("gb_not", lambda: cf_gb(P, ["t"])),
                     ("gb_noctrl", lambda: cf_gb(P, phases=("pre",))), ("gb_nocovid", lambda: cf_gb(P, ["covid"])),
                     ("gb_smear_level", lambda: cf_gb(P, smear_level=True))]:
        t0 = time.time()
        try: cfs[name] = (fn(), time.time() - t0, "ok")
        except Exception as ex: cfs[name] = (None, time.time() - t0, f"{type(ex).__name__}: {ex}"[:200])
    for tau in TAUS:
        truth = dict(tau=tau, true_saving=-tau, true_log=float(np.log1p(tau)))
        for name, (cf, rt, st) in cfs.items():
            r = agg_from_cf(act0 * (1 + tau), cf, np.random.default_rng([seed, rep, 7]), shift=float(np.log1p(tau))) if cf is not None else None
            if r is None: rows.append(dict(base, **truth, estimator=name, status=st if st != "ok" else "no estimate", runtime_s=rt))
            else: rows.append(dict(base, **truth, estimator=name, status="ok", runtime_s=rt, gb_smear=cf.attrs.get("smear", np.nan) if name.startswith("gb") else np.nan,
                                   ci_method="boot_treated_fixed_cf" if r["n_used"] >= MIN_BOOT_N else "none (1 building)", **r))
        B2, E2 = did_inputs(Bp, P, tau)
        ests = [("did", lambda: est_did(B2, E2, carrier, ctrl_trim=3)),
                ("did_usefe", lambda: est_did(B2, E2, carrier, fe="use", ctrl_trim=3, anticip=3))]
        if has_imp: ests.append(("imp", lambda: est_imp(B2, E2, carrier)))
        for name, fn in ests:
            t0 = time.time()
            try: rows.append(dict(base, **truth, estimator=name, status="ok", ci_method={"imp": "boot_building_refit_stratified"}.get(name, "cluster_building_pyfixest_default_ssc"), **fn(), runtime_s=time.time() - t0))
            except Exception as ex:
                rows.append(dict(base, **truth, estimator=name, status=f"{type(ex).__name__}: {ex}"[:200], runtime_s=time.time() - t0))
    return rows


def _empty(base, why):
    return pd.DataFrame([dict(base, tau=t, true_saving=-t, true_log=float(np.log1p(t)), estimator="(all)", status=why) for t in TAUS])

def selection_cohort(B, E, carrier, uses, seed, min_pre=None, min_post=None, trim=3):
    """All 2023 selections (aux_2023) pseudo-treated before their own start; controls = 2024-25 selections."""
    min_pre = A.MIN_WIN if min_pre is None else min_pre; min_post = A.MIN_WIN if min_post is None else min_post
    Bp, e = population(B, E, carrier, uses)
    base = dict(test="selection_cohort", carrier=carrier, rep=0, n_pop=len(Bp))
    T = B[(B.role == "treated") & B.start_ym.notna() & B.end_ym.notna()]
    idx = lambda v: (int(v) // 100) * 12 + int(v) % 100
    gap = {u: int(np.median([idx(b) - idx(a) + 1 for a, b in zip(g.start_ym, g.end_ym)])) for u, g in T.groupby("use_class")}
    have = e.groupby("pid").ym.apply(set).to_dict(); pseudo, gaps = {}, []
    for b in Bp[Bp.role == "aux_2023"].itertuples():
        pe = ym_add(int(b.start_ym), -(12 + trim + 1)); h = have.get(b.pid, set())
        if len(set(ym_range(ym_add(pe, 1), ym_add(pe, 12))) & h) < min_post: continue
        for g in range(gap.get(b.use_class, 0), -1, -1):
            ps = ym_add(pe, -g + 1)
            if len(set(ym_range(ym_add(ps, -12), ym_add(ps, -1))) & h) >= min_pre:
                pseudo[b.pid] = (ps, pe); gaps.append(g); break
    Bp = Bp[(Bp.role == "control") | Bp.pid.isin(list(pseudo))]; e = e[e.pid.isin(Bp.pid)]
    base.update(n_drawn=int((B.role == "aux_2023").sum()), n_pseudo=len(pseudo), n_control=int((Bp.role == "control").sum()),
                pseudo_gap_median=float(np.median(gaps)) if gaps else np.nan, real_gap_median=float(np.median(list(gap.values()))) if gap else np.nan)
    if len(pseudo) < 3 or base["n_control"] < 3: return _empty(base, f"not enough buildings (pseudo-treated {len(pseudo)}, controls {base['n_control']})")
    base.update({f"n_pseudo_{USE_EN.get(u, u)}": int(n) for u, n in Bp[Bp.pid.isin(list(pseudo))].use_class.value_counts().items()})
    return pd.DataFrame(evaluate(base, Bp, make_panel(e, pseudo), carrier, seed, 0))

def treated_pre_period(B, E, carrier, uses, seed, min_pre=None, min_post=8, anticip=3):
    """Real treated buildings, fake start at s-12: pseudo-pre = s-24..s-13 (collection pass 2), pseudo-post = s-12..s-(anticip+1);"""
    min_pre = A.MIN_WIN if min_pre is None else min_pre
    Bc, ec = population(B, E, carrier, uses)
    Tb = B[(B.role == "treated") & (B.link_grade == "A_cand") & B.use_class.isin(uses)]
    et = ML.prep(E[E.pid.isin(Tb.pid) & (E.phase == "pre")], carrier)
    base = dict(test="treated_pre_period", carrier=carrier, rep=0, n_pop=len(Bc) + len(Tb), n_drawn=len(Tb), n_control=len(Bc))
    have = et.groupby("pid").ym.apply(set).to_dict(); pseudo = {}
    for b in Tb.itertuples():
        s0 = int(b.start_ym); h = have.get(b.pid, set())
        if (len(set(ym_range(ym_add(s0, -24), ym_add(s0, -13))) & h) >= min_pre and
                len(set(ym_range(ym_add(s0, -12), ym_add(s0, -anticip - 1))) & h) >= min_post):
            pseudo[b.pid] = (ym_add(s0, -12), ym_add(s0, -13))
    base["n_pseudo"] = len(pseudo)
    if len(pseudo) < 3 or len(Bc) < 3: return _empty(base, f"not enough buildings (treated with months s-24..s-13: {len(pseudo)}, controls {len(Bc)}); needs pass 2")
    et = et[et.pid.isin(list(pseudo))]; et = et[et.ym < et.start_ym.map(lambda v: ym_add(int(v), -anticip))]
    Bp = pd.concat([Bc, Tb[Tb.pid.isin(list(pseudo))]], ignore_index=True); e = pd.concat([ec, et], ignore_index=True)
    base.update({f"n_pseudo_{USE_EN.get(u, u)}": int(n) for u, n in Tb[Tb.pid.isin(list(pseudo))].use_class.value_counts().items()})
    return pd.DataFrame(evaluate(base, Bp, make_panel(e, pseudo), carrier, seed, 0))


def run(B, E, carrier, uses, reps, seed, date_pool, flush=None, feasible_dates=False):
    Bp, e = population(B, E, carrier, uses)
    pools = date_pools(B, uses, date_pool)
    print(f"[{carrier}] population {len(Bp)} buildings {Bp.use_class.value_counts().to_dict()}; "
          f"date pool sizes { {u: len(v) for u, v in pools.items()} }", flush=True)
    has_imp = hasattr(A, "imputation_did")
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng([seed, rep])
        pseudo, n_drawn = draw(Bp, e, pools, rng, feasible_dates)
        base = dict(test="random", carrier=carrier, rep=rep, n_pop=len(Bp), n_drawn=n_drawn, n_pseudo=len(pseudo), n_control=len(Bp) - len(pseudo))
        if not pseudo:
            rows += [dict(base, tau=t, estimator="(all)", status="no feasible pseudo-treated building") for t in TAUS]; continue
        base.update({f"n_pseudo_{USE_EN.get(u, u)}": int(n) for u, n in Bp[Bp.pid.isin(list(pseudo))].use_class.value_counts().items()})
        um = Bp.set_index("pid").use_class
        DATES.extend(dict(carrier=carrier, rep=rep, pid=p, use=USE_EN.get(um.get(p), um.get(p)), pseudo_start=s, pseudo_end=en) for p, (s, en) in pseudo.items())
        rows += evaluate(base, Bp, make_panel(e, pseudo), carrier, seed, rep, has_imp)
        if (rep + 1) % 10 == 0:
            print(f"[{carrier}] rep {rep + 1}/{reps}", flush=True)
            if flush: flush(rows)
    return pd.DataFrame(rows)


DATES = []

def date_table(B, uses):
    """Distribution of pseudo start years-quarters against the real treated one (share of buildings), by carrier."""
    q = lambda v: f"{int(v) // 100}Q{(int(v) % 100 - 1) // 3 + 1}"
    T = B[(B.role == "treated") & (B.link_grade == "A_cand") & B.use_class.isin(uses) & B.start_ym.notna()]
    out = [pd.DataFrame({"group": "real treated (grade A, same uses)", "carrier": "all", "start_quarter": T.start_ym.map(q)})]
    if DATES:
        D = pd.DataFrame(DATES); out.append(pd.DataFrame({"group": "pseudo-treated (all replications)", "carrier": D.carrier, "start_quarter": D.pseudo_start.map(q)}))
    X = pd.concat(out, ignore_index=True)
    t = X.groupby(["group", "carrier", "start_quarter"]).size().rename("n").reset_index()
    t["share"] = t.n / t.groupby(["group", "carrier"]).n.transform("sum")
    return t


def summarise(R):
    R = R[(R.estimator != "(all)") | (R.test != "random")].copy()
    R["err"] = R.est - R.true_saving; R["err_log"] = R.est_log - R.true_log
    if "lo_log" not in R: R["lo_log"] = np.nan; R["hi_log"] = np.nan
    R["has_ci"] = R.lo.notna() & R.hi.notna()
    R["cover"] = np.where(R.has_ci, ((R.lo <= R.true_saving) & (R.true_saving <= R.hi)).astype(float), np.nan)
    R["rej"] = np.where(R.has_ci, ((R.lo > 0) | (R.hi < 0)).astype(float), np.nan)
    ucols = [c for c in R.columns if c.startswith("n_pseudo_")]
    rows = []
    for (test, car, tau, est), d in R.groupby(["test", "carrier", "tau", "estimator"], sort=False):
        ok = d[d.status == "ok"]; ci = ok[ok.has_ci]
        row = dict(test=test, test_label=TESTS.get(test, test), carrier=car, tau=tau, estimator=est, label=LABELS.get(est, est),
                   n_reps=len(d), n_ok=len(ok), n_fail=len(d) - len(ok),
                   mean_n_pseudo=d.n_pseudo.mean(), mean_n_used=ok.n_used.mean() if len(ok) else np.nan,
                   true_saving=-tau, mean_est=ok.est.mean(), bias=ok.err.mean(), sd=ok.err.std(), rmse=np.sqrt((ok.err ** 2).mean()),
                   mae=ok.err.abs().mean(), bias_log=ok.err_log.mean(), rmse_log=np.sqrt((ok.err_log ** 2).mean()), sd_log=ok.err_log.std(),
                   mean_ci_width_log=(ci.hi_log - ci.lo_log).mean() if len(ci) else np.nan, mean_est_log=ok.est_log.mean() if len(ok) else np.nan,
                   primary_scale="log (bias_log, rmse_log, sd_log); percent = exp(mean log effect) - 1",
                   coverage95=ci.cover.mean() if len(ci) else np.nan, n_ci=len(ci), mean_ci_width=(ci.hi - ci.lo).mean() if len(ci) else np.nan,
                   reject0=ci.rej.mean() if len(ci) else np.nan, mean_lo=ci.lo.mean() if len(ci) else np.nan, mean_hi=ci.hi.mean() if len(ci) else np.nan,
                   true_change=tau, mean_est_change=-ok.est.mean() if len(ok) else np.nan,
                   ci_method=ok.ci_method.value_counts().index[0] if len(ok) else "",
                   ci_note="" if len(ci) else ("no interval: " + ("no successful estimate" if not len(ok) else "fewer than 2 pseudo-treated buildings")),
                   mean_runtime_s=d.runtime_s.mean() if "runtime_s" in d else np.nan,
                   fail_reason=d[d.status != "ok"].status.value_counts().index[0] if (d.status != "ok").any() else "")
        row.update({"mean_" + c: d[c].mean() for c in ucols})
        rows.append(row)
    return pd.DataFrame(rows)


def figure(S, path, carrier="elec"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = S[(S.carrier == carrier) & (S.n_ok > 0) & (S.test == "random")]
    if s.empty: return False
    order = [k for k in LABELS if k in set(s.estimator)]
    style = {0.0: dict(color="#0072B2", marker="o", label=r"no effect ($\tau=0$)"),
             -0.10: dict(color="#D55E00", marker="s", label=r"injected saving ($\tau=-0.10$)"),
             -0.05: dict(color="#009E73", marker="^", label=r"injected saving ($\tau=-0.05$)")}
    fig, ax = plt.subplots(figsize=(4.8, 0.36 * len(order) + 1.0))
    for j, (tau, st) in enumerate(style.items()):
        d = s[np.isclose(s.tau, tau)].set_index("estimator").reindex(order)
        y = np.arange(len(order)) + (j - (len(style) - 1) / 2) * 0.25
        ax.errorbar(d.bias_log * 100, y, xerr=d.rmse_log * 100, fmt=st["marker"], color=st["color"], ms=5, lw=1.4, capsize=2.5, label=st["label"])
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LABELS[k] for k in order], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Bias of the estimated log effect (marker) ± RMSE (bar), log points × 100", fontsize=8)
    ax.tick_params(axis="x", labelsize=8); ax.grid(axis="x", color="0.9", lw=0.6); ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100); ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--uses", default="보건,어린이집"); ap.add_argument("--date-pool", default="use", choices=["use", "all"])
    ap.add_argument("--feasible-dates", action="store_true"); ap.add_argument("--with-placebos", action="store_true")
    ap.add_argument("--tag", default=""); ap.add_argument("--min-gas", type=int, default=30)
    a = ap.parse_args(); uses = a.uses.split(","); t_all = time.time()
    B, E = A.load()
    print("imputation_did available:", hasattr(A, "imputation_did"),
          inspect.signature(A.imputation_did) if hasattr(A, "imputation_did") else "", flush=True)
    f1 = RES / f"B1_benchmark_reps{a.tag}.csv"; f2 = RES / f"B2_benchmark_summary{a.tag}.csv"
    parts = []
    for car in ["elec", "gas"]:
        n = len(population(B, E, car, uses)[0])
        if n < 2 or (car == "gas" and n < a.min_gas): print(f"[{car}] skipped: {n} buildings in population"); continue
        done = list(parts)
        parts.append(run(B, E, car, uses, a.reps, a.seed, a.date_pool, feasible_dates=a.feasible_dates,
                         flush=lambda rows: pd.concat(done + [pd.DataFrame(rows)]).to_csv(f1, index=False)))
    for car in ["elec", "gas"]:
        for fn in ((selection_cohort, treated_pre_period) if a.with_placebos else ()):
            if car == "gas" and len(population(B, E, car, uses)[0]) < a.min_gas: continue
            try: parts.append(fn(B, E, car, uses, a.seed))
            except Exception as ex:
                parts.append(_empty(dict(test=fn.__name__, carrier=car, rep=0), f"{type(ex).__name__}: {ex}"[:200]))
    if not parts: sys.exit("no carrier with a usable population")
    R = pd.concat(parts, ignore_index=True)
    for c in ["est", "est_log", "lo", "hi", "lo_log", "hi_log", "n_used", "ci_method", "runtime_s", "n_pseudo", "true_saving", "true_log"]:
        if c not in R.columns: R[c] = np.nan
    R.to_csv(f1, index=False)
    S = summarise(R); S.to_csv(f2, index=False)
    import json
    (RES / f"B2_benchmark_meta{a.tag}.json").write_text(json.dumps(dict(
        generated=time.strftime("%Y-%m-%d %H:%M"), reps=a.reps, seed=a.seed, uses=uses, date_pool=a.date_pool, feasible_dates=bool(a.feasible_dates),
        taus=list(TAUS), frac_pseudo_treated=FRAC, population_trim_months_before_real_start=POP_TRIM, min_win=A.MIN_WIN,
        stacked_did=dict(did="province x month x stack effects, ctrl_trim=3", did_usefe="use x month x stack effects, ctrl_trim=3, treated months -3..-1 excluded",
                         estimate="treated-count-weighted average of cohort coefficients", se_convention=getattr(A, "SE_PYFIXEST", "")),
        imputation=dict(nboot=IMP_NBOOT, bootstrap="buildings resampled within role x use type, full refit"),
        gradient_boosting=dict(params={k: v for k, v in ML.PARAMS.items() if k != "device"}, rounds=600, encodings=ML.ENC_INFO,
                               effect="mean over pseudo-post building-months of (log actual - log predicted); no level retransformation is needed",
                               retransformation="none for the reported rows (log scale). Sensitivity row gb_smear_level: level counterfactual x Duan smearing factor "
                                                f"mean(exp(residual)) from {SMEAR_SENS_FOLDS}-fold out-of-fold residuals of the training rows",
                               mean_smear_factor_sensitivity_row=float(R[R.estimator == "gb_smear_level"].gb_smear.dropna().mean())
                               if "gb_smear" in R and R[R.estimator == "gb_smear_level"].gb_smear.notna().any() else None),
        primary_scale=dict(scale="log", columns=["bias_log", "rmse_log", "sd_log", "mean_ci_width_log", "mean_est_log"],
                           percent="exp(mean log effect) - 1 for every estimator (columns est / mean_est are 1 - exp(.), the saving convention; mean_est_change = -mean_est)",
                           truth="log(1 + tau)", note="est_agg_ratio in B1 is the former aggregate-ratio estimate, kept for reference only"),
        intervals=dict(naive_dd_gb="percentile bootstrap over pseudo-treated buildings, counterfactual held fixed (interval over buildings only)",
                       did="cluster-robust by building, pyfixest default small-sample correction", imp="building bootstrap with refit, stratified by role x use")),
        indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    try:
        pd.DataFrame(DATES).to_csv(RES / f"B3_benchmark_dates{a.tag}.csv", index=False)
        date_table(B, uses).to_csv(RES / f"B3_benchmark_date_distribution{a.tag}.csv", index=False)
    except Exception as ex: print("date table failed", ex)
    fig = ROOT / "paper/figs" / f"fig_benchmark{a.tag}.pdf"
    made = not S.empty and figure(S, fig)
    pd.set_option("display.width", 250)
    if not S.empty:
        print(S[["test", "carrier", "tau", "estimator", "n_ok", "n_fail", "mean_n_used", "bias", "rmse", "bias_log", "coverage95", "n_ci", "reject0",
                 "mean_runtime_s", "fail_reason"]].round(4).to_string(index=False))
    print(f"wrote {f1}\n      {f2}\n      {fig if made else '(no figure: no successful estimator)'}\n"
          f"total runtime {time.time() - t_all:.0f} s")
