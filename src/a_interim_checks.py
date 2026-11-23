"""Shared imputation helpers used by the descriptive and comparison scripts."""
import os, sys, json, time, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC
import s04_analysis as A

LABEL = "INTERIM, exploratory (post hoc, after the interim estimates were seen)"
NB = int(os.environ.get("IMP_NBOOT", "300"))
HC, DC = "보건", "어린이집"
SAMPLES = {"pooled": (HC, DC), "health_centre": (HC,), "daycare": (DC,)}
EARLY, LATE = ("2021H2", "2022H1", "2022H2"), ("2023H2", "2024H1")
cname = lambda c: f"{int(c) // 10}H{int(c) % 10 + 1}"
pct = lambda x: float(np.exp(x) - 1) if x is not None and np.isfinite(x) else np.nan
mi = lambda v: (v // 100) * 12 + v % 100
GEN = time.strftime("%Y-%m-%d %H:%M")


def groups_att(coh):
    """Cohort ATTs -> overall and early / late group ATT (treated-building-count weights, as in the headline)."""
    d = {cname(c["cohort"]): (c["att"], c["n_bld"]) for c in coh}
    def g(keys):
        k = [x for x in keys if x in d]; w = sum(d[x][1] for x in k)
        return sum(d[x][0] * d[x][1] for x in k) / w if w > 0 else np.nan
    out = {k: v[0] for k, v in d.items()}
    out["early_2021H2_2022H2"] = g(EARLY); out["late_2023H2_2024H1"] = g(LATE)
    out["late_minus_early"] = out["late_2023H2_2024H1"] - out["early_2021H2_2022H2"]
    return out, {k: v[1] for k, v in d.items()}


def estimate(B, E, carrier="elec", nboot=None, seed=0, **kw):
    """Point estimate, cohort / group ATTs and percentile bootstrap intervals (buildings, role x use strata, full refit)."""
    nboot = NB if nboot is None else nboot
    S = A.imputation_setup(B, E, carrier, **dict(A.PRIMARY_IMP, **kw))
    if S is None: return None
    f = S["fit"](); a, _, coh = S["att"](f); g0, nb = groups_att(coh)
    rng = np.random.default_rng(seed); bs, bg = [], []
    for _ in range(nboot):
        fb = S["fit"](A.boot_weights(S, rng), tol=1e-9); ab, _, cb = S["att"](fb); bs.append(ab); bg.append(groups_att(cb)[0])
    bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else [np.nan, np.nan]
    G = pd.DataFrame(bg)
    m = S["post"] & f["ok"]; fitc = f["F"] & S["isc"]
    res = dict(att_log=float(a), pct=pct(a), ci_log=q(bs), pct_ci=[pct(v) for v in q(bs)], n_treated=int(pd.unique(S["pi"][m]).size),
               n_control=int(pd.unique(S["pi"][fitc]).size), support_rows=float(m.sum() / max(int(S["post"].sum()), 1)), nboot=int(len(bs)),
               groups={k: dict(att_log=float(v), pct=pct(v), ci_log=q(G[k].dropna().to_numpy()) if k in G else [np.nan, np.nan],
                               n_bld=nb.get(k)) for k, v in g0.items()})
    for k, v in res["groups"].items(): v["pct_ci"] = [pct(x) for x in v["ci_log"]]
    return res, S, f


def rows_from(res, **tags):
    out = [dict(tags, level="all", att_log=res["att_log"], pct=res["pct"], lo_log=res["ci_log"][0], hi_log=res["ci_log"][1],
                pct_lo=res["pct_ci"][0], pct_hi=res["pct_ci"][1], n_treated=res["n_treated"], n_control=res["n_control"],
                support_rows=res["support_rows"], nboot=res["nboot"], label=LABEL, generated=GEN)]
    for k, v in res["groups"].items():
        out.append(dict(tags, level=k, att_log=v["att_log"], pct=v["pct"], lo_log=v["ci_log"][0], hi_log=v["ci_log"][1],
                        pct_lo=v["pct_ci"][0], pct_hi=v["pct_ci"][1], n_treated=v["n_bld"], n_control=res["n_control"],
                        support_rows=res["support_rows"], nboot=res["nboot"], label=LABEL, generated=GEN))
    return out


def control_classes(B, E):
    """Control-pool buildings (grade A_cand, both uses) by source of their untreated electricity months."""
    S = A.imputation_setup(B, E, "elec", **A.PRIMARY_IMP); D = S["D"]; c = D[D.ctrl == 1]
    bulk = c.pass_.astype(str) == "bulk"
    n_api = c[~bulk].groupby("pid").ym.nunique(); n_bulk = c[bulk].groupby("pid").ym.nunique()
    ids = pd.Index(c.pid.unique()); X = pd.DataFrame({"n_api": n_api.reindex(ids).fillna(0), "n_bulk": n_bulk.reindex(ids).fillna(0)})
    X["cls"] = np.select([X.n_api >= 12, X.n_api >= 1], ["long_api", "few_api"], "bulk_only")
    Bi = B.drop_duplicates("pid").set_index("pid")
    for k in ["use_class", "role", "reg_pk", "name_list", "start_ym"]: X[k] = Bi[k].reindex(X.index)
    return X, S


def task1(B, E, out, Braw, Eraw):
    Bi = B.drop_duplicates("pid").set_index("pid"); Bi_raw = Braw.drop_duplicates("pid").set_index("pid")
    Xraw, Sraw = control_classes(Braw, Eraw); Draw = Sraw["D"]
    X, S0 = control_classes(B, E); f0 = S0["fit"](); D = S0["D"].copy(); D["res"] = f0["res"]; D["ok"] = f0["ok"]; D["F"] = f0["F"]
    D["use"] = D.pid.map(Bi.use_class); D["src"] = np.where(D.pass_.astype(str) == "bulk", "bulk", "api"); D["coh"] = D.cohort.map(cname)
    long_, few = set(X[X.cls == "long_api"].index), set(X[X.cls != "long_api"].index)
    g = Xraw[Xraw.reg_pk.notna()].groupby("reg_pk")
    dup = g.filter(lambda d: len(d) > 1).sort_values(["reg_pk", "start_ym"])
    keep = X.sort_values("start_ym").groupby(X.reg_pk.fillna(pd.Series(X.index, index=X.index).astype(str)), sort=False).head(1).index
    dedup_drop = set(X.index) - set(keep)
    first_start = dup.groupby("reg_pk").start_ym.min()
    cont = Draw[(Draw.ctrl == 1) & Draw.pid.isin(dup.index)].copy(); cont["first_start"] = cont.pid.map(Bi_raw.reg_pk).map(first_start)
    cont = cont[cont.ym >= cont.first_start]
    out["duplicates"] = dict(physical_buildings_twice=int(dup.reg_pk.nunique()), pids=int(len(dup)),
                             by_roles=dup.groupby("reg_pk").role.agg(lambda s: "+".join(sorted(s))).value_counts().to_dict(),
                             by_use=dup.drop_duplicates("reg_pk").use_class.map({HC: "health_centre", DC: "daycare"}).value_counts().to_dict(),
                             control_months_at_or_after_first_start=int(len(cont)), buildings_with_such_months=int(cont.pid.nunique()),
                             basis="records as stored (before D23), control pool of the primary imputation frame",
                             after_d23_duplicates_left_in_control_pool=int(len(X) - len(keep)),
                             note="months of the later pid that fall at or after the first recorded start of the same building are in the fit as "
                                  "'untreated' although works of the earlier record may have begun; dedup_controls keeps the earliest pid only")
    dup.assign(use=dup.use_class.map({HC: "health_centre", DC: "daycare"}))[["reg_pk", "role", "use", "name_list", "start_ym", "n_api", "n_bulk", "cls"]] \
        .to_csv(RES / "a_interim_control_duplicates.csv")
    raw = pd.read_csv(PROC / "energy_monthly.csv"); el = raw[(raw.carrier == "elec") & (raw.status == "ok") & (raw.kwh > 0)]
    el = el.merge(B[["pid", "role", "use_class", "link_grade"]], on="pid"); el["bulk"] = el.pass_.astype(str) == "bulk"
    cov = []
    for role, d in el.groupby("role"):
        napi = d[~d.bulk].groupby("pid").ym.nunique(); nb = d[d.bulk].groupby("pid").ym.nunique(); ids = d.pid.unique()
        cov.append(dict(role=role, buildings=int(len(ids)), bulk_only_buildings=int((napi.reindex(ids).fillna(0) == 0).sum()),
                        api_only_buildings=int((nb.reindex(ids).fillna(0) == 0).sum()), months_api=int((~d.bulk).sum()), months_bulk=int(d.bulk.sum()),
                        api_months_from_202311=int(((~d.bulk) & (d.ym >= 202311)).sum()), bulk_first=int(d[d.bulk].ym.min()) if d.bulk.any() else None,
                        api_last=int(d[~d.bulk].ym.max())))
    out["coverage_by_role_all_buildings"] = cov
    out["control_pool_classes"] = X.assign(use=X.use_class.map({HC: "health_centre", DC: "daycare"})).groupby(["use", "role", "cls"]) \
        .agg(n=("n_api", "size"), mean_api_months=("n_api", "mean"), mean_bulk_months=("n_bulk", "mean")).round(2).reset_index().to_dict("records")
    c = D[D.ctrl == 1].copy(); c["cls"] = c.pid.map(X.cls)
    R = []
    for (u, ym), d in c.groupby(["use", "ym"]):
        r = dict(use={HC: "health_centre", DC: "daycare"}[u], ym=int(ym), n_ctrl=int(d.pid.nunique()), mean_log_intensity=float(d.y.mean()),
                 share_bulk=float((d.src == "bulk").mean()))
        for k in ["long_api", "few_api", "bulk_only"]:
            dk = d[d.cls == k]; r[f"n_{k}"] = int(dk.pid.nunique()); r[f"mean_log_{k}"] = float(dk.y.mean()) if len(dk) else np.nan
        R.append(r)
    R = pd.DataFrame(R)
    lam = pd.Series(f0["l"], index=S0["tks"]); R["lambda_fit"] = [lam.get(f"{ {'health_centre': HC, 'daycare': DC}[u] }|{ym}", np.nan) for u, ym in zip(R.use, R.ym)]
    tr = D[(D.ctrl == 0) & D.F].groupby(["use", "ym"]).pid.nunique()
    R["n_treated_fit_rows"] = [int(tr.get(({'health_centre': HC, 'daycare': DC}[u], ym), 0)) for u, ym in zip(R.use, R.ym)]
    for u_en, u in [("health_centre", HC), ("daycare", DC)]:
        w = c[(c.use == u) & (c.cls == "long_api")].pivot_table(index="pid", columns="ym", values="y"); w = w.sub(w.mean(axis=1), axis=0)
        for stat, fn in [("mean", w.mean()), ("median", w.median())]:
            R.loc[R.use == u_en, f"long_api_demeaned_{stat}"] = R.loc[R.use == u_en, "ym"].map(fn)
    R["label"] = LABEL; R.to_csv(RES / "a_interim_coverage_by_month.csv", index=False)
    seam = {}
    for u in ["health_centre", "daycare"]:
        r = R[R.use == u].set_index("ym")
        seam[u] = dict(n_ctrl_202310=int(r.n_ctrl.get(202310, 0)), n_ctrl_202311=int(r.n_ctrl.get(202311, 0)),
                       median_n_ctrl_202007_202310_ex_jan=float(r[(r.index < 202311) & (r.index % 100 != 1) & (r.index != 202110)].n_ctrl.median()),
                       median_n_ctrl_202311_202410=float(r[(r.index >= 202311) & (r.index <= 202410)].n_ctrl.median()),
                       last_month_with_ge10_controls=int(r[r.n_ctrl >= 10].index.max()))
        l = r.lambda_fit; yo = {}
        for y1, y0 in [(2022, 2021), (2023, 2022), (2024, 2023), (2024, 2022), (2025, 2024)]:
            for sname, months in [("jan", [1]), ("feb_jun", [2, 3, 4, 5, 6]), ("jul_sep", [7, 8, 9])]:
                d = [l.get(y1 * 100 + m_) - l.get(y0 * 100 + m_) for m_ in months if np.isfinite(l.get(y1 * 100 + m_, np.nan)) and np.isfinite(l.get(y0 * 100 + m_, np.nan))]
                yo[f"{y1}-{y0}_{sname}"] = float(np.mean(d)) if d else None
        seam[u]["lambda_same_month_change"] = yo
    out["control_pool_seam"] = seam
    key = X.reg_pk.fillna(pd.Series(X.index, index=X.index).astype(str)); INF = []
    base = {}
    for sname in ["health_centre", "pooled"]:
        S = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=SAMPLES[sname])); f = S["fit"](); a, _, coh = S["att"](f)
        base[sname] = (a, groups_att(coh)[0])
    cres = D[D.ctrl == 1].groupby("pid").res.agg(["std", "min", "max", "size"])
    for k, ids in key.groupby(key).groups.items():
        ids = set(ids); u = X.use_class.reindex(list(ids)).iloc[0]
        r = dict(reg_pk=k, pids="+".join(sorted(ids)), n_pids=len(ids), name=str(X.name_list.reindex(list(ids)).iloc[0]), use={HC: "health_centre", DC: "daycare"}[u],
                 cls="+".join(sorted(set(X.cls.reindex(list(ids))))), n_fit_rows=int(cres["size"].reindex(list(ids)).sum()),
                 resid_sd=float(cres["std"].reindex(list(ids)).max()), resid_min=float(cres["min"].reindex(list(ids)).min()))
        for sname in ["health_centre", "pooled"]:
            if sname == "health_centre" and u != HC: continue
            S = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=SAMPLES[sname], drop=ids)); f = S["fit"](); a, _, coh = S["att"](f)
            g = groups_att(coh)[0]
            r.update({f"{sname}_d_att": a - base[sname][0], f"{sname}_d_early": g["early_2021H2_2022H2"] - base[sname][1]["early_2021H2_2022H2"],
                      f"{sname}_d_late": g["late_2023H2_2024H1"] - base[sname][1]["late_2023H2_2024H1"]})
        INF.append(r)
    INF = pd.DataFrame(INF); INF["abs_gap_change"] = (INF.health_centre_d_late - INF.health_centre_d_early).abs()
    INF = INF.sort_values("abs_gap_change", ascending=False); INF["label"] = LABEL; INF.to_csv(RES / "a_interim_influence.csv", index=False)
    top = INF.iloc[0]; top_ids = set(top.pids.split("+"))
    hi = INF[INF.use == "health_centre"]
    loo = {}
    for sname in ["health_centre", "pooled"]:
        x = INF[INF[f"{sname}_d_att"].notna()].copy(); x["a"] = base[sname][0] + x[f"{sname}_d_att"]
        t3 = x.reindex(x[f"{sname}_d_att"].abs().sort_values(ascending=False).index).head(3)
        loo[sname] = dict(att_pct=pct(base[sname][0]), min_pct=pct(x.a.min()), max_pct=pct(x.a.max()), min_log=float(x.a.min()), max_log=float(x.a.max()),
                          n_control_buildings_left_out=int(len(x)),
                          top3=[dict(pid=r.pids, name=r["name"], use=r.use, d_att_log=float(r[f"{sname}_d_att"]), att_pct_without=pct(r.a), resid_min=float(r.resid_min)) for _, r in t3.iterrows()],
                          label="sensitivity: leave-one-control-building-out on the D23 sample; no building is removed from the headline")
    out["leave_one_out"] = loo
    out["influence"] = dict(n_physical_control_buildings=int(len(INF)), top=dict(name=top["name"], pids=top.pids, cls=top.cls, resid_min=top.resid_min,
                            d_att_health=float(top.health_centre_d_att), d_early_health=float(top.health_centre_d_early), d_late_health=float(top.health_centre_d_late),
                            d_att_pooled=float(top.pooled_d_att)),
                            health_att_leave_one_out_range_log=[float(base["health_centre"][0] + hi.health_centre_d_att.min()), float(base["health_centre"][0] + hi.health_centre_d_att.max())],
                            second_largest_abs_d_att_health=float(hi.health_centre_d_att.abs().sort_values().iloc[-2]),
                            note="drop of one physical building (all its pids) from the control pool; d_* = change of the log ATT")
    Ea = E[E.pass_.astype(str) != "bulk"]
    S_d = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, drop=dedup_drop)); f_d = S_d["fit"]()
    Dd = S_d["D"]; bad = Dd[(Dd.ctrl.to_numpy() == 1) & (np.abs(f_d["res"]) > 1.0)][["pid", "ym"]]
    key_bad = set(zip(bad.pid, bad.ym)); Etrim = E[~((E.carrier == "elec") & pd.Series(list(zip(E.pid, E.ym)), index=E.index).isin(key_bad))]
    out["trim_ctrl_outlier_rows"] = dict(rows_removed=int(len(bad)), buildings=int(bad.pid.nunique()), control_fit_rows=int((Dd.ctrl == 1).sum()))
    VAR = [("base", B, E, {}), ("no_dedup", Braw, Eraw, {}), ("api_only", B, Ea, {}), ("api_only_nowin", B, Ea, dict(min_win=0)),
           ("long_api_controls", B, E, dict(drop=few)), ("few_api_controls", B, E, dict(drop=long_)),
           ("drop_top_control", B, E, dict(drop=dedup_drop | top_ids)), ("trim_ctrl_outlier_rows", B, Etrim, dict(drop=dedup_drop))]
    rows, keep_fit = [], {}
    for vname, Bv, Ev, kw in VAR:
        for sname, uses in SAMPLES.items():
            try:
                r = estimate(Bv, Ev, "elec", uses=uses, **kw)
                if r is None: rows.append(dict(variant=vname, sample=sname, level="all", status="no treated rows", label=LABEL)); continue
                rows += rows_from(r[0], variant=vname, sample=sname, status="ok"); keep_fit[(vname, sname)] = r
            except Exception as ex:
                rows.append(dict(variant=vname, sample=sname, level="all", status=f"error: {type(ex).__name__}: {str(ex)[:100]}", label=LABEL))
            print(f"  cohort variant {vname:24s} {sname:14s} done", flush=True)
    C = pd.DataFrame(rows); C.to_csv(RES / "a_interim_cohort.csv", index=False)
    out["cohort_key"] = {f"{v}|{s}": {lv: [round(float(x), 4) for x in C[(C.variant == v) & (C["sample"] == s) & (C.level == lv)][["pct", "pct_lo", "pct_hi"]].iloc[0]]
                                     for lv in ["all", "early_2021H2_2022H2", "late_2023H2_2024H1", "late_minus_early"]
                                     if len(C[(C.variant == v) & (C["sample"] == s) & (C.level == lv)])}
                         for v, _, _, _ in VAR for s in SAMPLES}
    EV = []
    for vname in ["base", "drop_top_control"]:
        _, S, f = keep_fit[(vname, "pooled")]; d = S["D"].copy(); d["res"] = f["res"]; d["ok"] = f["ok"]; d["F"] = f["F"]
        d = d[d.ctrl == 0]; d["use"] = d.pid.map(Bi.use_class); d["coh"] = d.cohort.map(cname)
        d["grp"] = np.where(d.coh.isin(EARLY), "early_2021H2_2022H2", np.where(d.coh.isin(LATE), "late_2023H2_2024H1", "other"))
        d["kind"] = np.where(d.post == 1, "post (imputed effect)", np.where(d.ant == 1, "held-out s-3..s-1 (imputed)", "fit residual (in sample)"))
        d = d[d.ok & np.isfinite(d.res)]
        for (u, gname, kind, rel), x in d.groupby(["use", "grp", "kind", "rel"]):
            bm = x.groupby("pid").res.mean()
            EV.append(dict(variant=vname, use={HC: "health_centre", DC: "daycare"}[u], cohort_group=gname, kind=kind, rel_month=int(rel),
                           mean_log=float(x.res.mean()), pct=pct(x.res.mean()), median_log=float(x.res.median()), n_buildings=int(x.pid.nunique()),
                           se_naive=float(x.res.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan, label=LABEL))
    pd.DataFrame(EV).to_csv(RES / "a_interim_event_time.csv", index=False)
    CT = []
    for vname in ["base", "drop_top_control"]:
        _, S, f = keep_fit[(vname, "pooled")]
        a = pd.Series(f["a"], index=S["pids"]); l = pd.Series(f["l"], index=S["tks"]); ncell = pd.Series(f["nctrl"], index=S["tks"])
        Ec = A.carrier_frame(E, "elec"); T = Ec[Ec.pid.isin(S["T"].pid)].copy(); T["y"] = np.log(T.kwh / T.gr_gfa)
        T["use"] = T.pid.map(Bi.use_class); T["tk"] = T.use + "|" + T.ym.astype(str)
        T["res"] = T.y - T.pid.map(a) - T.tk.map(l); T["nc"] = T.tk.map(ncell).fillna(0)
        T["re"] = mi(T.ym) - mi(T.end_ym.astype(int)); T["coh"] = ((T.end_ym // 100) * 10 + ((T.end_ym % 100) > 6)).astype(int).map(cname)
        P = T[(T.re >= 1) & (T.nc >= 1) & np.isfinite(T.res)].copy(); P["half"] = (P.ym // 100).astype(str) + np.where(P.ym % 100 > 6, "H2", "H1")
        P["within_12"] = P.re <= 12
        for (u, co, h), x in P.groupby(["use", "coh", "half"]):
            CT.append(dict(variant=vname, use={HC: "health_centre", DC: "daycare"}[u], cohort=co, calendar_half=h, mean_log=float(x.res.mean()), pct=pct(x.res.mean()),
                           median_bld_log=float(x.groupby("pid").res.mean().median()), n_buildings=int(x.pid.nunique()), n_rows=int(len(x)),
                           share_rows_within_12_months=float(x.within_12.mean()), mean_ctrl_per_cell=float(x.nc.mean()), label=LABEL))
    pd.DataFrame(CT).to_csv(RES / "a_interim_calendar_tau.csv", index=False)
    w = el.assign(y=np.log(el.kwh / el.pid.map(Bi.area_m2))).pivot_table(index="pid", columns="ym", values="y")
    Ts = S0["T"]; cohT = ((Ts.end_ym // 100) * 10 + ((Ts.end_ym % 100) > 6)).astype(int).map(cname); cohT.index = Ts.pid
    def chg(pids, y1, y0, months):
        ww = w[w.index.isin(pids)]; d = [ww[y1 * 100 + m_] - ww[y0 * 100 + m_] for m_ in months if y1 * 100 + m_ in ww and y0 * 100 + m_ in ww]
        if not d: return None
        x = pd.concat(d, axis=1); v = x.mean(axis=1)[x.notna().sum(axis=1) >= 3]
        return dict(mean=float(v.mean()), median=float(v.median()), n=int(len(v))) if len(v) else None
    RAW = []
    for u, u_en in [(HC, "health_centre"), (DC, "daycare")]:
        G = {f"treated_{co}": set(cohT[cohT == co].index) & set(Ts[Ts.use_class == u].pid) for co in ["2022H1", "2022H2", "2023H2", "2024H1"]}
        G["controls_long_api"] = set(X[(X.cls == "long_api") & (X.use_class == u)].index) - dedup_drop
        ctrl_w = D[(D.ctrl == 1) & (D.use == u)].pivot_table(index="pid", columns="ym", values="y")
        for gname, ids in G.items():
            for y1, y0 in [(2022, 2021), (2023, 2022), (2024, 2022), (2024, 2023), (2024, 2021)]:
                for sname, months in [("jan_jun", range(1, 7)), ("jul_dec", range(7, 13))]:
                    if gname.startswith("controls"):
                        ww = ctrl_w[ctrl_w.index.isin(ids)]; d = [ww[y1 * 100 + m_] - ww[y0 * 100 + m_] for m_ in months if y1 * 100 + m_ in ww and y0 * 100 + m_ in ww]
                        r = None
                        if d:
                            x = pd.concat(d, axis=1); v = x.mean(axis=1)[x.notna().sum(axis=1) >= 3]
                            r = dict(mean=float(v.mean()), median=float(v.median()), n=int(len(v))) if len(v) else None
                    else: r = chg(ids, y1, y0, months)
                    if r: RAW.append(dict(use=u_en, group=gname, years=f"{y1} vs {y0}", months=sname, **r, label=LABEL))
    pd.DataFrame(RAW).to_csv(RES / "a_interim_raw_changes.csv", index=False)
    Tm = Ts.assign(coh=cohT.reindex(Ts.pid).to_numpy()); Tm["grp"] = np.where(Tm.coh.isin(EARLY), "early", np.where(Tm.coh.isin(LATE), "late", "other"))
    Tm["dur"] = mi(Tm.end_ym.astype(int)) - mi(Tm.start_ym.astype(int))
    Tm["sub_post"] = Tm.name_list.astype(str).str.contains("진료소").astype(float)
    pre = A.pre_intensity(B, E, "elec")[0]; Tm["pre_kwh_m2_y"] = Tm.pid.map(pre)
    mm = Tm.groupby([Tm.use_class.map({HC: "health_centre", DC: "daycare"}), "grp"]).agg(
        n=("pid", "size"), selection_2022_share=("year", lambda s: float((s == 2022).mean())), works_months_median=("dur", "median"),
        hvac=("hvac", "mean"), boiler=("boiler", "mean"), renewable=("renewable", "mean"), led=("led", "mean"), wall=("wall", "mean"), win=("win", "mean"),
        post_share=("sub_post", "mean"), pre_kwh_m2_y_median=("pre_kwh_m2_y", "median"), area_median=("area_m2", "median")).round(3).reset_index()
    out["measure_mix_by_cohort_group"] = mm.to_dict("records")
    return X, dedup_drop, top_ids


def task2(B, E, out):
    """Daycare centres with a confirmed gas meter: electricity, gas and total on the same building-months."""
    Bi = B.drop_duplicates("pid").set_index("pid")
    el = E[(E.carrier == "elec") & (E.kwh > 0)][["pid", "ym"]]; ga = E[(E.carrier == "gas") & (E.kwh > 0)][["pid", "ym"]]
    both = set(map(tuple, el.merge(ga, on=["pid", "ym"]).to_numpy()))
    Eb = E[pd.Series(list(zip(E.pid, E.ym)), index=E.index).isin(both)]
    rows, fits = [], {}
    for name, Ev, car in [("same_months", Eb, "elec"), ("same_months", Eb, "gas"), ("same_months", Eb, "total"),
                          ("all_available_months", E, "elec"), ("all_available_months", E, "gas"), ("all_available_months", E, "total")]:
        r = estimate(B, Ev, car, uses=(DC,))
        if r is None: rows.append(dict(panel=name, carrier=car, level="all", status="no treated rows")); continue
        rows += rows_from(r[0], panel=name, carrier=car, status="ok"); fits[(name, car)] = r
        print(f"  daycare {name} {car}: {r[0]['pct']:+.3f} [{r[0]['pct_ci'][0]:+.3f}, {r[0]['pct_ci'][1]:+.3f}] nT {r[0]['n_treated']} nC {r[0]['n_control']}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(RES / "a_interim_daycare_fuel.csv", index=False)
    res = {}
    if ("same_months", "gas") in fits:
        r0, S, f = fits[("same_months", "gas")]; m = S["post"] & f["ok"]; tp = set(S["pids"][pd.unique(S["pi"][m])])
        e = Eb[Eb.pid.isin(tp)]; e = e[(mi(e.ym) - mi(e.start_ym.astype(int))).between(-12, -1)]
        lv = e.assign(i=e.kwh / e.gr_gfa).groupby(["pid", "carrier"]).i.mean().unstack()
        res["pre_level_kwh_m2_month"] = dict(elec=float(lv.elec.mean()), gas=float(lv.gas.mean()), gas_share_of_total=float((lv.gas / (lv.gas + lv.elec)).mean()),
                                             n_buildings=int(len(lv)))
        pe, pg, pt = [fits[("same_months", c)][0]["pct"] for c in ["elec", "gas", "total"]]
        res["implied_level_change_kwh_m2_month"] = dict(elec=float(lv.elec.mean() * pe), gas=float(lv.gas.mean() * pg), sum=float(lv.elec.mean() * pe + lv.gas.mean() * pg),
                                                        total_from_total_att=float((lv.elec + lv.gas).mean() * pt),
                                                        note="mean pre-period intensity x percent effect; the log ATTs are building-month means, so the sum is only indicative")
        bt = {c: fits[("same_months", c)][1]["building_tau"](fits[("same_months", c)][2]).tau_mean for c in ["elec", "gas", "total"]}
        bt = pd.DataFrame(bt).dropna()
        res["building_level"] = dict(n=int(len(bt)), corr_elec_gas=float(bt.elec.corr(bt.gas)), share_elec_down_gas_up=float(((bt.elec < 0) & (bt.gas > 0)).mean()),
                                     share_both_down=float(((bt.elec < 0) & (bt.gas < 0)).mean()), share_elec_up_gas_down=float(((bt.elec > 0) & (bt.gas < 0)).mean()),
                                     median_elec=float(bt.elec.median()), median_gas=float(bt.gas.median()), median_total=float(bt.total.median()))
        X = Bi.loc[sorted(tp)]; txt = X.measures_txt.astype(str)
        flags = pd.DataFrame({"boiler": X.boiler.astype(int), "hvac_replacement": X.hvac.astype(int),
                              "heat_source_change_stated": txt.str.contains("열원").astype(int),
                              "electric_capacity_upgrade": txt.str.contains(r"전기\s*용량|전기\s*증설|수전").astype(int),
                              "instant_water_heater": txt.str.contains("순간온수기").astype(int), "renewable_entry": X.renewable.astype(int)})
        flags["any_heating_system_entry"] = ((flags.boiler + flags.hvac_replacement + flags.heat_source_change_stated + flags.electric_capacity_upgrade) > 0).astype(int)
        res["works_text_share_gas_sample"] = {k: float(v) for k, v in flags.mean().items()}; res["works_text_n_gas_sample"] = int(len(flags))
        allT = B[(B.role == "treated") & (B.use_class == DC) & (B.link_grade == "A_cand")].drop_duplicates("pid"); t2 = allT.measures_txt.astype(str)
        res["works_text_share_all_daycare_A"] = dict(n=int(len(allT)), boiler=float(allT.boiler.mean()), hvac_replacement=float(allT.hvac.mean()),
                                                     heat_source_change_stated=float(t2.str.contains("열원").mean()),
                                                     electric_capacity_upgrade=float(t2.str.contains(r"전기\s*용량|전기\s*증설|수전").mean()),
                                                     instant_water_heater=float(t2.str.contains("순간온수기").mean()))
        hc = A.heating_carrier(B, E).reindex(sorted(tp)); res["pre_heating_class_gas_sample"] = hc.value_counts().to_dict()
        sub = []
        GR = {"boiler_entry": set(flags[flags.boiler == 1].index), "no_boiler_entry": set(flags[flags.boiler == 0].index),
              "hvac_entry": set(flags[flags.hvac_replacement == 1].index), "no_hvac_entry": set(flags[flags.hvac_replacement == 0].index),
              "hvac_no_boiler": set(flags[(flags.hvac_replacement == 1) & (flags.boiler == 0)].index),
              "pre_heating_gas": set(hc[hc == "gas"].index), "pre_heating_not_gas": set(hc[hc != "gas"].index)}
        for gname, ids in GR.items():
            for car in ["elec", "gas", "total"]:
                if len(ids) < 10: sub.append(dict(group=gname, carrier=car, n_subset=len(ids), status="skipped (< 10 treated)")); continue
                r = estimate(B, Eb, car, uses=(DC,), treated_keep=ids, nboot=min(NB, 200))
                sub.append(dict(group=gname, carrier=car, n_subset=len(ids), status="ok", pct=r[0]["pct"], pct_lo=r[0]["pct_ci"][0], pct_hi=r[0]["pct_ci"][1],
                                n_treated=r[0]["n_treated"], n_control=r[0]["n_control"], label=LABEL))
        pd.DataFrame(sub).to_csv(RES / "a_interim_daycare_fuel_groups.csv", index=False)
        seas = {}
        for car in ["elec", "gas", "total"]:
            _, S_, f_ = fits[("same_months", car)]; mon = S_["D"].ym.to_numpy() % 100; hm = np.isin(mon, [11, 12, 1, 2, 3])
            seas[car] = dict(heating_nov_mar=pct(S_["att"](f_, S_["post"] & hm)[0]), other_months=pct(S_["att"](f_, S_["post"] & ~hm)[0]))
        res["season_pct_same_months"] = seas
        Sd = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=(DC,))); td = set(Sd["T"].pid); split = {}
        gas_any = set(E[(E.carrier == "gas")].pid)
        for gname, ids in [("in_gas_sample", td & tp), ("not_in_gas_sample", td - tp), ("no_gas_month_at_all", td - gas_any)]:
            r = estimate(B, E, "elec", uses=(DC,), treated_keep=ids)[0]
            split[gname] = dict(pct=r["pct"], pct_ci=r["pct_ci"], n_treated=r["n_treated"], n_control=r["n_control"])
        res["elec_all_months_by_gas_sample"] = split
        res["control_pool_same_months"] = dict(n=int(r0["n_control"]), note="gas-metered control-pool daycares with both carriers observed; gas comes from the API only (no bulk file)")
    res["row_label"] = "Gas-heated daycare centres (confirmed gas meter): electricity, gas and total on the same buildings and months"
    res["side_by_side_same_months"] = {c: dict(pct=fits[("same_months", c)][0]["pct"], pct_ci=fits[("same_months", c)][0]["pct_ci"], n_treated=fits[("same_months", c)][0]["n_treated"],
                                                 n_control=fits[("same_months", c)][0]["n_control"]) for c in ["elec", "gas", "total"] if ("same_months", c) in fits}
    out["daycare_fuel"] = dict(res, label=LABEL, rows_file="results/a_interim_daycare_fuel.csv")


def level_E(E):
    """Frame whose log intensity equals the LEVEL intensity (kWh/m2 per month): kwh' = area x exp(kwh / area). imputation_setup then"""
    Z = E.copy(); Z["kwh"] = np.where(Z.kwh > 0, Z.gr_gfa * np.exp(np.clip(Z.kwh / Z.gr_gfa, None, 600)), Z.kwh); return Z


def task3(B, E, out):
    res = {"unit": "kWh per m2 of register floor area per month", "label": LABEL,
           "keei_2025": "health centres -1.6 to -2.7; static pooled -0.73 to -2.47 (as given by the PI from KEEI 25-14; completed 2020-2021)"}
    EL = level_E(E)
    for sname, uses in SAMPLES.items():
        r, S, f = estimate(B, E, "elec", uses=uses); m = S["post"] & f["ok"]; tp = set(S["pids"][pd.unique(S["pi"][m])])
        e = E[(E.carrier == "elec") & (E.kwh > 0) & E.pid.isin(tp)]; e = e[(mi(e.ym) - mi(e.start_ym.astype(int))).between(-12, -1)]
        lv = e.assign(i=e.kwh / e.gr_gfa).groupby("pid").i.mean()
        D = S["D"][m]; obs = (D.kwh / D.gr_gfa).to_numpy(); cf = obs / np.exp(f["res"][m])
        rl, Sl, fl = estimate(B, EL, "elec", uses=uses)
        res[sname] = dict(n_treated=r["n_treated"], pct=r["pct"], pct_ci=r["pct_ci"], pre_mean_kwh_m2_month=float(lv.mean()), pre_median_kwh_m2_month=float(lv.median()),
                          level_from_pct_x_pre_mean=float(lv.mean() * r["pct"]), level_from_pct_x_pre_mean_ci=[float(lv.mean() * r["pct_ci"][0]), float(lv.mean() * r["pct_ci"][1])],
                          level_from_pct_x_pre_median=float(lv.median() * r["pct"]),
                          mean_observed_minus_imputed_post_level=float((obs - cf).mean()),
                          level_outcome_imputation=dict(att=rl["att_log"], ci=rl["ci_log"], n_treated=rl["n_treated"],
                                                        groups={k: dict(att=v["att_log"], ci=v["ci_log"]) for k, v in rl["groups"].items() if k.startswith(("early", "late"))},
                                                        note="same estimator with the outcome in kWh/m2 per month (levels); cohort ATTs, treated-count weights"))
    Ts = A.imputation_setup(B, E, "elec", **A.PRIMARY_IMP)["T"]
    res["cohort_overlap"] = dict(ours_n_treated_sample=int(len(Ts)), ours_selection_year=Ts.year.value_counts().sort_index().to_dict(),
                                 ours_completion_year={int(k): int(v) for k, v in (Ts.end_ym // 100).astype(int).value_counts().sort_index().items()},
                                 ours_completed_2020_2021=int(((Ts.end_ym // 100) <= 2021).sum()),
                                 ours_completed_2021_by_use=Ts[(Ts.end_ym // 100) <= 2021].use_class.map({HC: "health_centre", DC: "daycare"}).value_counts().to_dict(),
                                 note="KEEI: 522 buildings completed 2020-2021 (daycare 358, health 138, hospital 26). Overlap = our 2021 selections completed by "
                                      "2021-12; our sample has no 2020 selection. Building-level overlap cannot be checked (their list is not public).")
    out["keei_units"] = res


def task4(B, E, out, dedup_drop, top_ids):
    Bi = B.drop_duplicates("pid").set_index("pid"); rows = []
    S = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=(HC,))); f = S["fit"](); D = S["D"]
    tp = set(S["pids"][pd.unique(S["pi"][S["post"] & f["ok"]])]); cp = set(S["pids"][pd.unique(S["pi"][f["F"] & S["isc"]])])
    ZEN = {"중부1_춘천": "central1", "중부2_서울": "central2", "남부_부산": "southern"}
    r = estimate(B, E, "elec", uses=(HC,)); rows += rows_from(r[0], spec="health_base", detail="primary settings, health centres")
    for z, zen in ZEN.items():
        ids_out = set(Bi[Bi.zone != z].index); nt, nc = len(tp - ids_out), len(cp - ids_out)
        if nt < 15 or nc < 10:
            rows.append(dict(spec=f"within_zone_{zen}", level="all", detail=f"skipped: {nt} treated / {nc} control-pool buildings (need >= 15 / >= 10)", n_treated=nt, n_control=nc, label=LABEL)); continue
        r = estimate(B, E, "elec", uses=(HC,), drop=ids_out); rows += rows_from(r[0], spec=f"within_zone_{zen}", detail="treated and control pool both restricted to the zone")
    zr = [(x["att_log"], x["n_treated"]) for x in rows if str(x.get("spec", "")).startswith("within_zone") and x.get("level") == "all" and "att_log" in x]
    if zr:
        a = sum(v * n for v, n in zr) / sum(n for _, n in zr)
        rows.append(dict(spec="within_zone_weighted", level="all", detail=f"treated-count weighted mean of {len(zr)} estimable zone ATTs (no interval)", att_log=a, pct=pct(a),
                         n_treated=sum(n for _, n in zr), label=LABEL))
    pint, _ = A.pre_intensity(B, E, "elec")
    Fc = D[f["F"] & S["isc"]].sort_values("ym").groupby("pid").tail(12); g = Fc.groupby("pid").agg(n=("kwh", "size"), m=("kwh", "mean"), area=("gr_gfa", "first"))
    cint = (12 * g.m / g.area)[g.n >= A.MIN_WIN]
    t = pint.reindex(sorted(tp)).dropna(); c = cint.reindex(sorted(cp)).dropna()
    lo, hi = max(t.quantile(.05), c.quantile(.05)), min(t.quantile(.95), c.quantile(.95))
    keep_t = set(t[(t >= lo) & (t <= hi)].index); keep_c = set(c[(c >= lo) & (c <= hi)].index)
    smd = lambda a_, b_: float((a_.mean() - b_.mean()) / np.sqrt((a_.var(ddof=1) + b_.var(ddof=1)) / 2))
    info = dict(definition="kWh/m2 per year, 12 x mean monthly kWh / area; treated s-12..s-1, control pool last 12 fit months, >= 11 months (a_balance_within.py)",
                bounds_kwh_m2_y=[float(lo), float(hi)], treated_with_intensity=int(len(t)), treated_kept=int(len(keep_t)), controls_with_intensity=int(len(c)),
                controls_kept=int(len(keep_c)), controls_without_12_month_intensity=int(len(cp) - len(c)),
                smd_before=smd(t, c), smd_after=smd(t[list(keep_t)], c[list(keep_c)]))
    drop1 = (cp - keep_c); drop2 = set(c.index) - keep_c
    r = estimate(B, E, "elec", uses=(HC,), drop=drop1 | (tp - keep_t)); rows += rows_from(r[0], spec="common_support_strict",
                 detail="treated and controls inside the overlap of their 5th-95th percentiles; controls without a 12-month intensity dropped")
    r = estimate(B, E, "elec", uses=(HC,), drop=drop2 | (tp - keep_t)); rows += rows_from(r[0], spec="common_support_keep_unmeasured",
                 detail="as above, but control-pool buildings without a 12-month intensity are kept")
    r = estimate(B, E, "elec", uses=(HC,), drop=drop1 | (tp - keep_t) | dedup_drop | top_ids); rows += rows_from(r[0], spec="common_support_strict_drop_top_control",
                 detail="common_support_strict + dedup of the control pool + most influential control building removed")
    by = pd.to_numeric(Bi.built_year.astype(str).str[:4], errors="coerce"); bt = by.reindex(sorted(tp)).dropna(); bc = by.reindex(sorted(cp)).dropna()
    lo_b, hi_b = max(bt.quantile(.05), bc.quantile(.05)), min(bt.quantile(.95), bc.quantile(.95))
    out_b = set(by[(by < lo_b) | (by > hi_b)].index) & (tp | cp)
    r = estimate(B, E, "elec", uses=(HC,), drop=out_b); rows += rows_from(r[0], spec="built_year_common_support", detail=f"built year inside [{lo_b:.0f}, {hi_b:.0f}] on both sides")
    info["built_year_bounds"] = [float(lo_b), float(hi_b)]
    R = pd.DataFrame(rows); R["label"] = LABEL; R.to_csv(RES / "a_interim_balance_sens.csv", index=False)
    out["balance_sens"] = dict(info, rows={x["spec"]: [x.get("pct"), x.get("pct_lo"), x.get("pct_hi"), x.get("n_treated"), x.get("n_control")] for x in rows if x.get("level") == "all"})


def task5(B, E, out, dedup_drop, top_ids):
    ren = set(B[B.renewable == 1].pid); rows = []; key = {}
    for vname, extra in [("base", set()), ("drop_top_control", dedup_drop | top_ids)]:
        for sname, uses in SAMPLES.items():
            S = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=uses, drop=extra)); tids = set(S["T"].pid)
            r_all = estimate(B, E, "elec", uses=uses, drop=extra)[0]
            r_no = estimate(B, E, "elec", uses=uses, drop=extra | ren)[0]
            r_no_t = estimate(B, E, "elec", uses=uses, drop=extra, treated_keep=tids - ren)[0]
            r_ren = estimate(B, E, "elec", uses=uses, drop=extra, treated_keep=tids & ren)[0]
            for nm, r in [("all", r_all), ("drop_renewable_both_sides", r_no), ("treated_without_renewable_entry", r_no_t), ("treated_with_renewable_entry", r_ren)]:
                rows.append(dict(control_variant=vname, sample=sname, subset=nm, att_log=r["att_log"], pct=r["pct"], pct_lo=r["pct_ci"][0], pct_hi=r["pct_ci"][1],
                                 n_treated=r["n_treated"], n_control=r["n_control"], label=LABEL))
            sh = lambda r: float(1 - r["att_log"] / r_all["att_log"]) if r_all["att_log"] != 0 else np.nan
            key[f"{vname}|{sname}"] = dict(pct_all=r_all["pct"], pct_drop_both=r_no["pct"], pct_treated_no_ren=r_no_t["pct"], pct_treated_ren=r_ren["pct"],
                                           share_of_log_att_removed_drop_both=sh(r_no), share_of_log_att_removed_treated_side=sh(r_no_t),
                                           renewable_share_of_treated=float(len(tids & ren) / max(len(tids), 1)), n_treated_ren=r_ren["n_treated"], n_treated_no_ren=r_no_t["n_treated"],
                                           n_control_all=r_all["n_control"], n_control_drop_both=r_no["n_control"])
            print(f"  renewable {vname} {sname} done", flush=True)
    pd.DataFrame(rows).to_csv(RES / "a_interim_renewable.csv", index=False)
    C = B[B.role.isin(["control", "aux_2023"]) & (B.link_grade == "A_cand")]
    out["renewable"] = dict(key=key, control_pool_renewable_entry_share={"health_centre": float(C[C.use_class == HC].renewable.mean()), "daycare": float(C[C.use_class == DC].renewable.mean())},
                            pv_stated_explicitly_treated=int(B[(B.role == "treated") & (B.pv == 1)].pid.nunique()),
                            caveat="metered electricity is grid supply net of on-site generation consumed on site (behind-the-meter PV lowers the meter without lowering use); "
                                   "the works text states 'renewable energy' without the type, so the split cannot separate PV from solar thermal or geothermal; "
                                   "'drop both sides' also removes control-pool buildings with a renewable entry in their own (later) project, which changes the control pool",
                            label=LABEL)


def main():
    import a_dedup
    t0 = time.time(); Braw, Eraw = A.load(dedup=False)
    B, Dd = a_dedup.dedup_physical(Braw); E = Eraw[Eraw.pid.isin(set(B.pid))].copy()
    out = dict(label=LABEL, generated=GEN, nboot=NB, data=dict(energy_monthly_mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(PROC / "energy_monthly.csv"))),
               elec_rows_ok=int(((E.carrier == "elec") & (E.kwh > 0)).sum()), gas_rows_ok=int(((E.carrier == "gas") & (E.kwh > 0)).sum())))
    out["d23"] = dict(rule="one record per physical building (reg_pk): earliest selection year, then earliest start; a_dedup.dedup_physical, "
                           "applied by s04_analysis.load() since 11/14 (commit 4b5810d)", records_dropped=int(len(Dd)), physical_buildings=int(Dd.reg_pk.nunique()),
                      records_of_those_buildings=int(Dd.drop_duplicates("reg_pk").records_of_building.sum()),
                      by_roles={str(k): int(v) for k, v in Dd.drop_duplicates("reg_pk").roles_of_building.value_counts().items()},
                      dropped_by_role={str(k): int(v) for k, v in Dd.role.value_counts().items()}, file="results/a_dedup.csv",
                      declared="change made after the interim results were seen (PLAN.md 11/14 02:19)")
    NBH = int(os.environ.get("IMP_NBOOT_HEAD", "1000")); head = {}
    for tag, Bv, Ev in [("d23", B, E), ("no_dedup", Braw, Eraw)]:
        for sname, uses in SAMPLES.items():
            if tag == "no_dedup" and sname == "daycare": continue
            r = A.imputation_did(Bv, Ev, "elec", nboot=NBH if tag == "d23" else NB, **dict(A.PRIMARY_IMP, uses=uses))
            head[f"{tag}|{sname}"] = dict(pct=r["pct"], pct_ci=r["pct_ci"], att_log=r["att_log"], ci_log=r["ci_log"], n_treated=r["n_treated"], n_control=r["n_control"],
                                          pretrend_p=r["pretrend_p"], anticipation_gap=r["anticipation_gap"], anticipation_gap_ci=r["anticipation_gap_ci"],
                                          median_bld_pct=r["median_bld_pct"], support_rows=r["support_rows"], nboot=r["nboot"],
                                          cohorts={cname(c["cohort"]): dict(pct=pct(c["att"]), n_bld=c["n_bld"]) for c in r["cohorts"]})
            print(f"  headline {tag} {sname}: {r['pct']:+.3f} [{r['pct_ci'][0]:+.3f}, {r['pct_ci'][1]:+.3f}] pretrend p {r['pretrend_p']}", flush=True)
    out["headline_d23"] = dict(head, label=LABEL + "; electricity, imputation estimator, primary settings; d23 = deduplicated sample, no_dedup = records as stored")
    only = set(os.environ.get("INTERIM_TASKS", "1,2,3,4,5").split(","))
    X, dedup_drop, top_ids = task1(B, E, out, Braw, Eraw); print(f"task 1 done {time.time() - t0:.0f}s", flush=True)
    if "2" in only: task2(B, E, out); print(f"task 2 done {time.time() - t0:.0f}s", flush=True)
    if "3" in only: task3(B, E, out); print(f"task 3 done {time.time() - t0:.0f}s", flush=True)
    if "4" in only: task4(B, E, out, dedup_drop, top_ids); print(f"task 4 done {time.time() - t0:.0f}s", flush=True)
    if "5" in only: task5(B, E, out, dedup_drop, top_ids); print(f"task 5 done {time.time() - t0:.0f}s", flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer): return int(o)
        return o
    json.dump(clean(out), open(RES / "a_interim_checks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(clean(out), ensure_ascii=False, indent=1)[:200000])


if __name__ == "__main__":
    main()
