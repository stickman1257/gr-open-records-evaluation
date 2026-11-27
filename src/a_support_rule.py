"""Headline rule: comparison-group support by facility type."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A

MIN_BLD, MIN_SHARE = 10, 0.80
USE_EN = {"보건": "health_centre", "어린이집": "daycare"}


def evaluate(B, E, carrier, ctrl_trim=3):
    S = A.imputation_setup(B, E, carrier, **dict(A.PRIMARY_IMP, ctrl_trim=ctrl_trim))
    if S is None: return None, None
    D = S["D"].copy(); f = S["fit"](); D["fit"] = f["F"]; D["ok"] = f["ok"]
    Bi = B.drop_duplicates("pid").set_index("pid")
    D["use"] = D.pid.map(Bi.use_class); D["role"] = D.pid.map(Bi.role)
    post = D[(D.ctrl == 0) & (D.post == 1)]; ctrl = D[(D.ctrl == 1) & D.fit]
    months_u = {u: set(d.ym) for u, d in post.groupby("use")}; months_all = set(post.ym)
    per_use = {}
    for u in A.PRIMARY_IMP["uses"]:
        M = months_u.get(u, set()); c = ctrl[ctrl.use == u]
        row = dict(n_treated=int(post[post.use == u].pid.nunique()), n_treated_post_calendar_months=len(M),
                   treated_post_first=int(min(M)) if M else None, treated_post_last=int(max(M)) if M else None,
                   n_control_pool=int(c.pid.nunique()),
                   n_control_pool_by_group={str(k): int(v) for k, v in c.drop_duplicates("pid").role.value_counts().items()})
        for tag, MM in [("", M), ("_pooled_months", months_all)]:
            cov = (c[c.ym.isin(MM)].groupby("pid").ym.nunique() / max(len(MM), 1)).reindex(c.pid.unique()).fillna(0.0)
            row[f"n_control_ge80{tag}"] = int((cov >= MIN_SHARE).sum())
            if tag == "":
                row["coverage_quantiles"] = {q: float(cov.quantile(v)) if len(cov) else None for q, v in [("p10", .1), ("p50", .5), ("p90", .9), ("max", 1.0)]}
                row["n_control_ge50"] = int((cov >= 0.5).sum())
                percell = c[c.ym.isin(M)].groupby("ym").pid.nunique().reindex(sorted(M)).fillna(0)
                row["controls_per_treated_post_month"] = dict(min=float(percell.min()) if len(percell) else None, median=float(percell.median()) if len(percell) else None,
                                                             months_with_0=int((percell == 0).sum()), months_with_lt5=int((percell < 5).sum()),
                                                             months_with_lt10=int((percell < 10).sum()))
        row["passes"] = bool(row["n_control_ge80"] >= MIN_BLD)
        per_use[USE_EN.get(u, u)] = row
    months = sorted(set(D.ym)); rows = []
    for u in A.PRIMARY_IMP["uses"]:
        c = ctrl[ctrl.use == u]; t = D[(D.ctrl == 0) & (D.use == u)]
        g = lambda d: d.groupby("ym").pid.nunique().reindex(months).fillna(0).astype(int)
        tab = pd.DataFrame({"ym": months, "use": USE_EN.get(u, u), "carrier": carrier,
                            "n_control_pool": g(c).values, "n_control_2024_25": g(c[c.role == "control"]).values,
                            "n_control_2023": g(c[c.role == "aux_2023"]).values,
                            "n_treated_post": g(t[t.post == 1]).values, "n_treated_post_supported": g(t[(t.post == 1) & t.ok]).values,
                            "n_treated_pre_fit": g(t[t.fit]).values})
        rows.append(tab)
    return per_use, pd.concat(rows, ignore_index=True)


def treated_support(B, E, carrier):
    """Treated buildings with >= 1 supported post month against the treated buildings of the analysed sample."""
    def one(D):
        t = D[(D.ctrl == 0) & (D.post == 1)]; g = t.groupby("pid").ok
        n_any, n_all = int(g.any().sum()), int(g.all().sum()); n = int(D[D.ctrl == 0].pid.nunique()); n_post = int(t.pid.nunique())
        return dict(n_treated_analysed=n, n_treated_with_post_month=n_post, n_treated_with_supported_post_month=n_any,
                    n_treated_without_supported_post_month=n - n_any, n_treated_all_post_months_supported=n_all,
                    share_treated_with_supported_post_month=float(n_any / n) if n else None,
                    n_post_months=int(len(t)), n_post_months_supported=int(t.ok.sum()),
                    share_post_months_supported=float(t.ok.mean()) if len(t) else None,
                    median_supported_post_months_per_building=float(g.sum().median()) if len(t) else None)
    out = {}
    Bi = B.drop_duplicates("pid").set_index("pid")
    for tag, kw, spec in [("headline_health_centre", dict(A.PRIMARY_IMP, uses=("보건",)), "imputation_health_only"), ("pooled", dict(A.PRIMARY_IMP), "imputation_use")]:
        S = A.imputation_setup(B, E, carrier, **kw)
        if S is None: out[tag] = dict(status="no treated rows", spec_in_G7=spec); continue
        D = S["D"].copy(); D["ok"] = S["fit"]()["ok"]; D["use"] = D.pid.map(Bi.use_class)
        row = dict(status="ok", spec_in_G7=spec, **one(D))
        if tag == "pooled": row["by_use"] = {USE_EN.get(u, u): one(D[D.use == u]) for u in A.PRIMARY_IMP["uses"]}
        out[tag] = row
    return out


def record_based(B):
    """Record-based count (NO meter data, no outcome): control-pool candidates whose COLLECTION WINDOW can reach the 80% criterion."""
    _argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        import s03_energy as S3
    finally: sys.argv = _argv
    from s03_energy import ym_add, ym_range
    Bi = B[(B.link_grade == "A_cand") & B.start_ym.notna() & B.end_ym.notna()].drop_duplicates("pid")
    P = S3.plan(Bi[Bi.role.isin(["control", "aux_2023"])]); months = {}
    for ps in (0, 1, 2, 5):
        for pid, ym in P[ps]: months.setdefault(pid, set()).add(ym)
    out = {}
    for u in A.PRIMARY_IMP["uses"]:
        T = Bi[(Bi.role == "treated") & (Bi.use_class == u)]; M = set()
        for t in T.itertuples(): M |= {m for m in ym_range(ym_add(int(t.end_ym), 1), ym_add(int(t.end_ym), 12)) if m <= S3.LATEST}
        C = Bi[Bi.role.isin(["control", "aux_2023"]) & (Bi.use_class == u)]; cov = {}
        for c in C.itertuples():
            w = {m for m in months.get(c.pid, ()) if m < ym_add(int(c.start_ym), -3)}
            cov[c.pid] = len(w & M) / len(M) if M else 0.0
        cv = pd.Series(cov, dtype=float); role = C.set_index("pid").role
        out[USE_EN.get(u, u)] = dict(n_treated_candidates=int(len(T)), n_treated_post_calendar_months=len(M),
                                     treated_post_first=int(min(M)) if M else None, treated_post_last=int(max(M)) if M else None,
                                     n_control_pool_candidates=int(len(C)), n_control_pool_candidates_by_group={str(k): int(v) for k, v in role.value_counts().items()},
                                     n_can_reach_80=int((cv >= MIN_SHARE).sum()),
                                     n_can_reach_80_by_group={str(k): int(v) for k, v in role[cv[cv >= MIN_SHARE].index].value_counts().items()},
                                     max_coverage=float(cv.max()) if len(cv) else None, median_coverage=float(cv.median()) if len(cv) else None,
                                     n_can_reach_50=int((cv >= 0.5).sum()), criterion_reachable=bool((cv >= MIN_SHARE).sum() >= MIN_BLD))
    return dict(definition=record_based.__doc__, last_collected_month=int(S3.LATEST), per_use=out,
                headline_use_reachable_from_records=("health_centre" if out.get("health_centre", {}).get("criterion_reachable") else "none"))


def main():
    B, E = A.load()
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M"), rule=dict(min_buildings=MIN_BLD, min_share_of_treated_post_months=MIN_SHARE,
               text="pooled electricity ATT is the headline only if each use has >= 10 control-pool buildings observed in >= 80% of the "
                    "treated post-period calendar months; otherwise health centres"), definition=__doc__, min_win=A.MIN_WIN)
    tabs = []
    for car in ["elec", "total"]:
        try:
            pu, tab = evaluate(B, E, car)
            if pu is None: out[car] = dict(status="no treated rows"); continue
            notrim, _ = evaluate(B, E, car, ctrl_trim=0)
            for u in pu: pu[u]["n_control_ge80_no_trim"] = notrim[u]["n_control_ge80"]
            out[car] = dict(status="ok", per_use=pu, rule_met=bool(all(v["passes"] for v in pu.values())))
            out[car]["rule_met_variants"] = dict(pooled_months=bool(all(v["n_control_ge80_pooled_months"] >= MIN_BLD for v in pu.values())),
                                                 no_trim=bool(all(v["n_control_ge80_no_trim"] >= MIN_BLD for v in pu.values())))
            tabs.append(tab)
        except Exception as ex:
            out[car] = dict(status=f"{type(ex).__name__}: {ex}"[:300])
    out["treated_support"] = dict(definition=treated_support.__doc__)
    for car in ["elec", "total"]:
        try: out["treated_support"][car] = treated_support(B, E, car)
        except Exception as ex: out["treated_support"][car] = dict(status=f"{type(ex).__name__}: {ex}"[:300])
    try: out["record_based"] = record_based(B)
    except Exception as ex: out["record_based"] = dict(status=f"{type(ex).__name__}: {ex}"[:300])
    e = out.get("elec", {})
    out["headline"] = ("pooled (health centres + daycares)" if e.get("rule_met") else "health centres (pooled estimate secondary)") if e.get("status") == "ok" else "undetermined"
    out["headline_spec_in_G7"] = ("imputation_use" if e.get("rule_met") else "imputation_health_only") if e.get("status") == "ok" else None
    out["note"] = "evaluated on the energy data on disk at run time; partial-data outcomes are for debugging and are not the final outcome of the rule"
    (RES / "a_support_rule.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    if tabs: pd.concat(tabs, ignore_index=True).to_csv(RES / "a_support_controls_by_month.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "definition"}, indent=1, ensure_ascii=False, default=float))

if __name__ == "__main__":
    main()
