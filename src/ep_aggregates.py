"""Aggregate-only versions of counts whose source files hold project codes."""
import json, os, pathlib, sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data/processed"
sys.path.insert(0, str(ROOT / "src"))
USE_EN = {"보건": "health_centre", "어린이집": "daycare", "경로당": "senior_centre", "기타": "other"}


def influence_range():
    J = json.loads((RES / "a_influence.json").read_text(encoding="utf-8")); rows = []
    for s, r in J["samples"].items():
        rows.append(dict(sample=s, n_control_buildings=r["n_control_buildings"], pct=r["pct"], min_pct=r["min_pct"], max_pct=r["max_pct"],
                         att_log=r["att_log"], min_log=r["min_log"], max_log=r["max_log"], max_abs_change_log=r["max_abs_d_att_log"],
                         source="results/a_influence.json (samples.<sample>); one control building left out at a time"))
    pd.DataFrame(rows).to_csv(RES / "ep_influence_range.csv", index=False); return rows


def use_grade_counts():
    import s02_link as s2
    G = pd.read_csv(PROC / "gr_projects_raw.csv")[["year", "seq", "area_txt", "name_list"]]; L = pd.read_csv(PROC / "gr_parcels.csv")
    D = G.merge(L.drop(columns=[c for c in ("name_list", "area_txt") if c in L]), on=["year", "seq"], how="left")
    D["use"] = D.name_list.apply(s2.use_class).map(USE_EN)
    D["sel"] = pd.cut(D.year, [0, 2022, 2023, 2100], labels=["2021-22", "2023", "2024-25"]).astype(str)
    D["matched"] = D.match_status == "MATCHED"
    g = [s2.regrade(x.name_list, x.reg_use, x.reg_area, x.area_txt, x.reg_n_main, x.reg_is_collective, x.bdKdcd, x.reg_area_share)[1] if x.matched else "" for x in D.itertuples()]
    D["grade"] = g
    def agg(d):
        return dict(records=int(len(d)), matched=int(d.matched.sum()), grade_A=int((d.grade == "A_cand").sum()), grade_B=int((d.grade == "B").sum()), grade_C=int((d.grade == "C").sum()))
    rows = [dict(facility_type=u, selection_years=s, **agg(d)) for (u, s), d in D.groupby(["use", "sel"])]
    rows += [dict(facility_type=u, selection_years="all", **agg(d)) for u, d in D.groupby("use")]
    rows += [dict(facility_type="all", selection_years=s, **agg(d)) for s, d in D.groupby("sel")] + [dict(facility_type="all", selection_years="all", **agg(D))]
    T = pd.DataFrame(rows); T["definition"] = "facility type from the listed name (s02_link.use_class); grades after s02_link.regrade; unmatched records have no grade"
    T.to_csv(RES / "ep_use_grade_counts.csv", index=False); return T


def construction_duration():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    import s04_analysis as A
    B, E = A.load(); mi = lambda v: (v.astype(int) // 100) * 12 + v.astype(int) % 100
    _, P = A.imputation_did(B, E, "elec", uses=("보건",), fe="use", nboot=0, pretrend=False, return_tau=True)
    tA = B[(B.role == "treated") & (B.link_grade == "A_cand")]
    groups = [("headline sample: treated health centres in the estimate", B[B.pid.isin(set(P.pid))]),
              ("treated health centres, grade A", tA[tA.use_class == "보건"]), ("treated daycare centres, grade A", tA[tA.use_class == "어린이집"]),
              ("all treated buildings, grade A", tA)]
    rows = []
    for name, d in groups:
        m = mi(d.end_ym) - mi(d.start_ym) + 1
        rows.append(dict(group=name, n=int(len(d)), median_months=float(m.median()), q25=float(m.quantile(.25)), q75=float(m.quantile(.75)), mean_months=round(float(m.mean()), 2),
                         definition="months from the first to the last month of the works, both included (end month - start month + 1)",
                         also_in="results/a_r5_pretrend.json (samples.<sample>...b_relative_magnitudes.median_construction_months)"))
    pd.DataFrame(rows).to_csv(RES / "ep_construction_duration.csv", index=False); return rows


if __name__ == "__main__":
    for r in influence_range(): print(f"leave-one-control-out {r['sample']}: {100 * r['min_pct']:.1f} to {100 * r['max_pct']:.1f}% (n controls {r['n_control_buildings']})")
    if (PROC / "gr_parcels.csv").exists():
        T = use_grade_counts(); print(T[T.selection_years == "2021-22"].drop(columns="definition").to_string(index=False))
        for r in construction_duration(): print(f"{r['group']}: n {r['n']}, median {r['median_months']} months (q25 {r['q25']}, q75 {r['q75']})")
    else:
        print("data/processed not present: ep_use_grade_counts.csv and ep_construction_duration.csv not written")
