"""Project-parcel-register crosswalk for deposit."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC
import s02_link as s2, s04_analysis as A, r3_analysis as R
L = pd.read_csv(PROC / "gr_parcels.csv", dtype={"sigunguCd": str, "bjdongCd": str, "platGbCd": str, "bun": str, "ji": str})
Gp = pd.read_csv(PROC / "gr_projects_raw.csv")[["year", "seq", "area_txt", "name_list"]]
D = L.merge(Gp, on=["year", "seq"], how="left", suffixes=("", "_g")); D["pid"] = D.year.astype(str) + "_" + D.seq.astype(str)
m = D.match_status == "MATCHED"
r = D[m].apply(lambda x: s2.regrade(x.name_list, x.reg_use, x.reg_area, x.area_txt, x.reg_n_main, x.reg_is_collective, x.bdKdcd, x.reg_area_share), axis=1)
D.loc[m, "use_class_regraded"] = [u for u, _, _ in r]; D.loc[m, "link_grade_regraded"] = [g for _, g, _ in r]
mt = D.platGbCd.map({"0": "1", "1": "2"}).fillna(""); ok = D.sigunguCd.notna() & D.bjdongCd.notna() & D.bun.notna()
D["pnu"] = (D.sigunguCd.fillna("") + D.bjdongCd.fillna("") + mt + D.bun.fillna("").str.zfill(4) + D.ji.fillna("0").str.zfill(4)).where(ok, "")
B, E = A.load(); Bi = B.drop_duplicates("pid").set_index("pid")
dropped = set(getattr(A, "DEDUP_DROPPED", None)["pid"]) if isinstance(getattr(A, "DEDUP_DROPPED", None), pd.DataFrame) and "pid" in A.DEDUP_DROPPED else set()
G = pd.read_csv(RES / "r3_building_gaps.csv"); head = set(G.pid); R.TREATED_HEAD.update(head)
o, ev, S = R.event_long(B, E, cluster="sigungu", treated=head); comp = set(S[S.treat == 0].pid)
raw = pd.read_csv(PROC / "analysis_buildings.csv", usecols=["year", "seq"] + [c for c in ("role",) if c in pd.read_csv(PROC / "analysis_buildings.csv", nrows=1).columns])
D["role"] = D.pid.map(Bi.role)
D["dedup_dropped"] = D.pid.isin(dropped).astype(int)
D["headline_treated"] = D.pid.isin(head).astype(int); D["headline_comparison"] = D.pid.isin(comp).astype(int)
cols = ["pid", "year", "pnu", "bdMgtSn", "reg_pk", "reg_n_main", "match_status", "link_grade", "use_class_regraded", "link_grade_regraded", "role", "dedup_dropped", "headline_treated", "headline_comparison"]
out = D[cols].rename(columns={"link_grade": "link_grade_first_assigned"}); out.to_csv(RES / "crosswalk_project_parcel_register.csv", index=False)
print(len(out), out.link_grade_regraded.value_counts().to_dict(), "2021-22 A:", int(((out.year.isin([2021, 2022])) & (out.link_grade_regraded == "A_cand")).sum()),
      "dropped", int(out.dedup_dropped.sum()), "flags", int(out.headline_treated.sum()), int(out.headline_comparison.sum()), "dedup type", type(getattr(A, "DEDUP_DROPPED", None)))
