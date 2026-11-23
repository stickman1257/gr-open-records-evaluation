"""Locator records against the operator's counts by selection year."""
import sys, re, zipfile, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent; RES, PROC = ROOT / "results", ROOT / "data/processed"
sys.path.insert(0, str(HERE)); _a, sys.argv = sys.argv, sys.argv[:1]
import s04_analysis as A4, s02_link as L2, a_program_descriptives as PD
from a_dedup import dedup_physical
from common import sido_of
from s03_energy import parse_period, ym_add
sys.argv = _a
S = dict(dtype={"sigunguCd": str, "bjdongCd": str, "platGbCd": str, "bun": str, "ji": str, "bdMgtSn": str})

g, _ = PD.load_projects(); d = PD.build(g); d = d[d.year.isin([2021, 2022])].copy(); d["pid"] = d.year.astype(str) + "_" + d.seq.astype(str)
d["sido"] = d.address_list.apply(sido_of); H = d[d.use_class == "보건"].copy(); H["grp"] = np.where(H.sido == "전남", "Jeonnam", "Other provinces")
P = pd.read_csv(PROC / "gr_parcels.csv", **S); P["pid"] = P.year.astype(str) + "_" + P.seq.astype(str)
G = pd.read_csv(PROC / "gr_projects_raw.csv")
Pg = P.merge(G[["year", "seq", "area_txt", "name_list", "period_txt"]].drop_duplicates(["year", "seq"]), on=["year", "seq"], how="left")
Pg["g2"] = [L2.regrade(x.name_list, x.reg_use, x.reg_area, x.area_txt, x.reg_n_main, x.reg_is_collective, x.bdKdcd, x.reg_area_share)[1] if x.match_status == "MATCHED" else None for x in Pg.itertuples()]
se = Pg.period_txt.apply(parse_period); Pg["s"] = [x[0] for x in se]; Pg["e"] = [x[1] for x in se]
B = pd.read_csv(PROC / "analysis_buildings.csv", **S); Rg = A4.regrade_buildings(B[B.role.isin(["treated", "control", "aux_2023"])]); Rd, _ = dedup_physical(Rg)
E = pd.read_csv(PROC / "energy_monthly.csv", dtype={"pid": str, "pass_": str}); El = E[E.carrier == "elec"]
obs = set(El[(El.status == "ok") & (El.kwh > 0)].pid); asked = set(El.pid)
B4, E4 = A4.load(); okw = set(A4.complete_windows(A4.carrier_frame(E4, "elec"), B4))
S4 = A4.imputation_setup(B4, E4, "elec", **A4.PRIMARY_IMP); f4 = S4["fit"](); an = set(S4["pids"][pd.unique(S4["pi"][S4["post"] & f4["ok"]])])
sets = [("1 Projects with a detail record", set(H.pid)),
        ("2 Address matched to one parcel", set(P[~P.match_status.isin(["ADDR_NOT_FOUND", "ADDR_AMBIGUOUS", "JUSO_ERROR"])].pid)),
        ("3 Register record found", set(P[P.match_status == "MATCHED"].pid)),
        ("4 Grade A (regraded)", set(Pg[Pg.g2 == "A_cand"].pid)),
        ("5 Dates parsed", set(Pg[Pg.s.notna()].pid)),
        ("6 Completion leaves 12 post months (role treated)", set(Rg[Rg.role == "treated"].pid)),
        ("7 One record per building (D23)", set(Rd.pid)),
        ("8 Electricity requested", asked),
        ("9 At least one month returned", obs),
        ("10 11 of 12 months in both windows, QA gate", okw),
        ("11 Analysed", an)]
rows, cur = [], {k: set(v.pid) for k, v in H.groupby("grp")}
for lab, s in sets:
    r = dict(stage=lab)
    for k in ["Jeonnam", "Other provinces"]:
        cur[k] = cur[k] & s; r[k] = len(cur[k])
    rows.append(r)
T = pd.DataFrame(rows); n0 = T.iloc[0]
for k in ["Jeonnam", "Other provinces"]: T[k + " (%)"] = (100 * T[k] / n0[k]).round(1)
T.to_csv(RES / "ep_scope_jeonnam_stages.csv", index=False); print(T.to_string(index=False))

B["new"] = B.sigunguCd.str[:2]; B["old"] = B.bdMgtSn.str[:2]
B["grp"] = np.select([(B.new == "12") & (B.old == "46"), (B.new == "12") & (B.old == "29"), B.new == "12"], ["Jeonnam (12, former 46)", "Gwangju (12, former 29)", "code 12, other"], "All other provinces")
e = E.merge(B[["pid", "grp"]], on="pid"); e["src"] = np.where(e.pass_ == "bulk", "bulk file", "API call")
X = e.groupby(["grp", "carrier", "src"]).agg(buildings=("pid", "nunique"), building_months=("pid", "size"), ok=("status", lambda s: int((s == "ok").sum())),
                                              empty=("status", lambda s: int((s == "empty").sum())), other=("status", lambda s: int((~s.isin(["ok", "empty"])).sum()))).reset_index()
X["ok_share"] = (X.ok / X.building_months).round(4)
nb = B.groupby("grp").pid.nunique().rename("buildings_in_frame"); X = X.merge(nb, on="grp")
X.to_csv(RES / "ep_scope_jeonnam_energy.csv", index=False); print(X.to_string(index=False))
print("codes used for code-12 parcels:", e[e.grp.str.contains("12")].code_used.value_counts(dropna=False).to_dict())
print("old_code() for a code-12 district:", __import__("s03_energy").old_code("12740"))

BULK = pathlib.Path.home() / "data/gr_bulk/raw"; rowsb = []
J = B[B.new == "12"].drop_duplicates("pid"); new = {(r.sigunguCd, r.bjdongCd, r.platGbCd, r.bun, r.ji): r.pid for r in J.itertuples()}
old = {(r.bdMgtSn[:5], r.bdMgtSn[5:10], r.platGbCd, r.bun, r.ji): r.pid for r in J.itertuples() if isinstance(r.bdMgtSn, str)}
for ym in ["202311", "202406", "202510"]:
    f = BULK / f"elec_{ym}" / "file.zip"
    if not f.exists(): rowsb.append(dict(ym=ym, status="file absent")); continue
    pref, hn, ho = {}, set(), set(); z = zipfile.ZipFile(f)
    with z.open(z.infolist()[0].filename) as fh:
        for line in fh:
            p = [x.strip() for x in line.decode("utf-8", "replace").rstrip("\r\n").rstrip("|").split("|")]
            if len(p) < 10: continue
            pref[p[2][:2]] = pref.get(p[2][:2], 0) + 1; k = (p[2], p[3], p[7], p[8], p[9])
            if k in new: hn.add(new[k])
            if k in old: ho.add(old[k])
    rowsb.append(dict(ym=ym, status="read", lines_prefix_12=pref.get("12", 0), lines_prefix_46=pref.get("46", 0), lines_prefix_29=pref.get("29", 0),
                      lines_all=sum(pref.values()), code12_parcels_in_frame=len(J), found_under_new_key=len(hn), found_under_former_key=len(ho),
                      found_former_grade_A_treated_health_2021_22=len(ho & set(H.pid) & set(Rd[(Rd.role == "treated") & (Rd.link_grade == "A_cand")].pid))))
Bk = pd.DataFrame(rowsb); Bk.to_csv(RES / "ep_scope_bulk_codes.csv", index=False); print(Bk.T.to_string())

KAL = pd.DataFrame([(2021, 895, 443, 395, 57, 0, 0, 0), (2022, 575, 271, 276, 28, 0, 0, 0), (2023, 617, 72, 127, 4, 38, 376, 0), (2024, 562, 58, 163, 1, 0, 340, 0), (2025, 268, 0, 40, 0, 12, 191, 25)],
                   columns=["year", "total", "daycare", "health", "medical", "library", "senior", "other"])
G["nm"] = G.name_list.str.replace(r"\s*\((작성완료|작업완료)\)", "", regex=True).str.strip()
def typ(n):
    n = str(n)
    if "어린이집" in n: return "daycare"
    if "경로당" in n or "노인" in n or "쉼터" in n: return "senior"
    if "의료원" in n or "병원" in n: return "medical"
    if "보건" in n or "진료소" in n: return "health"
    if "도서관" in n: return "library"
    return "other"
G["typ"] = G.nm.map(typ); Lc = pd.crosstab(G.year, G.typ); Lc["total"] = Lc.sum(axis=1)
out = []
for r in KAL.itertuples():
    for c in ["total", "daycare", "health", "medical", "library", "senior", "other"]:
        out.append(dict(year=r.year, type=c, locator=int(Lc.loc[r.year, c]) if c in Lc else 0, kalis=int(getattr(r, c))))
O = pd.DataFrame(out); O["diff"] = O.locator - O.kalis; O.to_csv(RES / "ep_scope_locator_kalis.csv", index=False)
print(O.pivot(index="year", columns="type", values="locator")[["total", "daycare", "health", "medical", "library", "senior", "other"]].to_string())
print(O.pivot(index="year", columns="type", values="diff")[["total", "daycare", "health", "medical", "library", "senior", "other"]].to_string())
st = pd.to_numeric(G.period_txt.str.extract(r"(20\d\d)")[0], errors="coerce"); cost0 = pd.to_numeric(G.cost_txt.str.replace(r"[^\d]", "", regex=True), errors="coerce").fillna(0).eq(0)
multi = G.groupby("address_list").year.transform("nunique") > 1
F = pd.DataFrame(dict(records=G.groupby("year").size(), distinct_name_address=G.drop_duplicates(["year", "nm", "address_list"]).groupby("year").size(),
                      distinct_address=G.drop_duplicates(["year", "address_list"]).groupby("year").size(),
                      test_entries=G[G.nm.str.contains("테스트", na=False)].groupby("year").size(), signature_type=G[G.ptype.str.contains("시그니처", na=False)].groupby("year").size(),
                      no_period=G[st.isna()].groupby("year").size(), zero_cost=G[cost0].groupby("year").size(),
                      start_year_after_selection_year=G[st > G.year].groupby("year").size(), address_also_in_another_year=G[multi].groupby("year").size())).fillna(0).astype(int)
F["repeated_entries"] = F.records - F.distinct_name_address; F = F.merge(KAL.set_index("year")[["total"]].rename(columns={"total": "kalis"}), left_index=True, right_index=True)
F["excess"] = F.records - F.kalis; F["excess_after_repeats_and_tests"] = F.distinct_name_address - F.test_entries - F.kalis
F.to_csv(RES / "ep_scope_locator_flags.csv"); print(F.T.to_string()); print(F.sum().to_dict())
print(pd.crosstab(Rd.year, Rd.role)); print(pd.crosstab(B4[B4.pid.isin(set(S4["pids"]))].year, B4[B4.pid.isin(set(S4["pids"]))].role))
