"""Rule-based classification of links against the register."""
import json, hashlib, re, sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROC, RAW, RES
import s04_analysis as A
import s02_link as L2

SEED, N_GRADE, N_CTRL = 0, 40, 20
LABEL = "exploratory, post hoc (verification round V5)"
TYPE_WORDS = {"보건": ["보건진료소", "보건지소", "보건소", "보건의료원", "진료소", "보건"], "어린이집": ["어린이집"],
              "경로당": ["경로당", "노인정", "노인회관", "노인"], "기타": []}
STRIP = ["건강생활지원센터", "보건진료소", "보건지소", "보건의료원", "보건소", "진료소", "어린이집", "경로당", "국공립", "구립", "시립", "군립",
         "도립", "공립", "국립", "시립", "직장"]


def old_code(sig):
    if sig[:2] == "51": return "42" + sig[2:]
    if sig[:2] == "52": return "45" + sig[2:]
    if sig in ("41192", "41194", "41196"): return "41190"
    if sig == "27720": return "47720"
    return None


def cache(sub, ident):
    p = RAW / sub / (hashlib.sha1(ident.encode()).hexdigest()[:16] + ".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def nz(x): return re.sub(r"\s+", "", str(x)) if isinstance(x, str) else ""


def core_name(name):
    n = nz(re.sub(r"\(.*?\)", "", str(name)))
    for w in STRIP: n = n.replace(w, "")
    return n


def energy_index():
    """parcel key -> max items per ok electricity call, distinct response addresses (cached fields: useQty, newPlatPlc only)."""
    ix = {}
    for f in glob.glob(str(RAW / "energy" / "calls_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("status") != "ok" or r.get("op") != "getBeElctyUsgInfo": continue
            k = (r["sig"], r["bjd"], r["plat"], r["bun"], r["ji"])
            e = ix.setdefault(k, dict(n=0, calls=0, addr=set()))
            e["n"] = max(e["n"], int(r.get("n_items") or 0)); e["calls"] += 1
            e["addr"].update(a for a in (r.get("addr") or []) if a)
    return ix


def evidence(r, EIX, EM):
    sig = str(r.sigunguCd).split(".")[0].zfill(5); bjd = str(int(float(r.bjdongCd))).zfill(5); plat = str(int(float(r.platGbCd)))
    bun, ji = f"{int(float(r.bun)):04d}", f"{int(float(r.ji)):04d}"
    o = dict(pid=r.pid, role=r.role, link_grade=r.link_grade, use_class=r.use_class, sel_year=int(r.year),
             prog_name=r.name_list, prog_addr=r.addr_query, prog_area_txt=r.area_txt,
             detail_record_cached=(RAW / "gr_sites" / "detail" / f"{r.pid}.html").exists())
    t = cache("bldrgst", f"getBrTitleInfo|{sig}|{bjd}|{plat}|{bun}|{ji}")
    items = (t or {}).get("items") or []
    main = [x for x in items if x.get("mainAtchGbCdNm", "주건축물") == "주건축물"]
    hit = [x for x in items if str(x.get("mgmBldrgstPk")) == str(int(float(r.reg_pk)))] if r.reg_pk == r.reg_pk else []
    m = hit[0] if hit else {}
    others = [x for x in items if x is not m]
    desc = lambda x: f"{nz(x.get('bldNm'))}/{nz(x.get('dongNm'))}|{x.get('mainAtchGbCdNm')}|{x.get('mainPurpsCdNm')}|{x.get('etcPurps')}|{x.get('totArea')}"
    o.update(register_cached=t is not None, n_titles=len(items), n_main_titles=len(main), n_attached_titles=len(items) - len(main),
             collective=bool(any(x.get("regstrGbCdNm") == "집합" for x in items) or str(r.bdKdcd) in ("1", "1.0")),
             reg_name=f"{nz(m.get('bldNm'))}/{nz(m.get('dongNm'))}", reg_use=m.get("mainPurpsCdNm"), reg_use_detail=m.get("etcPurps"),
             reg_area=m.get("totArea"), reg_area_share_of_main=r.reg_area_share,
             other_main_titles=" ; ".join(desc(x) for x in others if x in main),
             attached_titles=" ; ".join(desc(x) for x in others if x not in main),
             n_other_main_uses=len({x.get("mainPurpsCdNm") for x in others if x in main} - {m.get("mainPurpsCdNm")}))
    cands = [a for a in L2.parse_areas(r.area_txt) if a > 0]; ra = float(m.get("totArea") or 0)
    ok = lambda a: ra > 0 and abs(ra - a) / a <= 0.30
    o["area_ratio_reg_over_gfa"] = round(ra / cands[-1], 3) if cands and ra > 0 else np.nan
    o["area_ratio_reg_over_first"] = round(ra / cands[0], 3) if len(cands) == 2 and ra > 0 else np.nan
    o["area_status"] = ("no_project_area" if not cands else "within30_gfa" if ok(cands[-1]) else
                        "footprint_only" if ok(cands[0]) else "outside")
    uc = r.use_class
    o["use_status"] = ("facility_type_other" if uc not in L2.USE_OK else
                       "agree" if any(k in str(m.get("mainPurpsCdNm")) for k in L2.USE_OK[uc]) else "conflict")
    j = cache("juso", str(r.addr_query)) or {}
    js = [x for x in ((j.get("results") or {}).get("juso") or [])]
    tail = nz(re.sub(r"^.*?\d+(-\d+)?(?=\s|$)", "", str(r.jibunAddr), count=0)) if isinstance(r.jibunAddr, str) else ""
    tail = nz(str(r.jibunAddr).split()[-1]) if isinstance(r.jibunAddr, str) and not re.match(r"^(산)?\d+(-\d+)?$", str(r.jibunAddr).split()[-1]) else ""
    src_title = [nz(m.get("bldNm")), nz(m.get("dongNm"))]
    src_addr = [nz(x.get("bdNm")) for x in js] + [nz(x.get("detBdNmList")) for x in js] + [tail]
    src_other = [nz(x.get("bldNm")) + "/" + nz(x.get("dongNm")) for x in others]
    core = core_name(r.name_list); tw = TYPE_WORDS.get(uc, [])
    has = lambda L_: len(core) >= 2 and any(core in s for s in L_ if s)
    typ = lambda L_: any(w in s for s in L_ if s for w in tw)
    allsrc = src_title + src_addr
    o["addr_search_name"] = "|".join(dict.fromkeys(s for s in src_addr if s))
    o["name_core"] = core
    o["name_status"] = ("match" if has(allsrc) else "blank" if not any(allsrc) else "type_only" if typ(allsrc) else "other")
    o["name_on_other_title"] = bool(any(src_title) and not has(src_title) and has(src_other))
    oc = old_code(sig); em = EM.get(r.pid)
    o.update(addr_search_n=int(r.juso_n) if r.juso_n == r.juso_n else np.nan, addr_search_exact=r.juso_exact,
             old_code_parcel=oc is not None,
             n_elec_months_ok=int(len(em)) if em is not None else 0,
             n_elec_months_old_code=int((em.code_used.astype(str).str.split(".").str[0] == oc).sum()) if em is not None and oc else 0)
    o["old_code_recovery"] = o["n_elec_months_old_code"] > 0
    e = EIX.get((sig, bjd, plat, bun, ji), None); eo = EIX.get((oc, bjd, plat, bun, ji), None) if oc else None
    addrs = set().union(*(x["addr"] for x in (e, eo) if x))
    o.update(energy_calls_ok=sum(x["calls"] for x in (e, eo) if x), energy_items_max=max([x["n"] for x in (e, eo) if x], default=0),
             energy_resp_addr="|".join(sorted(addrs)),
             energy_resp_addr_eq_register=(any(nz(a) == nz(m.get("newPlatPlc")) for a in addrs) if addrs and nz(m.get("newPlatPlc")) else np.nan),
             meter_building_attribution="none: request and cached response are keyed by parcel")
    con = [c for c, v in [("C1", o["name_status"] == "other"), ("C2", o["name_on_other_title"]),
                          ("C3", o["use_status"] == "conflict"), ("C4", o["area_status"] == "outside")] if v]
    unr = [c for c, v in [("U1", o["n_main_titles"] != 1), ("U2", o["collective"]), ("U3", o["name_status"] in ("blank", "type_only")),
                          ("U4", o["area_status"] in ("footprint_only", "no_project_area")),
                          ("U5", o["use_status"] == "facility_type_other")] if v]
    o["class_rule"] = "conflicting" if con else "unresolved" if unr else "record-consistent"
    o["class_reasons"] = ";".join(con + unr)
    return o


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(float((c - h) / d), 4), round(float((c + h) / d), 4)]


def shares(d):
    n = len(d)
    return dict(n=n, **{c: dict(k=int((d.class_rule == c).sum()), share=round(float((d.class_rule == c).mean()), 4) if n else None,
                                wilson95=wilson(int((d.class_rule == c).sum()), n))
                        for c in ["record-consistent", "unresolved", "conflicting"]},
                reasons={k: int(v) for k, v in pd.Series([x for s in d.class_reasons for x in s.split(";") if x]).value_counts().items()})


def signature(d, SG):
    """Existing diagnostic (results/a_signature_eval.csv, unchanged): per-building share of donor ratios above the real ratio"""
    s = d.merge(SG, on="pid")
    if s.empty: return dict(n_with_R=0)
    a = 1 - (s.perm_p * (s.n_donors + 1) - 1) / s.n_donors
    return dict(n_with_R=int(len(s)), auc_matched=round(float(a.mean()), 4), median_R=round(float(s.R_constr.median()), 4),
                share_R_le_0p7=round(float((s.R_constr <= 0.7).mean()), 4))


if __name__ == "__main__":
    B, E = A.load()
    g = pd.read_csv(PROC / "gr_projects_raw.csv", usecols=["year", "seq", "area_txt"]).drop_duplicates(["year", "seq"])
    if "area_txt" not in B: B = B.merge(g, on=["year", "seq"], how="left")
    EIX = energy_index()
    EMraw = pd.read_csv(PROC / "energy_monthly.csv")
    EM = {p: d for p, d in EMraw[(EMraw.carrier == "elec") & (EMraw.status == "ok")].groupby("pid")}
    B = B.sort_values("pid").reset_index(drop=True)
    T = B[B.role == "treated"]; Cc = B[B.role.isin(["control", "aux_2023"]) & (B.link_grade == "A_cand")]
    rng = np.random.default_rng(SEED); pick = []
    frame = {}
    for gr in ["A_cand", "B", "C"]:
        f = T[T.link_grade == gr]; frame[f"treated_{gr}"] = int(len(f))
        pick += [(p, f"treated_{gr}") for p in rng.choice(f.pid.to_numpy(), min(N_GRADE, len(f)), replace=False)]
    frame["control_pool_A_cand"] = int(len(Cc))
    pick += [(p, "control_A_cand") for p in rng.choice(Cc.pid.to_numpy(), min(N_CTRL, len(Cc)), replace=False)]
    Bi = B.set_index("pid", drop=False)
    cen = pd.DataFrame([evidence(r, EIX, EM) for r in pd.concat([T, Cc]).itertuples(index=False)])
    S = A.imputation_setup(B, E, "elec", uses=("보건",), fe="use")
    head_t = set(S["T"].pid); f0 = S["fit"](); head_c = set(S["pids"][pd.unique(S["pi"][f0["F"] & S["isc"]])])
    cen["headline_health"] = np.where(cen.pid.isin(head_t), "treated", np.where(cen.pid.isin(head_c), "control", ""))
    strat = dict(pick)
    cen["stratum"] = cen.pid.map(strat)
    cen.to_csv(RES / "v5_audit_census.csv", index=False, encoding="utf-8-sig")
    sheet = cen[cen.stratum.notna()].copy()
    sheet["order"] = sheet.pid.map({p: i for i, (p, _) in enumerate(pick)}); sheet = sheet.sort_values("order").drop(columns="order")
    sheet["label"] = LABEL; sheet["reviewer_class"] = ""; sheet["reviewer_note"] = ""
    first = ["pid", "stratum", "role", "link_grade", "use_class", "headline_health", "class_rule", "class_reasons", "reviewer_class", "reviewer_note"]
    sheet[first + [c for c in sheet.columns if c not in first]].to_csv(RES / "v5_audit_sheet.csv", index=False, encoding="utf-8-sig")

    SG = pd.read_csv(RES / "a_signature_eval.csv")[["pid", "R_constr", "perm_p", "n_donors"]]
    out = dict(label=LABEL, source="src/v5_linkage_audit.py", seed=SEED, frame=frame,
               sampling="default_rng(0).choice without replacement on rows sorted by pid; order A_cand, B, C, controls",
               n_sampled={k: int(v) for k, v in sheet.stratum.value_counts().items()},
               coverage={k: round(int((sheet.stratum == k).sum()) / frame[k.replace("control_A_cand", "control_pool_A_cand")], 4)
                         for k in sheet.stratum.unique()},
               rules=__doc__.split("CLASS RULES")[1].split("Outputs:")[0].strip(),
               sample={k: shares(d) for k, d in sheet.groupby("stratum")},
               census={"treated_" + k: shares(d) for k, d in cen[cen.role == "treated"].groupby("link_grade")},
               census_control_pool_A=shares(cen[cen.role != "treated"]),
               census_headline_health_treated=shares(cen[cen.headline_health == "treated"]),
               census_headline_health_control=shares(cen[cen.headline_health == "control"]),
               signature_all_with_R=signature(cen[cen.role == "treated"], SG),
               signature_by_grade_class={f"{gk}|{ck}": signature(d, SG) for (gk, ck), d in cen[cen.role == "treated"].groupby(["link_grade", "class_rule"])},
               signature_headline_by_class={ck: signature(d, SG) for ck, d in cen[cen.headline_health == "treated"].groupby("class_rule")},
               signature_sample_by_grade_class={f"{gk}|{ck}": signature(d, SG) for (gk, ck), d in sheet[sheet.role == "treated"].groupby(["link_grade", "class_rule"])},
               evidence_frame=dict(
                   treated_by_grade={gk: dict(n=int(len(d)), n_main_gt1=int((d.n_main_titles > 1).sum()), collective=int(d.collective.sum()),
                                              old_code_parcel=int(d.old_code_parcel.sum()), old_code_recovery=int(d.old_code_recovery.sum()),
                                              other_main_use_differs=int((d.n_other_main_uses > 0).sum()),
                                              energy_items_max_gt1=int((d.energy_items_max > 1).sum()),
                                              energy_addr_eq_register=int((d.energy_resp_addr_eq_register == True).sum()),
                                              energy_addr_differs=int((d.energy_resp_addr_eq_register == False).sum()),
                                              name=d.name_status.value_counts().to_dict(), area=d.area_status.value_counts().to_dict(),
                                              use=d.use_status.value_counts().to_dict())
                                     for gk, d in cen[cen.role == "treated"].groupby("link_grade")}),
               energy_api=dict(
                   request_keys="sigunguCd, bjdongCd, platGbCd, bun, ji, useYm (src/s03_energy.py); no building identifier can be passed",
                   cached_response_fields="useQty, newPlatPlc, number of items (data/raw/energy/calls_*.jsonl); the full response body was not cached",
                   meter_to_building_attribution_in_cache=False,
                   note="No cached field identifies a building, a meter or a customer. Whether the live response carries further fields cannot be checked offline; "
                        "the collection code read only useQty and newPlatPlc."),
               ai_audit_overlap=None)
    try:
        ai = pd.read_csv(RES / "a_audit_links.csv")[["pid", "ai_verdict"]].merge(cen[["pid", "link_grade", "class_rule"]], on="pid")
        out["ai_audit_overlap"] = dict(n=int(len(ai)), note="earlier AI-assisted labels (not human-verified) against this script's class, all 70",
                                       table={f"{a}|{c}": int(n) for (a, c), n in ai.groupby(["ai_verdict", "class_rule"]).size().items()})
    except Exception as ex:
        out["ai_audit_overlap"] = dict(error=str(ex)[:120])
    (RES / "v5_linkage_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["frame", "n_sampled", "coverage", "sample", "census", "census_control_pool_A",
                                          "census_headline_health_treated", "census_headline_health_control", "signature_all_with_R",
                                          "signature_by_grade_class", "signature_headline_by_class", "signature_sample_by_grade_class",
                                          "evidence_frame", "ai_audit_overlap"]}, indent=1, ensure_ascii=False, default=str))
