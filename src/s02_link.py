"""Step 2: address -> parcel -> building register; link grades A/B/C."""
import json, os, re, sys, time, hashlib, pathlib
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, PROC, RAW

def load_keys():
    p = pathlib.Path.home() / ".config/gr-energy/datagokr.env"
    if not p.exists(): return
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
load_keys()

def _cache(sub, ident):
    d = RAW / sub; d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(ident.encode()).hexdigest()[:16] + ".json")

NUM = re.compile(r"^(산)?\d+(-\d+)?(번지)?$")
def clean_address(a: str) -> str:
    """Keep address up to the last house/lot number token: drops appended building names, floors, wings."""
    a = re.sub(r"\(.*?\)", " ", str(a)); a = re.sub(r"[,]", " ", a)
    t = a.split()
    idx = [i for i, x in enumerate(t) if NUM.match(x)]
    if idx: t = t[: idx[0] + 1] if len(idx) and re.search(r"(로|길)$", t[idx[0] - 1] if idx[0] > 0 else "") else t[: idx[-1] + 1]
    return re.sub(r"(번지)$", "", " ".join(t)).strip()

def query_variants(q):
    t = q.split(); vs = [q]
    if any(re.search(r"(로|길)$", x) for x in t):
        vs.append(" ".join(x for x in t if not re.search(r"(읍|면|리)$", x)))
    vs.append(" ".join(t[-3:]) if len(t) >= 3 else q)
    return list(dict.fromkeys(vs))

def juso(keyword):
    p = _cache("juso", keyword)
    if p.exists(): return json.loads(p.read_text())
    d = None
    for attempt in range(5):
        try:
            r = requests.get("https://business.juso.go.kr/addrlink/addrLinkApi.do", timeout=30, params=dict(
                confmKey=os.environ["JUSO_KEY"], currentPage=1, countPerPage=10, keyword=keyword, resultType="json"))
            r.raise_for_status(); d = r.json(); break
        except Exception:
            time.sleep(2 * (attempt + 1))
    if d is None: raise RuntimeError("juso request failed after retries")
    d["_query"] = keyword; d["_retrieved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p.write_text(json.dumps(d, ensure_ascii=False)); time.sleep(0.15)
    return d

def bldrgst(op, sig, bjd, plat, bun, ji):
    ident = f"{op}|{sig}|{bjd}|{plat}|{bun}|{ji}"
    p = _cache("bldrgst", ident)
    if p.exists(): return json.loads(p.read_text())
    items, page, total = [], 1, None
    while True:
        d = None
        for attempt in range(5):
            try:
                r = requests.get(f"https://apis.data.go.kr/1613000/BldRgstHubService/{op}", timeout=60, params=dict(
                    serviceKey=os.environ["DATA_GO_KR_KEY"], sigunguCd=sig, bjdongCd=bjd, platGbCd=plat, bun=bun, ji=ji,
                    numOfRows=100, pageNo=page, _type="json"))
                if r.status_code == 200: d = r.json()["response"]; break
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
        if d is None: return dict(error="REQUEST_FAILED", text="non-JSON or HTTP error after retries", _ident=ident)
        hdr = d["header"]
        if hdr.get("resultCode") != "00": return dict(error=hdr.get("resultCode"), text=hdr.get("resultMsg"), _ident=ident)
        body = d.get("body", {}); total = int(body.get("totalCount", 0) or 0)
        it = (body.get("items") or {}).get("item", []) if body.get("items") else []
        items += it if isinstance(it, list) else [it]
        if len(items) >= total or not it: break
        page += 1; time.sleep(0.1)
    out = dict(items=items, totalCount=total, _ident=ident, _retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    p.write_text(json.dumps(out, ensure_ascii=False)); time.sleep(0.1)
    return out

USE_OK = {"어린이집": ["노유자"], "경로당": ["노유자", "근린생활"], "보건": ["의료", "근린생활", "업무"]}

def use_class(name):
    n = str(name)
    if "어린이집" in n: return "어린이집"
    if "경로당" in n or "노인" in n or "쉼터" in n: return "경로당"
    if "보건" in n or "진료소" in n: return "보건"
    return "기타"

def regrade(name_list, reg_use, reg_area, area_txt, reg_n_main, reg_is_collective, bdKdcd, reg_area_share=None):
    """Recompute use class and link grade from stored fields."""
    uc = use_class(name_list)
    use_ok = uc in USE_OK and any(k in str(reg_use) for k in USE_OK[uc])
    cands = [a for a in parse_areas(area_txt) if a > 0]
    swapped = bool(len(cands) == 2 and cands[0] > 1.5 * cands[1])
    ra = float(reg_area) if reg_area == reg_area and reg_area not in (None, "") else 0.0
    ok = lambda a: ra > 0 and abs(ra - a) / a <= 0.30
    gfa = cands[-1] if cands else None
    area_ok = bool(cands) and (ok(gfa) or (swapped and ok(cands[0])))
    flags = []
    if cands and not area_ok and any(ok(a) for a in cands): flags.append("footprint_only_area")
    if uc == "기타": flags.append("use_other")
    if str(reg_is_collective) == "True" or str(bdKdcd) in ("1", "1.0"): return uc, "C", flags
    if reg_n_main == 1 and use_ok and area_ok: return uc, "A_cand", flags
    sh = float(reg_area_share) if reg_area_share is not None and reg_area_share == reg_area_share else 0.0
    if sh >= 0.8 and use_ok and area_ok: return uc, "B", flags
    return uc, "C", flags

def parse_area(s):
    """'footprint㎡ / GFA㎡' -> GFA (last value)."""
    m = re.findall(r"([\d,\.]+)\s*㎡", str(s))
    return float(m[-1].replace(",", "")) if m else np.nan

def parse_areas(s):
    m = [float(x.replace(",", "")) for x in re.findall(r"([\d,\.]+)\s*㎡", str(s))]
    return m if m else []

def link_one(row):
    out = dict(year=row.year, seq=row.seq)
    raw = row.addr if isinstance(getattr(row, "addr", None), str) else row.address_list
    q0 = clean_address(raw); res = []
    for q in query_variants(q0):
        try: j = juso(q)
        except Exception as e: out["match_status"] = "JUSO_ERROR"; out["err"] = str(e)[:100]; return out
        res = (j.get("results") or {}).get("juso") or []
        if res: break
    out["addr_query"] = q
    out["juso_n"] = len(res)
    if not res: out["match_status"] = "ADDR_NOT_FOUND"; return out
    key2 = " ".join(q.split()[-2:])
    pat = re.compile(r"(^|\s)" + re.escape(key2) + r"(\s|$)")
    exact = [x for x in res if str(x.get("roadAddrPart1", "")).endswith(key2) or pat.search(str(x.get("jibunAddr", "")))]
    cands = exact if exact else res
    parcels = {(x["admCd"], x["mtYn"], x["lnbrMnnm"], x["lnbrSlno"]) for x in cands}
    out["juso_exact"] = len(exact); out["juso_cands"] = len(parcels)
    if len(parcels) > 1: out["match_status"] = "ADDR_AMBIGUOUS"; return out
    c = cands[0]
    sig, bjd = c["admCd"][:5], c["admCd"][5:]
    plat = "1" if str(c["mtYn"]) == "1" else "0"
    bun, ji = f"{int(c['lnbrMnnm']):04d}", f"{int(c['lnbrSlno'] or 0):04d}"
    out.update(sigunguCd=sig, bjdongCd=bjd, platGbCd=plat, bun=bun, ji=ji, roadAddr=c.get("roadAddr"),
               jibunAddr=c.get("jibunAddr"), bdMgtSn=c.get("bdMgtSn"), bdKdcd=c.get("bdKdcd"), siNm=c.get("siNm"), sggNm=c.get("sggNm"))
    t = bldrgst("getBrTitleInfo", sig, bjd, plat, bun, ji)
    if "error" in t: out["match_status"] = "REG_ERROR"; out["err"] = str(t.get("text"))[:100]; return out
    items = t["items"]; out["reg_n_title"] = len(items)
    if not items: out["match_status"] = "REG_EMPTY"; return out
    df = pd.DataFrame(items)
    df["totArea"] = pd.to_numeric(df.get("totArea"), errors="coerce")
    main = df[df.get("mainAtchGbCdNm", "주건축물") == "주건축물"] if "mainAtchGbCdNm" in df else df
    out["reg_n_main"] = len(main); out["reg_is_collective"] = bool((df.get("regstrGbCdNm") == "집합").any())
    gfa = parse_area(getattr(row, "area_txt", None)); out["gr_gfa"] = gfa
    if len(main):
        k = (main.totArea - gfa).abs().idxmin() if np.isfinite(gfa) else main.totArea.idxmax()
        b = main.loc[k]
        out.update(reg_pk=b.get("mgmBldrgstPk"), reg_use=b.get("mainPurpsCdNm"), reg_area=b.get("totArea"),
                   reg_apr=b.get("useAprDay"), reg_bldNm=b.get("bldNm"), reg_dongNm=b.get("dongNm"),
                   reg_area_share=float(b.totArea / main.totArea.sum()) if main.totArea.sum() > 0 else np.nan,
                   reg_parcel_total_area=float(main.totArea.sum()))
    uc = use_class(row.name_list); out["use_class"] = uc
    use_ok = any(k in str(out.get("reg_use", "")) for k in USE_OK.get(uc, [""]))
    cands = [a for a in parse_areas(getattr(row, "area_txt", None)) if a > 0]
    area_ok = bool(out.get("reg_area")) and any(abs(out["reg_area"] - a) / a <= 0.30 for a in cands)
    out["area_swapped_flag"] = bool(len(cands) == 2 and cands[0] > 1.5 * cands[1])
    if out["reg_is_collective"] or str(out.get("bdKdcd")) == "1":
        grade = "C"
    elif out["reg_n_main"] == 1 and use_ok and area_ok:
        grade = "A_cand"
    elif out.get("reg_area_share", 0) >= 0.8 and use_ok and area_ok:
        grade = "B"
    else:
        grade = "C"
    out["link_grade"] = grade; out["use_ok"] = use_ok; out["area_ok"] = bool(area_ok)
    out["match_status"] = "MATCHED"
    return out

if __name__ == "__main__":
    f = PROC / "gr_projects_raw.csv"
    if f.exists(): g = pd.read_csv(f)
    else:
        import s01_gr_sites as s1
        g = s1.table_from_cache()
    years = [int(y) for y in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2021", "2022", "2023", "2024"])]
    g = g[g.year.isin(years)]
    rows = []
    for k, r in enumerate(g.itertuples(index=False)):
        rows.append(link_one(r))
        if k % 100 == 0: print(k, flush=True)
    L = pd.DataFrame(rows); L.to_csv(PROC / "gr_parcels.csv", index=False)
    print(L.match_status.value_counts().to_dict()); print(L.get("link_grade", pd.Series()).value_counts().to_dict())
