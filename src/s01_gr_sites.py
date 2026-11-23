"""Step 1: programme project records from the public project locator."""
import re, json, time, hashlib, pathlib, csv, sys, requests, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/gr_sites"; DET = RAW / "detail"; DET.mkdir(parents=True, exist_ok=True)
LIST = sorted(RAW.glob("g_int4000_*.html"))[-1]
UA = {"User-Agent": "Mozilla/5.0 (research data collection; low rate)"}
BASE = "https://www.greenremodeling.or.kr/n1/green_intro/g_int4000_10.asp"
YEARS = [int(y) for y in (sys.argv[1].split(",") if len(sys.argv) > 1 else "2021,2022,2023,2024".split(","))]

def parse_list():
    h = LIST.read_text(encoding="utf-8", errors="replace")
    objs = re.findall(r'\{"year": ?(\d+),"seq": ?(\d+),"buildName": ?"(.*?)","signature_div_kor": ?"(.*?)","address": ?"(.*?)"\}', h)
    df = pd.DataFrame(objs, columns=["year", "seq", "name_list", "project_type", "address_list"]).astype({"year": int, "seq": int})
    return df.drop_duplicates(["year", "seq"]).reset_index(drop=True)

FIELDS = ["건축물명", "사업유형", "주소", "담당기관", "건축면적/연면적", "준공년도", "규모(층수)", "총 사업기간", "적용 공사", "총사업비", "에너지절감률"]
ASCII = dict(zip(FIELDS, ["name", "ptype", "addr", "agency", "area_txt", "built_year", "floors_txt", "period_txt", "measures_txt", "cost_txt", "pred_saving_txt"]))
def parse_detail(h):
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t).replace("&nbsp", " "); t = re.sub(r"\s+", " ", t)
    out = {}
    for i, f in enumerate(FIELDS):
        nxt = FIELDS[i + 1] if i + 1 < len(FIELDS) else "닫기"
        m = re.search(re.escape(f) + r"\s(.*?)\s" + re.escape(nxt), t)
        out[ASCII[f]] = m.group(1).strip() if m else None
    return out

def fetch(seq, year):
    p = DET / f"{year}_{seq}.html"
    if p.exists() and p.stat().st_size > 500: return p.read_text(encoding="utf-8", errors="replace"), False
    r = requests.post(BASE, params={"seq": seq, "year": year}, data=b"", headers=UA, timeout=30)
    r.raise_for_status(); p.write_bytes(r.content); return r.content.decode("utf-8", "replace"), True

def table_from_cache():
    """Offline: list x cached detail pages (no network)."""
    df = parse_list(); rows = []
    for p in sorted(DET.glob("*.html")):
        y, q = p.stem.split("_"); d = parse_detail(p.read_text(encoding="utf-8", errors="replace"))
        d.update(year=int(y), seq=int(q), retrieved_ok=True); rows.append(d)
    return df.merge(pd.DataFrame(rows), on=["year", "seq"], how="inner")

if __name__ == "__main__":
    df = parse_list(); df.to_csv(ROOT / "data/processed/gr_list_all.csv", index=False)
    todo = df[df.year.isin(YEARS)]
    print("list", len(df), "todo", len(todo), flush=True)
    rows = []
    for k, r in enumerate(todo.itertuples()):
        try:
            h, net = fetch(r.seq, r.year)
            d = parse_detail(h); d.update(year=r.year, seq=r.seq, retrieved_ok=True)
        except Exception as e:
            d = dict(year=r.year, seq=r.seq, retrieved_ok=False, error=str(e)[:200]); net = True
        rows.append(d)
        if net: time.sleep(1.5)
        if k % 100 == 0: print(k, flush=True)
    out = table_from_cache()
    out.to_csv(ROOT / "data/processed/gr_projects_raw.csv", index=False)
    with open(ROOT / "manifests/sha256_gr_sites.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "sha256"])
        for p in [LIST] + sorted(DET.glob("*.html")): w.writerow([str(p.relative_to(ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()])
    print("done", len(out), out.groupby("year").size().to_dict())
