"""Monthly parcel energy from the public bulk files (2023-11 to 2025-10)."""
import hashlib, os, re, sys, time, zipfile, json
from pathlib import Path
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, PROC

HUB = "https://www.hub.go.kr"
LIST = HUB + "/portal/opn/lps/idx-lgcpt-pvsn-srvc-list.do"
BULK = Path.home() / "data/gr_bulk/raw"
CAR = {"전기": ("elec", "0801"), "가스": ("gas", "0802")}
KEY = ["sig", "bjd", "plat", "bun", "ji"]
AGG_RULE = os.environ.get("BULK_AGG", "max")
AGG = {"first": lambda v: v[0], "max": max, "sum_distinct": lambda v: sum(sorted(set(v)))}[AGG_RULE]

def ym_ok(ym, a, b): return a <= ym <= b

def listing(s):
    r = s.get(LIST, timeout=30); tok = re.search(r'name="_csrf" value="([^"]+)"', r.text).group(1)
    files = {}
    for page in range(1, 20):
        r = s.post(LIST, data={"_csrf": tok, "pageIndex": page, "opnLgcptTaskSeCd": "08", "srvcNm": "",
                               "startDay": "2015-01-01", "endDay": time.strftime("%Y-%m-%d"), "pageCountPerPage": "40"}, timeout=30)
        names = re.findall(r"지번별 에너지 사용량 \((전기|가스), (\d{4})\.(\d{2})\)", r.text)
        ids = re.findall(r"fnDownloadPop\('08','(\d+)','(OPN\d+)'\)", r.text)
        if not ids: break
        names = names[::2] if len(names) == 2 * len(ids) else names
        new = 0
        for (c, y, m), (op, fid) in zip(names, ids):
            k = (CAR[c][0], int(y + m))
            if k not in files: files[k] = (op, fid); new += 1
        if new == 0: break
    return tok, files

def download(a=202311, b=202510):
    s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0"
    tok, files = listing(s)
    man = ROOT / "manifests" / "sha256_bulk.csv"
    rows = pd.read_csv(man).to_dict("records") if man.exists() else []
    done = {(r["carrier"], int(r["ym"])) for r in rows}
    cars = os.environ.get("BULK_CARRIERS", "elec,gas").split(",")
    todo = sorted(k for k in files if ym_ok(k[1], a, b) and k not in done and k[0] in cars)
    print(f"listing: {len(files)} files; to download {len(todo)} ({a}..{b})", flush=True)
    h = {"X-CSRF-TOKEN": tok, "X-Requested-With": "XMLHttpRequest"}
    for car, ym in todo:
        op, fid = files[(car, ym)]
        d = BULK / f"{car}_{ym}"; d.mkdir(parents=True, exist_ok=True); fn = d / "file.zip"
        s.post(HUB + "/portal/opn/lps/idx-srvc-hstry-insert.do",
               data={"opnLgcptTaskSeCd": "08", "opnTaskCd": op, "opnPrcusePrpsCd": "4", "_csrf": tok}, headers=h, timeout=30)
        with s.post(HUB + "/cmm/fms/fileOpnDown.do", data={"srvrFileNm": fid, "_csrf": tok}, stream=True, timeout=300) as r:
            r.raise_for_status(); cd = r.headers.get("Content-Disposition", "")
            with open(fn, "wb") as f:
                for ch in r.iter_content(1 << 20): f.write(ch)
        if not zipfile.is_zipfile(fn): print("NOT A ZIP:", car, ym, cd); continue
        sha = hashlib.sha256(fn.read_bytes()).hexdigest()
        rows.append(dict(carrier=car, ym=ym, file_id=fid, served_name=cd.split("filename=")[-1], bytes=fn.stat().st_size,
                         sha256=sha, downloaded=time.strftime("%Y-%m-%dT%H:%M:%S")))
        pd.DataFrame(rows).sort_values(["carrier", "ym"]).to_csv(man, index=False)
        print(car, ym, fn.stat().st_size, sha[:12], flush=True); time.sleep(2)

def old_code(sig):
    import s03_energy as s3
    return s3.old_code(sig)

def parcels():
    L = pd.read_csv(PROC / "gr_parcels.csv", dtype=str)
    L["pid"] = L["year"] + "_" + L["seq"]
    L = L.rename(columns={"sigunguCd": "sig", "bjdongCd": "bjd", "platGbCd": "plat"})
    for k in ["bun", "ji"]: L[k] = L[k].str.zfill(4)
    L["reg"] = [k[:5] if isinstance(k, str) and len(k) == 25 and k[:5].isdigit() else None for k in L["bdMgtSn"]]
    return L[L.sig.notna()][["pid"] + KEY + ["reg"]]

def extract():
    L = parcels()
    L["old0"] = [old_code(x)[0] for x in L.sig]
    rg = L[L.reg.notna() & (L.reg != L.sig) & (L.reg != L.old0)]
    cand = [rg.assign(sig=rg.reg, code_used=rg.reg)[["pid"] + KEY + ["code_used"]], L.assign(code_used=L.sig)[["pid"] + KEY + ["code_used"]]]
    L = L[["pid"] + KEY]
    o = L.copy(); o[["old", "cut"]] = [old_code(x) for x in o.sig]; o = o[o.old.notna()]
    cand.append(o.assign(sig=o.old, code_used=o.old)[["pid"] + KEY + ["code_used", "cut"]])
    C = pd.concat(cand, ignore_index=True)
    want = set(map(tuple, C[KEY].values))
    out = {"elec": [], "gas": []}
    cols = ["ym", "type", "sig", "bjd", "sido", "sgg", "dong", "plat", "bun", "ji"]
    for d in sorted(BULK.glob("*_*")):
        car, ym = d.name.split("_"); ym = int(ym); z = zipfile.ZipFile(d / "file.zip")
        agg = {}
        with z.open(z.infolist()[0].filename) as fh:
            for line in fh:
                p = [f.strip() for f in line.decode("utf-8", "replace").rstrip("\r\n").rstrip("|").split("|")]
                k = (p[2], p[3], p[7], p[8], p[9])
                if k in want:
                    try: q = float(p[-1])
                    except ValueError: continue
                    agg.setdefault(k, []).append(q)
        for r in C.itertuples():
            k = (r.sig, r.bjd, r.plat, r.bun, r.ji)
            if k in agg:
                v = agg[k]
                out[car].append(dict(pid=r.pid, carrier=car, ym=ym, status="ok", kwh=AGG(v), n_items=len(v),
                                     n_distinct=len(set(v)), kwh_sum_lines=sum(v), code_used=r.code_used, source="bulk"))
        print(car, ym, "parcels found", len(agg), flush=True)
    for car, rows in out.items():
        if not any(BULK.glob(f"{car}_*")): continue
        E = pd.DataFrame(rows)
        if len(E):
            E = E.sort_values(["pid", "ym"]).drop_duplicates(["pid", "ym"], keep="first")
        months = sorted({int(d.name.split("_")[1]) for d in BULK.glob(f"{car}_*")})
        full = pd.MultiIndex.from_product([sorted(L.pid.unique()), months], names=["pid", "ym"]).to_frame(index=False)
        E = full.merge(E, on=["pid", "ym"], how="left")
        E["carrier"] = car; E["source"] = "bulk"; E["status"] = E.status.fillna("empty")
        E.to_csv(PROC / f"bulk_monthly_{car}.csv", index=False)
        print(car, "rows", len(E), "ok", int((E.status == "ok").sum()), flush=True)

def validate():
    res = {"source": "src/s07_bulk.py validate", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for car in ("elec", "gas"):
        f = PROC / f"bulk_monthly_{car}.csv"
        if not f.exists(): continue
        Bk = pd.read_csv(f, dtype={"pid": str}); A = pd.read_csv(PROC / f"energy_monthly_{car}.csv", dtype={"pid": str})
        J = A.merge(Bk[["pid", "ym", "status", "kwh"]], on=["pid", "ym"], suffixes=("_api", "_bulk"))
        both = J[(J.status_api == "ok") & (J.status_bulk == "ok")]
        eq = np.isclose(both.kwh_api, both.kwh_bulk, rtol=1e-6, atol=1e-6)
        ratio = (both.kwh_bulk / both.kwh_api).replace([np.inf, -np.inf], np.nan)
        res[car] = dict(overlap_pid_months=int(len(J)),
                        status_agreement=pd.crosstab(J.status_api, J.status_bulk).to_dict(),
                        both_ok=int(len(both)), exact_equal=int(eq.sum()),
                        share_exact=float(eq.mean()) if len(both) else None,
                        ratio_quantiles=ratio.quantile([0, .01, .1, .5, .9, .99, 1]).round(4).to_dict() if len(both) else {},
                        months=sorted(map(int, J.ym.unique())))
    (ROOT / "results" / "a_bulk_validation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:3000])

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    if cmd == "download":
        a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (202311, 202510); download(a, b)
    elif cmd == "extract": extract()
    elif cmd == "validate": validate()
