"""Step 3: monthly electricity and gas per parcel from the building-energy service."""
import json, os, sys, time, glob, re, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, PROC, RAW
import s02_link

OPS = {"elec": "getBeElctyUsgInfo", "gas": "getBeGasUsgInfo"}
URL = "https://apis.data.go.kr/1613000/BldEngyHubService/"
LATEST = int(os.environ.get("ENERGY_LATEST", "202510"))
EARLIEST = 202001
OUT = RAW / "energy"; OUT.mkdir(parents=True, exist_ok=True)

def old_code(sig):
    if sig[:2] == "51": return "42" + sig[2:], 202305
    if sig[:2] == "52": return "45" + sig[2:], 202401
    if sig in ("41192", "41194", "41196"): return "41190", 202401
    if sig == "27720": return "47720", 202307
    return None, None

def ym_add(ym, k):
    y, m = divmod(int(ym), 100); t = y * 12 + (m - 1) + k; return (t // 12) * 100 + t % 12 + 1

def ym_range(a, b):
    out = []; x = a
    while x <= b: out.append(x); x = ym_add(x, 1)
    return out

def parse_period(s):
    """'2021-08-23 ~ 2021-12-04' | '20211108 ~ 20220616' | '202410 ~ 202512' -> (start_ym, end_ym)."""
    parts = re.split(r"~", str(s))
    if len(parts) != 2: return None, None
    out = []
    for t in parts:
        m = re.search(r"(20\d{2})[-./]?\s*(\d{1,2})(?:[-./]?\s*(\d{1,2}))?", t.strip())
        if not m: return None, None
        y, mo = int(m.group(1)), int(m.group(2))
        if not (2019 <= y <= 2027 and 1 <= mo <= 12): return None, None
        out.append(y * 100 + mo)
    a, b = out
    dur = (b // 100 * 12 + b % 100) - (a // 100 * 12 + a % 100)
    return (a, b) if 0 <= dur <= 36 else (None, None)

OFFLINE = threading.Event()

class Store:
    def __init__(self, tag=""):
        self.done = {}
        for f in sorted(glob.glob(str(OUT / "calls_*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                self.done[(r["op"], r["sig"], r["bjd"], r["plat"], r["bun"], r["ji"], r["ym"])] = r
        self.fh = open(OUT / f"calls_{time.strftime('%Y%m%d')}{tag}.jsonl", "a", encoding="utf-8")
        self.n_new = 0; self.lock = threading.Lock()
    def get(self, *k): return self.done.get(k)
    def put(self, r):
        with self.lock:
            self.done[(r["op"], r["sig"], r["bjd"], r["plat"], r["bun"], r["ji"], r["ym"])] = r
            self.fh.write(json.dumps(r, ensure_ascii=False) + "\n"); self.fh.flush(); self.n_new += 1

class QuotaExceeded(Exception): pass

class RateLimiter:
    def __init__(self, per_sec):
        self.gap = 1.0 / per_sec; self.lock = threading.Lock(); self.next = time.time()
    def wait(self):
        with self.lock:
            now = time.time(); t = max(now, self.next); self.next = t + self.gap
        time.sleep(max(0.0, t - now))
RL = RateLimiter(float(os.environ.get("ENERGY_RPS", "5")))

def call(store, op, sig, bjd, plat, bun, ji, ym):
    hit = store.get(op, sig, bjd, plat, bun, ji, ym)
    if hit is not None and hit.get("status") in ("ok", "empty"): return hit
    if OFFLINE.is_set(): return dict(status="missing")
    for attempt in range(8):
        try:
            RL.wait()
            r = requests.get(URL + op, timeout=40, params=dict(serviceKey=os.environ["DATA_GO_KR_KEY"], sigunguCd=sig,
                             bjdongCd=bjd, platGbCd=plat, bun=bun, ji=ji, useYm=ym, numOfRows=100, pageNo=1, _type="json"))
            txt = r.text
            if "PER_SECOND" in txt or '"returnReasonCode": "23"' in txt:
                time.sleep(1.0 + attempt); continue
            if "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS" in txt or '"resultCode":"22"' in txt: raise QuotaExceeded(txt[:200])
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1)); continue
            d = r.json(); hdr = d.get("response", {}).get("header", d.get("header", {}))
            code = hdr.get("resultCode")
            if code == "22": raise QuotaExceeded(str(hdr))
            if code != "00": time.sleep(2); continue
            body = d["response"].get("body", {})
            it = (body.get("items") or {}).get("item", []) if body.get("items") else []
            it = it if isinstance(it, list) else [it]
            rec = dict(op=op, sig=sig, bjd=bjd, plat=plat, bun=bun, ji=ji, ym=ym, status="ok" if it else "empty",
                       n_items=len(it), qty=[i.get("useQty") for i in it],
                       addr=[i.get("newPlatPlc") for i in it][:3], t=time.strftime("%Y-%m-%dT%H:%M:%S"))
            store.put(rec); return rec
        except QuotaExceeded: raise
        except Exception:
            time.sleep(3 * (attempt + 1))
    rec = dict(op=op, sig=sig, bjd=bjd, plat=plat, bun=bun, ji=ji, ym=ym, status="error", t=time.strftime("%Y-%m-%dT%H:%M:%S"))
    store.put(rec); return rec

def reg_code(p):
    """District code held by the building-register key (first 5 digits of bdMgtSn), or None."""
    k = p.get("bdMgtSn") if hasattr(p, "get") else None
    k = str(k) if k is not None else ""
    return k[:5] if len(k) == 25 and k[:5].isdigit() else None

def lookup_codes(p, ym, d24=True):
    """Ordered district codes to query for parcel p in month ym. d24=False gives the v1 lookup (before 2026-11-17)."""
    sig = p["sigunguCd"]; old, cut = old_code(sig)
    order = [sig] if old is None else ([old, sig] if ym < cut else [sig, old])
    rc = reg_code(p) if d24 else None
    if rc and rc != sig and rc not in order: order = [rc] + order
    return order

def fetch_month(store, carrier, p, ym):
    """Returns (status, kwh, n_items, code_used). Register-key code first when it differs from the address code (D24);"""
    op = OPS[carrier]; order = lookup_codes(p, ym)
    for c in order:
        r = call(store, op, c, p["bjdongCd"], p["platGbCd"], p["bun"], p["ji"], ym)
        if r["status"] == "ok":
            q = [float(x) for x in r["qty"] if x is not None]
            return "ok", sum(q), r["n_items"], c
    return r["status"], np.nan, 0, None

def plan(parcels):
    """parcels: rows with role, start_ym, end_ym. Returns dict pass -> list of (pid, ym)."""
    P = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    full = []
    for p in parcels.itertuples():
        if p.role == "treated":
            s, e = int(p.start_ym), int(p.end_ym)
            P[0] += [(p.pid, ym_add(s, -7)), (p.pid, ym_add(e, 6))]
            P[1] += [(p.pid, ym) for ym in ym_range(ym_add(s, -12), ym_add(s, -1)) + ym_range(ym_add(e, 1), ym_add(e, 12))]
            P[2] += [(p.pid, ym) for ym in ym_range(ym_add(s, -24), ym_add(s, -13))]
            P[3] += [(p.pid, ym) for ym in ym_range(ym_add(e, 13), ym_add(e, 24))]
            cm = ym_range(s, e)
            if len(cm) > 3:
                inner = cm[1:-1] if len(cm) >= 5 else cm
                cm = [inner[i] for i in sorted(set(np.linspace(0, len(inner) - 1, 3).round().astype(int)))]
            P[4] += [(p.pid, ym) for ym in cm]
        elif p.role == "control":
            last = min(LATEST, ym_add(int(p.start_ym), -1))
            P[0] += [(p.pid, 202101), (p.pid, 202301)]
            P[1] += [(p.pid, ym) for ym in ym_range(202007, last)]
            P[2] += [(p.pid, ym) for ym in ym_range(202001, 202006)]
        elif p.role == "aux_2023":
            last = min(LATEST, ym_add(int(p.start_ym), -1))
            P[0] += [(p.pid, 202101), (p.pid, 202201)]
            P[5] += [(p.pid, ym) for ym in ym_range(202007, last)]
        if p.use_class in ("보건", "어린이집"):
            if p.role == "treated":
                s, e = int(p.start_ym), int(p.end_ym)
                full += [(p.pid, ym) for ym in ym_range(EARLIEST, ym_add(s, -1)) + ym_range(ym_add(e, 1), LATEST)]
            elif p.role in ("control", "aux_2023"):
                full += [(p.pid, ym) for ym in ym_range(EARLIEST, min(LATEST, ym_add(int(p.start_ym), -1)))]
    for k in P: P[k] = [(i, ym) for i, ym in P[k] if EARLIEST <= ym <= LATEST]
    planned = {x for k in range(6) for x in P[k]}
    P[6] = [x for x in dict.fromkeys(full) if EARLIEST <= x[1] <= LATEST and x not in planned]
    return P

def build_parcels():
    g = pd.read_csv(PROC / "gr_projects_raw.csv") if (PROC / "gr_projects_raw.csv").exists() else None
    L = pd.read_csv(PROC / "gr_parcels.csv", dtype={"sigunguCd": str, "bjdongCd": str, "platGbCd": str, "bun": str, "ji": str})
    if g is None:
        import s01_gr_sites as s1; g = s1.table_from_cache()
    d = L.merge(g[["year", "seq", "period_txt", "measures_txt", "pred_saving_txt", "name_list", "built_year", "cost_txt", "floors_txt", "agency", "project_type"]], on=["year", "seq"], how="left")
    se = d.period_txt.apply(parse_period); d["start_ym"] = [x[0] for x in se]; d["end_ym"] = [x[1] for x in se]
    d = d[(d.match_status == "MATCHED") & d.link_grade.isin(["A_cand", "B"])].copy()
    d["role"] = np.select([d.year <= 2022, d.year == 2023], ["treated", "aux_2023"], "control")
    d.loc[(d.role == "treated") & (d.end_ym > ym_add(LATEST, -12)), "role"] = "treated_short"
    d.loc[(d.role == "control") & (d.start_ym < 202407), "role"] = "control_early"
    d["pid"] = d.year.astype(str) + "_" + d.seq.astype(str)
    d = d[d.role.isin(["treated", "control", "aux_2023"]) & d.start_ym.notna()].copy()
    import s02_link as L2
    d["use_class"] = d.name_list.apply(L2.use_class)
    d = d[(d.role == "treated") | d.use_class.isin(["보건", "어린이집"])]
    return d

def run(passes=(0, 1, 2, 3), carriers=("elec", "gas"), max_new=None, offline=False):
    if offline: OFFLINE.set()
    d = build_parcels(); store = Store("" if len(carriers) > 1 else "_" + carriers[0])
    d["prio"] = ((d.link_grade != "A_cand").astype(int) * 10 + d.role.map({"treated": 0, "control": 1, "aux_2023": 2})
                 + ((d.role == "control") & d.use_class.isin(["경로당", "기타"])).astype(int))
    d = d.sort_values(["prio", "year", "seq"])
    P = plan(d); byid = d.set_index("pid").to_dict("index")
    sub = os.environ.get("ENERGY_PIDS_FILE")
    if sub:
        keep = set(pd.read_csv(sub).pid.astype(str))
        P = {k: [(i, ym) for i, ym in v if str(i) in keep] for k, v in P.items()}
        print(f"subsample mode: {len(keep)} buildings from {sub}; outputs not written", flush=True)
    bulk = {c: pd.read_csv(f, dtype={"pid": str}) for c in carriers if (f := PROC / f"bulk_monthly_{c}.csv").exists()}
    bulk_months = {c: set(b.ym.astype(int)) for c, b in bulk.items()}
    if bulk: print("bulk months skipped:", {c: (min(m), max(m), len(m)) for c, m in bulk_months.items()}, flush=True)
    avail = {}
    rows = []
    stop = threading.Event(); rlock = threading.Lock()
    def work(item):
        ps, carrier, pid, ym = item
        try:
            st, kwh, n, code = fetch_month(store, carrier, byid[pid], ym)
        except QuotaExceeded as e:
            if not stop.is_set(): print("STOP (quota; continuing cache-only):", str(e)[:160], flush=True)
            stop.set(); OFFLINE.set()
            st, kwh, n, code = fetch_month(store, carrier, byid[pid], ym)
        if st == "missing": return
        with rlock:
            if ps == 0: avail[(pid, carrier)] = avail.get((pid, carrier), False) or st == "ok"
            rows.append(dict(pid=pid, carrier=carrier, ym=ym, status=st, kwh=kwh, n_items=n, code_used=code, pass_=ps))
    for ps in passes:
        for carrier in carriers:
            if ps == 4 and carrier != "elec": continue
            items = [(ps, carrier, pid, ym) for pid, ym in P[ps] if not (ps > 0 and avail.get((pid, carrier)) is False)
                     and not (ps > 0 and ym in bulk_months.get(carrier, ()))]
            with ThreadPoolExecutor(int(os.environ.get("ENERGY_WORKERS", "4"))) as ex: list(ex.map(work, items))
            print(f"pass {ps} {carrier} done; new calls so far {store.n_new}{' (cache-only)' if OFFLINE.is_set() else ''}", flush=True)
    E = pd.DataFrame(rows, columns=["pid", "carrier", "ym", "status", "kwh", "n_items", "code_used", "pass_"])
    if sub:
        print(E.groupby(["carrier", "pass_", "status"]).size().to_dict(), "subsample", "quota_stop" if stop.is_set() else "complete")
        return E
    E = E.drop_duplicates(["pid", "carrier", "ym"], keep="last")
    for c, b in bulk.items():
        b = b[["pid", "carrier", "ym", "status", "kwh", "n_items", "code_used"]].assign(pass_="bulk")
        have = set(zip(E[E.carrier == c].pid, E[E.carrier == c].ym))
        E = pd.concat([E, b[[(p_, y_) not in have for p_, y_ in zip(b.pid, b.ym)]]], ignore_index=True)
    for c in carriers:
        E[E.carrier == c].to_csv(PROC / f"energy_monthly_{c}.csv", index=False)
    parts = [pd.read_csv(f) for c in OPS if (f := PROC / f"energy_monthly_{c}.csv").exists()]
    pd.concat(parts, ignore_index=True).to_csv(PROC / "energy_monthly.csv", index=False)
    d.to_csv(PROC / "analysis_buildings.csv", index=False)
    print(E.groupby(["carrier", "status"]).size().to_dict(), "offline rebuild" if offline else ("quota_stop" if stop.is_set() else "complete"))
    return E

def d24_affected():
    """D24: parcels of the collection frame whose lookup codes change under the register-code rule (no network)."""
    import s02_link as L2
    PROV = {"11": "Seoul", "26": "Busan", "27": "Daegu", "28": "Incheon", "29": "Gwangju", "30": "Daejeon", "31": "Ulsan", "36": "Sejong", "41": "Gyeonggi",
            "42": "Gangwon", "43": "Chungbuk", "44": "Chungnam", "45": "Jeonbuk", "46": "Jeonnam", "47": "Gyeongbuk", "48": "Gyeongnam", "50": "Jeju"}
    d = build_parcels(); rows = []
    for r in d.to_dict("records"):
        v1, v2 = lookup_codes(r, 202201, d24=False), lookup_codes(r, 202201)
        if v1 != v2:
            rows.append(dict(pid=r["pid"], year=r["year"], seq=r["seq"], role=r["role"], use_class=r["use_class"], link_grade=r["link_grade"],
                             province=PROV.get(reg_code(r)[:2], reg_code(r)[:2]), register_prefix=reg_code(r)[:2], siNm=r.get("siNm"), sggNm=r.get("sggNm"),
                             code_address=r["sigunguCd"], code_register=reg_code(r), codes_v1="|".join(v1), codes_v2="|".join(v2)))
    A = pd.DataFrame(rows); res = ROOT / "results"
    A.to_csv(res / "d24_affected_parcels.csv", index=False)
    C = A.groupby(["province", "register_prefix", "role"]).size().unstack(fill_value=0); C["all"] = C.sum(axis=1)
    C.loc[("All", ""), :] = C.sum(); C.astype(int).to_csv(res / "d24_affected_counts.csv")
    print(len(A), "affected parcels of", len(d)); print(C.astype(int).to_string())
    return A

if __name__ == "__main__":
    if "--d24-affected" in sys.argv: d24_affected(); sys.exit(0)
    ps = tuple(int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "2", "3"]))
    cs = tuple(os.environ.get("ENERGY_CARRIERS", "elec,gas").split(","))
    run(ps, carriers=cs, offline="--offline" in sys.argv)
