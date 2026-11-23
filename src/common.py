"""Shared paths, weather and degree-day helpers."""
import json, pathlib, time, hashlib
import numpy as np, pandas as pd, requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC, RES = ROOT / "data/raw", ROOT / "data/processed", ROOT / "results"

SIDO_POINTS = {
    "서울": (37.5665, 126.9780), "부산": (35.1796, 129.0756), "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052), "광주": (35.1595, 126.8526), "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114), "세종": (36.4800, 127.2890), "경기": (37.2636, 127.0286),
    "강원": (37.8813, 127.7298), "충북": (36.6424, 127.4890), "충남": (36.6588, 126.6728),
    "전북": (35.8242, 127.1480), "전남": (34.9904, 126.4817), "경북": (36.5684, 128.7294),
    "경남": (35.2280, 128.6811), "제주": (33.4996, 126.5312),
}
SIDO_ALIASES = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
                "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
                "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
                "전라북도": "전북", "전북특별자치도": "전북", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
                "제주특별자치도": "제주", "제주도": "제주"}

def sido_of(addr: str):
    if not isinstance(addr, str): return None
    tok = addr.strip().split()[0] if addr.strip() else ""
    if tok in SIDO_POINTS: return tok
    return SIDO_ALIASES.get(tok)

def fetch_weather(start="2017-12-01", end="2025-10-31"):
    out = RAW / "weather"; out.mkdir(parents=True, exist_ok=True)
    frames = []
    for s, (lat, lon) in SIDO_POINTS.items():
        p = out / f"openmeteo_{s}_{start}_{end}.json"
        if not p.exists():
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params=dict(
                latitude=lat, longitude=lon, start_date=start, end_date=end,
                daily="temperature_2m_mean", timezone="Asia/Seoul"))
            r.raise_for_status(); p.write_bytes(r.content); time.sleep(1.0)
        d = json.loads(p.read_text())
        frames.append(pd.DataFrame({"sido": s, "date": pd.to_datetime(d["daily"]["time"]),
                                    "tmean": d["daily"]["temperature_2m_mean"]}))
    return pd.concat(frames, ignore_index=True)

H_BASES = [12, 14, 16, 18, 20]
C_BASES = [18, 20, 22, 24, 26]

def monthly_degree_days(daily: pd.DataFrame, key="sido"):
    d = daily.copy(); d["ym"] = d.date.dt.year * 100 + d.date.dt.month
    cols = {}
    for b in H_BASES: d[f"HDD{b}"] = np.clip(b - d.tmean, 0, None); cols[f"HDD{b}"] = "sum"
    for b in C_BASES: d[f"CDD{b}"] = np.clip(d.tmean - b, 0, None); cols[f"CDD{b}"] = "sum"
    cols.update(tmean="mean", date="count")
    m = d.groupby([key, "ym"]).agg(cols).rename(columns={"date": "days"}).reset_index()
    return m

def fit_dd_model(df: pd.DataFrame, y="kwh"):
    """df rows = months with columns days, HDDb, CDDb. Model on per-day rate:"""
    d = df.dropna(subset=[y]).copy()
    n = len(d)
    if n < 9: return None
    rate = (d[y] / d.days).values; w = d.days.values
    best = None
    cands = [("mean", None, None)] + [("H", hb, None) for hb in H_BASES] + [("C", None, cb) for cb in C_BASES] + \
            [("HC", hb, cb) for hb in H_BASES for cb in C_BASES if cb >= hb]
    for form, hb, cb in cands:
        X = [np.ones(n)]
        if hb is not None: X.append(d[f"HDD{hb}"].values / d.days.values)
        if cb is not None: X.append(d[f"CDD{cb}"].values / d.days.values)
        X = np.column_stack(X)
        W = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * W[:, None], rate * W, rcond=None)
        if (coef[1:] < 0).any(): continue
        pred = X @ coef
        resid = (rate - pred) * w
        sse = (resid ** 2).sum(); k = X.shape[1]
        bic = n * np.log(sse / n + 1e-12) + k * np.log(n)
        if best is None or bic < best["bic"]:
            ym = d[y].values
            cv = np.sqrt(sse / max(n - k, 1)) / ym.mean() * 100
            nmbe = resid.sum() / ((n - k) * ym.mean()) * 100
            r2 = 1 - sse / (((ym - ym.mean()) ** 2).sum() + 1e-12)
            best = dict(form=form, hb=hb, cb=cb, coef=coef.tolist(), bic=bic, cvrmse=cv, nmbe=nmbe, r2=r2, n=n)
    return best

def predict_dd(model, df):
    X = [np.ones(len(df))]
    if model["hb"] is not None: X.append(df[f"HDD{model['hb']}"].values / df.days.values)
    if model["cb"] is not None: X.append(df[f"CDD{model['cb']}"].values / df.days.values)
    X = np.column_stack(X)
    return X @ np.array(model["coef"]) * df.days.values

def sha256(p: pathlib.Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()

COVID_MONTHS = set(range(202003, 202013)) | {202101, 202102, 202107, 202108, 202109, 202110, 202201, 202202, 202203}
