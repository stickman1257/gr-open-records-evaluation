"""Analysed vs non-analysed projects selected in 2021-22 (Table 1)."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, sido_of
import a_program_descriptives as PD
import s04_analysis as A

USES = [("보건", "health_centre"), ("어린이집", "daycare")]
MEAS9 = ["wall", "roof", "cool_roof", "win", "shade", "vent_generic", "led", "hvac", "boiler", "renewable"]
COLS = ["use", "group", "variable", "kind", "n_all", "n_analysed", "n_not_analysed", "all", "analysed", "not_analysed", "median_all", "median_analysed", "smd_analysed_vs_not"]


def smd(a, b, kind):
    a, b = a.dropna(), b.dropna()
    if not len(a) or not len(b): return np.nan
    va = a.mean() * (1 - a.mean()) if kind == "share" else a.var(ddof=1); vb = b.mean() * (1 - b.mean()) if kind == "share" else b.var(ddof=1)
    den = np.sqrt((va + vb) / 2)
    return float((a.mean() - b.mean()) / den) if den and den > 0 else (0.0 if a.mean() == b.mean() else np.nan)


def main():
    g, src = PD.load_projects(); d = PD.build(g)
    d = d[d.year.isin([2021, 2022])].copy(); d["pid"] = d.year.astype(str) + "_" + d.seq.astype(str)
    d["sido"] = d.address_list.apply(sido_of) if "address_list" in d else None
    B, E = A.load(); S = A.imputation_setup(B, E, "elec", **A.PRIMARY_IMP)
    an = set()
    if S is not None:
        f = S["fit"](); an = set(S["pids"][pd.unique(S["pi"][S["post"] & f["ok"]])])
    rows = []
    for u, en in USES:
        D = d[d.use_class == u]; a_ = D[D.pid.isin(an)]; n_ = D[~D.pid.isin(an)]
        def add(group, var, kind, col):
            x, xa, xn = pd.to_numeric(D[col], errors="coerce"), pd.to_numeric(a_[col], errors="coerce"), pd.to_numeric(n_[col], errors="coerce")
            rows.append(dict(use=en, group=group, variable=var, kind=kind, n_all=int(x.notna().sum()), n_analysed=int(xa.notna().sum()), n_not_analysed=int(xn.notna().sum()),
                             all=x.mean(), analysed=xa.mean(), not_analysed=xn.mean(), median_all=x.median() if kind == "cont" else np.nan,
                             median_analysed=xa.median() if kind == "cont" else np.nan, smd_analysed_vs_not=smd(xa, xn, kind)))
        rows.append(dict(use=en, group="N", variable="Projects selected 2021-22 with a detail record", kind="count", n_all=len(D), n_analysed=len(a_), n_not_analysed=len(n_),
                         analysed=len(a_) / len(D) if len(D) else np.nan))
        add("Size, cost, age", "Floor area (m2)", "cont", "gfa_m2"); add("Size, cost, age", "Cost per m2 (KRW)", "cont", "cost_per_m2")
        add("Size, cost, age", "Construction year", "cont", "built_year_num")
        top = list(D.sido.value_counts().index[:6])
        for s_ in top:
            D = D.assign(_r=(D.sido == s_).astype(float)); a_ = D[D.pid.isin(an)]; n_ = D[~D.pid.isin(an)]
            add("Region (share)", PD_SIDO.get(s_, str(s_)), "share", "_r")
        D = D.assign(_r=(~D.sido.isin(top)).astype(float)); a_ = D[D.pid.isin(an)]; n_ = D[~D.pid.isin(an)]
        add("Region (share)", "Other provinces", "share", "_r")
        for m in MEAS9: add("Measures (share)", PD.MEAS_LABEL.get(m, m), "share", m)
        D = D.assign(_r=D.pred_saving_valid.notna().astype(float)); a_ = D[D.pid.isin(an)]; n_ = D[~D.pid.isin(an)]
        add("Records", "Predicted saving published", "share", "_r")
        D = D.assign(_r=(D.year == 2021).astype(float)); a_ = D[D.pid.isin(an)]; n_ = D[~D.pid.isin(an)]
        add("Records", "Selected in 2021", "share", "_r")
    hc = d[d.use_class == "보건"]
    rows.append(dict(use="all_uses", group="N", variable="All projects selected 2021-22 with a detail record; analysed = health centres in the headline sample",
                     kind="count", n_all=len(d), n_analysed=int(hc.pid.isin(an).sum()), n_not_analysed=len(d) - int(hc.pid.isin(an).sum()),
                     analysed=float(hc.pid.isin(an).sum() / len(d)) if len(d) else np.nan))
    rows.append(dict(use="all_uses", group="N", variable="All projects selected 2021-22 with a detail record; analysed = health centres and daycares",
                     kind="count", n_all=len(d), n_analysed=int(d.pid.isin(an).sum()), n_not_analysed=len(d) - int(d.pid.isin(an).sum()),
                     analysed=float(d.pid.isin(an).sum() / len(d)) if len(d) else np.nan))
    R = pd.DataFrame(rows).reindex(columns=COLS); R.to_csv(RES / "a_analysed_vs_all.csv", index=False)
    pd.set_option("display.width", 230)
    print(f"source: {src}; generated {time.strftime('%Y-%m-%d %H:%M')}; analysed buildings (both uses): {len(an)}")
    print(R.round(3).to_string(index=False))

PD_SIDO = {"서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon", "광주": "Gwangju", "대전": "Daejeon", "울산": "Ulsan", "세종": "Sejong",
           "경기": "Gyeonggi", "강원": "Gangwon", "충북": "Chungbuk", "충남": "Chungnam", "전북": "Jeonbuk", "전남": "Jeonnam", "경북": "Gyeongbuk",
           "경남": "Gyeongnam", "제주": "Jeju"}

if __name__ == "__main__":
    main()
