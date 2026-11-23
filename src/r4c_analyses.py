"""Mislinkage tolerance of the main estimator."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC
import s04_analysis as A, r3_analysis as R
from s04_analysis import ym_add, MIN_WIN
B, E = A.load(); G = pd.read_csv(RES / "r3_building_gaps.csv"); head = sorted(set(G.pid)); R.TREATED_HEAD.update(head)
Bi = B.drop_duplicates("pid").set_index("pid")
o, ev, S = R.event_long(B, E, cluster="sigungu", treated=set(head)); comp = set(S[S.treat == 0].pid)
cw = pd.read_csv(RES / "crosswalk_project_parcel_register.csv", dtype=str)
cw["link_grade_analysis"] = cw.pid.map(Bi.link_grade); cw["use_class_analysis"] = cw.pid.map(Bi.use_class); cw["role"] = cw.pid.map(Bi.role)
cw["kept_after_dedup"] = cw.pid.isin(Bi.index).astype(int)
cw["headline_treated"] = cw.pid.isin(set(head)).astype(int); cw["headline_comparison"] = cw.pid.isin(comp).astype(int)
cw.to_csv(RES / "crosswalk_project_parcel_register.csv", index=False)
print("crosswalk", int(cw.headline_treated.sum()), int(cw.headline_comparison.sum()), cw.link_grade_analysis.value_counts().to_dict(), flush=True)
Ec = A.carrier_frame(E, "elec"); T = Bi.loc[head]
C = Bi[Bi.role.isin(("control", "aux_2023")) & (Bi.use_class == "보건") & (Bi.link_grade == "A_cand")]
don = Ec[Ec.pid.isin(C.index)]; don = don[don.ym < don.start_ym.map(lambda v: ym_add(int(v), -3))]
dmon = {p: set(d.ym) for p, d in don.groupby("pid")}; drow = {p: d for p, d in don.groupby("pid")}
elig = {}
for p, r in T.iterrows():
    s_, e_ = int(r.start_ym), int(r.end_ym); pre = {ym_add(s_, -k) for k in range(1, 13)}; post = {ym_add(e_, k) for k in range(1, 13)}
    elig[p] = [q for q, m in dmon.items() if len(m & pre) >= MIN_WIN and len(m & post) >= MIN_WIN]
E_other = E[~((E.carrier == "elec") & E.pid.isin(head))]; E_t = {p: d for p, d in E[(E.carrier == "elec") & E.pid.isin(head)].groupby("pid")}
rng = np.random.default_rng(20261119); rows = []; n = len(head)
for pi in (0.3, 0.4, 0.5, 0.6):
    k = int(round(pi * n))
    for rep in range(20):
        sel = set(rng.choice(head, k, replace=False)); parts = [E_other] + [E_t[p] for p in head if p not in sel]
        for p in sel:
            if not elig[p]: parts.append(E_t[p]); continue
            d = drow[elig[p][rng.integers(len(elig[p]))]].copy(); d["pid"] = p; d["role"] = "treated"
            d["start_ym"] = T.start_ym[p]; d["end_ym"] = T.end_ym[p]; d["phase"] = np.select([d.ym < d.start_ym, d.ym > d.end_ym], ["pre", "post"], "constr")
            parts.append(d)
        o, ev, S = R.event_long(B, pd.concat(parts, ignore_index=True), cluster="sigungu", treated=set(head))
        q = o["post_1_12"]; rows.append(dict(pi=pi, rep=rep, k=k, pct=q["pct"], lo=q["pct_ci"][0], hi=q["pct_ci"][1], includes_zero=int(q["pct_ci"][1] >= 0)))
        pd.DataFrame(rows).to_csv(RES / "r4c_mislink_es.csv", index=False)
    print(pi, flush=True)
D = pd.DataFrame(rows)
Sm = D.groupby("pi").agg(k=("k", "first"), draws=("pct", "size"), median_pct=("pct", "median"), median_hi=("hi", "median"), share_includes_zero=("includes_zero", "mean")).reset_index()
Sm.to_csv(RES / "r4c_mislink_es_summary.csv", index=False); print(Sm.round(3).to_string())
