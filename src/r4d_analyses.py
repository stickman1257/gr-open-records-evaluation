"""Estimate without the calendar windows of largest cohort drift."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A, r3_analysis as R
from s03_energy import ym_add
B, E = A.load(); G = pd.read_csv(RES / "r3_building_gaps.csv"); head = set(G.pid); R.TREATED_HEAD.update(head); Bi = B.drop_duplicates("pid").set_index("pid")
out = {"label": "post hoc"}
win = set(Bi.loc[list(head)][(Bi.loc[list(head)].start_ym >= 202207) & (Bi.loc[list(head)].start_ym <= 202306)].index)
o, ev, S = R.event_long(B, E, cluster="sigungu", treated=head - win)
out["without_start_2022H2_2023H1"] = dict(n_dropped=len(win), n_treated=o["n_treated"], post_1_12=o["post_1_12"], pretrend_p=o["pretrend"]["p_F"])
print("1", out["without_start_2022H2_2023H1"], flush=True)
o0, ev0, _ = R.event_long(B, E, cluster="sigungu", treated=head)
x = o0["_x"]; b = o0["_b"]; V = o0["_V"]; leads = [i for i, t in enumerate(x) if int(t.split("_")[2]) <= -7]
t = np.array([(int(x[i].split("_")[1]) + 1 + 5) / 3.0 for i in leads])
Xq = np.column_stack([t, t ** 2]); Vl = V[np.ix_(leads, leads)]; Vi = np.linalg.inv(Vl)
cov = np.linalg.inv(Xq.T @ Vi @ Xq); beta = cov @ Xq.T @ Vi @ b[leads]
out["gls_quadratic_leads"] = dict(slope=float(beta[0]), curvature=float(beta[1]), curvature_se=float(np.sqrt(cov[1, 1])),
                                  curvature_ci=[float(beta[1] - 1.96 * np.sqrt(cov[1, 1])), float(beta[1] + 1.96 * np.sqrt(cov[1, 1]))],
                                  second_difference_equivalent=float(2 * beta[1]), note="second difference of a quadratic = 2 x curvature")
print("2", out["gls_quadratic_leads"], flush=True)
g = G.set_index("pid").pct
for q in (0.05, 0.116):
    lo, hi = g.quantile(q), g.quantile(1 - q); keep = set(g[(g > lo) & (g < hi)].index)
    o, ev, S = R.event_long(B, E, cluster="sigungu", treated=keep)
    out[f"trim_{q}"] = dict(n_treated=o["n_treated"], cut_lo=float(lo), cut_hi=float(hi), post_1_12=o["post_1_12"]); print("3", q, o["post_1_12"]["pct"], flush=True)
out["building_gap_median_pct"] = float(g.median())
Cc = A.carrier_frame(E, "elec"); ids = set(Bi[Bi.role.isin(("control", "aux_2023")) & (Bi.use_class == "보건") & (Bi.link_grade == "A_cand")].index)
Cc = Cc[Cc.pid.isin(ids)]; Cc = Cc[Cc.ym < Cc.start_ym.map(lambda v: ym_add(int(v), -3))].sort_values(["pid", "ym"]); r = []
for pid, d in Cc.groupby("pid"):
    d = d.reset_index(drop=True)
    for i in range(12, len(d)):
        if 202201 <= int(d.ym[i]) <= 202510: r.append(d.kwh[i] < 0.25 * d.kwh[i - 12:i].median())
out["comparison_base_rate_months_below_25pct"] = dict(share=float(np.mean(r)), n_months=len(r)); print("4", out["comparison_base_rate_months_below_25pct"], flush=True)
(RES / "r4d_analyses.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
