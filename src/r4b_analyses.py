"""Removal of 2025 comparison records; trend tests across cohorts."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A, r3_analysis as R
B, E = A.load(); G = pd.read_csv(RES / "r3_building_gaps.csv"); head = set(G.pid); Bi = B.drop_duplicates("pid").set_index("pid")
ctrl = Bi[Bi.role.isin(("control", "aux_2023")) & (Bi.use_class == "보건") & (Bi.link_grade == "A_cand")]; c25 = sorted(ctrl[ctrl.year == 2025].index); out = {"n_2025": len(c25)}
rng = np.random.default_rng(20261119); rows = []
for k in (18, 36, 54, 72):
    draws = 1 if k >= len(c25) else 5
    for d in range(draws):
        drop = set(c25) if k >= len(c25) else set(rng.choice(c25, k, replace=False))
        o, ev, S = R.event_long(B, E, cluster="sigungu", treated=head, drop_ctrl=drop)
        rows.append(dict(k=k, draw=d, pct=o["post_1_12"]["pct"], lo=o["post_1_12"]["pct_ci"][0], hi=o["post_1_12"]["pct_ci"][1], support=o["support_share"], n_control=o["n_control"]))
        print(rows[-1], flush=True)
out["drop_2025"] = rows
o, ev, S = R.event_long(B, E, cluster="sigungu", treated=head, drop_ctrl=set(ctrl[ctrl.year != 2024].index))
out["controls_2024_only"] = dict(post_1_12=o["post_1_12"], support=o["support_share"], n_control=o["n_control"], pretrend_p=o["pretrend"]["p_F"],
                                 treated_per_post_bin={r.term: int(r.n_treated_bld) for r in ev[ev.a >= 1].itertuples()})
X = pd.read_csv(RES / "x08_ep_ashrae.csv"); X = X[(X.rep < 100) & (X.use == "보건")].copy(); X["pass"] = X.cvrmse <= 15
rb = np.random.default_rng(7); pids = X.pid.unique(); res = {}
for g, d in X.groupby("pass"):
    med = float(np.median(d.gap_log)); bs = []
    byp = dict(tuple(d.groupby("pid")))
    for _ in range(1000):
        s = rb.choice(list(byp), len(byp), replace=True); bs.append(np.median(np.concatenate([byp[p].gap_log.to_numpy() for p in s])))
    res["pass" if g else "fail"] = dict(median_pct=float(np.exp(med) - 1), ci=[float(np.exp(np.percentile(bs, 2.5)) - 1), float(np.exp(np.percentile(bs, 97.5)) - 1)],
                                        mean_pct=float(np.exp(d.gap_log.mean()) - 1), n_fits=int(len(d)), n_bld=int(d.pid.nunique()))
pb = X.groupby("pid")["pass"].mean(); res["share_fits_pass"] = float(X["pass"].mean()); res["share_bld_pass_majority"] = float((pb >= 0.5).mean()); res["n_bld"] = int(pb.size)
out["ashrae_health"] = res; print(res, flush=True)
(RES / "r4b_analyses.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
