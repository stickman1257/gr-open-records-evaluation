"""Additional health-post checks: stack weights, 2024 cohort, distribution, 36 buildings."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
import r3_analysis as R
import pyfixest as pf
from s03_energy import ym_add
warnings.filterwarnings("ignore")
pct = lambda a: float(np.exp(a) - 1)
out = {"label": "post hoc (Energy Policy review R4, referee round 1)"}
B, E = A.load(); G = pd.read_csv(RES / "r3_building_gaps.csv"); head = set(G.pid); R.TREATED_HEAD.update(head)
Bi = B.drop_duplicates("pid").set_index("pid")

def post_lin(b, V, x, a_lo, a_hi, crit):
    ix = [i for i, t in enumerate(x) if a_lo <= int(t.split("_")[1]) <= a_hi]
    l = np.zeros(len(x)); l[ix] = 1 / len(ix); th = float(l @ b); se = float(np.sqrt(l @ V @ l))
    return dict(pct=pct(th), pct_ci=[pct(th - crit * se), pct(th + crit * se)], theta_log=th, se_log=se, bins=[x[i] for i in ix])

S, x, support = R.long_stack(B, E, treated=head)
cnt = S.groupby(["stack", "treat"]).pid.nunique().unstack(fill_value=0); NT, NC = cnt[1].sum(), cnt[0].sum()
w_c = (cnt[1] / NT) / (cnt[0] / NC)
S["w"] = np.where(S.treat == 1, 1.0, S["stack"].map(w_c).to_numpy())
cl = R.cl_of(S, B, "sigungu")
d = pd.DataFrame({"y": S.y.to_numpy(float), "unit_fe": S.unit_fe.to_numpy(), "time_fe": S.time_fe.to_numpy(), "cl": np.asarray(cl).astype(str), "w": S.w.to_numpy(float)})
nm = [f"x{i}" for i in range(len(x))]
for n_, c in zip(nm, x): d[n_] = S[c].to_numpy(float)
m = pf.feols("y ~ " + " + ".join(nm) + " | unit_fe + time_fe", data=d, vcov={"CRV1": "cl"}, weights="w", demeaner=pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100000))
co = m.coef().reindex(nm).to_numpy(float); V = np.asarray(m._vcov)
crit = float(stats.t.ppf(0.975, int(d.cl.nunique()) - 1))
leads = [i for i, t in enumerate(x) if int(t.split("_")[2]) <= -7]; bl = co[leads]; Vl = V[np.ix_(leads, leads)]; W = float(bl @ np.linalg.solve(Vl, bl))
out["A_weighted_stack"] = dict(post_1_12=post_lin(co, V, x, 1, 12, crit), pretrend_p=float(stats.f.sf(W / len(leads), len(leads), int(d.cl.nunique()) - 1)),
                               stack_weights={str(k): float(v) for k, v in w_c.items()}, n_stacks=int(len(cnt)))
o0, ev0, _ = R.event_long(B, E, cluster="sigungu", treated=head)
out["A_unweighted_reference"] = dict(post_1_12=o0["post_1_12"], pretrend_p=o0["pretrend"]["p_F"])
coh = []
for c in sorted(S["stack"].unique()):
    Sc = S[S["stack"] == c]
    xc = [v for v in x if Sc[v].sum() > 0]
    try:
        F = A.fe_fit(Sc, xc, R.cl_of(Sc, B, "sigungu")); r = post_lin(F["b"], F["cov"], xc, 1, 12, F["crit"])
        coh.append(dict(stack=int(c), n_treated=int(Sc[Sc.treat == 1].pid.nunique()), theta_log=r["theta_log"], se_log=r["se_log"], pct=r["pct"]))
    except Exception as ex: coh.append(dict(stack=int(c), error=str(ex)[:100]))
C_ = pd.DataFrame([c for c in coh if "theta_log" in c]); wts = C_.n_treated / C_.n_treated.sum()
th = float((wts * C_.theta_log).sum()); se = float(np.sqrt((wts ** 2 * C_.se_log ** 2).sum()))
out["A_cohort_aggregate"] = dict(cohorts=coh, treated_weighted=dict(pct=pct(th), pct_ci=[pct(th - 1.96 * se), pct(th + 1.96 * se)], note="cohort s.e. treated as independent"))
print("A", out["A_weighted_stack"]["post_1_12"]["pct"], out["A_cohort_aggregate"]["treated_weighted"], flush=True)

bg = pd.read_csv(RES / "r3_benchmark_bldgap.csv"); bg["sel_year"] = bg.pid.str[:4].astype(int)
bg["pseudo_half"] = (bg.pseudo_start // 100).astype(str) + np.where(bg.pseudo_start % 100 <= 6, "H1", "H2")
def tab(g):
    r = g.groupby("rep").tau.mean()
    return dict(n_bld_obs=int(len(g)), reps=int(r.size), bias_log=float(g.tau.mean()), rep_mean=float(r.mean()), mc_se=float(r.std(ddof=1) / np.sqrt(r.size)) if r.size > 1 else None)
out["B_bias_by_selection_year"] = {f"{u}_{y}": tab(g) for (u, y), g in bg.groupby(["use", "sel_year"])}
out["B_bias_by_pseudo_half_health"] = {h: tab(g) for h, g in bg[bg.use == "보건"].groupby("pseudo_half")}
print("B", {k: round(v["bias_log"], 4) for k, v in out["B_bias_by_selection_year"].items()}, flush=True)

big = set(G[G.pct <= -0.5].pid); Ec = A.carrier_frame(E, "elec"); Ec = Ec[Ec.pid.isin(head)].copy()
pre_med = Ec[(Ec.ym < Ec.start_ym) & (R.mi(Ec.ym) >= R.mi(Ec.start_ym) - 12)].groupby("pid").kwh.median()
po = Ec[(Ec.ym > Ec.end_ym)].copy(); po["k"] = R.mi(po.ym) - R.mi(po.end_ym); po = po[(po.k >= 1) & (po.k <= 24)]
po["ratio"] = po.kwh / po.pid.map(pre_med); po["q"] = (po.k - 1) // 3 + 1; po["grp"] = np.where(po.pid.isin(big), "fall_ge50", "other")
prof = po.groupby(["grp", "q"]).agg(median_ratio=("ratio", "median"), share_below_25=("ratio", lambda v: float((v < 0.25).mean())), n_bld=("pid", "nunique")).reset_index()
out["C_profile"] = prof.to_dict("records")
out["C_donut_bins_7_12"] = post_lin(o0["_b"], o0["_V"], o0["_x"], 7, 12, o0["crit"])
out["C_renewable_share_big"] = float(Bi.loc[list(big)].renewable.mean()); out["C_n_big"] = len(big)
print("C", out["C_donut_bins_7_12"]["pct"], flush=True)

bal = set(Bi.loc[list(head)][Bi.loc[list(head)].start_ym >= 202201].index)
ob, evb, Sb = R.event_long(B, E, cluster="sigungu", treated=bal, pre_lo=-24, post_hi=12)
xb = ob["_x"]; pre_ix = [i for i, t in enumerate(xb) if int(t.split("_")[2]) <= -7]; post_ix = ob["_post12"]
own_b = R.own_breakdown(ob["_b"], ob["_V"], pre_ix, post_ix, [2, 3, 4, 5])
xm = o0["_x"]; pre_m = [i for i, t in enumerate(xm) if int(t.split("_")[2]) <= -7]
own_m = R.own_breakdown(o0["_b"], o0["_V"], pre_m, o0["_post12"], [2, 3, 4, 5])
out["D_balanced"] = dict(n_treated=ob["n_treated"], post_1_12=ob["post_1_12"], pretrend_p=ob["pretrend"]["p_F"], own_breakdown_adjacent=own_b,
                         maxd=R.rm_bounds(ob["_b"], ob["_V"], pre_ix, None, post_ix, [2, 3, 4, 5], 1.0, nsim=1000)["maxd"])
out["D_main_reference"] = dict(own_breakdown_adjacent=own_m, maxd=R.rm_bounds(o0["_b"], o0["_V"], pre_m, None, o0["_post12"], [2, 3, 4, 5], 1.0, nsim=1000)["maxd"])
print("D", ob["n_treated"], ob["post_1_12"]["pct"], own_b, own_m, flush=True)

Cc = A.carrier_frame(E, "elec").copy(); ids = set(Bi[Bi.role.isin(("control", "aux_2023")) & (Bi.use_class == "보건") & (Bi.link_grade == "A_cand")].index)
Cc = Cc[Cc.pid.isin(ids)].copy(); Cc = Cc[Cc.ym < Cc.start_ym.map(lambda v: ym_add(int(v), -3))]
Cc["y"] = np.log(Cc.kwh / Cc.gr_gfa); Cc["sel"] = Cc.pid.str[:4].astype(int)
per = Cc.groupby(["ym", "sel"]).pid.nunique().unstack(fill_value=0); okm = per[(per >= 10).all(axis=1)].index
D_ = Cc[Cc.ym.isin(okm)].copy(); D_["t"] = (R.mi(D_.ym) - R.mi(D_.ym).min()) / 12.0
for s_ in (2024, 2025): D_[f"t{s_}"] = D_.t * (D_.sel == s_)
D_["cl"] = D_.pid.map(Bi.sigunguCd.astype(str)); D_["ym_s"] = D_.ym.astype(str)
me = pf.feols("y ~ t2024 + t2025 | pid + ym_s", data=D_, vcov={"CRV1": "cl"})
cf = me.coef(); se_ = me.se(); Vt = np.asarray(me._vcov); bvec = cf.to_numpy(float); Wt = float(bvec @ np.linalg.solve(Vt, bvec))
out["E_cohort_trends"] = dict(months=[int(okm.min()), int(okm.max())], n_months=int(len(okm)), n_bld={int(k): int(v) for k, v in D_.groupby("sel").pid.nunique().items()},
                              slope_vs_2023_per_year={k: dict(coef=float(cf[k]), se=float(se_[k])) for k in cf.index}, joint_p=float(stats.chi2.sf(Wt, len(bvec))))
print("E", out["E_cohort_trends"], flush=True)

M = 0.008; steps = [2, 3, 4, 5]
out["F_smoothness"] = dict(M=M, mean_worst_case_deviation_log=float(np.mean([M * k * (k + 1) / 2 for k in steps])),
                           observed_abs_second_diff_pre=None)
ev = ev0.sort_values("a"); pre = ev[ev.b <= -7].coef.tolist() + [0.0]
sd = np.abs(np.diff(np.array(pre), 2)); out["F_smoothness"]["observed_abs_second_diff_pre"] = dict(median=float(np.median(sd)), max=float(sd.max()), n=int(sd.size))
print("F", out["F_smoothness"], flush=True)
(RES / "r4_analyses.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
