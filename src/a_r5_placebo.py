"""Placebo estimates and their power (Appendix Table D.2)."""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC
import s04_analysis as A
import s06_benchmark as BM
from s06_benchmark import ym_add, ym_range

LABEL = "post hoc (added 11/16 after the final run)"
HC, DC = "보건", "어린이집"
SAMPLES = {"health_centre": [HC], "pooled": [HC, DC]}
Z80 = 1.959964 + 0.841621


def pseudo_selection(B, Bp, e, trim=3):
    T = B[(B.role == "treated") & B.start_ym.notna() & B.end_ym.notna()]
    idx = lambda v: (int(v) // 100) * 12 + int(v) % 100
    gap = {u: int(np.median([idx(b) - idx(a) + 1 for a, b in zip(g.start_ym, g.end_ym)])) for u, g in T.groupby("use_class")}
    have = e.groupby("pid").ym.apply(set).to_dict(); pseudo = {}
    for b in Bp[Bp.role == "aux_2023"].itertuples():
        pe = ym_add(int(b.start_ym), -(12 + trim + 1)); h = have.get(b.pid, set())
        if len(set(ym_range(ym_add(pe, 1), ym_add(pe, 12))) & h) < A.MIN_WIN: continue
        for g in range(gap.get(b.use_class, 0), -1, -1):
            ps = ym_add(pe, -g + 1)
            if len(set(ym_range(ym_add(ps, -12), ym_add(ps, -1))) & h) >= A.MIN_WIN:
                pseudo[b.pid] = (ps, pe); break
    return pseudo


def pseudo_treated_pre(B, E, uses, carrier="elec", min_post=8, anticip=3):
    Tb = B[(B.role == "treated") & (B.link_grade == "A_cand") & B.use_class.isin(uses)]
    et = BM.ML.prep(E[E.pid.isin(Tb.pid) & (E.phase == "pre")], carrier)
    have = et.groupby("pid").ym.apply(set).to_dict(); pseudo = {}
    for b in Tb.itertuples():
        s0 = int(b.start_ym); h = have.get(b.pid, set())
        if (len(set(ym_range(ym_add(s0, -24), ym_add(s0, -13))) & h) >= A.MIN_WIN and
                len(set(ym_range(ym_add(s0, -12), ym_add(s0, -anticip - 1))) & h) >= min_post):
            pseudo[b.pid] = (ym_add(s0, -12), ym_add(s0, -13))
    et = et[et.pid.isin(list(pseudo))]; et = et[et.ym < et.start_ym.map(lambda v: ym_add(int(v), -anticip))]
    return pseudo, Tb[Tb.pid.isin(list(pseudo))], et


def estimate(Bp, e, pseudo, nboot, tau=0.0):
    P = BM.make_panel(e, pseudo); B2, E2 = BM.did_inputs(Bp, P, tau)
    r = A.imputation_did(B2, E2, "elec", uses=None, ctrl_roles=("control",), nboot=nboot, pretrend=False, min_win=0)
    lo, hi = r["ci_log"]; se = (hi - lo) / (2 * 1.959964)
    return dict(tau=tau, est_log=r["att_log"], ci_log=[lo, hi], change_pct=100 * (np.exp(r["att_log"]) - 1), ci_pct=[100 * (np.exp(lo) - 1), 100 * (np.exp(hi) - 1)],
                ci_excludes_zero=bool(lo > 0 or hi < 0), se_log_from_interval=float(se), detectable_log=float(Z80 * se),
                detectable_pct=float(100 * (1 - np.exp(-Z80 * se))), n_pseudo_treated_used=int(r["n_treated"]), n_control=int(r["n_control"]),
                n_pseudo_by_use={BM.USE_EN.get(k, k): v for k, v in r["n_treated_by_use"].items()}, support_rows=r["support_rows"], nboot=int(nboot), nboot_ok=r["nboot_ok"])


def main():
    NB = int(os.environ.get("R5_NBOOT", "1000")); t0 = time.time()
    B, E = A.load()
    pub = json.loads((RES / "a_placebo_summary.json").read_text(encoding="utf-8"))
    prec = json.loads((RES / "a_precision.json").read_text(encoding="utf-8"))
    out = dict(label=LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), definition=__doc__, nboot=NB,
               panel_sha256=hashlib.sha256((PROC / "energy_monthly.csv").read_bytes()).hexdigest(), samples={})
    rows = []
    for s, uses in SAMPLES.items():
        Bc, ec = BM.population(B, E, "elec", uses)
        ps_sel = pseudo_selection(B, Bc, ec)
        ps_pre, Tb, et = pseudo_treated_pre(B, E, uses)
        assert not (set(ps_sel) & set(ps_pre)), "a building is pseudo-treated twice"
        ctrl = Bc[Bc.role == "control"]
        designs = {
            "selection_cohort": (Bc[(Bc.role == "control") | Bc.pid.isin(list(ps_sel))], ec, ps_sel),
            "treated_pre_period": (pd.concat([Bc, Tb], ignore_index=True), pd.concat([ec, et], ignore_index=True), ps_pre),
            "treated_pre_period_controls_2024_25_only": (pd.concat([ctrl, Tb], ignore_index=True), pd.concat([ec[ec.pid.isin(ctrl.pid)], et], ignore_index=True), ps_pre),
            "union": (pd.concat([ctrl, Bc[Bc.pid.isin(list(ps_sel))], Tb], ignore_index=True),
                      pd.concat([ec[ec.pid.isin(ctrl.pid) | ec.pid.isin(list(ps_sel))], et], ignore_index=True), {**ps_sel, **ps_pre}),
        }
        res = {}; rep = {}
        for name, (Bp, e, pseudo) in designs.items():
            e = e[e.pid.isin(Bp.pid)]
            r = estimate(Bp, e, pseudo, NB); r.update(n_pseudo_selection_cohort=len(set(pseudo) & set(ps_sel)), n_pseudo_treated_pre=len(set(pseudo) & set(ps_pre)))
            r10 = estimate(Bp, e, pseudo, min(NB, 200), tau=-0.10)
            r["recovery_tau_minus10"] = dict(change_pct=r10["change_pct"], ci_pct=r10["ci_pct"], nboot=r10["nboot"])
            res[name] = r; rows.append(dict(sample=s, design=name, label=LABEL, **{k: v for k, v in r.items() if not isinstance(v, dict)}))
            print(f"{s} {name}: {r['change_pct']:+.2f}% ({r['ci_pct'][0]:+.1f}, {r['ci_pct'][1]:+.1f}); pseudo {r['n_pseudo_treated_used']} "
                  f"(sel {r['n_pseudo_selection_cohort']}, pre {r['n_pseudo_treated_pre']}), controls {r['n_control']}; detectable {r['detectable_pct']:.1f}%; "
                  f"tau -10%: {r10['change_pct']:+.1f}%; {time.time() - t0:.0f} s", flush=True)
            if name in ("selection_cohort", "treated_pre_period"):
                r2 = estimate(Bp, e, pseudo, int(pub.get("imp_nboot", 200)))
                p = [q for q in pub["tests"][name]["by_sample"][s]["placebo_tau0"] if q["estimator"] == "imp" and q["carrier"] == "elec"][0]
                pd_ = pub["tests"][name]["detectable_differential_trend"][s]["by_carrier"]["elec"]["detectable_pct"]
                rep[name] = dict(published_change_pct=p["change_pct"], published_ci_pct=p["ci_pct"], published_detectable_pct=pd_, published_n=p["n_pseudo_treated_used"],
                                 here_change_pct=r2["change_pct"], here_ci_pct=r2["ci_pct"], here_detectable_pct=r2["detectable_pct"], here_n=r2["n_pseudo_treated_used"],
                                 identical=bool(abs(p["change_pct"] - r2["change_pct"]) < 1e-6 and np.allclose(p["ci_pct"], r2["ci_pct"], atol=1e-6)))
                print("   reproduction:", rep[name], flush=True)
        u = res["union"]
        cmp_ = dict(union_detectable_pct=u["detectable_pct"], selection_cohort_detectable_pct=res["selection_cohort"]["detectable_pct"],
                    treated_pre_period_detectable_pct=res["treated_pre_period"]["detectable_pct"],
                    treated_pre_period_controls_2024_25_only_detectable_pct=res["treated_pre_period_controls_2024_25_only"]["detectable_pct"],
                    published_selection_cohort_detectable_pct=rep["selection_cohort"]["published_detectable_pct"],
                    published_treated_pre_period_detectable_pct=rep["treated_pre_period"]["published_detectable_pct"],
                    design_value_selection_cohort_pct=(prec.get("selection_cohort_placebo_detectable", {}).get("detectable_differential_trend_pct", {}).get("health_centre") if s == "health_centre" else None),
                    headline_att_pct=None)
        try:
            g7 = {r.get("spec"): r for r in json.loads((RES / "G7_imputation.json").read_text(encoding="utf-8")) if isinstance(r, dict) and r.get("carrier") == "elec"}
            cmp_["headline_att_pct"] = 100 * g7["imputation_health_only" if s == "health_centre" else "imputation_use"]["pct"]
            cmp_["union_detects_headline_size"] = bool(u["detectable_pct"] < abs(cmp_["headline_att_pct"]))
        except Exception as ex: print("G7 not read:", ex)
        out["samples"][s] = dict(uses=uses, designs=res, reproduction=rep, comparison=cmp_)
    pd.DataFrame(rows).to_csv(RES / "a_r5_placebo.csv", index=False)
    out["runtime_s"] = round(time.time() - t0, 1)
    (RES / "a_r5_placebo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("written results/a_r5_placebo.json, .csv;", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
