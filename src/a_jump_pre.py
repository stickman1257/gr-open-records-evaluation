"""Outlier filter on all months vs pre-works months only."""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC
import s04_analysis as A

LABEL = "post-hoc (added 11/16 after the final run)"
HC, DC = "보건", "어린이집"
SAMPLES = {"health_centre": (HC,), "pooled": (HC, DC)}
G7ROW = {"health_centre": "imputation_health_only", "pooled": "imputation_use"}
K = 3.0
pct = lambda a: float(np.exp(a) - 1) if a is not None and np.isfinite(a) else None


def offending(E, k=K, phases=None):
    """Electricity rows (kWh > 0) outside [median / k, k x median] of the building; median over the rows kept by `phases`."""
    d = E[(E.carrier == "elec") & (E.kwh > 0)]
    if phases is not None: d = d[d.phase.isin(phases)]
    m = d.groupby("pid").kwh.transform("median")
    return d[(d.kwh > k * m) | (d.kwh < m / k)]


def row(sample, filt, r, n_flag_t=None, n_flag_c=None):
    return dict(sample=sample, filter=filt, label=LABEL, pct=r["pct"], pct_lo=r["pct_ci"][0], pct_hi=r["pct_ci"][1], att_log=r["att_log"],
                ci_log_lo=r["ci_log"][0], ci_log_hi=r["ci_log"][1], n_treated=r["n_treated"], n_control=r["n_control"],
                support_rows=r["support_rows"], pretrend_p=r["pretrend_p"],
                pretrend_leads_log=(r.get("pretrend") or {}).get("coef"), nboot=r["nboot"], nboot_ok=r["nboot_ok"],
                n_treated_flagged=n_flag_t, n_control_flagged=n_flag_c,
                median_bld_pct=r.get("median_bld_pct"), anticipation_gap=r.get("anticipation_gap"))


def main():
    NBH = int(os.environ.get("IMP_NBOOT_HEAD", "1000")); t0 = time.time()
    B, E = A.load(); Bi = B.drop_duplicates("pid").set_index("pid")
    orig = A.jumpy_buildings(E)
    off_all = offending(E); off_pre = offending(E, phases=["pre"])
    assert set(off_all.pid) == orig, "all-month rule here differs from s04_analysis.jumpy_buildings"
    pre = set(off_pre.pid)
    n_pre_months = E[(E.carrier == "elec") & (E.kwh > 0) & (E.phase == "pre")].groupby("pid").ym.nunique()
    g7 = {}
    try:
        for r in json.loads((RES / "G7_imputation.json").read_text(encoding="utf-8")):
            if isinstance(r, dict) and r.get("carrier") == "elec": g7[r.get("spec")] = r
    except Exception as ex: print("G7_imputation.json not read:", type(ex).__name__, ex)
    out = dict(label=LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), definition=__doc__, k=K, nboot=NBH,
               panel_sha256=hashlib.sha256((PROC / "energy_monthly.csv").read_bytes()).hexdigest(),
               n_buildings_flagged_any_role=dict(original_all_months=len(orig), pre_months_only=len(pre), both=len(orig & pre),
                                                 pre_only_not_original=len(pre - orig)), samples={})
    rows = []
    for s, uses in SAMPLES.items():
        S0 = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=uses)); f0 = S0["fit"]()
        pi, pids = S0["pi"], S0["pids"]
        T0 = set(pids[pd.unique(pi[S0["post"] & f0["ok"]])]); C0 = set(pids[pd.unique(pi[f0["F"] & S0["isc"]])])
        res = {}
        for filt, drop in [("pre_months_only", pre), ("original_all_months", orig)]:
            r = A.imputation_did(B, E, "elec", nboot=NBH, fe="use", uses=uses, drop=drop)
            res[filt] = row(s, filt, r, len(T0 & drop), len(C0 & drop)); rows.append(res[filt])
            print(f"{s} {filt}: flagged T {len(T0 & drop)} of {len(T0)}, C {len(C0 & drop)} of {len(C0)}; kept T {r['n_treated']}, C {r['n_control']}; "
                  f"ATT {100 * r['pct']:+.1f}% ({100 * r['pct_ci'][0]:+.1f}, {100 * r['pct_ci'][1]:+.1f}); pre-trend p {r['pretrend_p']}; {time.time() - t0:.0f} s", flush=True)
        h = g7.get(G7ROW[s])
        if h:
            res["none_headline_G7"] = dict(sample=s, filter="none_headline_G7", label="final run (G7_imputation.json), copied for reference",
                                           pct=h["pct"], pct_lo=h["pct_ci"][0], pct_hi=h["pct_ci"][1], att_log=h["att_log"], n_treated=h["n_treated"],
                                           n_control=h["n_control"], support_rows=h.get("support_rows"), pretrend_p=h.get("pretrend_p") if h.get("pretrend_p") is not None else (h.get("pretrend") or {}).get("p_joint"), nboot=h.get("nboot"),
                                           n_treated_flagged=0, n_control_flagged=0)
            rows.append(res["none_headline_G7"])
        g = g7.get(G7ROW[s].replace("_only", "").replace("imputation_use", "imputation") + "_drop_jumpy")
        oa = off_all[off_all.pid.isin(T0)]; ph = oa.groupby("pid").phase.agg(set); fl = sorted(T0 & orig)
        cnt = lambda cond: int(sum(cond(ph[p]) for p in fl))
        dec = dict(n_headline_treated=len(T0), n_flagged_original=len(fl), n_flagged_pre_rule=len(T0 & pre),
                   only_post=cnt(lambda v: v == {"post"}), only_constr=cnt(lambda v: v == {"constr"}),
                   only_constr_or_post=cnt(lambda v: "pre" not in v), constr_and_post_no_pre=cnt(lambda v: v == {"constr", "post"}),
                   any_pre=cnt(lambda v: "pre" in v), any_post=cnt(lambda v: "post" in v), any_constr=cnt(lambda v: "constr" in v),
                   not_flagged_by_pre_rule=len((T0 & orig) - pre), flagged_by_pre_rule_not_original=len((T0 & pre) - orig),
                   offending_months_by_phase={k_: int(v) for k_, v in oa.phase.value_counts().items()},
                   offending_low_vs_high=dict(below_median_over_k=int((oa.kwh < oa.groupby("pid").kwh.transform("median")).sum()), n=int(len(oa))),
                   median_pre_months_per_headline_treated=float(n_pre_months.reindex(sorted(T0)).fillna(0).median()),
                   note="phases by the recorded dates: pre = before the start month, constr = start..completion month, post = after completion; "
                        "offending months use the all-month median (original rule)")
        op = oa[oa.phase == "post"]; med = E[(E.carrier == "elec") & (E.kwh > 0)].groupby("pid").kwh.median()
        dec["post_offending_months_low"] = int((op.kwh < op.pid.map(med) / K).sum()); dec["post_offending_months_high"] = int((op.kwh > K * op.pid.map(med)).sum())
        cdec = dict(n_headline_control=len(C0), n_flagged_original=len(C0 & orig), n_flagged_pre_rule=len(C0 & pre),
                    not_flagged_by_pre_rule=len((C0 & orig) - pre))
        out["samples"][s] = dict(uses=list(uses), rows=res, treated_flags=dec, control_flags=cdec,
                                 G7_drop_jumpy_200_draws=(dict(pct=g["pct"], pct_ci=g["pct_ci"], n_treated=g["n_treated"], n_control=g["n_control"],
                                                               pretrend_p=g.get("pretrend_p"), nboot=g.get("nboot")) if g else None))
        print(f"{s} decomposition: {json.dumps({k_: v for k_, v in dec.items() if k_ != 'note'}, ensure_ascii=False)}; controls {cdec}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "a_jump_pre.csv", index=False)
    out["runtime_s"] = round(time.time() - t0, 1)
    (RES / "a_jump_pre.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("written results/a_jump_pre.csv, .json;", out["runtime_s"], "s")

if __name__ == "__main__":
    main()
