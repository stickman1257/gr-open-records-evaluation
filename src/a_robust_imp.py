"""Imputation-estimator robustness rows (Appendix Table D.2)."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A

NB = int(os.environ.get("IMP_NBOOT", "200"))
MIN_SIG = 30; MIN_TOTAL_MAIN = 30; MIN_TOTAL_CTRL = 10
_ROWS = {"": "primary", "controls_2024_25_only": "controls_2024_25_only", "drop_renewable": "drop_renewable", "donut3": "donut3", "use_sido": "fe_use_province",
         "minctrl5": "support_ge5", "minctrl10": "support_ge10", "strict_12_12": "strict_12_12", "no_anticip_holdout": "no_anticipation_holdout",
         "drop_jumpy": "drop_jumpy", "fit_s15_s4": "fit_window_s15_s4_calendar_matched"}
G7_LABEL = {"imputation_use": ("pooled", "primary"), "imputation_health_only": ("health_centre", "primary"), "imputation_daycare_only": ("daycare", "primary"),
            "imputation_use_sido": ("pooled", "fe_use_province"), "imputation_use_minctrl5": ("pooled", "support_ge5"),
            "imputation_use_minctrl10": ("pooled", "support_ge10"), "imputation_use_no_anticip_holdout": ("pooled", "no_anticipation_holdout")}
for _k, _v in _ROWS.items():
    if _k:
        G7_LABEL.setdefault(f"imputation_{_k}", ("pooled", _v)); G7_LABEL[f"imputation_health_{_k}"] = ("health_centre", _v)
SAMPLES = [("health_centre", "primary column (headline sample)", dict(uses=("보건",))), ("pooled", "secondary column", {})]
KEEP = ["att_log", "pct", "ci_log", "pct_ci", "att_log_rowmean", "n_treated", "n_treated_total", "n_control", "support_rows", "min_ctrl",
        "min_win", "anticipation_gap", "pretrend_p", "nboot_ok", "nboot", "fit_window", "median_bld_pct", "median_bld_pct_ci",
        "pct_heating_nov_mar", "pct_heating_nov_mar_lo", "pct_heating_nov_mar_hi", "pct_cooling_jun_sep", "pct_cooling_jun_sep_lo", "pct_cooling_jun_sep_hi",
        "pct_shoulder_apr_may_oct", "pct_shoulder_apr_may_oct_lo", "pct_shoulder_apr_may_oct_hi", "n_treated_by_use", "n_control_by_use", "cohorts"]


def row(spec, car, source, r=None, status="ok", note="", sample="pooled"):
    d = dict(sample=sample, column={"health_centre": "primary", "pooled": "secondary"}.get(sample, "use-specific"), spec=spec, carrier=car, source=source, status=status, note=note)
    if r:
        d.update({k: r.get(k) for k in KEEP})
        d["pct_lo"], d["pct_hi"] = (r.get("pct_ci") or [np.nan, np.nan])
    return d


def signature_sets(B):
    """(evaluable pids, confirmed pids) among treated buildings, from results/a_linkage_signature.csv."""
    f = RES / "a_linkage_signature.csv"
    if not f.exists(): return set(), set(), "results/a_linkage_signature.csv absent"
    L = pd.read_csv(f)
    if "signature" not in L or L.empty: return set(), set(), "no signature column / empty file"
    sig = L.signature.map(lambda v: {"true": True, "false": False}.get(str(v).strip().lower(), np.nan))
    ev = set(L.pid[sig.notna()]); ok = set(L.pid[sig == True])
    return ev, ok, f"{len(ev)} buildings evaluable in the signature file, {len(ok)} with signature"


def main():
    B, E = A.load(); rows = []
    g7 = RES / "G7_imputation.json"
    for r in (json.loads(g7.read_text()) if g7.exists() else []):
        if r.get("spec") in G7_LABEL:
            smp, lab = G7_LABEL[r["spec"]]
            rows.append(row(lab, r.get("carrier"), "results/G7_imputation.json", r if "att_log" in r else None,
                            status=r.get("status", "ok") if "att_log" not in r else "ok", sample=smp))
    hc = A.heating_carrier(B, E); hc_thr = {thr: A.heating_carrier(B, E, gas_ratio=thr) for thr in (1.5, 3.0)}
    Tb = B[(B.role == "treated") & (B.link_grade == "A_cand") & B.use_class.isin(A.PRIMARY_IMP["uses"])]
    ev, conf, sig_note = signature_sets(B)
    for smp, _col, skw in SAMPLES:
      for car in ["elec", "total"]:
        base = dict(A.PRIMARY_IMP, **skw)
        S0 = A.imputation_setup(B, E, car, **base)
        if S0 is None:
            rows.append(row("(own rows)", car, "a_robust_imp.py", status="not estimable: no treated building in this sample and carrier", sample=smp)); continue
        sample_t = set(S0["T"].pid)
        try:
            rows.append(row("original_link_grades", car, "a_robust_imp.py", A.imputation_did(A.original_grades(B), E, car, nboot=NB, **base),
                            note="link_grade_v1 / use_class_v1 (before s02_link.regrade)", sample=smp))
        except Exception as ex: rows.append(row("original_link_grades", car, "a_robust_imp.py", status=f"{type(ex).__name__}: {ex}"[:200], sample=smp))
        try:
            if "_ND" not in globals(): globals()["_ND"] = A.load(dedup=False)
            Bn, En = globals()["_ND"]
            rows.append(row("no_dedup", car, "a_robust_imp.py", A.imputation_did(Bn, En, car, nboot=NB, **base),
                            note="records as stored, before D23 (one physical building = one unit)", sample=smp))
        except Exception as ex: rows.append(row("no_dedup", car, "a_robust_imp.py", status=f"{type(ex).__name__}: {ex}"[:200], sample=smp))
        specs = [("grades_A+B", dict(grades=("A_cand", "B")), ""),
                 ("stable_meter_count", dict(drop=A.unstable_meter_buildings(E, "elec") | (A.unstable_meter_buildings(E, "gas") if car == "total" else set())),
                  "n_items constant over all observed months")]
        for cl in ["gas", "elec", "other", "unknown"]:
            keep = set(hc[hc == cl].index) & sample_t
            specs.append((f"heat_{cl}", dict(treated_keep=keep), f"exploratory; {len(keep)} treated buildings of the analysis sample in this class"))
        for thr in (1.5, 3.0):
            keep = set(hc_thr[thr][hc_thr[thr] == "gas"].index) & sample_t
            specs.append((f"heat_gas_thr{thr}", dict(treated_keep=keep), f"exploratory sensitivity; gas class with winter/summer gas ratio >= {thr}: {len(keep)} treated buildings"))
        for name, kw, note in specs:
            if "treated_keep" in kw and len(kw["treated_keep"]) < 3:
                rows.append(row(name, car, "a_robust_imp.py", status="not estimated: fewer than 3 treated buildings", note=note, sample=smp)); continue
            try: rows.append(row(name, car, "a_robust_imp.py", A.imputation_did(B, E, car, nboot=NB, **dict(base, **kw)), note=note, sample=smp))
            except Exception as ex: rows.append(row(name, car, "a_robust_imp.py", status=f"{type(ex).__name__}: {ex}"[:200], note=note, sample=smp))
        n_ev = len(ev & sample_t); n_conf = len(conf & sample_t)
        note = f"{sig_note}; in the analysis sample ({smp}, {car}): {n_ev} evaluable, {n_conf} confirmed; reported only if evaluable >= {MIN_SIG}"
        if n_ev < MIN_SIG:
            rows.append(row("signature_confirmed", car, "a_robust_imp.py", status=f"not reported: {n_ev} evaluable treated buildings (< {MIN_SIG})", note=note, sample=smp))
        else:
            try: rows.append(row("signature_confirmed", car, "a_robust_imp.py",
                                 A.imputation_did(B, E, car, nboot=NB, **dict(base, treated_keep=conf & sample_t)), note=note, sample=smp))
            except Exception as ex: rows.append(row("signature_confirmed", car, "a_robust_imp.py", status=f"{type(ex).__name__}: {ex}"[:200], note=note, sample=smp))
    R = pd.DataFrame(rows)
    Ts = B[B.pid.isin(Tb.pid)]
    heat_counts = {"threshold_2.0 (main)": {str(k): int(v) for k, v in hc.reindex(Ts.pid).value_counts().items()},
                   **{f"threshold_{thr}": {str(k): int(v) for k, v in hc_thr[thr].reindex(Ts.pid).value_counts().items()} for thr in (1.5, 3.0)}}
    (RES / "a_robust_imp.json").write_text(json.dumps(dict(generated=time.strftime("%Y-%m-%d %H:%M"), nboot=NB, min_win=A.MIN_WIN, definition=__doc__,
                                                           heating_class_rule="gas: winter (Dec-Feb) / summer (Jun-Aug) mean gas >= 2.0 with >= 10 pre months of gas; "
                                                           "else elec: winter / shoulder (Apr, May, Oct) mean electricity >= 1.3; else other; unknown: < 10 pre months of electricity",
                                                           heating_class_counts_treated_candidates=heat_counts,
                                                           rows=rows), indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    flat = R.drop(columns=[c for c in ["cohorts", "n_treated_by_use", "n_control_by_use", "ci_log", "pct_ci", "median_bld_pct_ci"] if c in R])
    flat.to_csv(RES / "a_robust_imp.csv", index=False)
    prim = {r["carrier"]: r for r in rows if r["spec"] == "primary" and r["sample"] == "pooled"}
    t = prim.get("total", {}); e = prim.get("elec", {})
    gas_ok = set(E[(E.carrier == "gas")].pid)
    tot = dict(generated=time.strftime("%Y-%m-%d %H:%M"),
               rule="total metered energy = electricity + gas for buildings with a confirmed gas meter (>= 1 gas month with status ok after the QA gate), "
                    "months with both carriers observed, a month with zero gas kept as zero; main text only if >= 30 treated buildings AND >= 10 gas-metered "
                    "control-pool buildings in the fit (n_control), else repository appendix",
               unit_note="both fields are summed as delivered by the API (no conversion); see results/a_gas_unit.md",
               n_treated=t.get("n_treated"), n_control=t.get("n_control"), n_treated_by_use=t.get("n_treated_by_use"), n_control_by_use=t.get("n_control_by_use"),
               n_treated_electricity_sample=e.get("n_treated"),
               n_treated_sample_with_gas_meter=int(Tb.pid.isin(gas_ok).sum()), n_treated_sample_candidates=int(len(Tb)),
               gas_months_ok_zero=int(((E.carrier == "gas") & (E.kwh == 0)).sum()), gas_months_ok_negative_dropped=int(((E.carrier == "gas") & (E.kwh < 0)).sum()),
               condition_treated_ge30=bool((t.get("n_treated") or 0) >= MIN_TOTAL_MAIN),
               condition_gas_metered_control_pool_ge10=bool((t.get("n_control") or 0) >= MIN_TOTAL_CTRL),
               main_text=bool((t.get("n_treated") or 0) >= MIN_TOTAL_MAIN and (t.get("n_control") or 0) >= MIN_TOTAL_CTRL),
               placement="main text (secondary outcome)" if ((t.get("n_treated") or 0) >= MIN_TOTAL_MAIN and (t.get("n_control") or 0) >= MIN_TOTAL_CTRL) else "repository appendix only",
               estimate_row="results/a_robust_imp.csv: sample=pooled, spec=primary, carrier=total (same as results/G7_imputation.csv spec=imputation_use, carrier=total)",
               status=t.get("status", "no estimate"))
    EN = {"보건": "health centres", "어린이집": "daycare centres", "경로당": "senior centres", "기타": "other facilities"}
    comp = lambda d: {EN.get(k, str(k)): int(v) for k, v in (d or {}).items() if int(v) > 0}
    ct, cc = comp(t.get("n_treated_by_use")), comp(t.get("n_control_by_use"))
    tot["use_composition"] = dict(treated=ct, control_pool=cc)
    if len(ct) == 1: lab = f"{list(ct)[0]} only"
    elif len(ct) > 1: lab = " and ".join(f"{k} ({v})" for k, v in sorted(ct.items(), key=lambda kv: -kv[1]))
    else: lab = None
    tot["suggested_row_label"] = None if lab is None else f"Total metered energy, gas-metered buildings: {lab}"
    tot["label_rule"] = ("PI decision 8 (11/13): the row is named after the use types that actually contribute treated buildings; "
                         "'<use> only' when one use contributes; control-pool composition is given for the table note")
    (RES / "a_total_energy.json").write_text(json.dumps(tot, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    flat = flat.sort_values(["carrier", "sample"], key=lambda c: c.map({"health_centre": 0, "pooled": 1, "daycare": 2, "elec": 0, "total": 1}).fillna(9), kind="stable")
    flat.to_csv(RES / "a_robust_imp.csv", index=False)
    print(flat[["sample", "spec", "carrier", "status", "pct", "pct_lo", "pct_hi", "n_treated", "n_control", "support_rows"]].round(4).to_string(index=False))
    print(json.dumps(tot, indent=1, ensure_ascii=False, default=float))

if __name__ == "__main__":
    main()
