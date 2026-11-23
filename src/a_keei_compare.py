"""Comparison with the agency (KEEI) estimate in kWh/m2 per month (Table 5)."""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy import stats as _st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
from a_interim_checks import estimate, level_E, SAMPLES, HC, DC, mi

LABEL = "added after the KEEI discovery and after the interim look, before the final data"
EXPL = LABEL + "; EXPLORATORY split of the treated daycares by confirmed gas meter"
DESCR = "descriptive only: different cohorts, data, control groups and estimators; our s.e. approximated from the percentile interval; not a test of equality"
KEEI = dict(bibkey="keei2025gr", source="KEEI Basic Research Report 25-14 (2025), ch. 4; values read by the PI from the report (paper/refs_verified.md)",
            unit="kWh per m2 per month, electricity",
            table="report Table 4-7 (PDF p. 106), stacked DID by use; s.e. clustered by building and year-month",
            sample="522 public buildings SELECTED in 2020 and 2021 (Table 4-2: 289 of the 2020 selection, 233 of the 2021 selection; daycare 358, health 138, "
                   "hospital 26); completions from early 2021 to late 2023",
            sample_by_selection_year={"2020": dict(total=289, health_centre=66, daycare=216), "2021": dict(total=233, health_centre=72, daycare=142)},
            se_clustering="building and year-month",
            comparable_no_temperature_interactions=dict(note="comparable to a simple ATT; these columns still include temperature spline controls",
                                                        daycare=dict(coef=-0.429, se=0.270, significant=False, n_obs=33043),
                                                        health_centre=dict(coef=-1.612, se=0.702, significant=True, n_obs=10419),
                                                        hospital=dict(coef=-0.236, se=0.974, significant=False)),
            with_temperature_interactions=dict(note="policy-effect main term of a model with temperature interactions; NOT directly comparable with an ATT",
                                               daycare=dict(coef=-1.014, se=0.503), health_centre=dict(coef=-2.712, se=1.334), hospital=dict(coef=-0.742, se=0.777)),
            pooled_static=[-0.732, -2.471], pooled_dynamic_about=-2.0, gas="not significant",
            footnote_165="most gas-using buildings are daycares (not: most daycares use gas)",
            controls="not-yet-started buildings of the same two selection years (ours: later selection years 2023-25 of the same use)")
for _b in ("comparable_no_temperature_interactions", "with_temperature_interactions"):
    for _u, _v in KEEI[_b].items():
        if isinstance(_v, dict): _v["ci95"] = [round(_v["coef"] - 1.96 * _v["se"], 3), round(_v["coef"] + 1.96 * _v["se"], 3)]
NBH = int(os.environ.get("IMP_NBOOT_HEAD", "1000"))
MIN_T = 15
ROLE = {2021: "reproduction cohort (selection year common with KEEI)", 2022: "extension cohort (not in the KEEI sample)"}


def one(B, E, EL, row, sample, keei_key, ids=None, label=LABEL, **tags):
    """One comparison row. ids = treated buildings kept (None = all of the sample)."""
    base = dict(row=row, sample=sample, carrier="elec", unit="kWh/m2 per month", label=label, **tags)
    if ids is not None and len(ids) < MIN_T: return dict(base, status=f"not estimated: {len(ids)} treated buildings (< {MIN_T})", n_treated_subset=len(ids))
    kw = dict(uses=SAMPLES[sample]); kw.update({} if ids is None else dict(treated_keep=set(ids)))
    r, S, f = estimate(B, E, "elec", nboot=NBH, **kw); rl, _, _ = estimate(B, EL, "elec", nboot=NBH, **kw)
    if r["n_treated"] < MIN_T: return dict(base, status=f"not estimated: {r['n_treated']} treated buildings with a supported post month (< {MIN_T})", n_treated=r["n_treated"])
    tp = set(S["pids"][pd.unique(S["pi"][S["post"] & f["ok"]])])
    e = E[(E.carrier == "elec") & (E.kwh > 0) & E.pid.isin(tp)]; e = e[(mi(e.ym) - mi(e.start_ym.astype(int))).between(-12, -1)]
    lv = e.assign(i=e.kwh / e.gr_gfa).groupby("pid").i.mean()
    d = dict(base, status="ok", n_treated=rl["n_treated"], n_control=rl["n_control"], support_rows=rl["support_rows"], nboot=rl["nboot"],
             level_att=rl["att_log"], level_att_lo=rl["ci_log"][0], level_att_hi=rl["ci_log"][1],
             pct=r["pct"], pct_lo=r["pct_ci"][0], pct_hi=r["pct_ci"][1], pre_mean=float(lv.mean()), pre_median=float(lv.median()),
             pct_x_pre_mean=float(lv.mean() * r["pct"]), pct_x_pre_mean_lo=float(lv.mean() * r["pct_ci"][0]), pct_x_pre_mean_hi=float(lv.mean() * r["pct_ci"][1]),
             pct_x_pre_mean_note="approximate")
    for g_, nm in [("early_2021H2_2022H2", "early"), ("late_2023H2_2024H1", "late")]:
        v = rl["groups"].get(g_, {}); ci_ = v.get("ci_log") or [None, None]
        d.update({f"level_att_{nm}": v.get("att_log"), f"level_att_{nm}_lo": ci_[0], f"level_att_{nm}_hi": ci_[1]})
    kc = KEEI["comparable_no_temperature_interactions"].get(keei_key) if keei_key else None
    kt = KEEI["with_temperature_interactions"].get(keei_key) if keei_key else None
    if kc:
        se_o = (rl["ci_log"][1] - rl["ci_log"][0]) / (2 * 1.96); diff = rl["att_log"] - kc["coef"]; se_d = float(np.sqrt(se_o ** 2 + kc["se"] ** 2)); z = diff / se_d
        d.update(keei_coef=kc["coef"], keei_se=kc["se"], keei_ci_lo=kc["ci95"][0], keei_ci_hi=kc["ci95"][1], keei_significant=kc["significant"],
                 keei_scope="both KEEI selection years (2020 and 2021) together; no coefficient by selection year is reported",
                 diff_ours_minus_keei=float(diff), se_ours_approx=float(se_o), se_diff=se_d, mdd_80=float(2.8 * se_d),
                 abs_diff_below_mdd_80=bool(abs(diff) < 2.8 * se_d),
                 ours_point_inside_keei_ci=bool(kc["ci95"][0] <= rl["att_log"] <= kc["ci95"][1]),
                 keei_coef_inside_our_ci=bool(rl["ci_log"][0] <= kc["coef"] <= rl["ci_log"][1]),
                 intervals_overlap=bool(rl["ci_log"][0] <= kc["ci95"][1] and rl["ci_log"][1] >= kc["ci95"][0]),
                 our_ci_includes_zero=bool(rl["ci_log"][0] <= 0 <= rl["ci_log"][1]),
                 z_approx=float(z), p_approx=float(2 * _st.norm.sf(abs(z))), do_not_quote=True, comparison_label=DESCR)
    if kt: d.update(keei_temp_coef=kt["coef"], keei_temp_se=kt["se"], keei_temp_ci_lo=kt["ci95"][0], keei_temp_ci_hi=kt["ci95"][1],
                    keei_temp_note="with temperature interactions; not directly comparable")
    if sample == "pooled": d.update(keei_static_pooled_lo=min(KEEI["pooled_static"]), keei_static_pooled_hi=max(KEEI["pooled_static"]), keei_dynamic_about=KEEI["pooled_dynamic_about"])
    print(f"{row}: n {d['n_treated']}  level {d['level_att']:+.2f} [{d['level_att_lo']:+.2f}, {d['level_att_hi']:+.2f}]  KEEI {d.get('keei_coef')}  mdd80 {d.get('mdd_80')}", flush=True)
    return d


def main():
    B, E = A.load(); EL = level_E(E); Bi = B.drop_duplicates("pid").set_index("pid")
    out = dict(label=LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), nboot=NBH, keei=KEEI, rows={})
    Sd = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=SAMPLES["daycare"])); td = set(Sd["T"].pid)
    Sh = A.imputation_setup(B, E, "elec", **dict(A.PRIMARY_IMP, uses=SAMPLES["health_centre"])); th = set(Sh["T"].pid)
    gas_any = set(E[E.carrier == "gas"].pid)
    GROUPS = [("health_centre", "health_centre", "health_centre", None, LABEL, "all health centres"),
              ("daycare", "daycare", "daycare", None, LABEL, "all daycare centres"),
              ("daycare_confirmed_gas_meter", "daycare", "daycare", td & gas_any, EXPL, "daycares with a confirmed gas meter (>= 1 gas month with status ok)"),
              ("daycare_no_confirmed_gas_meter", "daycare", "daycare", td - gas_any, EXPL, "daycares without a confirmed gas meter")]
    rows = []
    for row, sample, kk, ids, lab, desc in GROUPS:
        rows.append(one(B, E, EL, row, sample, kk, ids, lab, selection_year="2021+2022", cohort_role="all selection years", group=desc))
    rows.append(one(B, E, EL, "pooled", "pooled", None, None, LABEL, selection_year="2021+2022", cohort_role="all selection years", group="health centres and daycares"))
    for row, sample, kk, ids, lab, desc in GROUPS:
        allids = (th if sample == "health_centre" else td) if ids is None else ids
        for y in (2021, 2022):
            keep = {p for p in allids if int(Bi.year[p]) == y}
            rows.append(one(B, E, EL, f"{row}_sel{y}", sample, kk, keep, lab, selection_year=str(y), cohort_role=ROLE[y], group=desc))
    out["rows"] = {r["row"]: r for r in rows}
    S = A.imputation_setup(B, E, "elec", **A.PRIMARY_IMP); Ts = S["T"]; u_en = Ts.use_class.map({HC: "health_centre", DC: "daycare"})
    ct = pd.crosstab(Ts.year.astype(int), u_en); ours = {str(y): {u: int(ct.loc[y, u]) if u in ct.columns else 0 for u in ["health_centre", "daycare"]} for y in ct.index}
    k21 = KEEI["sample_by_selection_year"]["2021"]
    cy = pd.crosstab(Ts.year.astype(int), (Ts.end_ym // 100).astype(int))
    out["selection_overlap"] = dict(
        ours_treated_by_selection_year_and_use=ours, ours_n_treated_sample=int(len(Ts)),
        keei_by_selection_year_and_use={y: {u: v[u] for u in ["health_centre", "daycare"]} for y, v in KEEI["sample_by_selection_year"].items()},
        common_selection_year=2021,
        max_shared_2021={u: int(min(ours.get("2021", {}).get(u, 0), k21[u])) for u in ["health_centre", "daycare"]},
        ours_selection_2020=0, keei_selection_2022=0,
        ours_completion_year_by_selection_year={str(y): {str(c): int(cy.loc[y, c]) for c in cy.columns if cy.loc[y, c] > 0} for y in cy.index},
        note="KEEI: selections of 2020-21; ours: selections of 2021-22; the 2021 selection is common. max_shared_2021 = minimum of the two counts per use, "
             "an upper limit; building-level overlap cannot be checked (the KEEI list is not public)")
    gas = None
    gf = RES / "a_gas_heated_daycares.json"
    if gf.exists():
        gj = json.loads(gf.read_text(encoding="utf-8")).get("both_carriers_same_months", {}).get("gas")
        if gj: gas = dict(sample="daycares with both carriers observed in the same months", pct=gj.get("pct"), pct_ci=gj.get("pct_ci"), n_treated=gj.get("n_treated"),
                          source="results/a_gas_heated_daycares.json")
    out["gas"] = dict(keei="not significant", ours=gas)
    pd.DataFrame(rows).to_csv(RES / "a_keei_compare.csv", index=False)
    (RES / "a_keei_compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(out["selection_overlap"], ensure_ascii=False))

if __name__ == "__main__":
    main()
