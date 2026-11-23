"""Appendix D tables from the result files."""
import json, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent; RES = HERE.parent / "results"
def m(x, d=1):
    v = round(100 * float(x), d); s = f"{v:.{d}f}"
    if s.startswith("-") and float(s) == 0: s = s[1:]
    return f"${s}$" if s.startswith("-") else s
def ci(lo, hi): return f"{m(lo)}, {m(hi)}"
def mm(v, d=1):
    v = round(float(v), d); s = f"{v:.{d}f}"
    if s.startswith("-") and float(s) == 0: s = s[1:]
    return f"${s}$" if s.startswith("-") else s
G7 = {(r.get("spec"), r.get("carrier")): r for r in json.loads((RES / "G7_imputation.json").read_text())}
h = G7[("imputation_health_only", "elec")]

S = pd.read_csv(RES / "r3_benchmark_summary.csv"); S = S[S.subset.str.startswith("reps 0..4")].set_index("estimator")
V = pd.read_csv(RES / "v6_benchmark_table.csv").set_index("estimator")
rows = [("Before--after", "naive", "no"), ("Degree-day baseline", "dd", "no"), ("Gradient boosting$^{a}$", "gb", "no"), ("Synthetic DiD$^{a}$", "sdid", "yes"),
        ("Stacked DiD, type $\\times$ month", "did_usefe", "yes"), ("Stacked DiD, province $\\times$ month", "did", "yes"), ("Imputation (original specification)", "imp", "yes"),
        ("Imputation with degree-day slopes$^{a}$", "mvi", "yes")]
t1 = []
for lab, k, cg in rows:
    if k in S.index:
        r = S.loc[k]; b, lo, hi, rm = r.bias_pct, r.mc_lo_pct, r.mc_hi_pct, r.rmse_log
    else:
        r = V.loc[k]; se = r.mc_se; b = np.exp(r.bias_log) - 1; lo = np.exp(r.bias_log - 1.96 * se) - 1; hi = np.exp(r.bias_log + 1.96 * se) - 1; rm = r.rmse_log
    v = V.loc[k]
    t1.append(f"{lab} & {cg} & {m(b)} & {ci(lo, hi)} & {rm:.3f} & {v.size_tau0:.2f} & {v.power_minus10:.2f}\\\\")
n500 = int(S.loc["imp", "reps"]); npseudo = float(V.loc["imp", "mean_n_pseudo"])
D1 = r"""\begin{table}[h]
\caption{Benchmark on non-retrofitted programme buildings, electricity. Bias: estimated change at a true effect of zero, mean over %d replications with its 95\%% Monte Carlo interval ($^{a}$100 replications). Rejects: share of the 100 replications of the earlier run whose 95\%% interval excludes zero, at a true effect of zero and of $-10\%%$ (on average %.1f pseudo-retrofitted buildings).}\label{tab:bench}
\centering\footnotesize
%% src: results/r3_benchmark_summary.csv (subset reps 0..%d: bias_pct, mc_lo_pct, mc_hi_pct, rmse_log); results/v6_benchmark_table.csv (gb, sdid, mvi: bias_log, mc_se, rmse_log; all rows: size_tau0, power_minus10, mean_n_pseudo)
\resizebox{\textwidth}{!}{\begin{tabular}{@{}lcrrrrr@{}}
\toprule
 & Comparison & Bias & Monte Carlo & RMSE & \multicolumn{2}{c}{Rejects at true effect}\\
Method & group & (\%%) & 95\%% interval & (log) & 0 & $-10\%%$\\
\midrule
""" % (n500, npseudo, n500 - 1) + "\n".join(t1) + r"""
\bottomrule
\end{tabular}}
\end{table}
\emph{Before--after} compares the same calendar months before and after the works; the \emph{degree-day baseline} and \emph{gradient boosting} predict post-works use from a model of the building's pre-works use and the weather. The high rejection rates of the first three methods at $-10\%$ reflect their bias, not power.
"""

RI = pd.read_csv(RES / "a_robust_imp.csv"); ri = lambda smp, sp: RI[(RI["sample"] == smp) & (RI.spec == sp)]
G2 = pd.read_csv(RES / "G2_did.csv"); pl = G2[(G2.spec == "originally_planned_feasible") & (G2.carrier == "elec") & (G2.term == "post")].iloc[0]
JC = pd.read_csv(RES / "a_jump_pre.csv")
def cell(smp, sp, show_n=False):
    d = ri(smp, sp)
    if not len(d): return ""
    d = d.iloc[0]; s = f"{m(d.pct)} ({ci(d.pct_lo, d.pct_hi)})"
    return s + (f" [{int(d.n_treated)}]" if show_n else "")
grid = [("Original specification (within type, grade A)", "primary", False), ("Link grades A and B", "grades_A+B", False), ("Link grades as first assigned", "original_link_grades", False),
        ("No renewable-energy entry", "drop_renewable", True), ("Months $-3..-1$ and $+1..+3$ dropped", "donut3", False), ("Strict 12+12 windows (planned rule)", "strict_12_12", True),
        ("Fit window $s-15..s-4$", "fit_window_s15_s4_calendar_matched", False), ("Anticipation months kept in the fit", "no_anticipation_holdout", False),
        ("Support: $\\ge$10 comparison buildings per cell", "support_ge10", False), ("Stable meter count only", "stable_meter_count", False),
        ("Outlier filter, all months", "drop_jumpy", True), ("Pre-period heating class: electric (exploratory)", "heat_elec", True),
        ("Pre-period heating class: unmetered fuel (exploratory)", "heat_other", True), ("Links with a construction dip only (exploratory)", "signature_confirmed", True)]
t2 = [f"{lab} & {cell('health_centre', sp, n)} & {cell('pooled', sp, n)}\\\\" for lab, sp, n in grid]
jp = JC[JC["filter"] == "pre_months_only"].set_index("sample")
t2.insert(11, "Outlier filter, pre-works months only & " + " & ".join(f"{m(jp.loc[s, 'pct'])} ({ci(jp.loc[s, 'pct_lo'], jp.loc[s, 'pct_hi'])}) [{int(jp.loc[s, 'n_treated'])}]" if s in jp.index else "" for s in ("health_centre", "pooled")) + "\\\\")
t2.insert(1, f"Feasible version of the planned specification$^{{\\dagger}}$ & & {m(pl.pct_cohort_w if 'pct_cohort_w' in pl and pd.notna(pl.pct_cohort_w) else pl.pct_effect)} ({ci(pl.pct_cohort_w_lo, pl.pct_cohort_w_hi) if 'pct_cohort_w_lo' in pl and pd.notna(pl.pct_cohort_w_lo) else ci(pl.pct_lo, pl.pct_hi)})\\\\")
D2 = r"""\begin{table}[h]
\caption{Robustness of the original specification (imputation estimator, electricity; \% with 95\% interval, building bootstrap; in brackets: retrofitted buildings where the sample changes). $^{\dagger}$Stacked DiD, pooled sample: same-type 2024--25 selections as controls, province $\times$ month effects.}\label{tab:grid}
\centering\footnotesize
% src: results/a_robust_imp.csv (sample health_centre, pooled: pct, pct_lo, pct_hi, n_treated); results/G2_did.csv (originally_planned_feasible); results/a_jump_pre.csv (pre_months_only)
\resizebox{\textwidth}{!}{\begin{tabular}{@{}lrr@{}}
\toprule
Variant & Health posts & Both types pooled\\
\midrule
""" + "\n".join(t2) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""

MV = pd.read_csv(RES / "a_r6_mvi_att.csv"); DR = pd.read_csv(RES / "a_r6_did_real.csv"); V3 = pd.read_csv(RES / "v3_table.csv"); CS = pd.read_csv(RES / "a_cohort_split.csv")
MS = pd.read_csv(RES / "v5_mislink_sim.csv")
def row(lab, pct, lo, hi, nt, nc, p=""):
    return f"{lab} & {m(pct)} & {ci(lo, hi)} & {int(nt)}/{int(nc)} & {p}\\\\"
t3 = []
c = ri("health_centre", "controls_2024_25_only").iloc[0]; t3.append(row("Controls: 2024--25 selections only", c.pct, c.pct_lo, c.pct_hi, c.n_treated, c.n_control, f"{c.pretrend_p:.3f}"))
c = ri("health_centre", "no_dedup").iloc[0]; t3.append(row("Project records as stored (duplicates kept)", c.pct, c.pct_lo, c.pct_hi, c.n_treated, c.n_control, f"{c.pretrend_p:.3f}"))
c = CS[(CS["sample"] == "health_centre") & (CS.spec == "controls_pre_202311")]
if len(c): c = c.iloc[0]; t3.append(row("Controls observed before Nov 2023 only", c.pct, c.pct_lo, c.pct_hi, c.n_treated, c.n_control))
c = V3[(V3["sample"] == "health_centre") & (V3.estimator == "stacked") & (V3.weights == "B1")].iloc[0]; t3.append(row("Stacked, comparison buildings reweighted", c.pct, c.pct_lo, c.pct_hi, c.n_treated, c.n_control))
c = DR[(DR["sample"] == "health_centre") & (DR.spec == "did_province_month")].iloc[0]; t3.append(row("Stacked, province-by-month effects", c.pct, c.pct_lo, c.pct_hi, c.n_treated, c.n_control, f"{c.pretrend_p_s04_definition:.3f}"))
mv = MV[MV["sample"] == "health_centre"].set_index("variant")
for key, lab in (("imp_s24", "Imputation, fit from month $-24$$^{b}$"), ("mvi", "\\quad same, building-specific degree-day slopes$^{b}$")):
    if key in mv.index:
        c = mv.loc[key]; t3.append(row(lab, c.pct, c.pct_lo, c.pct_hi, c.n_treated, c.n_control, f"{c.pretrend_p:.3f} ({c.pretrend_p_two_leads_r5:.3f})"))
for pi in (0.1, 0.3):
    c = MS[np.isclose(MS.pi, pi)].iloc[-1]; t3.append(f"{int(pi * 100)}\\% of retrofitted meters mislinked$^{{a}}$ & {mm(c.att_pct_median)} & {mm(c.att_pct_p2_5)}, {mm(c.att_pct_p97_5)} & 310/241 & \\\\")
D3 = r"""\begin{table}[h]
\caption{Further specifications of the health-post estimate (electricity, \%; building bootstrap). $p$: pre-trend test of the row's specification. $^{a}$Median and percentile range over 200 replacement draws. $^{b}$Fit window from month $-24$, tested on six three-month periods; in parentheses the test on the two periods of the original window.}\label{tab:more}
\centering\footnotesize
% src: results/a_robust_imp.csv (controls_2024_25_only, no_dedup); results/a_cohort_split.csv (controls_pre_202311); results/v3_table.csv (stacked, B1); results/a_r6_did_real.csv (did_province_month); results/a_r6_mvi_att.csv (imp_s24, mvi); results/v5_mislink_sim.csv (pi = 0.1, 0.3)
\resizebox{\textwidth}{!}{\begin{tabular}{@{}lrrrr@{}}
\toprule
Specification & Change & 95\% CI & $N_R$/$N_C$ & $p$\\
\midrule
""" + "\n".join(t3) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""

se = h["season"]
D4 = r"""\begin{table}[h]
\caption{Original specification by season of the post-completion month and as the median over buildings (health posts, electricity, \%; building bootstrap). Declared before the final run as exploratory analyses.}\label{tab:season}
\centering\footnotesize
% src: results/G7_imputation.json (imputation_health_only, elec: season.*.pct, pct_ci, n_rows; median_bld_pct, median_bld_pct_ci)
\begin{tabular}{@{}lrr@{}}
\toprule
 & Change & 95\% CI\\
\midrule
""" + f"""Heating months (November--March) & {m(se['heating_nov_mar']['pct'])} & {ci(*se['heating_nov_mar']['pct_ci'])}\\\\
Cooling months (June--September) & {m(se['cooling_jun_sep']['pct'])} & {ci(*se['cooling_jun_sep']['pct_ci'])}\\\\
Shoulder months (April, May, October) & {m(se['shoulder_apr_may_oct']['pct'])} & {ci(*se['shoulder_apr_may_oct']['pct_ci'])}\\\\
Median over buildings & {m(h['median_bld_pct'])} & {ci(*h['median_bld_pct_ci'])}\\\\
""" + r"""\bottomrule
\end{tabular}
\end{table}
"""

c = CS[CS["sample"] == "health_centre"]; prim = c[c.spec == "primary"].set_index("level")
order = [l for l in ["2021H2", "2022H1", "2022H2", "2023H1", "2023H2", "2024H1", "2024H2"] if l in prim.index]
coh = "; ".join(f"{l} {mm(100 * prim.loc[l, 'pct'])} ({mm(100 * prim.loc[l, 'pct_lo'])}, {mm(100 * prim.loc[l, 'pct_hi'])}; {int(prim.loc[l, 'n_treated'])})" for l in order)
lme = c[c.spec.isin(["early_2021H2_2022H2", "late_2023H2_2024H1", "late_minus_early"])].set_index("spec")
PL = pd.read_csv(RES / "a_r5_placebo.csv"); un = PL[(PL["sample"] == "health_centre") & (PL.design == "union")].iloc[0]
uci = [float(x) for x in str(un.ci_pct).strip("[]").replace("np.float64(", "").replace(")", "").split(",")]
CY = pd.read_csv(RES / "r3_calendar_by_selection_year.csv")
def cy(y, q): d = CY[(CY.selection_year == y) & (CY.quarter == q)]; return mm(100 * d.pct.iloc[0]) if len(d) and d.n_buildings.iloc[0] >= 20 else "--"
JP = json.loads((RES / "a_jump_pre.json").read_text())["samples"]["health_centre"]; tf, cf = JP["treated_flags"], JP.get("control_flags", {})
D5 = r"""
\paragraph{Completion cohorts} Original specification by half-year of completion (\% with 95\% interval; buildings): """ + coh + r""". """ + (
    f"The estimate is larger for late completions ({mm(100 * lme.loc['early_2021H2_2022H2', 'pct'])}\\% for 2021H2--2022H2 against {mm(100 * lme.loc['late_2023H2_2024H1', 'pct'])}\\% for 2023H2--2024H1), but the difference, {mm(100 * lme.loc['late_minus_early', 'pct'])} points ({mm(100 * lme.loc['late_minus_early', 'pct_lo'])} to {mm(100 * lme.loc['late_minus_early', 'pct_hi'])}), cannot be distinguished from zero." if len(lme) == 3 else "") + r""" % src: results/a_cohort_split.csv (health_centre)

\paragraph{Combined placebo} Combining both placebos gives """ + f"{mm(un.change_pct)}\\% ({mm(uci[0])} to {mm(uci[1])}) with a smallest detectable effect of {un.detectable_pct:.1f}\\%" + r"""; its gain in power comes from the pre-period placebo, whose evaluated months are fitted months of the original specification. % src: results/a_r5_placebo.csv (health_centre, union)

\paragraph{Waiting buildings by selection year} The calendar profile of Section~\ref{sec:res-fail}, estimated separately for comparison health posts selected in 2023, 2024 and 2025 (\% against July 2020 to June 2021): 2022Q4 """ + f"{cy(2023, '2022Q4')}, {cy(2024, '2022Q4')}, {cy(2025, '2022Q4')}; 2023Q2 {cy(2023, '2023Q2')}, {cy(2024, '2023Q2')}, {cy(2025, '2023Q2')}; 2024Q1 {cy(2023, '2024Q1')}, {cy(2024, '2024Q1')}, {cy(2025, '2024Q1')}; 2024Q4 {cy(2023, '2024Q4')}, {cy(2024, '2024Q4')}, {cy(2025, '2024Q4')}; 2025Q1 {cy(2023, '2025Q1')}, {cy(2024, '2025Q1')}, {cy(2025, '2025Q1')}" + r""" (--: fewer than 20 buildings still waiting). All three groups fell in 2022--23; the 2025 selections, the only group left from 2024, kept falling. This profile was computed after the revision results had been seen. % src: results/r3_calendar_by_selection_year.csv

\paragraph{Outlier filter} The planned filter flags """ + f"{tf['n_flagged_original']} of the {tf['n_headline_treated']}" + r""" retrofitted health posts. Of the months it flags among retrofitted buildings, """ + f"{tf['offending_months_by_phase'].get('pre', '?')} lie before the works, {tf['offending_months_by_phase'].get('constr', '?')} during them and {tf['offending_months_by_phase'].get('post', '?')} after completion, {tf.get('post_offending_months_low', '?')} of the latter being low months." + r""" % src: results/a_jump_pre.json (samples.health_centre.treated_flags)
"""
(HERE / "appendix_D_r3.tex").write_text("\n".join([D1, D2, D3, D4, D5]), encoding="utf-8")
print((HERE / "appendix_D_r3.tex").read_text()[:600]); print("written")
