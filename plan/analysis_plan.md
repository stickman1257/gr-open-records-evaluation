# Analysis plan — fixed 2026-11-13 00:42 KST, before any treated-vs-control effect estimate on the GR panel was computed
(Only the Mendeley 92-building pilot and the decision sweep had been run. SHA-256 of this file is logged in PLAN.md.)

## Samples
- Treated: GR projects selected 2021–2022 with parseable start/end dates, link grade A_cand (single main building on the parcel, register use consistent with facility type, register floor area within ±30% of program floor area), ≥12 observed months before start and ≥12 after completion (completion ≤ 2024-10; latest meter month 2025-10).
- Controls (not yet treated): GR projects selected 2024–2025 whose construction started ≥ 2024-07, same link-grade rule; only months before their own start are used.
- Construction months (start..end) are excluded for treated buildings. Months with kWh ≤ 0 are excluded (0 ≠ missing is logged).
- Primary carrier: electricity. Gas: buildings with ≥ 10 of 12 pre months observed; reported as secondary.

## Estimands and estimators
- Primary: ATT on log monthly electricity intensity, 12 months post-completion vs 12 months pre-start, stacked DiD (Cengiz et al. 2019): stacks = completion half-year cohorts; FE = building×stack and province×month×stack; SE clustered by building.
- Event study (same design, 3-month bins, reference = months −3..−1): pre-trend test = joint Wald test of pre bins.
- Secondary estimators (same buildings/windows): (a) per-building degree-day baseline (change-point grid, BIC selection), fit gate CV(RMSE) ≤ 20% on the baseline period (sensitivity 15% / none); (b) global gradient-boosting counterfactual trained on controls + treated pre months, with 200-replicate building-cluster bootstrap.
- Placebo calibration (no treatment): P1 controls' last 12 observed months; P2 treated months s−12..s−1 predicted from s−24..s−13. Reported as bias (median) and dispersion (MAE) for (a) and (b).

## Measure-level analysis
- post × measure interactions (wall, roof, window, shading, HRV, LED, HVAC replacement, boiler, PV) in the stacked DiD; Holm adjustment across the 9 terms; interpreted as conditional associations within the observed bundles.
- Realisation: per-building realised savings (kWh/m²·y) regressed on simulated Shapley contributions of envelope and systems measures (archetype matched by use × era × climate zone); heating carrier assigned from data (gas meter present → gas; else electric if pre-period winter/shoulder electricity ratio ≥ 1.3, otherwise unmetered fuel → heating terms dropped). Coefficients = realisation rates ρ_env, ρ_sys with bootstrap CIs.
- Decision impact: plug (ρ_env, ρ_sys) and its bootstrap draws into the 27×2,048 decision set; report mean/P90 regret and share of changed optima.

## Robustness (all pre-specified)
A+B link grades; drop PV projects; drop 2020-03..2021-02 (COVID) from baselines; 24-month windows; province×month vs month-only FE; use-type×month FE; exclude buildings with any monthly jump > 3× median.
