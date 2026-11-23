#!/usr/bin/env bash
# Rebuilds every result file of the paper from the collected data (data/processed). About 3 hours.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="python -I"
HONEST="${HONEST_PY:-python} -I"

$PY src/s04_analysis.py
$PY src/s05_ml.py
$PY src/s06_benchmark.py
for s in a_analysed_vs_all a_attrition a_balance_within a_baselines_real a_cohort_split a_dedup a_gas_daycare \
         a_jump_pre a_keei_compare a_program_descriptives a_r5_placebo a_r5_pretrend a_r6_did_real a_r6_mvi \
         a_robust_imp a_support_rule pi_api_probe pi_regrade_all \
         v3_balance v5_linkage_audit v5_mislink_sensitivity v6_benchmark_table ep_aggregates ep_scope; do
  $PY src/$s.py
done

$PY src/r3_analysis.py
$HONEST src/r3_honest.py
$HONEST src/r3_honest_sd_fine.py
for s in r3_es_variants r3_balance r3_benchmark r3_descriptive r3_byyear \
         r4_analyses r4b_analyses r4c_analyses r4d_analyses r4e_crosswalk x08_ep_v3; do
  $PY src/$s.py
done

$PY paper/make_appD.py
$PY paper/make_figs_ep.py
