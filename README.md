# Evaluating public building retrofit programmes with open administrative data

Reproduction code for the paper *Evaluating public building retrofit programmes with open administrative data: Evidence and limits from South Korea*.

The repository holds the code that produces the paper's numbers, tables and figures, the analysis plan, the collection manifests, the aggregate result files and the project–parcel–register crosswalk. Raw data are not redistributed (see *Data*).

## Layout

| Path | Content |
|---|---|
| `src/s01`–`s03`, `s07` | data collection: project records, address → parcel → building register, monthly electricity and gas |
| `src/s04`–`s06` | panel, estimators (stacked DiD, imputation), counterfactual models, benchmark |
| `src/a_*`, `v*`, `ep_*`, `pi_*` | descriptives, robustness checks, linkage audit, agency comparison |
| `src/r3_*`, `r4*`, `x08_ep_v3.py` | main event study, sensitivity analysis, benchmark and the remaining reliability checks (`x01`, `x03` hold helpers for `x08`) |
| `paper/` | figures 1–5 and Appendix D tables from `results/` |
| `results/` | aggregate result files behind the paper; no building-level rows |
| `plan/analysis_plan.md` | analysis plan fixed before the main estimates |
| `manifests/` | SHA-256 manifests of the collected records and bulk files |

## Tables and figures

| Paper | Script | Result file |
|---|---|---|
| Fig. 1 | `a_attrition.py` | `a_attrition.csv` |
| Table 1 | `a_analysed_vs_all.py`, `ep_aggregates.py` | `a_analysed_vs_all.csv`, `ep_use_grade_counts.csv` |
| Table 2 | `a_balance_within.py`, `v3_balance.py` | `a_balance_within.csv`, `v3_smd.csv`, `v3_table.csv` |
| Table 3 | `r3_analysis.py`, `r4_analyses.py`, `s04_analysis.py`, `a_gas_daycare.py` | `r3_main.json`, `r4_analyses.json`, `G7_imputation.json`, `a_gas_heated_daycares.json` |
| Fig. 2 | `r3_analysis.py` | `r3_event_long.csv` (panel b needs building-level data) |
| Table 4 | `r3_analysis.py`, `r3_honest.py`, `r3_balance.py`, `a_r5_placebo.py`, `r3_es_variants.py` | `r3_main.json`, `r3_honestdid*.csv`, `r3_balance.json`, `a_r5_placebo.csv`, `r3_es_variants.csv` |
| Fig. 3 | `r3_honest.py` | `r3_honestdid.csv` |
| Fig. 4 | `r3_benchmark.py`, `v6_benchmark_table.py`, `x08_ep_v3.py` | `r3_benchmark_summary.csv`, `v6_benchmark_table.csv`, `x08_ep_summary.csv` |
| Fig. 5 | `r3_descriptive.py`, `r3_byyear.py` | `r3_calendar_profile.csv`, `r3_calendar_by_selection_year.csv` |
| Table 5 | `a_keei_compare.py` | `a_keei_compare.csv` |
| §4.3–4.4 | `r4b_analyses.py`, `r4c_analyses.py`, `r4d_analyses.py`, `x08_ep_v3.py`, `v5_*.py` | `r4b_analyses.json`, `r4c_mislink_es_summary.csv`, `r4d_analyses.json`, `x08_ep_summary.csv`, `v5_*.{csv,json}` |
| Appendix D | `paper/make_appD.py` | inputs listed in the script |

## Running

Python 3.13 with `requirements.txt`. The sensitivity analysis (`r3_honest*.py`) needs the `honestdid` package in a separate environment; point `HONEST_PY` to its Python. `s05_ml.py` uses a GPU by default; set `XGB_DEVICE=cpu` otherwise.

1. Service keys: get your own keys for the public data portal (building register, building energy) and the address-search service, and put them in `~/.config/gr-energy/datagokr.env` as `DATA_GO_KR_KEY=...` and `JUSO_KEY=...`. Placeholder values are enough for steps 3–4, which make no request.
2. Collect (writes `data/`):
   ```bash
   python -I src/s01_gr_sites.py 2021,2022,2023,2024,2025
   python -I src/s02_link.py 2021,2022,2023,2024,2025
   ENERGY_LATEST=202510 python -I src/s03_energy.py 0,1,5,2,4,3
   ENERGY_PIDS_FILE=manifests/d25_health_first.csv ENERGY_CARRIERS=elec ENERGY_LATEST=202510 python -I src/s03_energy.py 0,1,5,2,4,3,6
   python -I src/s07_bulk.py download && python -I src/s07_bulk.py extract
   ENERGY_LATEST=202510 python -I src/s03_energy.py 0,1,5,4,2,3 --offline
   ```
   The energy service allows about 10,000 calls per operation and day, so `s03_energy.py` is resumable and is repeated until complete. A later rebuild can differ if the services revise their data; `manifests/` records what was collected.
3. Analyse: `bash run_all.sh` (about 3 hours).
4. Without the data, `paper/make_figs_ep.py` and `paper/make_appD.py` redraw the figures and Appendix D from `results/`.

## Data

Project records, building-register records and building-energy records are public but are retrieved with the user's own service key under the portal's terms, so they are not redistributed. `results/` holds aggregate files only; `results/crosswalk_project_parcel_register.csv` links the project records of these public buildings to parcels and register records.
