# Starlink-like UEMR QA for 21-cm Delay-Spectrum Analyses

This repository contains the scripts, configuration files, summary outputs, and figure-generation code for the coverage-robustness analysis of Starlink-like UEMR quality-assurance diagnostics in a HERA-like 21-cm delay-spectrum setting.

The active release is centered on the question of whether local-max statistical excursions survive beam/model robustness checks and whether they correspond to integrated bandpower contamination. The corrected result is conservative: full-bin satellite-exposure metadata changes 15 of 27 selected LST-stratum cells, the 648-row coarse coverage grid leaves one statistical `PTE_global,max < 0.01` trial, and the `B_rel > 1e-3` near-threshold 1000-null refinement leaves zero strict local-max candidates. The analysis reports `B_floor = 1e-2` as a tested-regime reporting gate, not as a universal physical threshold.

## Reproducibility Level

This archive supports manuscript-level numerical inspection and figure regeneration from archived CSV outputs. It does not support a complete raw-data rerun unless the external HERA-like UVH5 backgrounds, native geometry NPZ files, and TLE products are supplied at the paths described in `REPRODUCIBILITY.md`. Complete reruns should use the archived TLE catalog from the original experiment, such as `tle/starlink_jan2026_LEO_only.tle`, rather than refreshing orbital elements online.

## Active Analysis Files

Use these files for the current submitted coverage/PTE analysis:

- `configs/coverage_robustness.yaml`
- `scripts/build_lst_metadata.py`
- `scripts/select_lst_bins.py`
- `scripts/run_coverage_grid.py`
- `scripts/summarize_coverage_grid.py`
- `scripts/run_coverage_tail_refined_near_threshold.py`
- `scripts/run_coverage_tail_resolution_check.py`
- `outputs/coverage_robustness_trials.csv`
- `outputs/coverage_tail_refined_near_threshold.csv`
- `outputs/coverage_tail_refined_near_threshold_summary.csv`
- `outputs/coverage_candidate_counts_by_floor_extended.csv`
- `figures/coverage_robustness/`

Older manuscript material from a morphology matched-null draft has been moved to `archive/old_manuscripts/`, and older Path B morphology/beam release artifacts have been moved to `archive/old_pathb_release/`, so they are not confused with the active coverage/PTE release.

## Key Archived Outputs

The provided summary outputs are sufficient to inspect the manuscript-level claims without rerunning the external raw-data pipeline.

| File | Purpose |
| --- | --- |
| `outputs/lst_bin_metadata.csv` | 10-minute LST-bin metadata used for bin selection |
| `outputs/lst_bin_selection.csv` | selected quiet/typical/stress LST bins |
| `outputs/coverage_robustness_trials.csv` | 648-row reduced coverage grid, `N_null = 100` |
| `outputs/coverage_summary_by_factor.csv` | factor-level summaries from the coverage grid |
| `outputs/coverage_summary_by_baseline_lst.csv` | baseline/LST-cell summaries |
| `outputs/coverage_candidate_counts_by_floor_extended.csv` | candidate counts versus integrated-bias floor, including `B_floor = 1e-2` |
| `outputs/coverage_tail_refined_near_threshold.csv` | near-threshold refinement rows, `N_null = 1000` |
| `outputs/coverage_tail_refined_near_threshold_summary.csv` | final refined candidate counts |

Current refined summary:

- `n_refined_cases = 4`
- `n_refined_PTE_lt_001 = 0`
- `n_refined_physical_floor_1e3 = 0`
- `n_refined_physical_floor_1e2 = 0`
- `n_refined_beam_robust = 0`
- `n_full_polybeam_strict = 0`
- `n_frozen_only_strict = 0`

## Figures

The active coverage figures are:

- `figures/coverage_robustness/R1_lst_selection_map.png`
- `figures/coverage_robustness/R2_z_vs_pte_global.png`
- `figures/coverage_robustness/R3_bias_floor_sensitivity.png`
- `figures/coverage_robustness/R4_null_mad_diagnostic.png`

The corrected tail refinement leaves no strict candidates, so no strict-candidate delay-profile panels are included.

## Reproduction

Install the Python environment:

```bash
python -m pip install -r requirements.txt
```

The coarse summaries can be regenerated from the archived trial CSV:

```bash
python scripts/summarize_coverage_grid.py \
  --trials outputs/coverage_robustness_trials.csv \
  --out-dir outputs
```

Full reruns require external HERA-like visibility backgrounds and TLE products that are not redistributed here. The full-rerun commands are not expected to run out-of-the-box from this archive. See `REPRODUCIBILITY.md` for the raw-data limitation, seed policy, and rerun order.

## Repository Scope

This is the single public release repository for the active Starlink-like UEMR QA coverage analysis. Do not use older `starlink_comb` materials or archived manuscripts as the source of record for the submitted coverage/PTE claims.
