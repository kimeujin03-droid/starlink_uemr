# Four-flux and ZA-sweep experiment results

Date: 2026-07-14

This note records the experiments that were actually run after switching the full-catalog coverage grid to
`S_ref = {10, 30, 300, 1000} Jy` and adding the current-operator synthetic zenith-angle sweep.

## 1. Four-flux full-catalog coverage grid

Run command:

```powershell
python scripts/run_coverage_grid.py --config configs/coverage_robustness_all_tle.yaml --pathb-config configs/pathB_jan2026_main.yaml --out coverage_robustness_trials_fourflux.csv --null-dir coverage_null_global_stats_fourflux --resume
```

Completed successfully.

- Rows: 1296
- Paired beam rows: 648 pairs
- Nulls in main grid: 100
- Output: `coverage_robustness_trials_fourflux.csv`
- Metadata: `coverage_robustness_trials_fourflux.meta.json`
- Null directory: `coverage_null_global_stats_fourflux/`
- Runtime: 2991.4 s

Topline main-grid counts:

| Metric | Count |
|---|---:|
| Strict local rows | 30 |
| Strict integrated rows | 74 |
| Local physical rows, `B_rel > 1e-3` | 11 |
| Integrated physical rows, `B_rel > 1e-3` | 57 |
| Beam-robust local physical pairs, `B_rel > 1e-3` | 4 |
| Frozen-only local physical pairs, `B_rel > 1e-3` | 1 |
| Full-only local physical pairs, `B_rel > 1e-3` | 2 |
| Beam-robust integrated physical pairs, `B_rel > 1e-3` | 10 |
| Frozen-only integrated physical pairs, `B_rel > 1e-3` | 27 |
| Full-only integrated physical pairs, `B_rel > 1e-3` | 10 |

By flux tier:

| `S_ref` Jy | Rows | Strict local | Strict integrated | Local physical, `1e-3` | Integrated physical, `1e-3` | Local physical, `1e-2` | Integrated physical, `1e-2` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 324 | 9 | 18 | 0 | 6 | 0 | 0 |
| 30 | 324 | 8 | 17 | 2 | 12 | 0 | 2 |
| 300 | 324 | 6 | 16 | 4 | 16 | 2 | 10 |
| 1000 | 324 | 7 | 23 | 5 | 23 | 5 | 23 |

Bias-floor sensitivity:

| `B_floor` | Local physical | Integrated physical |
|---:|---:|---:|
| `1e-8` | 30 | 74 |
| `1e-7` | 30 | 74 |
| `1e-6` | 30 | 74 |
| `1e-5` | 28 | 74 |
| `1e-4` | 28 | 74 |
| `1e-3` | 11 | 57 |
| `1e-2` | 7 | 35 |
| `1e-1` | 1 | 13 |

By morphology at `B_rel > 1e-3`:

| Morphology | Local physical | Integrated physical |
|---|---:|---:|
| `smooth` | 3 | 16 |
| `lines` | 2 | 23 |
| `khz_comb` | 6 | 18 |

## 2. Four-flux ANOVA

Run command:

```powershell
python scripts/run_coverage_anova_typeII.py --trials coverage_robustness_trials_fourflux.csv --out fourflux_summary/coverage_anova_typeII_fourflux.csv
```

Completed successfully.

Key Type-II results for `log10(B_rel)`:

| Term | Partial eta squared | p-value |
|---|---:|---:|
| Beam model | 0.005780 | 0.00747 |
| Morphology | 0.001550 | 0.3837 |
| Flux tier | 0.695379 | `3.9e-318` |
| Multiplicity | 0.217617 | `7.46e-68` |
| LST stratum | 0.308760 | `9.35e-100` |
| Beam x morphology | 0.000093 | 0.9443 |
| Baseline length | 0.092973 | `5e-28` |

Model diagnostics:

- `R^2 = 0.7571`
- adjusted `R^2 = 0.7547`
- descriptive log-flux slope: 1.042 dex `B_rel` per dex flux
- 48 of 1296 rows had non-positive `B_rel` and were dropped from the log-response ANOVA.

## 3. Four-flux N=1000 tail refinement

### Local-tail selection

Run command:

```powershell
python scripts/run_coverage_tail_refined_near_threshold.py --selection outputs/lst_bin_selection.csv --trials coverage_robustness_trials_fourflux.csv --config configs/coverage_robustness_all_tle.yaml --pathb-config configs/pathB_jan2026_main.yaml --out coverage_tail_refined_fourflux_local.csv --summary-out coverage_tail_refined_fourflux_local_summary.csv --fig-dir fourflux_tail_profile_figures_local --n-null 1000 --selection-mode local --pte-cut 0.03 --z-cut 3.0 --bias-floor 1e-3 --expand-paired-beams
```

Completed successfully.

- Seed cases before paired expansion: 60
- Refined cases after paired expansion: 90
- Refined local `PTE < 0.01`: 21
- Refined local physical, `B_rel > 1e-3`: 17
- Refined local physical, `B_rel > 1e-2`: 9
- Refined local beam-robust: 10
- Full-PolyBeam local strict: 8
- Frozen-only local strict: 4

Integrated branch values inside this local-tail selection:

- Refined integrated `PTE < 0.01`: 13
- Refined integrated physical, `B_rel > 1e-3`: 13
- Refined integrated physical, `B_rel > 1e-2`: 6
- Refined integrated beam-robust: 0

### Integrated-tail selection

Run command:

```powershell
python scripts/run_coverage_tail_refined_near_threshold.py --selection outputs/lst_bin_selection.csv --trials coverage_robustness_trials_fourflux.csv --config configs/coverage_robustness_all_tle.yaml --pathb-config configs/pathB_jan2026_main.yaml --out coverage_tail_refined_fourflux_absint.csv --summary-out coverage_tail_refined_fourflux_absint_summary.csv --fig-dir fourflux_tail_profile_figures_absint --n-null 1000 --selection-mode absint --pte-cut 0.03 --bias-floor 1e-3 --expand-paired-beams
```

Completed successfully.

- Seed cases before paired expansion: 126
- Refined cases after paired expansion: 184
- Refined integrated `PTE < 0.01`: 62
- Refined integrated physical, `B_rel > 1e-3`: 61
- Refined integrated physical, `B_rel > 1e-2`: 40
- Refined integrated beam-robust: 22
- Full-PolyBeam integrated strict: 27
- Frozen-only integrated strict: 23

Local branch values inside this integrated-tail selection:

- Refined local `PTE < 0.01`: 4
- Refined local physical, `B_rel > 1e-3`: 3
- Refined local physical, `B_rel > 1e-2`: 1
- Refined local beam-robust: 0

## 4. Synthetic ZA sweep under the current PTE/B_rel hierarchy

New script:

```text
scripts/run_synthetic_za_sweep_current_operator.py
```

Run command:

```powershell
python scripts/run_synthetic_za_sweep_current_operator.py --n-null 100
```

Completed successfully.

Design actually run:

- `S_ref = 300 Jy`
- Altitude: 550 km
- Azimuth at transit: 90 deg
- Zenith-angle peaks: 5, 15, 30, 45, 60, 70, 78 deg
- Synthetic east-west baselines: 14.6, 140.4, 207.3 m
- Background crops: 5 deterministic selected LST-bin crops with `flag_fraction < 0.5`
- Beam conditions: full-chromatic PolyBeam and `B=1` geometry-only diagnostic
- Morphology: smooth broadband
- Nulls per case: 100
- Total cases: 210
- Output: `za_sweep_current_operator.csv`
- Summary: `za_sweep_current_operator_summary.csv`
- Metadata: `za_sweep_current_operator.meta.json`
- Runtime: 421.2 s

Topline ZA results:

| Beam condition | Cases | Local physical, `B_rel > 1e-3` | Integrated physical, `B_rel > 1e-3` | `B_rel > 1e-2` |
|---|---:|---:|---:|---:|
| Full PolyBeam | 105 | 0 | 0 | 24 |
| `B=1` diagnostic | 105 | 2 | 0 | 103 |

By beam and baseline:

| Beam condition | Baseline m | Local physical, `B_rel > 1e-3` | Integrated physical, `B_rel > 1e-3` |
|---|---:|---:|---:|
| Full PolyBeam | 14.6 | 0 | 0 |
| Full PolyBeam | 140.4 | 0 | 0 |
| Full PolyBeam | 207.3 | 0 | 0 |
| `B=1` diagnostic | 14.6 | 1 | 0 |
| `B=1` diagnostic | 140.4 | 1 | 0 |
| `B=1` diagnostic | 207.3 | 0 | 0 |

The paired beam-robust integrated count for the ZA sweep is 0.

The smallest integrated PTE in the full-PolyBeam ZA sweep was at `za=78 deg`, `baseline=140.4 m`, one background crop:

- `PTE_global_absint = 0.009901`
- `PTE_global_max = 0.920792`
- `B_rel = 3.9e-5`
- It is not a physical integrated candidate because it fails the `B_rel > 1e-3` gate.

## 5. Candidate-level N=1000 audit

The candidate-level audit was rerun against `coverage_robustness_trials_fourflux.csv`, not the old 648-row trial table.

Run command:

```powershell
python scripts/run_full_catalog_physical_candidate_audit.py --trials coverage_robustness_trials_fourflux.csv --out full_catalog_physical_candidate_audit_fourflux.csv --summary-out full_catalog_physical_candidate_audit_fourflux_summary.csv --n-null 1000
```

Completed successfully.

Summary:

| Metric | Count |
|---|---:|
| Candidate-audit cases | 7 |
| Beam-sensitive candidates | 0 |
| Beam-robust contamination candidates | 0 |
| Window-integrated candidates | 6 |
| Relative-bias candidates | 0 |
| Local-only QA candidates | 1 |
| TLE records loaded | 6364 |

The seven audited cases were the `B_rel > 1e-2` local-physical candidates from the four-flux main grid. After N=1000 paired-beam reevaluation, none are beam-robust contamination candidates. Six remain local-significant but integrated-branch non-significant, and one drops to local-only QA because its N=1000 local PTE is `0.01199`.

## 6. Candidate calibration-residual audit

The targeted calibration-residual audit was rerun using the four-flux candidate audit as input.

Run command:

```powershell
python scripts/run_targeted_candidate_calibration_audit.py --candidate-audit full_catalog_physical_candidate_audit_fourflux.csv --trials coverage_robustness_trials_fourflux.csv --out targeted_candidate_calibration_audit_fourflux.csv --summary-out targeted_candidate_calibration_audit_fourflux_summary.csv --n-null 1000
```

Completed successfully.

Design:

- Candidate cases: 7
- Nulls per candidate/residual condition: 1000
- Residual models: `white`, `smooth`
- `sigma_cal`: 0, `1e-4`, `1e-3`, `1e-2`
- Smooth-residual frequency scales: 0.5, 1.0, 5.0 MHz

Result:

- Every residual condition retained the same class split: 1 local-only QA candidate and 6 window-integrated candidates.
- No residual condition promoted any case to beam-sensitive or beam-robust contamination class.
- The calibration residual audit therefore does not change the four-flux candidate-level conclusion.

## 7. Generated outputs

Core CSV outputs:

- `coverage_robustness_trials_fourflux.csv`
- `coverage_robustness_trials_fourflux.meta.json`
- `coverage_tail_refined_fourflux_local.csv`
- `coverage_tail_refined_fourflux_local_summary.csv`
- `coverage_tail_refined_fourflux_absint.csv`
- `coverage_tail_refined_fourflux_absint_summary.csv`
- `za_sweep_current_operator.csv`
- `za_sweep_current_operator_summary.csv`
- `za_sweep_current_operator.meta.json`
- `full_catalog_physical_candidate_audit_fourflux.csv`
- `full_catalog_physical_candidate_audit_fourflux_summary.csv`
- `targeted_candidate_calibration_audit_fourflux.csv`
- `targeted_candidate_calibration_audit_fourflux_summary.csv`

Summary outputs:

- `fourflux_summary/fourflux_topline_summary.csv`
- `fourflux_summary/coverage_candidate_counts_by_floor_fourflux.csv`
- `fourflux_summary/coverage_summary_by_flux_fourflux.csv`
- `fourflux_summary/polybeam_pair_audit_fourflux.csv`
- `fourflux_summary/coverage_anova_typeII_fourflux.csv`

Figures already generated:

- `fourflux_figures/R1_lst_selection_map.png`
- `fourflux_figures/R2_z_vs_pte_global.png`
- `fourflux_figures/R3_bias_floor_sensitivity.png`
- `fourflux_figures/R4_null_mad_diagnostic.png`
- `fourflux_tail_profile_figures_local/`
- `fourflux_tail_profile_figures_absint/`

## 8. Interpretation guardrails

- The old 648-row result must not be described as a four-flux result.
- The old 210-case ZA sweep based on null-p95 excess was not reused for numerical counts.
- The new ZA sweep was rerun under the current `PTE_global,max`, `PTE_global,absint`, and `B_rel` hierarchy.
- The `B=1` ZA condition is only a geometry-only diagnostic comparator and should not be used as a physical candidate gate.
- The N=1000 tail refinement changes the candidate counts relative to the N=100 main grid and should be used for tail statements.
- Candidate-level calibration residual auditing has been rerun for the four-flux local-physical candidate set.
- Multiplicity bootstrap and any other downstream manuscript tables that depend on the old 648-row grid still need regeneration before they are used as four-flux results.
