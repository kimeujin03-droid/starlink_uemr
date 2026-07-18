# Beam-robust window-integrated candidate set: confirmation + calibration-residual audit

Date: 2026-07-18

This note (1) confirms the candidate set of four-flux `N_null=1000` beam-robust
window-integrated (`PTE_global_absint`) pairs by filtering the existing tail
re-evaluation output, and (2) runs a calibration-residual stress test on the
full set under the same design used for the earlier local-floor candidate
audit (`2026-07-14_fourflux_za_sweep_experiment_results.md`, section 6).

## 1. Candidate set confirmation (filter only, no new integration run)

Source: `coverage_tail_refined_fourflux_absint.csv` (already-existing
`--selection-mode absint`, `N_null=1000` tail re-evaluation output; 184 rows,
`n_seed_cases_before_pair_expansion=126`).

A `(frozen_polybeam, full_polybeam)` pair is "beam-robust integrated" at a
given `B_rel` floor if **both** beam models independently satisfy
`PTE_global_absint < 0.01` and `relative_abs_bias (B_rel) > floor`, matched on
`(baseline_id, lst_stratum, lst_bin_id, morphology, flux_jy, multiplicity)`.
This is the definition used by `scripts/run_coverage_tail_refined_near_threshold.py`
(the `retained_absint_floor_*` / beam-robust counters), independent of the
stricter near-bit-identical `classify_case` "beam-robust contamination"
definition used in the local-floor candidate audit.

| `B_rel` floor | Beam-robust rows (script's `n_refined_absint_beam_robust`) | Unique pairs |
|---:|---:|---:|
| `> 1e-3` (exploratory) | 22 | 11 |
| `> 1e-2` | 18 | 9 |

The `> 1e-3` count reproduces the documented value
(`coverage_tail_refined_fourflux_absint_summary.csv`:
`n_refined_absint_beam_robust = 22`). All 22 rows / 11 pairs were exported to
`outputs/beam_robust_absint_pairs_fourflux_1e3.csv` and used as the audit
target set, per the instruction to cover the full exploratory set rather than
a subsample.

The 11 pairs span 2 geometry/LST cells:

- `23_80 / typical / lst_bin 44`: 3 pairs (`smooth`, `lines`, `khz_comb`; all `1000 Jy`, `multi`)
- `4_196 / stress / lst_bin 0`: 8 pairs (`smooth`/`lines`/`khz_comb` x `{30, 300, 1000} Jy`, `multi`; the `khz_comb` tier has no `30 Jy` member)

9 of the 11 pairs also clear the stricter `B_rel > 1e-2` floor on both beam
models; the remaining 2 (pair 4: `4_196/smooth/30 Jy`, pair 7:
`4_196/lines/30 Jy`) sit between the `1e-3` and `1e-2` floors
(baseline `B_rel` in `[0.0019, 0.0024]`).

## 2. Calibration-residual audit design

New script: `scripts/run_beam_robust_residual_audit.py`.

Matches the design used for the local-floor candidate residual audit
(`targeted_candidate_calibration_audit_plan.md`), applied to the 11
beam-robust integrated pairs instead of the 7 local-floor candidates:

- Beam condition: paired `frozen_polybeam` / `full_polybeam`, same satellite
  selection, geometry, and TLE set as the source tail-refined run.
- `N_null = 1000` per beam per residual condition.
- `sigma_cal in {0, 1e-4, 1e-3, 1e-2}`.
- Residual models: `white` (uncorrelated) and `smooth` (chromatic, Gaussian
  spectral kernel) with `ell_nu in {0.5, 1.0, 5.0} MHz`.
- Calibration residual seed: `10` (matches the local-floor audit's primary
  seed).
- Classification per (pair, residual condition): a pair is **beam-robust
  retained** if both `frozen_polybeam` and `full_polybeam` independently keep
  `PTE_global_absint < 0.01` and `B_rel > 1e-3` under the perturbed
  background; otherwise `frozen-only`, `full-only`, or `dropped`.

Total matrix: 11 pairs x 16 residual conditions (4 white + 4 sigma x 3 `ell_nu`
smooth) x 2 beam models = 352 `N_null=1000` evaluations.

### Implementation note (performance)

A single `N_null=1000` evaluation with the 12-satellite `multi` stack used by
all 11 candidates costs ~66 s. Run naively (no caching, single process) the
full matrix was projected at ~6-8 hours. Two changes were applied, both
behavior-preserving:

- Satellite visibility depends only on background geometry (times,
  frequencies, baseline), not on the injected calibration residual, so it is
  built once per pair/beam and reused across all 16 residual conditions.
- `sigma_cal = 0` gives an identical zero residual regardless of
  `residual_model`/`ell_nu`, so it is computed once per pair and reused for
  the `white` and all three `smooth` `ell_nu` rows (verified byte-identical
  against a direct recompute in a correctness smoke test before the full run).
- The 11 pairs (independent work units) were farmed out across 6 worker
  processes.

Wall time for the full 352-evaluation matrix: ~83 minutes (two waves of
6 + 5 pairs, ~31-36 min per pair single-threaded).

Sigma-zero rows were cross-checked against the source tail-refined data
(`coverage_tail_refined_fourflux_absint.csv`) and reproduce it exactly, e.g.
pair 1 frozen `B_rel = 0.206251` in both files.

## 3. Result

Output: `outputs/beam_robust_residual_audit_fourflux.csv` (176 rows: 11 pairs
x 16 residual conditions), `outputs/beam_robust_residual_audit_fourflux_summary.csv`.

**All 176 (pair, residual condition) cells classify as `beam-robust
retained`.** No residual condition, at any tested `sigma_cal` or `ell_nu`,
demoted a single pair to `frozen-only`, `full-only`, or `dropped`.

| `residual_model` | `sigma_cal` | `ell_nu_mhz` | `beam-robust retained` pairs |
|---|---:|---:|---:|
| white | 0, 1e-4, 1e-3, 1e-2 | n/a | 11 / 11 (all four sigma levels) |
| smooth | 0, 1e-4, 1e-3, 1e-2 | 0.5, 1.0, 5.0 | 11 / 11 (all twelve sigma x ell_nu combinations) |

Margins to the classification gates, across the full 176-row matrix:

| Quantity | Min | Max |
|---|---:|---:|
| `PTE_global_absint` (both beams) | 0.000999 | 0.006993 (gate: `< 0.01`) |
| `B_rel` (both beams) | 0.001893 | 0.206879 (gate: `> 1e-3`) |

Even at the strongest tested stress amplitude (`sigma_cal = 1e-2`, either
residual model), `PTE_global_absint` never exceeds 0.006993 (30% below the
0.01 gate) and `B_rel` never drops below 0.001893 (still ~1.9x the `1e-3`
floor).

The two smallest-margin pairs at baseline (pair 4: `4_196/smooth/30 Jy`, pair
7: `4_196/lines/30 Jy`, baseline `B_rel` in `[0.0019, 0.0024]`) are also the
tightest under stress, but their `B_rel` only moves in the range
`[0.001893, 0.002380]` across the entire residual matrix -- it neither drops
to the `1e-3` floor nor crosses up into the `1e-2` tier. Their
`PTE_global_absint` stays at or below 0.006993 throughout.

The 9 pairs that also clear the stricter `B_rel > 1e-2` floor at baseline
retain `B_rel > 1e-2` (and `PTE_global_absint < 0.01`) on both beam models
across all 16 residual conditions as well (minimum observed `B_rel = 0.0174`
under stress, still comfortably above `1e-2`).

## 4. Interpretation

- The full exploratory-tail beam-robust window-integrated set (22 rows / 11
  pairs at `B_rel > 1e-3`) is unconditionally stable under the calibration
  residual stress family used elsewhere in this project (white + chromatic,
  `sigma_cal` up to `1e-2`, `ell_nu` from 0.5 to 5 MHz): no pair is
  reclassified away from beam-robust under any tested condition.
- This closes the reviewer question this audit was designed to preempt: the
  beam-robust integrated classification is not an artifact of the specific
  (unperturbed) calibration realization used in the main tail refinement, and
  it holds for the entire 22-row exploratory set, not just a representative
  subsample.
- The stricter `B_rel > 1e-2` sub-tier (9 of the 11 pairs) is likewise
  residual-stable, so tightening the paper's floor to `1e-2` would not change
  this robustness conclusion.
- Consistent with the earlier local-floor candidate calibration audit
  (`targeted_candidate_calibration_audit_plan.md`, `targeted_candidate_calibration_audit_fourflux_summary.csv`),
  no calibration-residual condition tested anywhere in this project has
  produced a beam-model-dependent (non-robust) flip for a physical candidate.

## 5. Generated outputs

- `outputs/beam_robust_absint_pairs_fourflux_1e3.csv` -- the 11-pair (22-row)
  candidate set with matched frozen/full seeds, derived from
  `coverage_tail_refined_fourflux_absint.csv`.
- `scripts/run_beam_robust_residual_audit.py` -- the residual-audit script.
- `outputs/beam_robust_residual_audit_fourflux.csv` -- full 176-row
  (pair x residual condition) result table.
- `outputs/beam_robust_residual_audit_fourflux_summary.csv` -- counts of
  `pair_status` by `(residual_model, sigma_cal, ell_nu_mhz)`.
