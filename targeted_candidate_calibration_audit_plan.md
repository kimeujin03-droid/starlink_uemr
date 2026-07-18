# Targeted Candidate Calibration Audit Plan

This note narrows the remaining follow-up work to the five full-catalog physical candidates only. The full 648-row grid does not need to be rerun at this stage.

## Goal

Check whether the five full-catalog physical candidates remain stable under a small calibration-residual injection family, while keeping:

- frozen/full PolyBeam pairing
- the same candidate-level geometry and TLE selection
- the same staged QA hierarchy used in the paper

## Target set

Use only the five full-catalog physical candidates from:

- `outputs/full_catalog_physical_candidate_audit.csv`

These are the cases that already survived the initial catalog-complete tail audit and therefore matter most for the final claim.

## Core factors

Vary only the following stress-test parameters:

| Parameter | Values |
| --- | --- |
| `sigma_cal` | `0, 1e-4, 1e-3, 1e-2` |
| frequency residual model | `white`, `smooth/chromatic` |
| random seeds | `10` |
| beam condition | frozen/full PolyBeam paired condition preserved |
| Doppler mode | representative class-stable mode, or the existing `none / constant / linear` result set |

Required outputs:

- `PTE_global_max^1000`
- `B_rel`
- `PTE_global_absint^1000`
- `final class`

## Why two residual models

The residual model should not be only white noise.

### Model A: uncorrelated residual

Use a thermal-like residual field:

```math
\eta_{\rm cal}(t,\nu) \sim \mathcal{CN}(0,\sigma_{\rm cal}^2)
```

This is a simple stress test. It checks whether the candidate is sensitive to generic unstructured perturbations.

### Model B: spectrally correlated residual

Use a chromatic residual field:

```math
\eta_{\rm cal}(t,\nu) \sim \mathrm{GP}\!\left(0,\sigma_{\rm cal}^2 K_\nu\right),
```

with

```math
K_{\nu\nu'} = \exp\!\left[-\frac{(\nu-\nu')^2}{2\ell_\nu^2}\right].
```

Use one of:

- `\ell_\nu = 0.5 MHz`
- `\ell_\nu = 1 MHz`
- `\ell_\nu = 5 MHz`

This is the more important model. In real calibration problems, spectral structure matters more than white residuals because small chromatic errors can break foreground smoothness and leak power into the EoR window.

## Suggested audit table

For each of the five candidates, report:

| case | beam | morphology | flux | multiplicity | residual model | `sigma_cal` | `ell_nu` | `PTE_max^1000` | `B_rel` | `PTE_absint^1000` | final class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

If Doppler is included in this pass, keep the mode fixed to the representative class-stable setting and note it in a separate column.

## Interpretation rule

Do not promote a case to beam-robust contamination unless it passes all of the following:

1. matched-null local gate
2. relative window-bias gate
3. window-integrated PTE gate
4. paired frozen/full PolyBeam comparison
5. calibration-residual stress test

## Practical priority

Recommended execution order:

1. run the five candidates with white residuals
2. rerun the same five candidates with smooth/chromatic residuals
3. only then decide whether a broader candidate-level sensitivity sweep is needed

This keeps the remaining work tightly scoped and aligned with the manuscript's central claim.

## Execution results

The targeted audit was executed with:

- `sigma_cal = 0, 1e-4, 1e-3, 1e-2`
- residual models: `white`, `smooth`
- chromatic scales: `ell_nu = 0.5, 1.0, 5.0 MHz`
- Doppler mode: `linear`
- random seed: `10`

Output files:

- `outputs/targeted_candidate_calibration_audit.csv`
- `outputs/targeted_candidate_calibration_audit_summary.csv`

Summary of the executed matrix:

| residual model | `sigma_cal` | `ell_nu` | final class | `n_cases` |
| --- | --- | --- | --- | ---: |
| white | `0, 1e-4, 1e-3, 1e-2` | `na` | beam-sensitive candidate | 1 |
| white | `0, 1e-4, 1e-3, 1e-2` | `na` | local-only QA candidate | 1 |
| white | `0, 1e-4, 1e-3, 1e-2` | `na` | window-integrated candidate | 3 |
| smooth | `0, 1e-4, 1e-3, 1e-2` | `0.5, 1.0, 5.0 MHz` | beam-sensitive candidate | 1 |
| smooth | `0, 1e-4, 1e-3, 1e-2` | `0.5, 1.0, 5.0 MHz` | local-only QA candidate | 1 |
| smooth | `0, 1e-4, 1e-3, 1e-2` | `0.5, 1.0, 5.0 MHz` | window-integrated candidate | 3 |

Observed behavior:

- the five physical candidates retained the same final class under all tested calibration-residual settings
- the white and smooth/chromatic residual families did not promote any case to `beam-robust contamination candidate`
- the strongest effect was small movement in `PTE_global_max^1000` and `B_rel` near the threshold in the beam-sensitive and local-only cases
- the window-integrated candidates remained window-integrated across the full matrix

## Multi-seed extension (stability check)

The execution above used a single calibration-residual seed (`seed=10`). A
single realization cannot rule out that "final class unchanged" was a
coincidence of that one draw. Two follow-up checks close this gap:

**Supplementary full-matrix run at a second seed.** Rerunning the entire
`sigma_cal x residual_model x ell_nu` matrix (80 rows) at `seed=1` reproduces
the `seed=10` final class for all 5 candidates with zero mismatches across
all 80 (case, model, sigma, ell) combinations
(`outputs/multiseed_calibration_audit/seed_01.csv`).

**10-seed stability check at the strongest tested stress amplitude.**
Because the full 80-row matrix is expensive to repeat many times, the
targeted 10-seed check (`scripts/run_calibration_multiseed_audit.py`,
`outputs/calibration_multiseed_audit.csv`) restricts to the two most
informative settings — `sigma_cal=1e-2` (the strongest tested amplitude) for
both `white` and `smooth` (`ell_nu=1.0 MHz`) residuals — and draws 10
independent calibration-residual seeds for each of the 5 candidates
(satellite visibility is built once per case/beam and reused across seeds,
since it does not depend on the calibration residual applied to the
background).

Result (100 evaluations: 5 candidates x 2 residual settings x 10 seeds):

| case | residual model | classes observed over 10 seeds |
| --- | --- | --- |
| 1 | white, smooth | beam-sensitive candidate (10/10 both) |
| 2 | smooth | local-only QA candidate (10/10) |
| 2 | white | local-only QA candidate (8/10), window-integrated candidate (2/10) |
| 3 | white, smooth | window-integrated candidate (10/10 both) |
| 4 | white, smooth | window-integrated candidate (10/10 both) |
| 5 | white, smooth | window-integrated candidate (10/10 both) |

No `beam-robust contamination candidate` classification occurred in any of
the 100 evaluations.

Interpretation:

- cases 1, 3, 4, and 5 are perfectly stable across all 10 seeds at the strongest tested stress amplitude, for both residual structures.
- case 2 shows label flicker under `white` noise: its `PTE_global_max` sits right at the 0.01 boundary (0.0100–0.0230 across seeds), so 2/10 realizations dip just under 0.01 and get labeled `window-integrated candidate` instead of `local-only QA candidate`. This is a cosmetic flip between two non-candidate labels, not a substantive one: `PTE_global_absint` (~0.037–0.039) and `B_rel` (~0.0287) barely move across all 10 seeds and stay far above the 0.01/1e-2 gates in every realization, so case 2 never approaches `beam-robust contamination candidate` regardless of which of the two labels applies in a given realization.
- combined with the zero-mismatch second-seed full-matrix run, "no beam-robust contamination candidate under calibration-residual stress" is not an artifact of the single `seed=10` realization; the one seed-sensitive case (case 2) is sensitive only in a label sense at the strict local-PTE boundary, not in the physical-floor or window-integrated statistics that actually determine the final verdict.
