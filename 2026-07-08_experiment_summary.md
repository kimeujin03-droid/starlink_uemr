# 2026-07-08 Experiment Summary

Consolidated record of the experiments requested and executed in this
session. Full detail and interpretation for each item lives in
`catalog_complete_followup_actions.md` (items 6–10) and
`targeted_candidate_calibration_audit_plan.md` ("Multi-seed extension"); this
file is a single-page index over that work.

## Requests

1. Re-evaluate the 2 first1200 PolyBeam absint floor-hit candidates at `N_null=1000` on the full_polybeam side (previously only at `N_null=100`).
2. Extend the targeted calibration-residual audit from a single seed (10) to multiple seeds.
3. Run 1–2 low-altitude (<30 deg) satellite-pass stress cases.
4. (Optional) Add a lightweight permutation/FWER correction using stored null draws.
5. (Added mid-session) Upgrade the coverage-grid ANOVA to a proper Type II decomposition of `log10(B_rel)`.

## 1. PolyBeam 2-case N=1000 re-evaluation

- Script: `scripts/run_polybeam_pair_n1000_recheck.py`
- Output: `outputs/polybeam_pair_n1000_recheck.csv`

| morphology | frozen PTE_max^1000 | frozen PTE_absint^1000 | full PTE_absint^100 (old) | full PTE_absint^1000 (new) | final class |
| --- | --- | --- | --- | --- | --- |
| smooth | 0.7363 | 0.006993 | 0.029703 | 0.038961 | local-only QA candidate |
| khz_comb | 0.7632 | 0.007992 | 0.019802 | 0.030969 | local-only QA candidate |

**Result:** both cases already fail the primary (frozen) local `PTE_max<0.01`
gate at N=1000, so they were never true physical candidates under the paper's
adopted local-max definition — they only ever qualified through a separate,
looser absint-floor-hit heuristic. Refining the full_polybeam absint PTE from
N=100 to N=1000 moves it *further* from the 0.01 threshold in both cases
(0.030→0.039, 0.020→0.031), ruling out an N=100 resolution-floor artifact.
This closes the last remaining N=100-resolution dependency in the "0
beam-robust candidates" claim.

## 2. Calibration-residual audit: single seed → multi-seed

- Script: `scripts/run_calibration_multiseed_audit.py`
- Outputs: `outputs/multiseed_calibration_audit/seed_01.csv`, `outputs/calibration_multiseed_audit.csv`, `outputs/calibration_multiseed_audit_stability.csv`

Two checks:

- **Second full-matrix seed** (`seed=1`, all `sigma_cal x model x ell_nu` combinations, 80 rows): reproduces the original `seed=10` final class for all 5 candidates, 0 mismatches.
- **10-seed sweep at the strongest tested stress amplitude** (`sigma_cal=1e-2`, white + smooth residuals, 5 candidates x 2 settings x 10 seeds = 100 evaluations, `N_null=1000`, satellite visibility cached per case/beam and reused across seeds):

| case | residual model | classes observed over 10 seeds |
| --- | --- | --- |
| 1 | white, smooth | beam-sensitive candidate (10/10 both) |
| 2 | smooth | local-only QA candidate (10/10) |
| 2 | white | local-only QA candidate (8/10), window-integrated candidate (2/10) |
| 3–5 | white, smooth | window-integrated candidate (10/10 both) |

**Result:** no `beam-robust contamination candidate` in any of the 100
evaluations. 4/5 candidates are perfectly stable across all 10 seeds. Case 2
shows a cosmetic label flip under white noise (`PTE_global_max` sits right at
the 0.01 boundary), but the statistics that actually gate the final verdict
(`PTE_global_absint`≈0.038, `B_rel`≈0.029) barely move and stay far from
threshold in every realization. "Grade retention" is not a single-seed
coincidence.

## 3. Low-altitude (<30 deg) satellite-pass stress test

- Script: `scripts/run_low_altitude_stress_case.py`
- Output: `outputs/low_altitude_stress_case.csv`

Reuses the same 2 background cells as the full-catalog physical-candidate
audit (no new background), injecting only satellites with `0 < peak_alt_deg < 30`
during the 10-minute window, `N_null=1000`, paired frozen/full PolyBeam.

| case | baseline/LST bin | morphology | peak alt range (deg) | eta_tau range | frozen (PTE_max/absint/B_rel) | full (PTE_max/absint/B_rel) | final class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 82_0/typical/18 | khz_comb | 28.5–29.7 | 0.49–1.00 | 0.0779 / 0.0100 / 2.1e-4 | 0.4645 / 0.1618 / 4.2e-4 | local-only QA candidate |
| 2 | 0_1/typical/44 | smooth | 28.6–30.0 | 0.85–1.00 | 0.6214 / 0.4915 / 7.6e-5 | 0.0090 / 0.0010 / 5.1e-4 | local-only QA candidate |

**Result:** several injected satellites reach `eta_tau` near/at 1.0 (the
`near_horizon` bin), confirming this exercises the intended riskiest
geometry. In case 2, full_polybeam's local PTE actually dips below the
strict 0.01 gate — the only place in this package's QA work where a
low-altitude injection alone crosses the local statistical gate — but
`B_rel` is ~20x below the `1e-2` physical-bias floor, so it does not qualify
as a physical candidate. Neither case produces a beam-robust candidate; this
demonstrates concretely how the staged QA hierarchy behaves at a geometry the
main grid (`alt_visible_deg>=70`) never tests.

## 4. Lightweight family-wise (grid-max) correction (optional — executed)

- Script: `scripts/run_fwer_permutation_check.py`
- Output: `outputs/fwer_permutation_check.csv`

Bootstrap (200,000 draws) approximation of the family-wise minimum-p null
distribution, built from the already-stored per-trial null ensembles
(`outputs/coverage_null_global_stats_all_tle_full/*.npz`, 100 draws/trial,
no new physics simulation): leave-one-out p-values per trial, resampled
independently per trial, minimum taken across the 648-row grid.

| statistic | observed min-p (matched denom=100) | bootstrap min-p median | FWER-corrected p-value |
| --- | --- | --- | --- |
| PTE_global_max | 0.01 | 0.01 | 0.998 |
| PTE_global_absint | 0.01 | 0.01 | 0.998 |

**Result:** under a global-null, trial-independence assumption, it is
virtually certain (99.8% of bootstrap draws) that at least one of the 648
trials hits its own resolution floor by chance alone — the observed grid
touching that floor is fully consistent with pure multiplicity noise.
**Caveat:** resamples trials independently, so it does not reproduce the real
positive correlation between trials sharing a background cell/beam/geometry;
lightweight approximation, not an exact joint permutation test.

## 5. Type II ANOVA of log10(B_rel) (added mid-session, revised)

- Script: `scripts/run_coverage_anova_typeII.py`
- Output: `outputs/coverage_anova_typeII.csv`

Replaces the earlier ad hoc OLS table (`outputs/coverage_anova_results.csv`,
raw `Z_PS,max` response, no Type II sums of squares, no effect size, no
residual diagnostics, first1200 catalog) with a minimal defensible
decomposition on the 648-row `all_available` grid:

- response: `log10(relative_abs_bias)`; 24/648 trials with non-positive `relative_abs_bias` dropped (log undefined) and reported.
- `flux_jy` modeled as a 2-level categorical factor (only 30 and 1000 Jy injected, ~1.5 dex apart — too few levels to identify a continuous dose-response shape).
- baseline length modeled as the **continuous** covariate `baseline_length_m` (a first pass used the derived 3-level `baseline_class` bucket, which mismatched this package's stated "length is continuous" convention; fixed by rerunning rather than editing text around the mismatch). No `frf_tier` field exists anywhere in this package's data products, so the FRF-loss-quality-tier factor described in the original baseline-selection design (`paper/paper.tex`) cannot be reconstructed here — flagged explicitly as a gap, not silently dropped.
- Type II ANOVA + partial eta-squared per term, including `beam_model:morphology`.
- residual diagnostics (one line): Shapiro-Wilk W=0.9613, p=9.33e-12 (non-normal); Breusch-Pagan LM=71.68, p=2.10e-11 (heteroscedastic).
- flux dose-response sanity check (new): implied slope of `log10(B_rel)` vs `log10(flux)`.

| term | partial eta-sq | PR(>F) |
| --- | --- | --- |
| flux_jy | 0.669 | ~0 |
| lst_stratum | 0.303 | ~0 |
| multiplicity | 0.210 | ~0 |
| baseline_length_m | 0.089 | 4.0e-14 |
| beam_model | 0.006 | 0.058 |
| morphology | 0.002 | 0.626 |
| beam_model:morphology | 0.0001 | 0.971 |

(R²=0.739, adj-R²=0.735.)

**Flux dose-response check:** the fitted step in `log10(B_rel)` between 30
and 1000 Jy (1.523 dex) is 1.615, giving an implied slope of **1.06
dex(B_rel)/dex(flux)** — consistent with the injected bias being dominated
by the coherent cross/interference term with the background (expected slope
~1), not by the satellite's own power `|V_sat|^2` (expected slope ~2).

**Result:** injected flux and background geometry dominate `B_rel` variance,
as physically expected; beam model and spectral morphology (and their
interaction) contribute negligibly. Because residuals are non-normal/
heteroscedastic and the 648 rows are not independent draws, this table is
reported as a **descriptive variance decomposition**, not a formal
inferential test.

**Scope note (avoids an apparent self-contradiction):** the negligible
`morphology` effect here is about window-integrated bias *magnitude*
(`B_rel`) only. It does not contradict `khz_comb` morphology being flagged
as a risk factor elsewhere in this package's local-excursion/PTE-based
candidate screening (`Z_PS,max`, `PTE_global,max`), which is a different
response variable (local-statistic detectability against the matched null,
not integrated bias size). The manuscript should state these as two
separate, response-specific claims rather than one undifferentiated
"morphology matters / does not matter."

## Net effect on the manuscript's central claim

Every frozen/full PolyBeam pairing behind the "0 beam-robust candidates"
verdict is now backed by `N_null=1000` on both sides (item 1), the result is
shown stable across independent calibration-noise realizations rather than
resting on one seed (item 2), the staged QA hierarchy's behavior at the
untested near-horizon geometry is now demonstrated rather than only flagged
as a limitation (item 3), the informal `648 x 0.01` multiplicity argument now
has a computed (if approximate) bootstrap FWER backing it (item 4), and the
factor-contribution claims for `B_rel` rest on a proper Type II decomposition
with effect sizes and residual diagnostics instead of a raw-response OLS
table (item 5).
