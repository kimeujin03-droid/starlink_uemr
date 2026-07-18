# Catalog-Complete Follow-up Actions

This note captures the remaining reviewer-critical work for the `paper_final_experiment_check` reproduction folder.

## Execution results so far

### Full-catalog QA rerun

Executed outputs:

- `outputs/coverage_robustness_trials_all_tle_full.csv`
- `outputs/all_tle_summary/coverage_candidate_counts_by_floor.csv`
- `outputs/all_tle_summary/coverage_summary_by_factor.csv`
- `outputs/all_tle_summary/coverage_summary_by_baseline_lst.csv`

Summary:

| floor | n_trials | n_statistical | n_physical | Pr_physical |
| --- | --- | --- | --- | --- |
| 1e-08 | 648 | 18 | 18 | 0.02778 |
| 1e-07 | 648 | 18 | 18 | 0.02778 |
| 1e-06 | 648 | 18 | 18 | 0.02778 |
| 1e-05 | 648 | 18 | 18 | 0.02778 |
| 1e-04 | 648 | 18 | 16 | 0.02469 |
| 1e-03 | 648 | 18 | 8 | 0.01235 |
| 1e-02 | 648 | 18 | 5 | 0.00772 |
| 1e-01 | 648 | 18 | 0 | 0.00000 |

Interpretation:

- full catalog changes the final gate outcome relative to archived `first1200`
- the 5 full-catalog physical candidates are the next audit target

### Paired frozen/full PolyBeam audit

Executed outputs:

- `outputs/polybeam_pair_audit.csv`
- `outputs/polybeam_pair_audit_summary.csv`

Summary:

| n_pairs | frozen-only candidates | full-only candidates | mismatches | median dZ_PS,max | median dPTE_absint | median dPTE_max | median dB_rel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 324 | 13 | 1 | 14 | -0.06174 | 0.00000 | 0.00000 | 0.000253 |

Interpretation:

- PolyBeam is beam-sensitive at the candidate level
- most frozen-only candidates disappear under full PolyBeam
- one full-only candidate appears, so the effect is not a one-way normalization artifact

### Full-catalog 5-candidate tail audit

Executed outputs:

- `outputs/full_catalog_physical_candidate_audit.csv`
- `outputs/full_catalog_physical_candidate_audit_summary.csv`

Summary:

| n_cases | beam-sensitive | beam-robust | window-integrated | relative-bias | local-only |
| --- | --- | --- | --- | --- | --- |
| 5 | 1 | 0 | 3 | 0 | 1 |

Case-level result:

| case | beam | paired beam | morphology | flux | multiplicity | PTE_max^1000 | B_rel | PTE_absint^1000 | final class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | frozen_polybeam | full_polybeam | khz_comb | 1000 | multi | 0.003996 | 0.026819 | 0.001000 | beam-sensitive candidate |
| 2 | frozen_polybeam | full_polybeam | smooth | 1000 | multi | 0.018981 | 0.028754 | 0.038961 | local-only QA candidate |
| 3 | frozen_polybeam | full_polybeam | khz_comb | 1000 | single | 0.002997 | 0.030349 | 0.493506 | window-integrated candidate |
| 4 | frozen_polybeam | full_polybeam | smooth | 1000 | single | 0.001998 | 0.031262 | 0.516484 | window-integrated candidate |
| 5 | full_polybeam | frozen_polybeam | khz_comb | 1000 | single | 0.002997 | 0.034878 | 0.577423 | window-integrated candidate |

Interpretation:

- none of the 5 full-catalog physical cases survives as a beam-robust contamination candidate
- one case remains beam-sensitive
- the rest are either window-integrated or local-only after tail refinement
- this is the cleanest table for the final manuscript discussion

## Must do

Status:

- 1. Full-catalog physical-candidate audit: completed
- 2. Full-catalog tail refinement: completed for the 5 full-catalog physical candidates
- 3. Full-catalog paired frozen/full PolyBeam audit: completed at the candidate level and the broader 324-pair audit
- 4. Random-1200 grid sanity runs: completed
- 5. Doppler comb-only rerun: completed for `none`, `constant`, and `linear` Doppler modes
- 6. first1200 absint floor-hit candidates, full_polybeam side at N_null=1000: completed
- 7. Low-altitude (<30 deg) satellite-pass stress test: completed
- 8. Lightweight family-wise (grid-max) correction from stored null draws: completed
- 9. Multi-seed extension of the targeted calibration-residual audit: completed
- 10. Type II ANOVA variance decomposition of log10(B_rel): completed

### 1. Full-catalog physical-candidate audit

The `all_available` rerun produced 18 statistical candidates and 5 physical candidates at `B_floor = 1e-2`. Those 5 cases must be traced to a final class.

Required columns:

| case | TLE set | beam | morphology | flux | multiplicity | PTE_max | B_rel | PTE_absint | paired beam | final class | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Use these final classes:

- `local-only QA candidate`
- `relative-bias candidate`
- `window-integrated candidate`
- `beam-sensitive candidate`
- `beam-robust contamination candidate`

Interpretation rule:

- `local-only QA candidate`: local `Z_PS,max` or `PTE_global,max` excursion only; fails downstream gates.
- `relative-bias candidate`: passes the statistical gate but fails `B_rel`.
- `window-integrated candidate`: passes `PTE_global,absint` but fails beam robustness.
- `beam-sensitive candidate`: changes class under frozen/full PolyBeam pairing.
- `beam-robust contamination candidate`: passes local, absint, floor, and paired-beam checks.

### 2. Full-catalog tail refinement

The 5 physical candidates from the full catalog must be rerun with `N_null = 1000`.

Required columns:

| case | PTE_global_max^1000 | PTE_global_absint^1000 | B_rel | final class |
| --- | --- | --- | --- | --- |

This is mandatory because `N_null = 100` has a minimum empirical PTE of `1/101 = 0.0099`, which is too coarse for a strict `PTE < 0.01` gate.

### 3. Full-catalog paired frozen/full PolyBeam audit

The 5 full-catalog physical candidates must be checked under both frozen and full PolyBeam.

Required verdicts:

- candidate appears only in frozen beam
- candidate appears only in full beam
- candidate passes in both beams
- candidate fails both beams but only clears the floor
- `PTE_global,absint` shift under full beam

This is the key beam-robustness audit for the final paper claim.

### 4. Random-1200 grid sanity runs

Executed outputs:

- `outputs/random1200_random1200_seed1.csv`
- `outputs/random1200_random1200_seed2.csv`
- `outputs/random1200_random1200_seed3.csv`

Summary:

| seed | n_trials | n_statistical | n_physical @ 1e-2 | min PTE_max | min PTE_absint |
| --- | --- | --- | --- | --- | --- |
| 1 | 648 | 8 | 0 | 0.009901 | 0.009901 |
| 2 | 648 | 0 | 0 | 0.019802 | 0.009901 |
| 3 | 648 | 0 | 0 | 0.019802 | 0.009901 |

Interpretation:

- random 1200-record subsets do change the statistical-candidate count
- none of the three subsets produce a physical candidate at `B_floor = 1e-2`
- this supports treating `all_available` as a catalog-complete stress case rather than a generic subset effect

### 5. Doppler comb-only rerun

Executed outputs:

- `outputs/doppler_comb_audit.csv`
- `outputs/doppler_comb_audit_summary.csv`

Summary:

| doppler_mode | beam-sensitive | local-only | window-integrated |
| --- | --- | --- | --- |
| none | 1 | 1 | 3 |
| constant | 1 | 1 | 3 |
| linear | 1 | 1 | 3 |

Interpretation:

- Doppler injection was implemented by time-dependent spectral-template shifting using the track range-rate
- for the candidate set tested here, Doppler mode does not change the final class
- `constant` Doppler slightly perturbs `B_rel` and `PTE_global,max` for one window-integrated case, but not enough to move class

The current result is still limited to the tested modes. If needed later, the same hook can be extended to an SGP4-derived custom range-rate series.

### 6. first1200 absint floor-hit candidates: full_polybeam side re-evaluated at N_null=1000

Executed outputs:

- `outputs/polybeam_pair_n1000_recheck.csv`
- `scripts/run_polybeam_pair_n1000_recheck.py`

Context: `outputs/absint_floor_recheck.csv` already reran the **frozen_polybeam** side
of the two first1200 absint floor-hit candidates (`11_10`/quiet/bin16, `smooth`
and `khz_comb`, 1000 Jy, multi) at `N_null=1000`. The **full_polybeam** paired
comparison that this catalog's "0 beam-robust candidates" verdict actually
depends on had only ever been evaluated at `N_null=100`
(`PTE_global_absint` = 0.029703 and 0.019802), which sits at the `1/101=0.0099`
resolution floor and was the one remaining N=100-resolution dependency in the
manuscript's central non-detection claim.

Summary:

| morphology | frozen PTE_max^1000 | frozen PTE_absint^1000 | full PTE_max^100 (old) | full PTE_absint^100 (old) | full PTE_max^1000 (new) | full PTE_absint^1000 (new) | final class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smooth | 0.7363 | 0.006993 | 0.881188 | 0.029703 | 0.801199 | 0.038961 | local-only QA candidate |
| khz_comb | 0.7632 | 0.007992 | 0.613861 | 0.019802 | 0.655345 | 0.030969 | local-only QA candidate |

Interpretation:

- both cases already fail the primary (frozen) local `PTE_global_max < 0.01` gate at N=1000 (0.74-0.76), so under the paper's adopted physical-candidate definition they were never true "physical candidates" in the first place — they only ever qualified via the separate, more lenient absint-floor-hit screening heuristic, not the local-max statistic used elsewhere.
- refining the full_polybeam absint PTE from N=100 to N=1000 moves it *further* from the 0.01 threshold in both cases (0.0297→0.0390, 0.0198→0.0310), not closer — ruling out the possibility that the N=100 resolution floor was masking a genuine near-threshold full-beam candidate.
- this closes the one remaining N=100-resolution dependency in the "0 beam-robust candidates" claim; every beam-robustness comparison in the manuscript is now backed by an N_null=1000 evaluation on both sides of the frozen/full pairing.

### 7. Low-altitude (<30 deg peak elevation) satellite-pass stress test

Executed outputs:

- `outputs/low_altitude_stress_case.csv`
- `scripts/run_low_altitude_stress_case.py`

Every trial in the main coverage grid only injects satellites with peak
elevation >= 70 deg (`alt_visible_deg: 70.0` in both `coverage_robustness.yaml`
and `coverage_robustness_all_tle.yaml`), i.e. near-zenith passes. Near-horizon
geometry is exactly the regime that maps most directly onto the
`tau_horizon`/window-boundary risk already tracked internally by
`pathb.satellite.window_geometry_metrics` (`eta_tau`, `horizon_proximity_bin`),
so it had never been exercised by the staged QA hierarchy. This test reuses
the same two background cells as the full-catalog physical-candidate audit (no
new background generation), injecting only satellites with `0 < peak_alt_deg < 30`
during the 10-minute window, at `N_null=1000`, paired frozen/full PolyBeam.

Summary:

| case | baseline/LST bin | morphology | n_sat | peak alt range (deg) | eta_tau range | frozen PTE_max / absint / B_rel | full PTE_max / absint / B_rel | final class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 82_0/typical/18 | khz_comb | 12 | 28.5–29.7 | 0.49–1.00 | 0.0779 / 0.0100 / 2.1e-4 | 0.4645 / 0.1618 / 4.2e-4 | local-only QA candidate |
| 2 | 0_1/typical/44 | smooth | 12 | 28.6–30.0 | 0.85–1.00 | 0.6214 / 0.4915 / 7.6e-5 | 0.0090 / 0.0010 / 5.1e-4 | local-only QA candidate |

Interpretation:

- several injected satellites reach `eta_tau` near or at 1.0 (the `near_horizon` bin, i.e. their delay track approaches `tau_horizon`), confirming this is the intended riskiest geometry, directly adjacent to the window boundary.
- in case 2, `full_polybeam`'s local PTE dips below the strict 0.01 gate (`PTE_max=0.0090`, `PTE_absint=0.0010`), the only instance anywhere in this package's QA work where a low-altitude injection alone crosses the local statistical gate; however `B_rel=5.1e-4` is roughly 20x below the `1e-2` physical-bias floor, so it does not qualify as a physical candidate under the adopted gate hierarchy.
- neither case produces a beam-robust contamination candidate; both fail at the primary local gate (case 1, judged on frozen) or the physical-floor gate (case 2, judged on full).
- this demonstrates concretely how the staged QA hierarchy behaves at the near-horizon geometry the main grid never tests, rather than only asserting it as an untested limitation.

### 8. Lightweight family-wise (grid-max) correction using stored null draws

Executed outputs:

- `outputs/fwer_permutation_check.csv`
- `scripts/run_fwer_permutation_check.py`

The manuscript's only stated multiplicity accounting is a descriptive
`648 x 0.01` expected-false-positive count, explicitly flagged as not a
rigorous FWER correction because the 648-row `all_available` grid shares
baselines, LST strata, beam models, morphologies, flux tiers, and
multiplicity settings across rows. A full joint permutation test would need a
shared randomization re-derived across all 648 trials, which is out of scope.
Instead, this check reuses the matched-null ensembles already saved per trial
(`outputs/coverage_null_global_stats_all_tle_full/*.npz`, 100 null draws each,
no new physical simulation needed): for each trial, a leave-one-out empirical
p-value is computed for each of its 100 null draws (putting every trial on a
common resolution-matched p-value scale), then 200,000 bootstrap draws
independently resample one such p-value per trial and take the grid-wide
minimum, building an approximate null distribution for "the smallest p-value
anywhere in the 648-row grid under a global null."

Result:

| statistic | observed min-p (matched denom=100) | bootstrap min-p median | FWER-corrected p-value |
| --- | --- | --- | --- |
| PTE_global_max | 0.01 | 0.01 | 0.998 |
| PTE_global_absint | 0.01 | 0.01 | 0.998 |

Interpretation:

- under a global-null, trial-independence assumption, it is virtually certain (99.8% of bootstrap draws) that at least one of the 648 trials hits its own resolution floor by chance alone.
- this makes the earlier informal `648 x 0.01` argument concrete: the observed grid touching the resolution floor is fully consistent with pure multiplicity noise and is not evidence of a real detection anywhere in the grid.
- caveat, stated explicitly in the script and the output metadata: the bootstrap resamples each trial independently and does not reproduce the real positive correlation between trials sharing a background cell/beam/geometry, so this is a lightweight approximation, not an exact joint permutation test. It should be reported as such (an improvement over the pure descriptive count, but still not a full joint FWER correction).

### 9. Multi-seed extension of the targeted calibration-residual audit

Executed outputs:

- `outputs/multiseed_calibration_audit/seed_01.csv` (supplementary full 80-row matrix at a second seed)
- `outputs/calibration_multiseed_audit.csv`, `outputs/calibration_multiseed_audit_stability.csv`
- `scripts/run_calibration_multiseed_audit.py`

See `targeted_candidate_calibration_audit_plan.md` ("Multi-seed extension"
section) for full detail. Summary: a second full-matrix run at `seed=1`
reproduces the original `seed=10` final class for all 5 candidates with 0
mismatches across 80 (case, model, sigma, ell) combinations; a targeted
10-seed sweep at the strongest tested stress amplitude (`sigma_cal=1e-2`,
white and smooth residuals, 100 evaluations total) finds no
`beam-robust contamination candidate` in any evaluation. 4 of 5 candidates are
perfectly stable across all 10 seeds; the 5th (case 2) shows a cosmetic label
flip between two non-candidate classes (`local-only QA candidate` /
`window-integrated candidate`) in 2/10 white-noise realizations, driven by
`PTE_global_max` sitting right at the 0.01 boundary, while its
`PTE_global_absint` and `B_rel` (the statistics that actually gate the final
verdict) stay stable and far from threshold in every realization. This
confirms "no beam-robust candidate under calibration stress" is not an
artifact of the original single-seed run.

### 10. Type II ANOVA variance decomposition of log10(B_rel)

Executed outputs:

- `outputs/coverage_anova_typeII.csv`, `outputs/coverage_anova_typeII.meta.json`
- `scripts/run_coverage_anova_typeII.py`

Upgrades the earlier ad hoc OLS coefficient table
(`outputs/coverage_anova_results.csv`, response = raw `Z_PS,max`, no Type II
sums of squares, no effect size, no residual diagnostics, first1200 catalog)
to a proper minimal-set variance decomposition on the 648-row `all_available`
grid (`outputs/coverage_robustness_trials_all_tle_full.csv`):

- response variable: `log10(relative_abs_bias)` ("log10 B_rel"); 24/648 trials with non-positive `relative_abs_bias` are dropped (log undefined) and reported as such.
- `flux_jy` modeled as a 2-level categorical factor: only 30 and 1000 Jy are injected (~1.5 dex apart), too few levels to identify a continuous dose-response shape, so a categorical contrast is the correct minimal treatment.
- baseline length modeled as the **continuous** covariate `baseline_length_m`, not the derived 3-level `baseline_class` bucket (a first pass used `baseline_class`, which is itself just a discretization of `baseline_length_m`; fixed to match this package's own stated preference for treating length as continuous). No `frf_tier` (fringe-rate-filter loss quality tier) field exists anywhere in this package's data products — the 9 native baselines were originally chosen to cross a length tier with an FRF-loss-quality tier (`paper/paper.tex`, FRF description), but that tier label itself was never computed into any CSV/metadata column, so it cannot be added as a factor without re-deriving it from the original baseline-selection process; flagged as a known gap between the design description and what is reproducible from this package's stored outputs, rather than silently mismatched.
- Type II ANOVA table (each term tested against the full model minus that term) with partial eta-squared per term.
- one-line residual diagnostic: Shapiro-Wilk W=0.9613, p=9.33e-12 (non-normal); Breusch-Pagan LM=71.68, p=2.10e-11 (heteroscedastic).
- `beam_model:morphology` interaction included explicitly.
- flux dose-response sanity check (see below).

Result:

| term | partial eta-sq | PR(>F) |
| --- | --- | --- |
| flux_jy | 0.669 | ~0 |
| lst_stratum | 0.303 | ~0 |
| multiplicity | 0.210 | ~0 |
| baseline_length_m | 0.089 | 4.0e-14 |
| beam_model | 0.006 | 0.058 |
| morphology | 0.002 | 0.626 |
| beam_model:morphology | 0.0001 | 0.971 |

(R^2=0.739, adj-R^2=0.735; full table in `outputs/coverage_anova_typeII.csv`.)

**Flux dose-response sanity check:** the fitted step in `log10(B_rel)` between
30 and 1000 Jy (1.523 dex of flux) is 1.615, giving an implied slope of
**1.06 dex(B_rel)/dex(flux)**. This is consistent with the injected bias
being dominated by the coherent cross/interference term between the injected
signal and the background (expected slope ~1), not by the satellite's own
power `|V_sat|^2` (which would give slope ~2) — a useful physical
consistency check on the injection model, not just a statistical aside.

Interpretation:

- injected flux and background geometry (LST stratum, multiplicity, baseline length) dominate the variance in `B_rel`, as physically expected; beam model and spectral morphology contribute negligibly, and their interaction is consistent with zero. Longer baselines are associated with slightly higher `log10(B_rel)` (coefficient +0.0031/m).
- the non-normal, heteroscedastic residuals mean nominal p-values should not be over-interpreted; combined with the fact that the 648 rows share baselines/LST strata/beam/morphology/flux/multiplicity structure and are not independent draws, this table is reported as a **descriptive variance decomposition** of factor contributions, not a formal inferential hypothesis test.
- **response-variable-specific scope, stated explicitly to avoid an apparent self-contradiction:** the negligible `morphology` effect found here describes the window-integrated bias *magnitude* (`B_rel`) only. It does not contradict `khz_comb`/spectral-comb morphology being flagged as a risk factor elsewhere in this package's local-excursion/PTE-based candidate screening (`Z_PS,max`, `PTE_global,max`) — that is a different response variable (spectral concentration / detectability of a local excursion against the matched-null distribution), not the integrated bias amplitude. The manuscript should state these as two separate, response-specific claims ("morphology affects local-statistic detectability, not window-integrated bias size") rather than a single undifferentiated "morphology matters / does not matter."

## Can stay as limitations

These do not need to be rerun for this paper.

- full HERA production pipeline
- physical calibration of `B_floor` to a thermal-noise or mK budget
- polarization leakage model
- satellite antenna pattern / attitude model
- full downstream `(k_\perp, k_\parallel)` propagation

## Reporting rule

Do not describe the current result as "no candidate" unless the case has passed:

1. local gate
2. `PTE_global,absint` gate
3. `B_rel` floor
4. paired frozen/full beam check
5. `N_null = 1000` tail refinement for boundary cases

That is the minimum defensible catalog-complete QA statement.

As of section 6, every frozen/full PolyBeam pairing used in the manuscript's
beam-robustness claim (the full-catalog 5-candidate audit and the first1200
absint floor-hit pair) has been evaluated with `N_null=1000` on both sides;
none is still resting on an `N_null=100` resolution floor. Section 7 shows how
the staged gates behave outside the grid's tested `alt_visible_deg>=70`
regime. Section 8 upgrades the descriptive `648 x 0.01` multiplicity count to
a bootstrap-based (independence-assumption) family-wise reference value.
