#!/usr/bin/env python3
"""Type II ANOVA variance decomposition of log10(B_rel) on the coverage grid
(`outputs/coverage_robustness_trials_all_tle_full.csv` by default).

This upgrades the earlier ad hoc OLS coefficient table
(`outputs/coverage_anova_results.csv`, response = raw Z_PS,max, no Type II
sums of squares, no effect-size measure, no residual diagnostics) with the
minimal set needed to report this as a defensible descriptive analysis:

- response variable: log10(relative_abs_bias) ("B_rel"), not the raw ratio,
  because B_rel spans several orders of magnitude across the grid and its
  OLS residuals would otherwise be dominated by the heaviest-tailed cells.
- flux_jy is modeled as a categorical factor in the main Type II table. This
  remains valid for both the historical two-flux grid and the four-flux
  integration grid. A separate descriptive log-flux slope check is reported
  only as a compact scaling diagnostic, not as the main ANOVA term.
- Type II sums of squares (each term tested against the full model minus
  that term, appropriate for an unbalanced factorial design) plus partial
  eta-squared per term.
- one-line residual diagnostic (Shapiro-Wilk normality, Breusch-Pagan
  heteroscedasticity).
- beam_model:morphology interaction, the one interaction this package's beam-
  robustness discussion actually depends on.
- explicit non-independence caveat: the grid rows share baselines, LST strata,
  beam models, morphologies, flux tiers and multiplicity settings, so this
  ANOVA is reported as a descriptive variance decomposition of factor
  contributions, not a formal hypothesis test with valid nominal p-values.

Baseline length is modeled as the continuous covariate `baseline_length_m`,
not the derived 3-level `baseline_class` (short/mid/long) bucketing used in
`scripts/summarize_coverage_grid.py`. `baseline_class` is itself just a
discretization of `baseline_length_m` (see `scripts/build_lst_metadata.py`),
so using the categorical bucket instead of the underlying continuous length
both throws away information and does not match this package's own stated
preference for treating baseline length as a continuous covariate. There is
no `frf_tier` (fringe-rate-filter loss quality tier) field anywhere in this
package's data products -- the 9 native baselines were originally chosen to
cross a length tier with an FRF-loss-quality tier (see `paper/paper.tex`
around the FRF description), but that FRF tier label itself was never
computed into any CSV/metadata column here, so it cannot be included as a
factor without re-deriving it from the original baseline-selection process
(out of scope for this script).
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_coverage_grid import to_local_path

FORMULA = (
    "log10_Brel ~ C(beam_model) + C(morphology) + C(flux_jy) + C(multiplicity) + "
    "baseline_length_m + C(lst_stratum) + C(beam_model):C(morphology)"
)
SLOPE_FORMULA = (
    "log10_Brel ~ C(beam_model) + C(morphology) + log10_flux_jy + C(multiplicity) + "
    "baseline_length_m + C(lst_stratum) + C(beam_model):C(morphology)"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials_all_tle_full.csv")
    ap.add_argument("--out", default="outputs/coverage_anova_typeII.csv")
    args = ap.parse_args()

    trials_path = to_local_path(args.trials)
    df = pd.read_csv(trials_path)
    n_total = len(df)

    rel_bias = pd.to_numeric(df["relative_abs_bias"], errors="coerce")
    positive = np.isfinite(rel_bias) & (rel_bias > 0)
    n_dropped = int((~positive).sum())
    df = df.loc[positive].copy()
    df["log10_Brel"] = np.log10(rel_bias.loc[positive].to_numpy(float))
    df["log10_flux_jy"] = np.log10(pd.to_numeric(df["flux_jy"], errors="coerce").to_numpy(float))
    flux_levels = sorted(float(x) for x in pd.Series(df["flux_jy"]).dropna().unique())

    fit = smf.ols(FORMULA, data=df).fit()
    aov = anova_lm(fit, typ=2)
    ss_resid = float(aov.loc["Residual", "sum_sq"])
    aov["partial_eta_sq"] = aov["sum_sq"] / (aov["sum_sq"] + ss_resid)
    aov = aov.reset_index().rename(columns={"index": "term"})

    slope_fit = smf.ols(SLOPE_FORMULA, data=df).fit()
    flux_slope_per_dex = float(slope_fit.params["log10_flux_jy"])
    flux_slope_line = (
        f"descriptive log10(B_rel) slope versus log10(flux_jy) = {flux_slope_per_dex:.3f} "
        f"dex(B_rel)/dex(flux) over flux levels {flux_levels}; "
        f"dex(B_rel)/dex(flux) (reference: ~1 if the cross/interference term with the background "
        f"dominates the injected bias, ~2 if the satellite's own power |V_sat|^2 dominates)"
    )

    resid = fit.resid.to_numpy(float)
    shapiro_stat, shapiro_p = stats.shapiro(resid if len(resid) <= 5000 else resid[:5000])
    bp_lm, bp_lm_p, _bp_f, _bp_f_p = het_breuschpagan(resid, fit.model.exog)
    residual_diag_line = (
        f"Shapiro-Wilk normality: W={shapiro_stat:.4f}, p={shapiro_p:.4g}; "
        f"Breusch-Pagan heteroscedasticity: LM={bp_lm:.3f}, p={bp_lm_p:.4g}"
    )

    aov_out = to_local_path(args.out)
    aov_out.parent.mkdir(parents=True, exist_ok=True)
    aov.to_csv(aov_out, index=False)

    meta = {
        "description": "Type II ANOVA variance decomposition of log10(relative_abs_bias) on the "
                        f"{n_total}-row coverage grid",
        "formula": FORMULA,
        "response_variable_definition": "log10_Brel = log10(relative_abs_bias); relative_abs_bias is the "
                                         "window-integrated |injected bias| / |background| ratio B_rel used "
                                         "elsewhere in this package's physical-candidate gate.",
        "n_trials_total": n_total,
        "n_trials_dropped_nonpositive_Brel": n_dropped,
        "flux_levels_jy": flux_levels,
        "flux_categorical_rationale": "flux_jy is modeled as a categorical factor in the Type II ANOVA. "
                                       "This avoids making the candidate hierarchy depend on a specific "
                                       "dose-response shape. The separate log-flux slope is descriptive only.",
        "residual_diagnostics": residual_diag_line,
        "non_independence_caveat": "The grid rows share baselines, LST strata, beam models, morphologies, "
                                    "flux tiers, and multiplicity settings and are not independent draws. "
                                    "This Type II ANOVA is reported as a descriptive variance decomposition "
                                    "of factor contributions to log10(B_rel), not as a formal hypothesis "
                                    "test with valid nominal p-values.",
        "baseline_length_note": "baseline_length_m is modeled as a continuous covariate, not the derived "
                                 "3-level baseline_class bucket; no frf_tier field exists anywhere in this "
                                 "package's data products (see module docstring).",
        "morphology_interpretation_note": "This ANOVA decomposes variance in log10(B_rel), the "
                                           "window-integrated |bias|/|background| ratio. A negligible "
                                           "morphology effect here describes bias MAGNITUDE only. It does "
                                           "not contradict morphology (e.g. khz_comb) being flagged as a "
                                           "risk factor elsewhere in this package's local-excursion/PTE-based "
                                           "candidate screening (Z_PS,max, PTE_global,max) -- that is a "
                                           "different response variable (spectral concentration / detectability "
                                           "of a local excursion against the matched-null distribution), not "
                                           "the integrated bias amplitude. The manuscript should state these as "
                                           "two separate, response-specific claims rather than one 'morphology "
                                           "matters / does not matter' statement.",
        "flux_slope_check": flux_slope_line,
        "flux_slope_per_dex": flux_slope_per_dex,
        "r_squared": float(fit.rsquared),
        "adj_r_squared": float(fit.rsquared_adj),
    }
    aov_out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(aov.to_string(index=False))
    print(f"\nR^2={fit.rsquared:.4f} adj-R^2={fit.rsquared_adj:.4f}")
    print(residual_diag_line)
    print(flux_slope_line)
    print(f"dropped {n_dropped}/{n_total} trials with non-positive relative_abs_bias (log10 undefined)")
    print(f"saved {aov_out}")


if __name__ == "__main__":
    main()
