#!/usr/bin/env python3
"""Lightweight family-wise (grid-max) correction using already-stored null draws.

The manuscript currently only offers a naive `648 x 0.01` expected-false-positive
count as a descriptive scale, and explicitly disclaims it as a rigorous FWER
correction because the 648 grid rows are correlated (shared baselines, LST
strata, beam models, morphologies, flux tiers, multiplicity). A true joint
permutation test would require re-deriving a shared randomization across all
648 trials, which is out of scope here. This script instead builds a bootstrap
approximation of the family-wise minimum-p-value null distribution directly
from the per-trial null ensembles already saved to
`outputs/coverage_null_global_stats_all_tle_full/*.npz` (100 matched-null
draws per trial, no new physical simulation needed):

1. For each of the 648 trials and each of its 100 null draws, compute a
   leave-one-out empirical p-value of that draw within its own trial's null
   ensemble. This puts every trial on a common p-value scale regardless of
   its own bias magnitude/scale.
2. Bootstrap B draws where, for each of the 648 trials independently, one of
   its 100 leave-one-out p-values is sampled; take the minimum across the 648
   trials. This approximates the null distribution of "the smallest p-value
   anywhere in the 648-row grid" under a global-null, trial-independence
   assumption.
3. Compare the grid's real observed minimum PTE_global_max / PTE_global_absint
   to that bootstrap null distribution.

Caveat (stated explicitly in the output): step 2 resamples each trial
independently, so it does not reproduce the real positive correlation between
trials that share a background cell / beam / geometry. It is a lightweight
approximation, not an exact joint permutation test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_coverage_grid import to_local_path

N_BOOT = 200_000
RNG_SEED = 20270708


def leave_one_out_pvalues(null_vals: np.ndarray) -> np.ndarray:
    n = null_vals.shape[0]
    ge = null_vals[:, None] >= null_vals[None, :]
    count_incl_self = ge.sum(axis=0)
    count_excl_self = count_incl_self - 1
    return (1.0 + count_excl_self) / float(n)


def run_for_statistic(trials: pd.DataFrame, null_key: str, obs_key: str, obs_col: str, pte_col: str, rng: np.random.Generator) -> dict:
    n_trials = len(trials)
    p_loo = np.empty((n_trials, 100), dtype=float)
    # Observed p-values recomputed with the SAME denominator (100, not the
    # pipeline's official 1+100=101) as the leave-one-out bootstrap values, so
    # the two are on a matched resolution scale. Using the pipeline's native
    # PTE (denom 101) here would make the real observation structurally
    # unable to ever be beaten by the leave-one-out floor (denom 100),
    # producing a spurious "significant" result driven purely by a ~1%
    # denominator mismatch rather than by any real effect.
    obs_p_rescaled = np.empty(n_trials, dtype=float)
    missing = 0
    for i, path in enumerate(trials["null_global_stats_path"].tolist()):
        p = to_local_path(path)
        if not p.exists():
            missing += 1
            p_loo[i, :] = np.nan
            obs_p_rescaled[i] = np.nan
            continue
        with np.load(p) as d:
            null_vals = np.asarray(d[null_key], dtype=float)
            obs_val = float(d[obs_key])
        p_loo[i, :] = leave_one_out_pvalues(null_vals)
        obs_p_rescaled[i] = (1.0 + np.sum(null_vals >= obs_val)) / 100.0

    valid_rows = ~np.isnan(p_loo).any(axis=1)
    p_loo_valid = p_loo[valid_rows]
    n_valid = int(valid_rows.sum())

    idx = rng.integers(0, 100, size=(N_BOOT, n_valid))
    sampled = p_loo_valid[np.arange(n_valid)[None, :], idx]
    min_p_boot = sampled.min(axis=1)

    observed_min_p_native = float(trials.loc[valid_rows, pte_col].min())
    observed_min_p = float(np.min(obs_p_rescaled[valid_rows]))
    fwer_p = float((1.0 + np.sum(min_p_boot <= observed_min_p)) / (1.0 + N_BOOT))

    return {
        "statistic": obs_col,
        "n_trials_total": n_trials,
        "n_trials_used": n_valid,
        "n_trials_missing_null_file": missing,
        "observed_min_p_native_denom101": observed_min_p_native,
        "observed_min_p_rescaled_denom100": observed_min_p,
        "boot_min_p_p05": float(np.percentile(min_p_boot, 5)),
        "boot_min_p_median": float(np.percentile(min_p_boot, 50)),
        "boot_min_p_p95": float(np.percentile(min_p_boot, 95)),
        "fwer_corrected_p_value": fwer_p,
        "n_boot": N_BOOT,
    }


def main() -> None:
    trials = pd.read_csv(to_local_path("outputs/coverage_robustness_trials_all_tle_full.csv"))
    rng = np.random.default_rng(RNG_SEED)

    result_max = run_for_statistic(trials, "null_max_bias", "obs_max_bias", "obs_max_bias", "PTE_global_max", rng)
    result_absint = run_for_statistic(trials, "null_abs_integrated", "obs_abs_integrated", "obs_abs_integrated", "PTE_global_absint", rng)

    out_df = pd.DataFrame([result_max, result_absint])
    out = to_local_path("outputs/fwer_permutation_check.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    meta = {
        "description": "Bootstrap approximation of the family-wise minimum-p null distribution for the "
                        "648-row all_available coverage grid, using each trial's own stored N_null=100 "
                        "matched-null draws (leave-one-out p-values), resampled assuming trial independence.",
        "caveat": "Resamples each of the 648 trials independently; does not reproduce the real positive "
                  "correlation between trials sharing a background cell/beam/geometry. Lightweight "
                  "approximation, not an exact joint permutation test.",
        "n_boot": N_BOOT,
        "results": [result_max, result_absint],
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(out_df.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
