#!/usr/bin/env python3
"""Multi-seed extension of the targeted calibration-residual audit.

`outputs/targeted_candidate_calibration_audit.csv` (seed=10 only) showed that
the 5 full-catalog physical candidates keep the same final class under a
sigma_cal x residual-model x ell_nu sweep. That is one noise realization per
setting. This script reruns the two most informative (strongest-stress)
settings from that sweep -- white sigma_cal=1e-2, and smooth/chromatic
sigma_cal=1e-2 at ell_nu=1.0 MHz -- across 10 independent calibration-residual
seeds, to check that "final class is unchanged" is not an artifact of the one
seed=10 realization.

Satellite visibility (the injected signal) depends only on the background
context's time/frequency/geometry axes, not on the calibration residual
applied to `ctx.vis_tf`, so it is built once per (case, beam) and reused
across all seeds -- avoiding the dominant per-satellite track/beam/spectral
cost that made a naive seed loop over the full script prohibitively slow.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.satellite import build_visibility_for_sat
from scripts.run_coverage_grid import configure_case, load_tle_fast, read_uvh5_bin, to_local_path
from scripts.run_coverage_tail_resolution_check import compute_tail_case
from scripts.run_full_catalog_physical_candidate_audit import classify_case
from scripts.run_targeted_candidate_calibration_audit import apply_residual, build_cal_residual, make_rng

N_NULL = 1000
DOPPLER_MODE = "linear"
SEEDS = list(range(1, 11))
SETTINGS = [
    {"model": "white", "sigma_cal": 1e-2, "ell_nu_mhz": None},
    {"model": "smooth", "sigma_cal": 1e-2, "ell_nu_mhz": 1.0},
]


def main() -> None:
    cfg_run = yaml.safe_load(to_local_path("configs/coverage_robustness_all_tle.yaml").read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path("configs/pathB_jan2026_main.yaml"))
    candidate_audit = pd.read_csv(to_local_path("outputs/full_catalog_physical_candidate_audit.csv"))
    selection = pd.read_csv(to_local_path("outputs/lst_bin_selection.csv"))
    trials = pd.read_csv(to_local_path("outputs/coverage_robustness_trials_all_tle_full.csv"))

    recs, _tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026.txt")),
        max_records=int(cfg_run.get("max_tle_records", 6364)),
    )
    rec_map = {r.norad_id: r for r in recs}

    rows = []
    t_start = time.time()
    for case_idx, row in candidate_audit.iterrows():
        baseline_id = str(row["baseline_id"])
        lst_stratum = str(row["lst_stratum"])
        lst_bin_id = int(row["lst_bin_id"])
        morphology = str(row["morphology"])
        flux = float(row["flux_jy"])
        multiplicity = str(row["multiplicity"])
        beam = str(row["beam"])
        paired_beam = str(row["paired_beam"])
        primary_seed = int(row["primary_seed"])
        paired_seed = int(row["paired_seed"])
        primary_norads = [x for x in str(row["primary_selected_norad_ids"]).split(";") if x]
        paired_norads = [x for x in str(row["paired_selected_norad_ids"]).split(";") if x]

        srows = selection[
            (selection["baseline_id"].astype(str) == baseline_id)
            & (selection["lst_stratum"].astype(str) == lst_stratum)
            & (selection["lst_bin_id"].astype(int) == lst_bin_id)
        ]
        if len(srows) != 1:
            raise ValueError(f"Expected one selection row for {(baseline_id, lst_stratum, lst_bin_id)}")
        ctx0 = read_uvh5_bin(srows.iloc[0])

        primary_trial = trials[
            (trials["baseline_id"].astype(str) == baseline_id)
            & (trials["lst_stratum"].astype(str) == lst_stratum)
            & (trials["lst_bin_id"].astype(int) == lst_bin_id)
            & (trials["beam_model"].astype(str) == beam)
            & (trials["morphology"].astype(str) == morphology)
            & (trials["flux_jy"].astype(float) == flux)
            & (trials["multiplicity"].astype(str) == multiplicity)
        ].iloc[0]
        paired_trial = trials[
            (trials["baseline_id"].astype(str) == baseline_id)
            & (trials["lst_stratum"].astype(str) == lst_stratum)
            & (trials["lst_bin_id"].astype(int) == lst_bin_id)
            & (trials["beam_model"].astype(str) == paired_beam)
            & (trials["morphology"].astype(str) == morphology)
            & (trials["flux_jy"].astype(float) == flux)
            & (trials["multiplicity"].astype(str) == multiplicity)
        ].iloc[0]

        primary_cfg = configure_case(base_cfg, beam, morphology)
        paired_cfg = configure_case(base_cfg, paired_beam, morphology)
        primary_cfg.setdefault("starlink", {}).setdefault("emission_model", {})["doppler_mode"] = DOPPLER_MODE
        paired_cfg.setdefault("starlink", {}).setdefault("emission_model", {})["doppler_mode"] = DOPPLER_MODE

        # Satellite visibility does not depend on the calibration residual
        # (it only perturbs ctx.vis_tf), so build it once per case/beam.
        t0 = time.time()
        primary_vis_list = [
            build_visibility_for_sat(rec_map[str(n)], ctx0, primary_cfg, s_ref_jy=flux)[0] for n in primary_norads
        ]
        paired_vis_list = [
            build_visibility_for_sat(rec_map[str(n)], ctx0, paired_cfg, s_ref_jy=flux)[0] for n in paired_norads
        ]
        print(f"[case {int(row['case'])}] built vis lists in {time.time()-t0:.1f}s", flush=True)

        for setting in SETTINGS:
            model = setting["model"]
            sigma_cal = setting["sigma_cal"]
            ell_nu_mhz = setting["ell_nu_mhz"]
            for seed in SEEDS:
                rng = make_rng(seed, case_idx, 0, 0, 0)
                residual = build_cal_residual(
                    shape=ctx0.vis_tf.shape,
                    sigma_cal=sigma_cal,
                    model=model,
                    rng=rng,
                    ell_nu_mhz=ell_nu_mhz,
                    freq_hz=ctx0.freqs_hz,
                )
                ctx = apply_residual(ctx0, residual)

                primary_profile = compute_tail_case(
                    ctx, primary_cfg, primary_vis_list, N_NULL, primary_seed,
                    str(primary_trial.get("injection_mode", "coherent_ab")),
                )
                paired_profile = compute_tail_case(
                    ctx, paired_cfg, paired_vis_list, N_NULL, paired_seed,
                    str(paired_trial.get("injection_mode", "coherent_ab")),
                )
                final_class = classify_case(primary_profile, paired_profile)
                rows.append({
                    "case": int(row["case"]),
                    "baseline_id": baseline_id,
                    "lst_stratum": lst_stratum,
                    "lst_bin_id": lst_bin_id,
                    "beam": beam,
                    "paired_beam": paired_beam,
                    "morphology": morphology,
                    "flux_jy": flux,
                    "multiplicity": multiplicity,
                    "residual_model": model,
                    "sigma_cal": sigma_cal,
                    "ell_nu_mhz": ell_nu_mhz,
                    "cal_seed": seed,
                    "PTE_max_1000": float(primary_profile["PTE_global_max"]),
                    "B_rel": float(primary_profile["relative_abs_bias"]),
                    "PTE_absint_1000": float(primary_profile["PTE_global_absint"]),
                    "paired_PTE_max_1000": float(paired_profile["PTE_global_max"]),
                    "paired_B_rel": float(paired_profile["relative_abs_bias"]),
                    "paired_PTE_absint_1000": float(paired_profile["PTE_global_absint"]),
                    "final_class": final_class,
                })
                print(
                    f"  case={int(row['case'])} model={model} sigma={sigma_cal:g} "
                    f"ell={ell_nu_mhz if ell_nu_mhz is not None else 'na'} seed={seed} "
                    f"class={final_class} PTE_max={float(primary_profile['PTE_global_max']):.5f} "
                    f"elapsed={time.time()-t_start:.0f}s",
                    flush=True,
                )
                # incremental checkpoint so partial progress is never lost
                pd.DataFrame(rows).to_csv(to_local_path("outputs/calibration_multiseed_audit.csv"), index=False)

    out = to_local_path("outputs/calibration_multiseed_audit.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    stability = (
        df.groupby(["case", "residual_model"])["final_class"]
        .nunique()
        .reset_index(name="n_distinct_classes_over_10_seeds")
    )
    stability.to_csv(to_local_path("outputs/calibration_multiseed_audit_stability.csv"), index=False)

    meta = {
        "description": "10-seed calibration-residual stability check for the 5 full-catalog physical candidates, "
                        "at the strongest tested stress amplitude (sigma_cal=1e-2) for white and smooth/chromatic residuals",
        "n_seeds": len(SEEDS),
        "settings": SETTINGS,
        "n_rows": len(df),
        "all_classes_stable": bool((stability["n_distinct_classes_over_10_seeds"] == 1).all()),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    print(stability.to_string(index=False))
    print(f"all classes stable across 10 seeds: {meta['all_classes_stable']}")


if __name__ == "__main__":
    main()
