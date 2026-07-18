#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

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


DOPPLER_MODES = ["none", "constant", "linear"]


def main() -> None:
    config_path = to_local_path("configs/coverage_robustness_all_tle.yaml")
    cfg_run = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path("configs/pathB_jan2026_main.yaml"))
    trials = pd.read_csv(to_local_path("outputs/coverage_robustness_trials_all_tle_full.csv"))
    selection = pd.read_csv(to_local_path("outputs/lst_bin_selection.csv"))
    candidate_audit = pd.read_csv(to_local_path("outputs/full_catalog_physical_candidate_audit.csv"))

    recs, tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026.txt")),
        max_records=int(cfg_run.get("max_tle_records", 6364)),
    )
    rec_map = {r.norad_id: r for r in recs}

    rows = []
    for idx, row in candidate_audit.iterrows():
        baseline_id = str(row["baseline_id"])
        lst_stratum = str(row["lst_stratum"])
        lst_bin_id = int(row["lst_bin_id"])
        beam = str(row["beam"])
        paired_beam = str(row["paired_beam"])
        morphology = str(row["morphology"])
        flux = float(row["flux_jy"])
        multiplicity = str(row["multiplicity"])
        primary_seed = int(row["primary_seed"])
        paired_seed = int(row["paired_seed"])

        srows = selection[
            (selection["baseline_id"].astype(str) == baseline_id)
            & (selection["lst_stratum"].astype(str) == lst_stratum)
            & (selection["lst_bin_id"].astype(int) == lst_bin_id)
        ]
        if len(srows) != 1:
            raise ValueError(f"Expected one selection row for {(baseline_id, lst_stratum, lst_bin_id)}, found {len(srows)}")
        ctx = read_uvh5_bin(srows.iloc[0])

        primary_trial = trials[
            (trials["baseline_id"].astype(str) == baseline_id)
            & (trials["lst_stratum"].astype(str) == lst_stratum)
            & (trials["lst_bin_id"].astype(int) == lst_bin_id)
            & (trials["beam_model"].astype(str) == beam)
            & (trials["morphology"].astype(str) == morphology)
            & (trials["flux_jy"].astype(float) == flux)
            & (trials["multiplicity"].astype(str) == multiplicity)
        ]
        paired_trial = trials[
            (trials["baseline_id"].astype(str) == baseline_id)
            & (trials["lst_stratum"].astype(str) == lst_stratum)
            & (trials["lst_bin_id"].astype(int) == lst_bin_id)
            & (trials["beam_model"].astype(str) == paired_beam)
            & (trials["morphology"].astype(str) == morphology)
            & (trials["flux_jy"].astype(float) == flux)
            & (trials["multiplicity"].astype(str) == multiplicity)
        ]
        if len(primary_trial) != 1 or len(paired_trial) != 1:
            raise ValueError(f"Expected one primary and one paired trial for case {baseline_id}/{lst_stratum}/{lst_bin_id}")
        primary_trial = primary_trial.iloc[0]
        paired_trial = paired_trial.iloc[0]

        norads = [x for x in str(row["primary_selected_norad_ids"]).split(";") if x]
        paired_norads = [x for x in str(row["paired_selected_norad_ids"]).split(";") if x]

        for mode in DOPPLER_MODES:
            def build_vis_list(cfg, norad_list):
                out = []
                for norad in norad_list:
                    vis, _track, _report = build_visibility_for_sat(
                        rec_map[str(norad)],
                        ctx,
                        cfg,
                        s_ref_jy=flux,
                    )
                    out.append(vis)
                return out

            primary_cfg = configure_case(base_cfg, beam, morphology)
            paired_cfg = configure_case(base_cfg, paired_beam, morphology)
            primary_cfg.setdefault("starlink", {}).setdefault("emission_model", {})["doppler_mode"] = mode
            paired_cfg.setdefault("starlink", {}).setdefault("emission_model", {})["doppler_mode"] = mode

            primary_profile = compute_tail_case(
                ctx,
                primary_cfg,
                build_vis_list(primary_cfg, norads),
                1000,
                primary_seed,
                str(primary_trial.get("injection_mode", "coherent_ab")),
            )
            paired_profile = compute_tail_case(
                ctx,
                paired_cfg,
                build_vis_list(paired_cfg, paired_norads),
                1000,
                paired_seed,
                str(paired_trial.get("injection_mode", "coherent_ab")),
            )

            final_class = classify_case(primary_profile, paired_profile)
            rows.append(
                {
                    "case": idx + 1,
                    "doppler_mode": mode,
                    "baseline_id": baseline_id,
                    "lst_stratum": lst_stratum,
                    "lst_bin_id": lst_bin_id,
                    "beam": beam,
                    "paired_beam": paired_beam,
                    "morphology": morphology,
                    "flux_jy": flux,
                    "multiplicity": multiplicity,
                    "PTE_max_1000": float(primary_profile["PTE_global_max"]),
                    "B_rel": float(primary_profile["relative_abs_bias"]),
                    "PTE_absint_1000": float(primary_profile["PTE_global_absint"]),
                    "paired_PTE_max_1000": float(paired_profile["PTE_global_max"]),
                    "paired_B_rel": float(paired_profile["relative_abs_bias"]),
                    "paired_PTE_absint_1000": float(paired_profile["PTE_global_absint"]),
                    "final_class": final_class,
                    "primary_seed": primary_seed,
                    "paired_seed": paired_seed,
                    "primary_selected_norad_ids": row["primary_selected_norad_ids"],
                    "paired_selected_norad_ids": row["paired_selected_norad_ids"],
                }
            )
            print(
                f"[{idx+1}/{len(candidate_audit)}] mode={mode} case={baseline_id}/{lst_stratum}/{lst_bin_id} "
                f"{beam}->{paired_beam} class={final_class} "
                f"PTE={float(primary_profile['PTE_global_max']):.5f}/{float(primary_profile['PTE_global_absint']):.5f}",
                flush=True,
            )

    out = to_local_path("outputs/doppler_comb_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)

    summary = (
        df_out.groupby(["doppler_mode", "final_class"])
        .size()
        .reset_index(name="n_cases")
        .sort_values(["doppler_mode", "final_class"])
    )
    summary.to_csv(to_local_path("outputs/doppler_comb_audit_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
