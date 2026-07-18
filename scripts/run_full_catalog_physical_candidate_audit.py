#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.satellite import build_visibility_for_sat
from scripts.run_coverage_grid import configure_case, load_tle_fast, rank_visible_sats, read_uvh5_bin, to_local_path
from scripts.run_coverage_tail_resolution_check import compute_tail_case


def classify_case(primary: dict, paired: dict) -> str:
    p_max = float(primary["PTE_global_max"])
    p_abs = float(primary["PTE_global_absint"])
    b_rel = float(primary["relative_abs_bias"])
    paired_max = float(paired["PTE_global_max"])
    paired_abs = float(paired["PTE_global_absint"])
    paired_brel = float(paired["relative_abs_bias"])

    if p_max >= 0.01:
        return "local-only QA candidate"
    if b_rel <= 1e-2:
        return "relative-bias candidate"
    if p_abs >= 0.01:
        return "window-integrated candidate"
    paired_pass = paired_max < 0.01 and paired_abs < 0.01 and paired_brel > 1e-2
    if paired_pass and abs(paired_max - p_max) < 1e-6 and abs(paired_abs - p_abs) < 1e-6:
        return "beam-robust contamination candidate"
    return "beam-sensitive candidate"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/coverage_robustness_all_tle.yaml")
    ap.add_argument("--pathb-config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials_all_tle_full.csv")
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--out", default="outputs/full_catalog_physical_candidate_audit.csv")
    ap.add_argument("--summary-out", default="outputs/full_catalog_physical_candidate_audit_summary.csv")
    ap.add_argument("--n-null", type=int, default=1000)
    args = ap.parse_args()

    config_path = to_local_path(args.config)
    cfg_run = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path(args.pathb_config))
    trials = pd.read_csv(to_local_path(args.trials))
    selection = pd.read_csv(to_local_path(args.selection))

    phys = trials[(trials["candidate_statistical"].astype(bool)) & (trials["relative_abs_bias"] > 1e-2)].copy()
    phys = phys.sort_values(["PTE_global_max", "PTE_global_absint", "relative_abs_bias"]).reset_index(drop=True)
    if phys.empty:
        raise RuntimeError("No full-catalog physical candidates found")

    recs, tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026.txt")),
        max_records=int(cfg_run.get("max_tle_records", 6364)),
    )
    rec_map = {r.norad_id: r for r in recs}

    rows = []
    for idx, row in phys.iterrows():
        sel = selection[
            (selection["baseline_id"].astype(str) == str(row["baseline_id"]))
            & (selection["lst_stratum"].astype(str) == str(row["lst_stratum"]))
            & (selection["lst_bin_id"].astype(int) == int(row["lst_bin_id"]))
        ]
        if len(sel) != 1:
            raise ValueError(f"Expected one selection row for {row['baseline_id']}/{row['lst_stratum']}/{row['lst_bin_id']}")
        ctx = read_uvh5_bin(sel.iloc[0])

        norads = [x for x in str(row["selected_norad_ids"]).split(";") if x]
        if not norads:
            raise ValueError(f"Empty selected_norad_ids for {row.to_dict()}")

        paired = trials[
            (trials["baseline_id"].astype(str) == str(row["baseline_id"]))
            & (trials["lst_stratum"].astype(str) == str(row["lst_stratum"]))
            & (trials["lst_bin_id"].astype(int) == int(row["lst_bin_id"]))
            & (trials["morphology"].astype(str) == str(row["morphology"]))
            & (trials["flux_jy"].astype(float) == float(row["flux_jy"]))
            & (trials["multiplicity"].astype(str) == str(row["multiplicity"]))
            & (trials["beam_model"].astype(str) != str(row["beam_model"]))
        ]
        if len(paired) != 1:
            raise ValueError(
                f"Expected 1 paired row for case {row['baseline_id']}/{row['lst_stratum']}/{row['lst_bin_id']}/"
                f"{row['morphology']}/{row['flux_jy']}/{row['multiplicity']}, found {len(paired)}"
            )
        paired_row = paired.iloc[0]

        primary_cfg = configure_case(base_cfg, str(row["beam_model"]), str(row["morphology"]))
        paired_cfg = configure_case(base_cfg, str(paired_row["beam_model"]), str(paired_row["morphology"]))

        def build_vis_list(cfg, flux: float) -> list:
            out = []
            for norad in norads:
                vis, _track, _report = build_visibility_for_sat(rec_map[str(norad)], ctx, cfg, s_ref_jy=float(flux))
                out.append(vis)
            return out

        primary_profile = compute_tail_case(
            ctx,
            primary_cfg,
            build_vis_list(primary_cfg, float(row["flux_jy"])),
            int(args.n_null),
            int(row["seed"]),
            str(row.get("injection_mode", "coherent_ab")),
        )
        paired_profile = compute_tail_case(
            ctx,
            paired_cfg,
            build_vis_list(paired_cfg, float(row["flux_jy"])),
            int(args.n_null),
            int(paired_row["seed"]),
            str(paired_row.get("injection_mode", "coherent_ab")),
        )

        final_class = classify_case(primary_profile, paired_profile)
        rows.append(
            {
                "case": idx + 1,
                "baseline_id": row["baseline_id"],
                "lst_stratum": row["lst_stratum"],
                "lst_bin_id": int(row["lst_bin_id"]),
                "TLE_set": "all_available",
                "beam": row["beam_model"],
                "paired_beam": paired_row["beam_model"],
                "morphology": row["morphology"],
                "flux_jy": float(row["flux_jy"]),
                "multiplicity": row["multiplicity"],
                "PTE_max_1000": float(primary_profile["PTE_global_max"]),
                "B_rel": float(primary_profile["relative_abs_bias"]),
                "PTE_absint_1000": float(primary_profile["PTE_global_absint"]),
                "paired_PTE_max_1000": float(paired_profile["PTE_global_max"]),
                "paired_B_rel": float(paired_profile["relative_abs_bias"]),
                "paired_PTE_absint_1000": float(paired_profile["PTE_global_absint"]),
                "final_class": final_class,
                "primary_seed": int(row["seed"]),
                "paired_seed": int(paired_row["seed"]),
                "primary_coarse_PTE_max": float(row["PTE_global_max"]),
                "primary_coarse_PTE_absint": float(row["PTE_global_absint"]),
                "paired_coarse_PTE_max": float(paired_row["PTE_global_max"]),
                "paired_coarse_PTE_absint": float(paired_row["PTE_global_absint"]),
                "primary_selected_norad_ids": row["selected_norad_ids"],
                "paired_selected_norad_ids": paired_row["selected_norad_ids"],
            }
        )
        print(
            f"[{idx+1}/{len(phys)}] case={row['baseline_id']}/{row['lst_stratum']}/{row['lst_bin_id']} "
            f"{row['beam_model']} {row['morphology']} {row['flux_jy']:g} {row['multiplicity']} "
            f"PTE1000={primary_profile['PTE_global_max']:.5f} abs1000={primary_profile['PTE_global_absint']:.5f} "
            f"class={final_class}",
            flush=True,
        )

    out = to_local_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)
    summary = pd.DataFrame(
        [
            {
                "n_cases": len(df_out),
                "n_beam_sensitive": int((df_out["final_class"] == "beam-sensitive candidate").sum()),
                "n_beam_robust": int((df_out["final_class"] == "beam-robust contamination candidate").sum()),
                "n_window_integrated": int((df_out["final_class"] == "window-integrated candidate").sum()),
                "n_relative_bias": int((df_out["final_class"] == "relative-bias candidate").sum()),
                "n_local_only": int((df_out["final_class"] == "local-only QA candidate").sum()),
                "tle_records_loaded_fast": tle_meta["tle_records_loaded_fast"],
            }
        ]
    )
    summary.to_csv(to_local_path(args.summary_out), index=False)
    print(summary.to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
