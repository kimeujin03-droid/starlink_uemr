#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.satellite import build_visibility_for_sat
from scripts.run_coverage_grid import (
    configure_case,
    load_tle_fast,
    rank_visible_sats,
    read_uvh5_bin,
    safe_token,
    to_local_path,
)
from scripts.run_coverage_tail_resolution_check import compute_tail_case, write_profile_figure


def select_near_threshold(df: pd.DataFrame, pte_cut: float, z_cut: float, bias_floor: float) -> pd.DataFrame:
    mask = ((df["PTE_global_max"] <= pte_cut) | (df["Z_PS_max"] > z_cut)) & (df["relative_abs_bias"] > bias_floor)
    return df.loc[mask].copy().reset_index(drop=False).rename(columns={"index": "old_row_index"})


def select_absint_tail(df: pd.DataFrame, pte_cut: float, bias_floor: float) -> pd.DataFrame:
    mask = (df["PTE_global_absint"] <= pte_cut) & (df["relative_abs_bias"] > bias_floor)
    return df.loc[mask].copy().reset_index(drop=False).rename(columns={"index": "old_row_index"})


def expand_to_paired_beams(df: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return cases
    settings = cases[
        ["baseline_id", "lst_stratum", "lst_bin_id", "morphology", "flux_jy", "multiplicity"]
    ].drop_duplicates()
    expanded = df.merge(
        settings,
        on=["baseline_id", "lst_stratum", "lst_bin_id", "morphology", "flux_jy", "multiplicity"],
        how="inner",
    )
    return expanded.copy().reset_index(drop=False).rename(columns={"index": "old_row_index"})


def paired_key(row: pd.Series, beam_model: str | None = None) -> tuple:
    return (
        str(row["baseline_id"]),
        str(row["lst_stratum"]),
        int(row["lst_bin_id"]),
        beam_model if beam_model is not None else str(row["beam_model"]),
        str(row["morphology"]),
        float(row["flux_jy"]),
        str(row["multiplicity"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials.csv")
    ap.add_argument("--config", default="configs/coverage_robustness.yaml")
    ap.add_argument("--pathb-config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--out", default="outputs/coverage_tail_refined_near_threshold.csv")
    ap.add_argument("--summary-out", default="outputs/coverage_tail_refined_near_threshold_summary.csv")
    ap.add_argument("--fig-dir", default="figures/coverage_robustness/candidate_delay_profiles_strict")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--selection-mode", choices=["local", "absint"], default="local")
    ap.add_argument("--pte-cut", type=float, default=0.03)
    ap.add_argument("--z-cut", type=float, default=3.0)
    ap.add_argument("--bias-floor", type=float, default=1e-3)
    ap.add_argument("--expand-paired-beams", action="store_true")
    ap.add_argument("--intrinsic-phase-mode", choices=["none", "random_per_satellite"], default="none")
    ap.add_argument("--intrinsic-phase-seed", type=int, default=20260709)
    args = ap.parse_args()

    cfg_run = yaml.safe_load(to_local_path(args.config).read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path(args.pathb_config))
    selection = pd.read_csv(to_local_path(args.selection))
    trials = pd.read_csv(to_local_path(args.trials))
    if args.selection_mode == "absint":
        cases = select_absint_tail(trials, args.pte_cut, args.bias_floor)
    else:
        cases = select_near_threshold(trials, args.pte_cut, args.z_cut, args.bias_floor)
    n_seed_cases = len(cases)
    if args.expand_paired_beams:
        cases = expand_to_paired_beams(trials, cases)
    if cases.empty:
        raise RuntimeError("No near-threshold cases selected")

    recs, _tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026_LEO_only.tle")),
        max_records=int(cfg_run.get("max_tle_records", 1200)),
    )
    rec_map = {r.norad_id: r for r in recs}

    contexts = {}
    rankings = {}
    sat_cache = {}
    profile_cache = {}
    rows = []

    for i, old_row in cases.iterrows():
        cell_key = (str(old_row["baseline_id"]), str(old_row["lst_stratum"]), int(old_row["lst_bin_id"]))
        if cell_key not in contexts:
            srows = selection[
                (selection["baseline_id"].astype(str) == cell_key[0])
                & (selection["lst_stratum"].astype(str) == cell_key[1])
                & (selection["lst_bin_id"].astype(int) == cell_key[2])
            ]
            if len(srows) != 1:
                raise ValueError(f"Expected one selection row for {cell_key}, found {len(srows)}")
            contexts[cell_key] = read_uvh5_bin(srows.iloc[0])
        ctx = contexts[cell_key]

        if cell_key not in rankings:
            rankings[cell_key] = rank_visible_sats(recs, ctx, base_cfg, float(cfg_run.get("alt_visible_deg", 70.0)))
        ranked = rankings[cell_key]
        if len(ranked) == 0:
            raise RuntimeError(f"No visible satellites for {cell_key}")

        multiplicity = str(old_row["multiplicity"])
        n_sat_target = 1 if multiplicity == "single" else int(cfg_run.get("max_multi_satellites", 12))
        chosen = ranked.head(max(1, min(n_sat_target, len(ranked))))
        beam = str(old_row["beam_model"])
        morphology = str(old_row["morphology"])
        flux = float(old_row["flux_jy"])
        cfg = configure_case(base_cfg, beam, morphology)

        sat_vis_list = []
        for norad in chosen["norad_id"].astype(str).tolist():
            cache_key = (cell_key, beam, morphology, flux, norad)
            if cache_key not in sat_cache:
                vis, _track, _report = build_visibility_for_sat(rec_map[norad], ctx, cfg, s_ref_jy=flux)
                sat_cache[cache_key] = vis
            sat_vis_list.append(sat_cache[cache_key])

        phase_seed = None
        if args.intrinsic_phase_mode == "random_per_satellite":
            phase_seed = int(args.intrinsic_phase_seed + int(old_row["old_row_index"]))

        profile = compute_tail_case(
            ctx,
            cfg,
            sat_vis_list,
            int(args.n_null),
            int(old_row["seed"]),
            str(old_row.get("injection_mode", "coherent_ab")),
            intrinsic_phase_seed=phase_seed,
        )
        case_id = (
            f"{old_row['baseline_id']}_lst{int(old_row['lst_bin_id']):03d}_{old_row['lst_stratum']}_"
            f"{beam}_{morphology}_S{flux:g}_{multiplicity}"
        )
        profile_cache[case_id] = profile

        rel_bias = float(profile["relative_abs_bias"])
        pte_new = float(profile["PTE_global_max"])
        pte_abs_new = float(profile["PTE_global_absint"])
        row = {
            "case_id": case_id,
            "old_row_index": int(old_row["old_row_index"]),
            "baseline_id": old_row["baseline_id"],
            "baseline_length_m": float(old_row["baseline_length_m"]),
            "lst_stratum": old_row["lst_stratum"],
            "lst_bin_id": int(old_row["lst_bin_id"]),
            "beam_model": beam,
            "morphology": morphology,
            "flux_jy": flux,
            "multiplicity": multiplicity,
            "N_null": int(args.n_null),
            "Z_PS_max_old": float(old_row["Z_PS_max"]),
            "Z_PS_max_new": float(profile["Z_PS_max"]),
            "PTE_global_max_old": float(old_row["PTE_global_max"]),
            "PTE_global_max_new": pte_new,
            "PTE_global_absint_old": float(old_row["PTE_global_absint"]),
            "PTE_global_absint_new": pte_abs_new,
            "relative_abs_bias": rel_bias,
            "window_abs_bias_sum": float(profile["window_abs_bias_sum"]),
            "window_bg_abs_sum": float(profile["window_bg_abs_sum"]),
            "null_mad_win": float(profile["null_mad_win"]),
            "n_null_exceed_max": int(profile["n_null_exceed_max"]),
            "n_null_exceed_absint": int(profile["n_null_exceed_absint"]),
            "selection_mode": args.selection_mode,
            "n_seed_cases_before_pair_expansion": int(n_seed_cases),
            "intrinsic_phase_mode": args.intrinsic_phase_mode,
            "intrinsic_phase_seed": profile["intrinsic_phase_seed"],
            "intrinsic_phases_rad": ";".join(f"{x:.8f}" for x in profile["intrinsic_phases_rad"]),
            "retained_PTE001": bool(pte_new < 0.01),
            "retained_PTE005": bool(pte_new < 0.05),
            "retained_absint_001": bool(pte_abs_new < 0.01),
            "retained_absint_003": bool(pte_abs_new <= 0.03),
            "retained_physical_floor_1e3": bool(pte_new < 0.01 and rel_bias > 1e-3),
            "retained_physical_floor_1e2": bool(pte_new < 0.01 and rel_bias > 1e-2),
            "retained_absint_floor_1e3": bool(pte_abs_new < 0.01 and rel_bias > 1e-3),
            "retained_absint_floor_1e2": bool(pte_abs_new < 0.01 and rel_bias > 1e-2),
        }
        rows.append(row)
        print(
            f"[{i + 1}/{len(cases)}] {case_id} PTE max {float(old_row['PTE_global_max']):.5f}->{pte_new:.5f} "
            f"absint {float(old_row['PTE_global_absint']):.5f}->{pte_abs_new:.5f}",
            flush=True,
        )

    refined = pd.DataFrame(rows)

    strict = refined[refined["retained_physical_floor_1e3"]].copy()
    strict_keys = {paired_key(r) for _, r in strict.iterrows()}
    beam_robust = 0
    frozen_only = 0
    for _, r in strict.iterrows():
        other = "full_polybeam" if r["beam_model"] == "frozen_polybeam" else "frozen_polybeam"
        if paired_key(r, beam_model=other) in strict_keys:
            beam_robust += 1
        elif r["beam_model"] == "frozen_polybeam":
            frozen_only += 1

    absint_strict = refined[refined["retained_absint_floor_1e3"]].copy()
    absint_strict_keys = {paired_key(r) for _, r in absint_strict.iterrows()}
    absint_beam_robust = 0
    absint_frozen_only = 0
    for _, r in absint_strict.iterrows():
        other = "full_polybeam" if r["beam_model"] == "frozen_polybeam" else "frozen_polybeam"
        if paired_key(r, beam_model=other) in absint_strict_keys:
            absint_beam_robust += 1
        elif r["beam_model"] == "frozen_polybeam":
            absint_frozen_only += 1

    summary = pd.DataFrame(
        [
            {"metric": "selection_mode", "value": args.selection_mode},
            {"metric": "n_seed_cases_before_pair_expansion", "value": int(n_seed_cases)},
            {"metric": "n_refined_cases", "value": int(len(refined))},
            {"metric": "n_refined_PTE_lt_001", "value": int(refined["retained_PTE001"].sum())},
            {"metric": "n_refined_physical_floor_1e3", "value": int(refined["retained_physical_floor_1e3"].sum())},
            {"metric": "n_refined_physical_floor_1e2", "value": int(refined["retained_physical_floor_1e2"].sum())},
            {"metric": "n_refined_beam_robust", "value": int(beam_robust)},
            {"metric": "n_full_polybeam_strict", "value": int(((strict["beam_model"] == "full_polybeam")).sum())},
            {"metric": "n_frozen_only_strict", "value": int(frozen_only)},
            {"metric": "n_refined_absint_lt_001", "value": int(refined["retained_absint_001"].sum())},
            {"metric": "n_refined_absint_floor_1e3", "value": int(refined["retained_absint_floor_1e3"].sum())},
            {"metric": "n_refined_absint_floor_1e2", "value": int(refined["retained_absint_floor_1e2"].sum())},
            {"metric": "n_refined_absint_beam_robust", "value": int(absint_beam_robust)},
            {"metric": "n_full_polybeam_absint_strict", "value": int(((absint_strict["beam_model"] == "full_polybeam")).sum())},
            {"metric": "n_frozen_only_absint_strict", "value": int(absint_frozen_only)},
        ]
    )

    out = to_local_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    refined.to_csv(out, index=False)
    summary_out = to_local_path(args.summary_out)
    summary.to_csv(summary_out, index=False)

    fig_dir = to_local_path(args.fig_dir)
    for j, (_, r) in enumerate(strict.iterrows(), start=1):
        profile = profile_cache[r["case_id"]]
        title = (
            f"{r['baseline_id']} LST {int(r['lst_bin_id'])} {r['beam_model']} "
            f"{r['morphology']} {float(r['flux_jy']):g} Jy {r['multiplicity']}"
        )
        fig_name = f"{j:02d}_{safe_token(r['case_id'])}.png"
        write_profile_figure(profile, fig_dir / fig_name, title)

    print(f"saved {out} rows={len(refined)}")
    print(f"saved {summary_out}")
    print(f"saved strict profile figures to {fig_dir}")


if __name__ == "__main__":
    main()
