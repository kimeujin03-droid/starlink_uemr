#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.metrics import window_mask
from pathb.pipeline import make_taper
from pathb.satellite import build_visibility_for_sat
from scripts.run_coverage_grid import (
    configure_case,
    load_tle_fast,
    pte_ge,
    rank_visible_sats,
    read_uvh5_bin,
    to_local_path,
)
from scripts.revision_matched_null_to_bandpower import cross_delay_power, dirty_for_mode, robust_z_scalar


def stable_u32(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def phase_for_sat(global_phase_seed: int, satellite_id: str) -> float:
    rng = np.random.default_rng(stable_u32("intrinsic-phase", global_phase_seed, satellite_id))
    return float(rng.uniform(0.0, 2.0 * np.pi))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def percentile_rank(samples: pd.Series, value: float, higher_is_more_extreme: bool = True) -> float:
    arr = pd.to_numeric(samples, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0 or not np.isfinite(value):
        return float("nan")
    if higher_is_more_extreme:
        return float(100.0 * np.mean(arr <= value))
    return float(100.0 * np.mean(arr >= value))


def paired_case_id(row: pd.Series) -> str:
    return (
        f"{row['baseline_id']}_lst{int(row['lst_bin_id']):03d}_{row['lst_stratum']}_"
        f"{row['morphology']}_S{float(row['flux_jy']):g}_{row['multiplicity']}"
    )


def classify_row(row: dict) -> str:
    if row["strict_local"] and row["strict_integrated"] and row["physical_1e3"]:
        return "local+integrated physical"
    if row["strict_integrated"] and row["physical_1e3"]:
        return "integrated physical"
    if row["strict_local"] and row["physical_1e3"]:
        return "local physical"
    if row["strict_local"] or row["strict_integrated"]:
        return "statistical QA flag"
    return "no strict excess"


def _delay_transform_nd(vis: np.ndarray, weights_tf: np.ndarray, taper: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(vis), vis, 0.0 + 0.0j) * np.clip(weights_tf, 0.0, 1.0) * taper
    return np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1)


def cross_delay_power_batch(vis_btf: np.ndarray, weights_tf: np.ndarray, taper: np.ndarray) -> np.ndarray:
    even = np.arange(weights_tf.shape[0]) % 2 == 0
    odd = ~even
    da = _delay_transform_nd(vis_btf[:, even, :], weights_tf[even][None, :, :], taper[None, None, :])
    db = _delay_transform_nd(vis_btf[:, odd, :], weights_tf[odd][None, :, :], taper[None, None, :])
    n = min(da.shape[1], db.shape[1])
    if n == 0:
        return np.full((vis_btf.shape[0], vis_btf.shape[2]), np.nan)
    return np.nanmean(np.real(da[:, :n, :] * np.conj(db[:, :n, :])), axis=1)


def dirty_batch_for_mode(bg_vis: np.ndarray, sat_vis_btf: np.ndarray, mode: str) -> np.ndarray:
    dirty = np.broadcast_to(bg_vis[None, :, :], sat_vis_btf.shape).copy()
    if mode == "coherent_ab":
        dirty += sat_vis_btf
    elif mode == "a_only":
        even = np.arange(bg_vis.shape[0]) % 2 == 0
        dirty[:, even, :] += sat_vis_btf[:, even, :]
    else:
        raise ValueError(f"Unknown injection mode: {mode}")
    return dirty


def compute_tail_case_fast(
    ctx,
    cfg,
    sat_vis_list: list[np.ndarray],
    n_null: int,
    seed: int,
    injection_mode: str,
    intrinsic_phase_seed: int,
    intrinsic_phases_rad: np.ndarray,
    batch_size: int = 100,
) -> dict:
    sat_stack = np.stack(sat_vis_list, axis=0)
    intrinsic_phases_rad = np.asarray(intrinsic_phases_rad, dtype=float)
    sat_stack = sat_stack * np.exp(1j * intrinsic_phases_rad)[:, None, None]
    sat_obs = np.sum(sat_stack, axis=0)

    taper_cfg = cfg.get("metrics", {}).get("window", {})
    taper = make_taper(len(ctx.freqs_hz), taper_cfg.get("taper", "blackman_harris"), taper_cfg)
    win = window_mask(ctx, cfg)

    p_bg = cross_delay_power(ctx.vis_tf, ctx.weights_tf, taper)
    p_dirty = cross_delay_power(dirty_for_mode(ctx.vis_tf, sat_obs, injection_mode), ctx.weights_tf, taper)
    bias_obs = p_dirty - p_bg
    bias_win = bias_obs[win]
    obs_max_bias = float(np.nanmax(bias_win))
    obs_abs_integrated = float(np.nansum(np.abs(bias_win)))

    rng = np.random.default_rng(seed)
    null_max_chunks = []
    null_abs_chunks = []
    remaining = int(n_null)
    while remaining > 0:
        b = min(int(batch_size), remaining)
        phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(b, sat_stack.shape[0])))
        sat_null = np.einsum("bs,stf->btf", phases, sat_stack, optimize=True)
        p_null = cross_delay_power_batch(dirty_batch_for_mode(ctx.vis_tf, sat_null, injection_mode), ctx.weights_tf, taper)
        null_win = p_null[:, win] - p_bg[None, win]
        null_max_chunks.append(np.nanmax(null_win, axis=1))
        null_abs_chunks.append(np.nansum(np.abs(null_win), axis=1))
        remaining -= b

    null_max_bias = np.concatenate(null_max_chunks)
    null_abs_integrated = np.concatenate(null_abs_chunks)
    return {
        "Z_PS_max": robust_z_scalar(obs_max_bias, null_max_bias),
        "PTE_global_max": pte_ge(obs_max_bias, null_max_bias),
        "PTE_global_absint": pte_ge(obs_abs_integrated, null_abs_integrated),
        "relative_abs_bias": float(np.nansum(np.abs(bias_win)) / max(np.nansum(np.abs(p_bg[win])), 1e-30)),
        "window_abs_bias_sum": float(np.nansum(np.abs(bias_win))),
        "window_bg_abs_sum": float(np.nansum(np.abs(p_bg[win]))),
        "null_mad_win": float(np.nanmedian(np.abs(null_max_bias - np.nanmedian(null_max_bias)))),
        "n_null_exceed_max": int(np.sum(null_max_bias >= obs_max_bias)),
        "n_null_exceed_absint": int(np.sum(null_abs_integrated >= obs_abs_integrated)),
        "intrinsic_phase_seed": intrinsic_phase_seed,
        "intrinsic_phases_rad": intrinsic_phases_rad,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials.csv")
    ap.add_argument("--case-table", default="coverage_absint_tail_refined_pte003_brel1e3.csv")
    ap.add_argument("--config", default="configs/coverage_robustness.yaml")
    ap.add_argument("--pathb-config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--out-trials", default="outputs/intrinsic_phase_multiseed_trials.csv")
    ap.add_argument("--out-summary", default="outputs/intrinsic_phase_multiseed_summary.csv")
    ap.add_argument("--out-pairs", default="outputs/intrinsic_phase_beam_pairs.csv")
    ap.add_argument("--out-counts", default="outputs/intrinsic_phase_class_counts.csv")
    ap.add_argument("--out-percentiles", default="outputs/intrinsic_phase_coherent_percentiles.csv")
    ap.add_argument("--out-md", default="2026-07-13_intrinsic_phase_multiseed_results.md")
    ap.add_argument("--n-phase", type=int, default=100)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--phase-seed-base", type=int, default=2026071300)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    cfg_run = yaml.safe_load(to_local_path(args.config).read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path(args.pathb_config))
    selection = pd.read_csv(to_local_path(args.selection))
    trials = pd.read_csv(to_local_path(args.trials))
    cases = pd.read_csv(to_local_path(args.case_table))
    cases = cases.drop_duplicates(
        ["baseline_id", "lst_stratum", "lst_bin_id", "beam_model", "morphology", "flux_jy", "multiplicity"]
    ).copy()
    if args.max_rows > 0:
        cases = cases.head(args.max_rows).copy()

    recs, _tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026_LEO_only.tle")),
        max_records=int(cfg_run.get("max_tle_records", 1200)),
    )
    rec_map = {r.norad_id: r for r in recs}

    contexts = {}
    rankings = {}
    sat_cache = {}
    rows = []

    for row_i, case in cases.reset_index(drop=True).iterrows():
        cell_key = (str(case["baseline_id"]), str(case["lst_stratum"]), int(case["lst_bin_id"]))
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

        multiplicity = str(case["multiplicity"])
        n_sat_target = 1 if multiplicity == "single" else int(cfg_run.get("max_multi_satellites", 12))
        chosen = ranked.head(max(1, min(n_sat_target, len(ranked))))
        satellite_ids = chosen["norad_id"].astype(str).tolist()
        beam = str(case["beam_model"])
        morphology = str(case["morphology"])
        flux = float(case["flux_jy"])
        cfg = configure_case(base_cfg, beam, morphology)

        trial_match = trials[
            (trials["baseline_id"].astype(str) == str(case["baseline_id"]))
            & (trials["lst_stratum"].astype(str) == str(case["lst_stratum"]))
            & (trials["lst_bin_id"].astype(int) == int(case["lst_bin_id"]))
            & (trials["beam_model"].astype(str) == beam)
            & (trials["morphology"].astype(str) == morphology)
            & (trials["flux_jy"].astype(float) == flux)
            & (trials["multiplicity"].astype(str) == multiplicity)
        ]
        if len(trial_match) != 1:
            raise ValueError(f"Expected one original trial for {case.to_dict()}, found {len(trial_match)}")
        trial_row = trial_match.iloc[0]
        injection_mode = str(trial_row.get("injection_mode", "coherent_ab"))
        original_seed = int(trial_row["seed"])

        sat_vis_list = []
        for norad in satellite_ids:
            cache_key = (cell_key, beam, morphology, flux, norad)
            if cache_key not in sat_cache:
                vis, _track, _report = build_visibility_for_sat(rec_map[norad], ctx, cfg, s_ref_jy=flux)
                sat_cache[cache_key] = vis
            sat_vis_list.append(sat_cache[cache_key])

        for phase_i in range(int(args.n_phase)):
            phase_seed = int(args.phase_seed_base + phase_i)
            phases = np.asarray([phase_for_sat(phase_seed, norad) for norad in satellite_ids], dtype=float)
            null_seed = stable_u32(
                "intrinsic-phase-null",
                phase_seed,
                case["baseline_id"],
                case["lst_stratum"],
                int(case["lst_bin_id"]),
                morphology,
                flux,
                multiplicity,
            )
            profile = compute_tail_case_fast(
                ctx,
                cfg,
                sat_vis_list,
                int(args.n_null),
                int(null_seed),
                injection_mode,
                phase_seed,
                phases,
            )
            pte_max = float(profile["PTE_global_max"])
            pte_abs = float(profile["PTE_global_absint"])
            brel = float(profile["relative_abs_bias"])
            rec = {
                "case_id": str(case["case_id"]),
                "paired_case_id": paired_case_id(case),
                "phase_seed": phase_seed,
                "phase_condition": "random_intrinsic",
                "satellite_count": len(satellite_ids),
                "satellite_ids": ";".join(satellite_ids),
                "beam_model": beam,
                "morphology": morphology,
                "flux_jy": flux,
                "multiplicity": multiplicity,
                "baseline_id": case["baseline_id"],
                "baseline_length_m": float(case["baseline_length_m"]),
                "lst_stratum": case["lst_stratum"],
                "lst_bin_id": int(case["lst_bin_id"]),
                "N_null": int(args.n_null),
                "pte_global_max": pte_max,
                "pte_global_absint": pte_abs,
                "relative_abs_bias": brel,
                "t_max": float("nan"),
                "t_absint": float(profile["window_abs_bias_sum"]),
                "z_ps_max": float(profile["Z_PS_max"]),
                "n_null_exceed_max": int(profile["n_null_exceed_max"]),
                "n_null_exceed_absint": int(profile["n_null_exceed_absint"]),
                "strict_local": bool(pte_max < 0.01),
                "strict_integrated": bool(pte_abs < 0.01),
                "physical_1e3": bool(brel > 1e-3),
                "physical_1e2": bool(brel > 1e-2),
                "integrated_physical_1e3": bool(pte_abs < 0.01 and brel > 1e-3),
                "integrated_physical_1e2": bool(pte_abs < 0.01 and brel > 1e-2),
                "margin_integrated": float(-np.log10(max(pte_abs, 1e-300)) - 2.0),
                "margin_local": float(-np.log10(max(pte_max, 1e-300)) - 2.0),
                "margin_bias_1e3": float(np.log10(max(brel, 1e-300) / 1e-3)),
                "margin_bias_1e2": float(np.log10(max(brel, 1e-300) / 1e-2)),
                "intrinsic_phases_rad": ";".join(f"{x:.8f}" for x in phases),
                "null_seed_base": int(null_seed),
                "original_trial_seed": original_seed,
            }
            rec["final_class"] = classify_row(rec)
            rows.append(rec)

        print(
            f"[{row_i + 1}/{len(cases)}] {case['case_id']} "
            f"done {args.n_phase} phase seeds x {args.n_null} nulls",
            flush=True,
        )

    trial_df = pd.DataFrame(rows)
    for path_arg in ["out_trials", "out_summary", "out_pairs", "out_counts", "out_percentiles"]:
        to_local_path(getattr(args, path_arg)).parent.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(to_local_path(args.out_trials), index=False)

    pair_rows = []
    for (pcase, phase_seed), grp in trial_df.groupby(["paired_case_id", "phase_seed"], dropna=False):
        if set(grp["beam_model"]) < {"frozen_polybeam", "full_polybeam"}:
            continue
        frozen = grp[grp["beam_model"] == "frozen_polybeam"].iloc[0]
        full = grp[grp["beam_model"] == "full_polybeam"].iloc[0]
        pair_rows.append(
            {
                "paired_case_id": pcase,
                "phase_seed": phase_seed,
                "morphology": frozen["morphology"],
                "flux_jy": frozen["flux_jy"],
                "multiplicity": frozen["multiplicity"],
                "baseline_id": frozen["baseline_id"],
                "lst_bin_id": frozen["lst_bin_id"],
                "pte_absint_frozen": frozen["pte_global_absint"],
                "pte_absint_full": full["pte_global_absint"],
                "brel_frozen": frozen["relative_abs_bias"],
                "brel_full": full["relative_abs_bias"],
                "strict_integrated_frozen": frozen["strict_integrated"],
                "strict_integrated_full": full["strict_integrated"],
                "integrated_physical_1e3_frozen": frozen["integrated_physical_1e3"],
                "integrated_physical_1e3_full": full["integrated_physical_1e3"],
                "beam_robust_integrated_1e3": bool(frozen["integrated_physical_1e3"] and full["integrated_physical_1e3"]),
                "frozen_only_integrated_1e3": bool(frozen["integrated_physical_1e3"] and not full["integrated_physical_1e3"]),
                "full_only_integrated_1e3": bool(full["integrated_physical_1e3"] and not frozen["integrated_physical_1e3"]),
                "delta_beam_logpte": float(np.log10(max(full["pte_global_absint"], 1e-300) / max(frozen["pte_global_absint"], 1e-300))),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(to_local_path(args.out_pairs), index=False)

    summary_rows = []
    for (pcase, beam), grp in trial_df.groupby(["paired_case_id", "beam_model"], dropna=False):
        n = int(len(grp))
        row = {
            "paired_case_id": pcase,
            "beam_model": beam,
            "n_phase_seeds": n,
            "n_strict_local": int(grp["strict_local"].sum()),
            "n_strict_integrated": int(grp["strict_integrated"].sum()),
            "n_physical_1e3": int(grp["physical_1e3"].sum()),
            "n_physical_1e2": int(grp["physical_1e2"].sum()),
            "n_integrated_physical_1e3": int(grp["integrated_physical_1e3"].sum()),
            "n_integrated_physical_1e2": int(grp["integrated_physical_1e2"].sum()),
        }
        for key in ["strict_local", "strict_integrated", "integrated_physical_1e3", "integrated_physical_1e2"]:
            k = int(grp[key].sum())
            lo, hi = wilson_ci(k, n)
            row[f"rate_{key}"] = k / n if n else float("nan")
            row[f"rate_{key}_wilson95_lo"] = lo
            row[f"rate_{key}_wilson95_hi"] = hi
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    if not pair_df.empty:
        pair_summary = pair_df.groupby("paired_case_id").agg(
            n_pair_phase_seeds=("phase_seed", "count"),
            n_beam_robust_integrated_1e3=("beam_robust_integrated_1e3", "sum"),
            n_frozen_only_integrated_1e3=("frozen_only_integrated_1e3", "sum"),
            n_full_only_integrated_1e3=("full_only_integrated_1e3", "sum"),
            median_delta_beam_logpte=("delta_beam_logpte", "median"),
        ).reset_index()
        summary_df = summary_df.merge(pair_summary, on="paired_case_id", how="left")
    summary_df.to_csv(to_local_path(args.out_summary), index=False)

    counts = (
        trial_df.groupby(["paired_case_id", "beam_model", "final_class"])
        .size()
        .reset_index(name="count")
        .sort_values(["paired_case_id", "beam_model", "final_class"])
    )
    counts.to_csv(to_local_path(args.out_counts), index=False)

    coherent = cases.copy()
    coherent["paired_case_id"] = coherent.apply(paired_case_id, axis=1)
    pct_rows = []
    for _, crow in coherent.iterrows():
        grp = trial_df[
            (trial_df["paired_case_id"] == crow["paired_case_id"])
            & (trial_df["beam_model"] == crow["beam_model"])
        ]
        pct_rows.append(
            {
                "paired_case_id": crow["paired_case_id"],
                "beam_model": crow["beam_model"],
                "coherent_pte_global_absint": float(crow["PTE_global_absint_new"]),
                "coherent_minuslog10_pte_absint": float(-np.log10(max(float(crow["PTE_global_absint_new"]), 1e-300))),
                "coherent_relative_abs_bias": float(crow["relative_abs_bias"]),
                "coherent_t_absint": float(crow["window_abs_bias_sum"]),
                "percentile_t_absint_vs_random": percentile_rank(grp["t_absint"], float(crow["window_abs_bias_sum"]), True),
                "percentile_minuslog10_pte_absint_vs_random": percentile_rank(
                    -np.log10(grp["pte_global_absint"].clip(lower=1e-300)),
                    -np.log10(max(float(crow["PTE_global_absint_new"]), 1e-300)),
                    True,
                ),
                "percentile_brel_vs_random": percentile_rank(grp["relative_abs_bias"], float(crow["relative_abs_bias"]), True),
            }
        )
    percentile_df = pd.DataFrame(pct_rows)
    percentile_df.to_csv(to_local_path(args.out_percentiles), index=False)

    total_pairs = int(len(pair_df))
    beam_robust = int(pair_df["beam_robust_integrated_1e3"].sum()) if total_pairs else 0
    frozen_only = int(pair_df["frozen_only_integrated_1e3"].sum()) if total_pairs else 0
    full_only = int(pair_df["full_only_integrated_1e3"].sum()) if total_pairs else 0
    strict_integrated = int(trial_df["strict_integrated"].sum())
    integrated_phys = int(trial_df["integrated_physical_1e3"].sum())
    md = f"""# Intrinsic Phase Multi-Seed Audit Results

Run date: 2026-07-13

## Scope

- Input case table: `{args.case_table}`
- Phase seeds: {args.n_phase}
- Null draws per row/phase: {args.n_null}
- Trial rows: {len(trial_df)}
- Paired frozen/full phase comparisons: {total_pairs}
- Phase seed base: `{args.phase_seed_base}`

## Top-Line Counts

| metric | count |
| --- | ---: |
| strict integrated rows | {strict_integrated} |
| strict integrated + `B_rel > 1e-3` rows | {integrated_phys} |
| paired beam-robust integrated + `B_rel > 1e-3` phase cases | {beam_robust} |
| frozen-only integrated + `B_rel > 1e-3` phase cases | {frozen_only} |
| full-only integrated + `B_rel > 1e-3` phase cases | {full_only} |

## Outputs

- `{args.out_trials}`
- `{args.out_summary}`
- `{args.out_pairs}`
- `{args.out_counts}`
- `{args.out_percentiles}`

## Interpretation Placeholder

Use the summary and paired-beam tables above for manuscript wording. If the
beam-robust count remains zero in the 100-seed run, describe the coherent-default
integrated excess as a phase-sensitive stress response under the sampled
intrinsic-phase model, not as a repeatedly reproduced beam-robust contamination
candidate.
"""
    to_local_path(args.out_md).write_text(md, encoding="utf-8")
    print(f"saved {args.out_trials} rows={len(trial_df)}")
    print(f"saved {args.out_summary}")
    print(f"saved {args.out_pairs}")
    print(f"saved {args.out_counts}")
    print(f"saved {args.out_percentiles}")
    print(f"saved {args.out_md}")


if __name__ == "__main__":
    main()
