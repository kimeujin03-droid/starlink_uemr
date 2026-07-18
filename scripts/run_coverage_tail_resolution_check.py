#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.metrics import window_mask
from pathb.pipeline import delay_axis_s, make_taper
from pathb.satellite import build_visibility_for_sat
from scripts.revision_matched_null_to_bandpower import (
    cross_delay_power,
    dirty_for_mode,
    k_axes,
    robust_z,
    robust_z_scalar,
)
from scripts.run_coverage_grid import (
    configure_case,
    independent_global_phase_sum,
    load_tle_fast,
    pte_ge,
    rank_visible_sats,
    read_uvh5_bin,
    safe_token,
    to_local_path,
)


TAIL_CASES = [
    ("frozen_polybeam", "smooth", 1000.0, "single"),
    ("frozen_polybeam", "lines", 30.0, "single"),
    ("frozen_polybeam", "lines", 1000.0, "single"),
    ("full_polybeam", "smooth", 1000.0, "single"),
    ("full_polybeam", "lines", 30.0, "single"),
    ("full_polybeam", "lines", 1000.0, "single"),
    ("frozen_polybeam", "khz_comb", 1000.0, "single"),
    ("frozen_polybeam", "smooth", 30.0, "single"),
]

PROFILE_CASES = {
    ("frozen_polybeam", "smooth", 1000.0, "single"),
    ("frozen_polybeam", "lines", 30.0, "single"),
    ("frozen_polybeam", "lines", 1000.0, "single"),
    ("full_polybeam", "smooth", 1000.0, "single"),
    ("full_polybeam", "lines", 30.0, "single"),
    ("full_polybeam", "lines", 1000.0, "single"),
}


def contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def compute_tail_case(
    ctx,
    cfg,
    sat_vis_list,
    n_null: int,
    seed: int,
    injection_mode: str,
    intrinsic_phase_seed: int | None = None,
    intrinsic_phases_rad: np.ndarray | None = None,
) -> dict:
    sat_stack = np.stack(sat_vis_list, axis=0)
    phase_array_provided = intrinsic_phases_rad is not None
    if intrinsic_phases_rad is not None:
        intrinsic_phases_rad = np.asarray(intrinsic_phases_rad, dtype=float)
        if intrinsic_phases_rad.shape != (sat_stack.shape[0],):
            raise ValueError(
                f"intrinsic_phases_rad shape {intrinsic_phases_rad.shape} "
                f"does not match satellite count {sat_stack.shape[0]}"
            )
        sat_stack = sat_stack * np.exp(1j * intrinsic_phases_rad)[:, None, None]
    else:
        intrinsic_phases_rad = np.zeros(sat_stack.shape[0], dtype=float)
    if intrinsic_phase_seed is not None and not phase_array_provided:
        phase_rng = np.random.default_rng(int(intrinsic_phase_seed))
        intrinsic_phases_rad = phase_rng.uniform(0.0, 2.0 * np.pi, size=sat_stack.shape[0])
        sat_stack = sat_stack * np.exp(1j * intrinsic_phases_rad)[:, None, None]
    sat_obs = np.sum(sat_stack, axis=0)
    taper_cfg = cfg.get("metrics", {}).get("window", {})
    taper = make_taper(len(ctx.freqs_hz), taper_cfg.get("taper", "blackman_harris"), taper_cfg)
    win = window_mask(ctx, cfg)
    delays = delay_axis_s(ctx.freqs_hz)
    kpar, _kperp, _cosmo = k_axes(ctx.freqs_hz, float(np.linalg.norm(ctx.baseline_enu_m)))

    p_bg = cross_delay_power(ctx.vis_tf, ctx.weights_tf, taper)
    p_dirty = cross_delay_power(dirty_for_mode(ctx.vis_tf, sat_obs, injection_mode), ctx.weights_tf, taper)
    bias_obs = p_dirty - p_bg

    rng = np.random.default_rng(seed)
    null_bias = []
    for _ in range(int(n_null)):
        sat_null = independent_global_phase_sum(sat_stack, rng)
        p_null = cross_delay_power(dirty_for_mode(ctx.vis_tf, sat_null, injection_mode), ctx.weights_tf, taper)
        null_bias.append(p_null - p_bg)
    null_bias = np.asarray(null_bias, dtype=float)

    bias_win = bias_obs[win]
    null_win = null_bias[:, win]
    obs_max_bias = float(np.nanmax(bias_win))
    obs_abs_integrated = float(np.nansum(np.abs(bias_win)))
    null_max_bias = np.nanmax(null_win, axis=1)
    null_abs_integrated = np.nansum(np.abs(null_win), axis=1)

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
        "bias_obs": bias_obs,
        "p_bg": p_bg,
        "null_bias": null_bias,
        "win": win,
        "delays_ns": delays * 1e9,
        "kpar": kpar,
        "z_delay": robust_z(bias_obs, null_bias),
    }


def write_profile_figure(profile: dict, fig_path: Path, title: str) -> None:
    delays = profile["delays_ns"]
    bias_obs = profile["bias_obs"]
    null_bias = profile["null_bias"]
    win = profile["win"]
    median = np.nanpercentile(null_bias, 50, axis=0)
    lo68, hi68 = np.nanpercentile(null_bias, [16, 84], axis=0)
    lo95, hi95 = np.nanpercentile(null_bias, [2.5, 97.5], axis=0)
    max_idx = int(np.flatnonzero(win)[np.nanargmax(bias_obs[win])])

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for a, b in contiguous_true_regions(win):
        ax.axvspan(delays[a], delays[b], color="0.92", zorder=0)
    ax.fill_between(delays, lo95, hi95, color="#9ecae1", alpha=0.35, linewidth=0, label="matched null 95%")
    ax.fill_between(delays, lo68, hi68, color="#3182bd", alpha=0.30, linewidth=0, label="matched null 68%")
    ax.plot(delays, median, color="#08519c", linewidth=1.2, label="matched null median")
    ax.plot(delays, bias_obs, color="#b22222", linewidth=1.2, label="observed residual")
    ax.axvline(delays[max_idx], color="black", linestyle="--", linewidth=1.0, label="max window bin")
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xlabel("delay (ns)")
    ax.set_ylabel("window residual statistic")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials.csv")
    ap.add_argument("--config", default="configs/coverage_robustness.yaml")
    ap.add_argument("--pathb-config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--out", default="outputs/coverage_tail_resolution_check.csv")
    ap.add_argument("--fig-dir", default="figures/coverage_robustness/candidate_delay_profiles")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--baseline-id", default="0_1")
    ap.add_argument("--lst-stratum", default="typical")
    ap.add_argument("--lst-bin-id", type=int, default=36)
    args = ap.parse_args()

    cfg_run = yaml.safe_load(to_local_path(args.config).read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path(args.pathb_config))
    sel = pd.read_csv(to_local_path(args.selection))
    trials = pd.read_csv(to_local_path(args.trials))

    srows = sel[
        (sel["baseline_id"].astype(str) == str(args.baseline_id))
        & (sel["lst_stratum"].astype(str) == str(args.lst_stratum))
        & (sel["lst_bin_id"].astype(int) == int(args.lst_bin_id))
    ]
    if len(srows) != 1:
        raise ValueError(f"Expected one selected LST bin row, found {len(srows)}")
    srow = srows.iloc[0]
    ctx = read_uvh5_bin(srow)

    recs, _tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026_LEO_only.tle")),
        max_records=int(cfg_run.get("max_tle_records", 1200)),
    )
    rec_map = {r.norad_id: r for r in recs}
    ranked = rank_visible_sats(recs, ctx, base_cfg, float(cfg_run.get("alt_visible_deg", 70.0)))
    if len(ranked) == 0:
        raise RuntimeError("No visible satellites for selected LST bin")
    chosen = ranked.head(1)

    rows = []
    for idx, (beam, morphology, flux, multiplicity) in enumerate(TAIL_CASES, start=1):
        old = trials[
            (trials["baseline_id"].astype(str) == str(args.baseline_id))
            & (trials["lst_stratum"].astype(str) == str(args.lst_stratum))
            & (trials["lst_bin_id"].astype(int) == int(args.lst_bin_id))
            & (trials["beam_model"].astype(str) == beam)
            & (trials["morphology"].astype(str) == morphology)
            & (trials["flux_jy"].astype(float) == float(flux))
            & (trials["multiplicity"].astype(str) == multiplicity)
        ]
        if len(old) != 1:
            raise ValueError(f"Expected one old trial for {(beam, morphology, flux, multiplicity)}, found {len(old)}")
        old_row = old.iloc[0]

        cfg = configure_case(base_cfg, beam, morphology)
        sat_vis_list = []
        for norad in chosen["norad_id"].astype(str).tolist():
            vis, _track, _report = build_visibility_for_sat(rec_map[norad], ctx, cfg, s_ref_jy=float(flux))
            sat_vis_list.append(vis)

        profile = compute_tail_case(
            ctx,
            cfg,
            sat_vis_list,
            int(args.n_null),
            int(old_row["seed"]),
            str(old_row.get("injection_mode", "coherent_ab")),
        )
        rel_bias = float(profile["relative_abs_bias"])
        pte_max_new = float(profile["PTE_global_max"])

        case_id = f"{beam}_{morphology}_S{float(flux):g}_{multiplicity}"
        rows.append(
            {
                "case_id": case_id,
                "baseline_id": args.baseline_id,
                "lst_bin_id": int(args.lst_bin_id),
                "beam_model": beam,
                "morphology": morphology,
                "flux_jy": float(flux),
                "multiplicity": multiplicity,
                "N_null": int(args.n_null),
                "Z_PS_max": profile["Z_PS_max"],
                "PTE_global_max_old": float(old_row["PTE_global_max"]),
                "PTE_global_max_new": pte_max_new,
                "PTE_global_absint_old": float(old_row["PTE_global_absint"]),
                "PTE_global_absint_new": profile["PTE_global_absint"],
                "relative_abs_bias": rel_bias,
                "window_abs_bias_sum": profile["window_abs_bias_sum"],
                "window_bg_abs_sum": profile["window_bg_abs_sum"],
                "null_mad_win": profile["null_mad_win"],
                "n_null_exceed_max": profile["n_null_exceed_max"],
                "n_null_exceed_absint": profile["n_null_exceed_absint"],
                "retained_PTE001": bool(pte_max_new < 0.01),
                "retained_PTE005": bool(pte_max_new < 0.05),
                "retained_physical_floor_1e3": bool(pte_max_new < 0.01 and rel_bias > 1e-3),
                "retained_physical_floor_1e2": bool(pte_max_new < 0.01 and rel_bias > 1e-2),
            }
        )

        if (beam, morphology, float(flux), multiplicity) in PROFILE_CASES:
            fig_name = f"{idx:02d}_{safe_token(case_id)}.png"
            title = f"{args.baseline_id} LST {args.lst_bin_id} {beam} {morphology} {float(flux):g} Jy {multiplicity}"
            write_profile_figure(profile, to_local_path(args.fig_dir) / fig_name, title)
        print(
            f"[{idx}/{len(TAIL_CASES)}] {case_id} PTE old={float(old_row['PTE_global_max']):.5f} "
            f"new={pte_max_new:.5f} exceed={profile['n_null_exceed_max']}/{args.n_null}",
            flush=True,
        )

    out = to_local_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
