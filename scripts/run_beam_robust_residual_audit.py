#!/usr/bin/env python3
"""Calibration-residual stress test for the four-flux N_null=1000 beam-robust
integrated (PTE_global_absint) candidate pairs.

Unlike scripts/run_targeted_candidate_calibration_audit.py (which reuses the
strict, near-bit-identical `classify_case` beam-robust *contamination*
definition from the local-floor candidate audit), this script tests the
looser "beam-robust integrated" definition used by
scripts/run_coverage_tail_refined_near_threshold.py in --selection-mode
absint: a (frozen_polybeam, full_polybeam) pair is beam-robust if BOTH beam
models independently satisfy PTE_global_absint < 0.01 and
relative_abs_bias (B_rel) > the floor, under N_null matched-null nulls.

Satellite visibilities depend only on background geometry (times, baseline,
frequencies), not on the injected calibration residual (which only
perturbs the background vis_tf amplitude), so they are built once per
pair/beam and reused across all sigma_cal x residual_model x ell_nu
conditions. Pairs are independent and are farmed out across worker
processes.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.io_background import BackgroundContext
from pathb.satellite import build_visibility_for_sat
from scripts.run_coverage_grid import configure_case, load_tle_fast, rank_visible_sats, read_uvh5_bin, to_local_path
from scripts.run_coverage_tail_resolution_check import compute_tail_case

DEFAULT_SIGMAS = [0.0, 1e-4, 1e-3, 1e-2]
DEFAULT_ELL_NU_MHZ = [0.5, 1.0, 5.0]
DEFAULT_MODELS = ["white", "smooth"]


def make_rng(seed: int, pair_idx: int, model_idx: int, sigma_idx: int, ell_idx: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(seed), int(pair_idx), int(model_idx), int(sigma_idx), int(ell_idx)])
    return np.random.default_rng(ss)


def build_cal_residual(
    shape: tuple[int, int],
    sigma_cal: float,
    model: str,
    rng: np.random.Generator,
    ell_nu_mhz: float | None = None,
    freq_hz: np.ndarray | None = None,
) -> np.ndarray:
    if sigma_cal <= 0.0:
        return np.zeros(shape, dtype=float)
    white = rng.normal(size=shape)
    model = model.lower()
    if model == "white":
        residual = white
    elif model in {"smooth", "chromatic", "smooth/chromatic"}:
        if freq_hz is None:
            raise ValueError("freq_hz is required for chromatic residuals")
        df_hz = float(np.median(np.diff(freq_hz))) if len(freq_hz) > 1 else 1.0
        ell_hz = float(ell_nu_mhz or 1.0) * 1e6
        sigma_chan = max(ell_hz / max(df_hz, 1e-30), 1e-6)
        residual = gaussian_filter1d(white, sigma=sigma_chan, axis=1, mode="reflect")
    else:
        raise ValueError(f"unknown residual model: {model}")

    resid_std = float(np.std(residual))
    if not np.isfinite(resid_std) or resid_std <= 1e-30:
        return np.zeros(shape, dtype=float)
    residual = residual / resid_std * float(sigma_cal)
    return residual.astype(float)


def apply_residual(ctx: BackgroundContext, residual: np.ndarray) -> BackgroundContext:
    vis = np.asarray(ctx.vis_tf, dtype=complex) * (1.0 + residual)
    return replace(ctx, vis_tf=vis)


def pair_status(frozen_ok: bool, full_ok: bool) -> str:
    if frozen_ok and full_ok:
        return "beam-robust retained"
    if frozen_ok and not full_ok:
        return "frozen-only"
    if full_ok and not frozen_ok:
        return "full-only"
    return "dropped"


def condition_list(models: list[str], sigmas: list[float], ell_nu_mhz: list[float]) -> list[tuple[int, str, int, float, int, float | None]]:
    conditions = []
    for model_idx, model in enumerate(models):
        ell_values = [None] if model == "white" else list(ell_nu_mhz)
        for sigma_idx, sigma_cal in enumerate(sigmas):
            for ell_idx, ell in enumerate(ell_values):
                conditions.append((model_idx, model, sigma_idx, sigma_cal, ell_idx, ell))
    return conditions


def process_pair(
    pair_row: dict,
    pair_idx: int,
    conditions: list[tuple[int, str, int, float, int, float | None]],
    args_ns: argparse.Namespace,
) -> list[dict]:
    t0 = time.time()
    cfg_run = yaml.safe_load(to_local_path(args_ns.config).read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path(args_ns.pathb_config))
    selection = pd.read_csv(to_local_path(args_ns.selection))
    recs, _tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026_LEO_only.tle")),
        max_records=int(cfg_run.get("max_tle_records", 1200)),
    )
    rec_map = {r.norad_id: r for r in recs}
    alt_visible_deg = float(cfg_run.get("alt_visible_deg", 70.0))
    max_multi_satellites = int(cfg_run.get("max_multi_satellites", 12))

    baseline_id = str(pair_row["baseline_id"])
    lst_stratum = str(pair_row["lst_stratum"])
    lst_bin_id = int(pair_row["lst_bin_id"])
    morphology = str(pair_row["morphology"])
    flux = float(pair_row["flux_jy"])
    multiplicity = str(pair_row["multiplicity"])
    seed_frozen = int(pair_row["seed_frozen"])
    seed_full = int(pair_row["seed_full"])
    inj_frozen = str(pair_row["injection_mode_frozen"])
    inj_full = str(pair_row["injection_mode_full"])

    srows = selection[
        (selection["baseline_id"].astype(str) == baseline_id)
        & (selection["lst_stratum"].astype(str) == lst_stratum)
        & (selection["lst_bin_id"].astype(int) == lst_bin_id)
    ]
    if len(srows) != 1:
        raise ValueError(f"Expected one selection row for {(baseline_id, lst_stratum, lst_bin_id)}, found {len(srows)}")
    ctx0 = read_uvh5_bin(srows.iloc[0])

    ranked = rank_visible_sats(recs, ctx0, base_cfg, alt_visible_deg)
    if len(ranked) == 0:
        raise RuntimeError(f"No visible satellites for {(baseline_id, lst_stratum, lst_bin_id)}")
    n_sat_target = 1 if multiplicity == "single" else max_multi_satellites
    chosen_norads = ranked.head(max(1, min(n_sat_target, len(ranked))))["norad_id"].astype(str).tolist()

    frozen_cfg = configure_case(base_cfg, "frozen_polybeam", morphology)
    full_cfg = configure_case(base_cfg, "full_polybeam", morphology)

    # satellite visibility depends only on ctx0 geometry (times/freqs/baseline),
    # not on the calibration residual applied to vis_tf, so build once per beam.
    frozen_vis_list = [
        build_visibility_for_sat(rec_map[n], ctx0, frozen_cfg, s_ref_jy=flux)[0] for n in chosen_norads
    ]
    full_vis_list = [
        build_visibility_for_sat(rec_map[n], ctx0, full_cfg, s_ref_jy=flux)[0] for n in chosen_norads
    ]

    rows = []
    seen_zero_sigma: dict[str, dict] = {}
    for model_idx, model, sigma_idx, sigma_cal, ell_idx, ell_nu_mhz in conditions:
        cache_key = None
        if float(sigma_cal) == 0.0:
            cache_key = "zero"
            if cache_key in seen_zero_sigma:
                cached = seen_zero_sigma[cache_key]
                row_out = dict(cached)
                row_out["residual_model"] = model
                row_out["ell_nu_mhz"] = float(ell_nu_mhz) if ell_nu_mhz is not None else None
                rows.append(row_out)
                continue

        rng = make_rng(args_ns.seed, pair_idx, model_idx, sigma_idx, ell_idx)
        residual = build_cal_residual(
            shape=ctx0.vis_tf.shape,
            sigma_cal=float(sigma_cal),
            model=model,
            rng=rng,
            ell_nu_mhz=ell_nu_mhz,
            freq_hz=ctx0.freqs_hz,
        )
        ctx = apply_residual(ctx0, residual)

        frozen_profile = compute_tail_case(
            ctx, frozen_cfg, frozen_vis_list, int(args_ns.n_null), seed_frozen, inj_frozen
        )
        full_profile = compute_tail_case(
            ctx, full_cfg, full_vis_list, int(args_ns.n_null), seed_full, inj_full
        )

        frozen_pte_abs = float(frozen_profile["PTE_global_absint"])
        frozen_brel = float(frozen_profile["relative_abs_bias"])
        full_pte_abs = float(full_profile["PTE_global_absint"])
        full_brel = float(full_profile["relative_abs_bias"])

        frozen_ok = bool(frozen_pte_abs < 0.01 and frozen_brel > args_ns.bias_floor)
        full_ok = bool(full_pte_abs < 0.01 and full_brel > args_ns.bias_floor)
        status = pair_status(frozen_ok, full_ok)

        row_out = {
            "pair": int(pair_row["pair"]),
            "baseline_id": baseline_id,
            "lst_stratum": lst_stratum,
            "lst_bin_id": lst_bin_id,
            "morphology": morphology,
            "flux_jy": flux,
            "multiplicity": multiplicity,
            "residual_model": model,
            "sigma_cal": float(sigma_cal),
            "ell_nu_mhz": float(ell_nu_mhz) if ell_nu_mhz is not None else None,
            "bias_floor": float(args_ns.bias_floor),
            "frozen_PTE_max_1000": float(frozen_profile["PTE_global_max"]),
            "frozen_PTE_absint_1000": frozen_pte_abs,
            "frozen_B_rel": frozen_brel,
            "frozen_retained": frozen_ok,
            "full_PTE_max_1000": float(full_profile["PTE_global_max"]),
            "full_PTE_absint_1000": full_pte_abs,
            "full_B_rel": full_brel,
            "full_retained": full_ok,
            "pair_status": status,
            "seed_frozen": seed_frozen,
            "seed_full": seed_full,
            "cal_seed": int(args_ns.seed),
        }
        rows.append(row_out)
        if cache_key is not None:
            seen_zero_sigma[cache_key] = row_out

    elapsed = time.time() - t0
    print(
        f"[pair {int(pair_row['pair'])}] {baseline_id}/{lst_stratum}/{lst_bin_id} {morphology} "
        f"{flux:g}Jy done in {elapsed:.1f}s ({len(rows)} rows)",
        flush=True,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/beam_robust_absint_pairs_fourflux_1e3.csv")
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--config", default="configs/coverage_robustness_all_tle.yaml")
    ap.add_argument("--pathb-config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--out", default="outputs/beam_robust_residual_audit_fourflux.csv")
    ap.add_argument("--summary-out", default="outputs/beam_robust_residual_audit_fourflux_summary.csv")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--sigmas", nargs="*", type=float, default=DEFAULT_SIGMAS)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--ell-nu-mhz", nargs="*", type=float, default=DEFAULT_ELL_NU_MHZ)
    ap.add_argument("--bias-floor", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    pairs = pd.read_csv(to_local_path(args.pairs))
    conditions = condition_list(args.models, args.sigmas, args.ell_nu_mhz)
    print(f"{len(pairs)} pairs x {len(conditions)} conditions = {len(pairs) * len(conditions)} (pair,condition) cells")

    all_rows: list[dict] = []
    if args.workers <= 1:
        for pair_idx, row in pairs.iterrows():
            all_rows.extend(process_pair(row.to_dict(), pair_idx, conditions, args))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(process_pair, row.to_dict(), pair_idx, conditions, args): pair_idx
                for pair_idx, row in pairs.iterrows()
            }
            for fut in as_completed(futs):
                all_rows.extend(fut.result())

    out = to_local_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows).sort_values(["pair", "residual_model", "sigma_cal", "ell_nu_mhz"], na_position="first")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["residual_model", "sigma_cal", "ell_nu_mhz", "pair_status"], dropna=False)
        .size()
        .reset_index(name="n_pairs")
        .sort_values(["residual_model", "sigma_cal", "ell_nu_mhz", "pair_status"])
    )
    summary.to_csv(to_local_path(args.summary_out), index=False)
    print(summary.to_string(index=False))
    print(f"saved {out}")
    print(f"saved {to_local_path(args.summary_out)}")


if __name__ == "__main__":
    main()
