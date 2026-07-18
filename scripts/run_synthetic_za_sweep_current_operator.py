#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.io_background import with_baseline
from pathb.metrics import window_mask
from pathb.satellite import build_synthetic_visibility
from scripts.run_coverage_grid import configure_case, read_uvh5_bin
from scripts.run_coverage_tail_resolution_check import compute_tail_case


ZA_VALUES = [5.0, 15.0, 30.0, 45.0, 60.0, 70.0, 78.0]
BASELINES_M = [14.6, 140.4, 207.3]
BEAMS = [
    ("full_polybeam", None, "full-chromatic HERA H2C CST PolyBeam"),
    ("unity_beam", "none", "B=1 geometry-only diagnostic"),
]


def select_background_rows(selection: pd.DataFrame, n_background: int) -> pd.DataFrame:
    usable = selection[np.asarray(selection["flag_fraction"], dtype=float) < 0.5].copy()
    if len(usable) < n_background:
        raise ValueError(f"Need {n_background} usable background rows, found {len(usable)}")
    # Deterministic spread over the existing selected crops.
    idx = np.linspace(0, len(usable) - 1, n_background).round().astype(int)
    return usable.iloc[idx].reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["beam_model", "baseline_length_m", "za_peak_deg"]
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        beam_model, baseline_length_m, za_peak_deg = keys
        strict_local = (grp["pte_global_max"] < 0.01) & (grp["relative_abs_bias"] > 1e-3)
        strict_absint = (grp["pte_global_absint"] < 0.01) & (grp["relative_abs_bias"] > 1e-3)
        rows.append(
            {
                "beam_model": beam_model,
                "baseline_length_m": baseline_length_m,
                "za_peak_deg": za_peak_deg,
                "n_background": int(len(grp)),
                "median_pte_global_max": float(np.nanmedian(grp["pte_global_max"])),
                "median_pte_global_absint": float(np.nanmedian(grp["pte_global_absint"])),
                "median_relative_abs_bias": float(np.nanmedian(grp["relative_abs_bias"])),
                "n_local_physical_1e3": int(strict_local.sum()),
                "n_absint_physical_1e3": int(strict_absint.sum()),
                "max_minus_log10_pte_absint": float(
                    np.nanmax(-np.log10(np.clip(grp["pte_global_absint"].astype(float), 1e-300, None)))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--out", default="za_sweep_current_operator.csv")
    ap.add_argument("--summary-out", default="za_sweep_current_operator_summary.csv")
    ap.add_argument("--meta-out", default="za_sweep_current_operator.meta.json")
    ap.add_argument("--n-background", type=int, default=5)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--s-ref-jy", type=float, default=300.0)
    ap.add_argument("--height-km", type=float, default=550.0)
    ap.add_argument("--az-transit-deg", type=float, default=90.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    selection = pd.read_csv(args.selection)
    bg_rows = select_background_rows(selection, args.n_background)
    base_cfg = read_config(args.config)

    records: list[dict] = []
    case_index = 0
    for bg_idx, srow in bg_rows.iterrows():
        native_ctx = read_uvh5_bin(srow)
        for baseline_m in BASELINES_M:
            ctx = with_baseline(
                native_ctx,
                np.asarray([baseline_m, 0.0, 0.0], dtype=float),
                group_id=f"ew_{baseline_m:.1f}m",
                baseline_id=f"synthetic_EW_{baseline_m:.1f}m",
            )
            win = window_mask(ctx, base_cfg)
            for za in ZA_VALUES:
                for beam_model, beam_override, beam_description in BEAMS:
                    cfg = configure_case(base_cfg, "full_polybeam", "smooth")
                    sat_vis, _track, sat_report = build_synthetic_visibility(
                        ctx,
                        cfg,
                        za_peak_deg=za,
                        az_transit_deg=args.az_transit_deg,
                        height_km=args.height_km,
                        s_ref_jy=args.s_ref_jy,
                        beam_mode_override=beam_override,
                    )
                    profile = compute_tail_case(
                        ctx,
                        cfg,
                        [sat_vis],
                        n_null=args.n_null,
                        seed=args.seed + case_index,
                        injection_mode="coherent_ab",
                    )
                    strict_local = (profile["PTE_global_max"] < 0.01) and (profile["relative_abs_bias"] > 1e-3)
                    strict_absint = (profile["PTE_global_absint"] < 0.01) and (profile["relative_abs_bias"] > 1e-3)
                    records.append(
                        {
                            "case_id": f"za{za:g}_bg{bg_idx}_b{baseline_m:.1f}_{beam_model}",
                            "background_index": int(bg_idx),
                            "background_id": native_ctx.bg_id,
                            "source_uvh5": native_ctx.source_path,
                            "lst_stratum": srow.get("lst_stratum", ""),
                            "lst_bin_id": int(srow["lst_bin_id"]),
                            "native_baseline_id": srow["baseline_id"],
                            "baseline_id": f"synthetic_EW_{baseline_m:.1f}m",
                            "baseline_length_m": float(baseline_m),
                            "beam_model": beam_model,
                            "beam_description": beam_description,
                            "morphology": "smooth",
                            "s_ref_jy": float(args.s_ref_jy),
                            "height_km": float(args.height_km),
                            "az_transit_deg": float(args.az_transit_deg),
                            "za_peak_deg": float(za),
                            "n_null": int(args.n_null),
                            "pte_global_max": profile["PTE_global_max"],
                            "pte_global_absint": profile["PTE_global_absint"],
                            "relative_abs_bias": profile["relative_abs_bias"],
                            "z_ps_max": profile["Z_PS_max"],
                            "window_abs_bias_sum": profile["window_abs_bias_sum"],
                            "window_bg_abs_sum": profile["window_bg_abs_sum"],
                            "n_null_exceed_max": profile["n_null_exceed_max"],
                            "n_null_exceed_absint": profile["n_null_exceed_absint"],
                            "strict_local_physical_1e3": bool(strict_local),
                            "strict_absint_physical_1e3": bool(strict_absint),
                            "physical_1e2": bool(profile["relative_abs_bias"] > 1e-2),
                            "window_bins": int(np.sum(win)),
                            "peak_abs_jy": sat_report.get("peak_abs_jy", np.nan),
                            "mean_abs_jy": sat_report.get("mean_abs_jy", np.nan),
                            "tau_min_ns": sat_report.get("tau_min_ns", np.nan),
                            "tau_max_ns": sat_report.get("tau_max_ns", np.nan),
                            "attenuation_median": sat_report.get("attenuation_median", np.nan),
                            "attenuation_max": sat_report.get("attenuation_max", np.nan),
                            "seed": int(args.seed + case_index),
                        }
                    )
                    case_index += 1
                    if case_index % 25 == 0:
                        print(f"[progress] completed {case_index} cases", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    summary = summarize(df)
    summary.to_csv(args.summary_out, index=False)
    meta = {
        "out": args.out,
        "summary_out": args.summary_out,
        "n_cases": int(len(df)),
        "n_background": int(args.n_background),
        "n_null": int(args.n_null),
        "za_values": ZA_VALUES,
        "baselines_m": BASELINES_M,
        "beams": [b[0] for b in BEAMS],
        "s_ref_jy": float(args.s_ref_jy),
        "height_km": float(args.height_km),
        "elapsed_s": float(time.time() - t0),
        "background_rows": bg_rows[
            ["baseline_id", "lst_stratum", "lst_bin_id", "source_uvh5", "flag_fraction"]
        ].to_dict(orient="records"),
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(df)} rows to {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
