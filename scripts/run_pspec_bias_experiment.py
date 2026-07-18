#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathb.config import read_config
from pathb.io_background import load_background_npz, with_baseline
from pathb.metrics import window_mask
from pathb.pipeline import C_M_PER_S, delay_axis_s, make_taper, pipeline, weighted_delay_transform
from pathb.runner import _baseline_vector
from pathb.satellite import build_starlink_visibility


NU21_HZ = 1_420_405_751.77
H0_KM_S_MPC = 67.74
OMEGA_M = 0.3089
OMEGA_L = 0.6911


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def flux_tag(flux_jy: float) -> str:
    flux = float(flux_jy)
    if flux.is_integer():
        return str(int(flux))
    return ("%g" % flux).replace(".", "p")


def e_z(z: float) -> float:
    return float(np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L))


def comoving_distance_mpc(z: float, n: int = 4096) -> float:
    c_km_s = C_M_PER_S / 1000.0
    grid = np.linspace(0.0, z, n)
    y = 1.0 / np.sqrt(OMEGA_M * (1.0 + grid) ** 3 + OMEGA_L)
    return float((c_km_s / H0_KM_S_MPC) * np.trapezoid(y, grid))


def k_axes(freqs_hz: np.ndarray, baseline_len_m: float) -> Tuple[np.ndarray, float, Dict[str, float]]:
    nu = float(np.nanmedian(freqs_hz))
    z = NU21_HZ / nu - 1.0
    x_mpc = comoving_distance_mpc(z)
    y_mpc_hz = (C_M_PER_S / 1000.0) * (1.0 + z) ** 2 / (H0_KM_S_MPC * e_z(z) * NU21_HZ)
    delays_s = delay_axis_s(freqs_hz)
    kpar = 2.0 * np.pi * delays_s / y_mpc_hz
    lam_m = C_M_PER_S / nu
    kperp = 2.0 * np.pi * baseline_len_m / (x_mpc * lam_m)
    return kpar, float(kperp), {"z_eff": z, "X_mpc": x_mpc, "Y_mpc_per_hz": y_mpc_hz, "nu_eff_hz": nu}


def delay_power(vis_tf: np.ndarray, ctx, cfg: Dict[str, Any]) -> np.ndarray:
    mc = cfg.get("metrics", {}).get("window", {})
    taper = make_taper(len(ctx.freqs_hz), mc.get("taper", "blackman_harris"), mc)
    dly = weighted_delay_transform(vis_tf, ctx.weights_tf, taper)
    return np.nanmean(np.abs(dly) ** 2, axis=0)


def scalar_window_bias(p_bg: np.ndarray, p_dirty: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    eps = 1e-30
    delta = p_dirty - p_bg
    bg = float(np.nansum(p_bg[mask]))
    abs_delta = float(np.nansum(np.abs(delta[mask])))
    signed_delta = float(np.nansum(delta[mask]))
    return {
        "B_win_abs_db": float(10.0 * np.log10(max(abs_delta, eps) / max(bg, eps))),
        "B_win_signed_db": float(10.0 * np.log10(max(abs(signed_delta), eps) / max(bg, eps))),
        "window_bg_power": bg,
        "window_abs_delta_power": abs_delta,
        "window_signed_delta_power": signed_delta,
    }


def stage1_plot(stage1: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stage1["tau_ns"], stage1["P_bg"], label="P_bg")
    ax.plot(stage1["tau_ns"], stage1["P_dirty"], label="P_dirty", alpha=0.9)
    ax.plot(stage1["tau_ns"], np.abs(stage1["DeltaP"]), label="|DeltaP|", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("delay tau [ns]")
    ax.set_ylabel("pspec-like power [arb.]")
    ax.axvspan(stage1.loc[stage1["window_mask"], "tau_ns"].min(), stage1["tau_ns"].min(), alpha=0.08)
    ax.axvspan(stage1.loc[stage1["window_mask"], "tau_ns"].max(), stage1["tau_ns"].max(), alpha=0.08)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_delay_pspec.png", dpi=200)
    plt.close(fig)


def cylindrical_plot(cyl: pd.DataFrame, out_dir: Path) -> None:
    pivot = cyl.pivot(index="kpar_abs_center", columns="kperp_center", values="Delta2_rel")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
        ],
    )
    ax.set_xlabel("k_perp [1/Mpc]")
    ax.set_ylabel("|k_parallel| [1/Mpc]")
    fig.colorbar(im, ax=ax, label="relative Delta^2 bias")
    fig.tight_layout()
    fig.savefig(out_dir / "stage3_cylindrical_delta2_rel.png", dpi=200)
    plt.close(fig)


def maybe_export_uvh5(ctx, proc_dirty: np.ndarray, out_path: Path) -> Dict[str, Any]:
    try:
        import importlib.util

        if importlib.util.find_spec("pyuvdata") is None:
            return {"exported": False, "reason": "pyuvdata_not_available"}
        # A full UVData construction needs antenna metadata that these compact NPZ
        # backgrounds do not carry. Record the blocker instead of inventing metadata.
        return {
            "exported": False,
            "reason": "compact_npz_missing_full_uvdata_metadata",
            "target_path": str(out_path),
            "required_extra_metadata": ["antenna_numbers", "antenna_names", "array_layout", "phase_center_catalog"],
            "dirty_shape": list(proc_dirty.shape),
            "source_path": ctx.source_path,
        }
    except Exception as exc:
        return {"exported": False, "reason": repr(exc), "target_path": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--results", default="results/main_jan2026/pathB_results.csv")
    ap.add_argument("--out", default="results/pspec_bias_experiment")
    ap.add_argument("--stage1-bg", default="bg_01_clean")
    ap.add_argument("--stage1-baseline", default="short_14p6m")
    ap.add_argument("--stage1-flux", type=float, default=100.0)
    args = ap.parse_args()

    cfg = read_config(args.config)
    prior = pd.read_csv(args.results)
    out_dir = ensure_dir(args.out)
    spectra_dir = ensure_dir(out_dir / "delay_spectra")

    try:
        import importlib.util

        hera_pspec_available = importlib.util.find_spec("hera_pspec") is not None
    except Exception:
        hera_pspec_available = False

    rows: List[Dict[str, Any]] = []
    stage1_written = False
    stage2_meta: Dict[str, Any] = {"hera_pspec_available": bool(hera_pspec_available)}

    for bg in cfg["backgrounds"]:
        ctx0 = load_background_npz(bg["path"], bg_id=bg.get("id"), product=bg.get("product", "unknown"), processing_history=bg.get("processing_history", "unspecified"))
        for group in cfg.get("baseline_groups", [{"id": "native", "use_native_baseline": True}]):
            group_id = group.get("id", group.get("name", "baseline"))
            ctx = ctx0 if group.get("use_native_baseline", False) else with_baseline(ctx0, _baseline_vector(group, ctx0.baseline_enu_m), group_id=group_id)
            proc_bg, _ = pipeline(ctx.vis_tf, ctx, cfg)
            p_bg = delay_power(proc_bg, ctx, cfg)
            delays = delay_axis_s(ctx.freqs_hz)
            mask = window_mask(ctx, cfg)
            kpar, kperp, cosmo = k_axes(ctx.freqs_hz, float(np.linalg.norm(ctx.baseline_enu_m)))
            tau_hor_s = float(np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S)
            buffer_s = float(cfg.get("metrics", {}).get("window", {}).get("buffer_ns", 100.0)) * 1e-9

            for flux in cfg.get("experiment", {}).get("flux_grid_jy", [100.0]):
                sat_vis, _track, sat_report = build_starlink_visibility(ctx, cfg, s_ref_jy=float(flux))
                proc_dirty, _ = pipeline(ctx.vis_tf + sat_vis, ctx, cfg)
                p_dirty = delay_power(proc_dirty, ctx, cfg)
                delta = p_dirty - p_bg
                bias = scalar_window_bias(p_bg, p_dirty, mask)
                old = prior[
                    (prior["bg_id"] == ctx.bg_id)
                    & (prior["baseline_group"] == group_id)
                    & (np.isclose(prior["S_ref_jy"].astype(float), float(flux)))
                ]
                null_p95 = float(old["null_p95"].iloc[0]) if len(old) else float("nan")
                win_int = float(np.nansum(np.abs(delta[mask])))
                wedge_int = float(np.nansum(np.abs(delta[~mask])))
                total_int = win_int + wedge_int
                k_abs = np.sqrt(kpar**2 + kperp**2)
                delta2_rel = (k_abs**3 / (2.0 * np.pi**2)) * delta / max(float(np.nanmean(p_bg)), 1e-30)

                rows.append({
                    "bg_id": ctx.bg_id,
                    "baseline_group": group_id,
                    "baseline_length_m": float(np.linalg.norm(ctx.baseline_enu_m)),
                    "S_ref_jy": float(flux),
                    "kperp": kperp,
                    "B_win_abs_db": bias["B_win_abs_db"],
                    "B_win_signed_db": bias["B_win_signed_db"],
                    "null_p95_db": null_p95,
                    "obs_minus_null_p95_db": bias["B_win_abs_db"] - null_p95,
                    "EoR_window_integrated_bias": win_int,
                    "wedge_integrated_bias": wedge_int,
                    "window_fraction_of_abs_bias": win_int / max(total_int, 1e-30),
                    "eta_tau": sat_report.get("eta_tau", float("nan")),
                    "tau_horizon_ns": tau_hor_s * 1e9,
                    "tau_window_boundary_ns": (tau_hor_s + buffer_s) * 1e9,
                    **cosmo,
                })

                spec = pd.DataFrame({
                    "tau_ns": delays * 1e9,
                    "kpar": kpar,
                    "kpar_abs": np.abs(kpar),
                    "kperp": kperp,
                    "P_bg": p_bg,
                    "P_dirty": p_dirty,
                    "DeltaP": delta,
                    "Delta2_rel": delta2_rel,
                    "wedge_mask": ~mask,
                    "window_mask": mask,
                })
                spec.to_csv(spectra_dir / f"{ctx.bg_id}__{group_id}__S{flux_tag(flux)}Jy_delay_pspec.csv", index=False)

                if (ctx.bg_id == args.stage1_bg and group_id == args.stage1_baseline and np.isclose(float(flux), args.stage1_flux)):
                    spec["B_win_abs_db"] = bias["B_win_abs_db"]
                    spec["null_p95_db"] = null_p95
                    spec.to_csv(out_dir / "stage1_single_baseline_delay_pspec.csv", index=False)
                    stage1_plot(spec, out_dir)
                    stage2_meta["uvh5_export"] = maybe_export_uvh5(ctx, proc_dirty, out_dir / "stage2_single_baseline_dirty.uvh5")
                    stage2_meta["surrogate_bandpower_source"] = "stage1_single_baseline_delay_pspec.csv"
                    stage1_written = True

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "stage2_3_bandpower_summary.csv", index=False)

    cyl_rows = []
    for (group_id, flux), sub in summary.groupby(["baseline_group", "S_ref_jy"]):
        files = [
            spectra_dir / f"{r.bg_id}__{r.baseline_group}__S{flux_tag(r.S_ref_jy)}Jy_delay_pspec.csv"
            for r in sub.itertuples()
        ]
        specs = [pd.read_csv(p) for p in files if p.exists()]
        if not specs:
            continue
        all_spec = pd.concat(specs, ignore_index=True)
        kpar_bins = np.linspace(0.0, max(float(all_spec["kpar_abs"].max()), 1e-6), 10)
        for i in range(len(kpar_bins) - 1):
            m = (all_spec["kpar_abs"] >= kpar_bins[i]) & (all_spec["kpar_abs"] < kpar_bins[i + 1])
            if not np.any(m):
                continue
            p_bg_bin = float(np.nanmean(all_spec.loc[m, "P_bg"]))
            p_dirty_bin = float(np.nanmean(all_spec.loc[m, "P_dirty"]))
            delta_bin = p_dirty_bin - p_bg_bin
            kpar_c = 0.5 * (kpar_bins[i] + kpar_bins[i + 1])
            kperp_c = float(all_spec["kperp"].median())
            k3 = (np.sqrt(kpar_c**2 + kperp_c**2)) ** 3
            cyl_rows.append({
                "baseline_group": group_id,
                "S_ref_jy": float(flux),
                "kperp_center": kperp_c,
                "kpar_abs_center": kpar_c,
                "P_bg": p_bg_bin,
                "P_dirty": p_dirty_bin,
                "DeltaP": delta_bin,
                "Delta2_rel": float(k3 * delta_bin / (2.0 * np.pi**2 * max(p_bg_bin, 1e-30))),
                "n_delay_samples": int(np.sum(m)),
            })
    cyl = pd.DataFrame(cyl_rows)
    cyl.to_csv(out_dir / "stage3_cylindrical_kperp_kpar.csv", index=False)
    if not cyl.empty:
        cylindrical_plot(cyl[cyl["S_ref_jy"] == args.stage1_flux], out_dir)

    conclusion = {
        "stage1_written": stage1_written,
        "stage2": stage2_meta,
        "hera_pspec_note": "hera_pspec was not available in this WSL environment; Stage 2 UVH5/hera_pspec execution was not completed." if not hera_pspec_available else "hera_pspec import is available.",
        "main_result": {
            "median_B_win_abs_db_by_flux": summary.groupby("S_ref_jy")["B_win_abs_db"].median().to_dict(),
            "max_obs_minus_null_p95_db": float(summary["obs_minus_null_p95_db"].max()),
            "cases_above_null_p95": int(np.sum(summary["obs_minus_null_p95_db"] > 0.0)),
            "n_cases": int(len(summary)),
            "median_window_fraction_of_abs_bias": float(summary["window_fraction_of_abs_bias"].median()),
            "max_window_fraction_of_abs_bias": float(summary["window_fraction_of_abs_bias"].max()),
        },
    }
    with open(out_dir / "conclusion.json", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=2)
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
