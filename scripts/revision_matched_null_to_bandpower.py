#!/usr/bin/env python3
"""Propagate matched-null QA exceedance into delay-spectrum bandpower bias.

This is a lightweight, HERA-like delay-spectrum propagation test. It is not a
full HERA cosmological bandpower pipeline. The goal is to test whether the
matched-null QA flag predicts downstream cross-delay false excess.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.io_background import load_background_npz, with_baseline
from pathb.metrics import window_mask
from pathb.pipeline import delay_axis_s, make_taper
from pathb.runner import _baseline_vector
from pathb.satellite import build_starlink_visibility
from scripts.run_pspec_bias_experiment import k_axes


MORPHOLOGY_INPUTS = {
    "smooth": None,
    "controlled_ripple_stress": "results/revision_morphology_comb/morphology_comparison.csv",
    "literature_lofar_khz_comb": "results/revision_morphology_literature_khz/morphology_comparison.csv",
    "literature_lofar_lines": "results/revision_morphology_literature_lines/morphology_comparison.csv",
    "bursty": "results/revision_morphology_bursty/morphology_comparison.csv",
}


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).astype(int).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "t", "1", "yes", "y"})


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_morphology(cfg: Dict[str, Any], morphology: str) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    em = out.setdefault("starlink", {}).setdefault("emission_model", {})
    em["spectral_morphology"] = morphology
    if morphology == "controlled_ripple_stress":
        em["comb_delay_ns"] = 800.0
    if morphology in {"literature_lofar_khz_comb", "literature_lofar_lines"}:
        em["highres_channel_average"] = True
    return out


def load_qa_lookup(path: str | None, smooth_csv: Path, morphology: str) -> pd.DataFrame:
    if path is None:
        df = pd.read_csv(smooth_csv)
        return pd.DataFrame(
            {
                "bg_id": df["bg_id"],
                "baseline_group": df["baseline_group"],
                "S_ref_jy": df["S_ref_jy"].astype(float),
                "morphology": morphology,
                "qa_delta_null_db": df["obs_minus_null_p95_db"].astype(float),
                "qa_null_positive": df["obs_minus_null_p95_db"].astype(float) > 0.0,
                "qa_S_win_db": df["S_win_abs_delta_power_db"].astype(float),
                "qa_null_p95_db": df["null_p95"].astype(float),
            }
        )
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    df = pd.read_csv(p)
    return pd.DataFrame(
        {
            "bg_id": df["bg_id"],
            "baseline_group": df["baseline_group"],
            "S_ref_jy": df["S_ref_jy"].astype(float),
            "morphology": morphology,
            "qa_delta_null_db": df["S_win_morphology_db"].astype(float) - df["null_p95_db"].astype(float),
            "qa_null_positive": bool_series(df["morphology_exceeds_null"]),
            "qa_S_win_db": df["S_win_morphology_db"].astype(float),
            "qa_null_p95_db": df["null_p95_db"].astype(float),
        }
    )


def _delay_transform(vis_tf: np.ndarray, weights_tf: np.ndarray, taper: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(vis_tf), vis_tf, 0.0 + 0.0j) * np.clip(weights_tf, 0.0, 1.0) * taper[None, :]
    return np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)


def cross_delay_power(vis_tf: np.ndarray, weights_tf: np.ndarray, taper: np.ndarray) -> np.ndarray:
    even = np.arange(vis_tf.shape[0]) % 2 == 0
    odd = ~even
    da = _delay_transform(vis_tf[even], weights_tf[even], taper)
    db = _delay_transform(vis_tf[odd], weights_tf[odd], taper)
    n = min(len(da), len(db))
    if n == 0:
        return np.full(vis_tf.shape[1], np.nan)
    return np.nanmean(np.real(da[:n] * np.conj(db[:n])), axis=0)


def dirty_for_mode(bg_vis: np.ndarray, sat_vis: np.ndarray, mode: str) -> np.ndarray:
    dirty = np.array(bg_vis, copy=True)
    if mode == "coherent_ab":
        dirty += sat_vis
    elif mode == "a_only":
        even = np.arange(bg_vis.shape[0]) % 2 == 0
        dirty[even] += sat_vis[even]
    else:
        raise ValueError(f"Unknown injection mode: {mode}")
    return dirty


def robust_z(obs: np.ndarray, nulls: np.ndarray) -> np.ndarray:
    med = np.nanmedian(nulls, axis=0)
    mad = np.nanmedian(np.abs(nulls - med[None, :]), axis=0)
    sigma = 1.4826 * mad
    fallback = np.nanstd(nulls, axis=0)
    sigma = np.where(sigma > 1e-30, sigma, fallback)
    sigma = np.where(sigma > 1e-30, sigma, np.nan)
    return (obs - med) / sigma


def robust_z_scalar(obs: float, nulls: np.ndarray) -> float:
    med = float(np.nanmedian(nulls))
    mad = float(np.nanmedian(np.abs(nulls - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-30:
        sigma = float(np.nanstd(nulls))
    if not np.isfinite(sigma) or sigma <= 1e-30:
        return float("nan")
    return float((obs - med) / sigma)


def phase_randomized_sat(sat_vis: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "global":
        phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
    elif mode == "per_time":
        phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(sat_vis.shape[0], 1)))
    elif mode == "per_freq":
        phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(1, sat_vis.shape[1])))
    elif mode == "per_pixel":
        phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=sat_vis.shape))
    else:
        raise ValueError(mode)
    return sat_vis * phase


def iter_cases(cfg: Dict[str, Any], limit: int | None = None) -> Iterable[tuple[dict, dict, float]]:
    fluxes = cfg.get("experiment", {}).get("flux_grid_jy", [100.0])
    count = 0
    for bg in cfg["backgrounds"]:
        for group in cfg.get("baseline_groups", [{"id": "native", "use_native_baseline": True}]):
            for flux in fluxes:
                yield bg, group, float(flux)
                count += 1
                if limit is not None and count >= limit:
                    return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--smooth-csv", default="results/main_jan2026/pathB_results.csv")
    ap.add_argument("--out", default="results/revision_bandpower_propagation")
    ap.add_argument("--morphologies", nargs="*", default=list(MORPHOLOGY_INPUTS.keys()))
    ap.add_argument("--injection-modes", nargs="*", default=["coherent_ab", "a_only"])
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--phase-mode", default="per_time", choices=["global", "per_time", "per_freq", "per_pixel"])
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--limit-cases", type=int, default=None)
    args = ap.parse_args()

    base_cfg = read_config(args.config)
    out_dir = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    smooth_csv = Path(args.smooth_csv)
    if not smooth_csv.is_absolute():
        smooth_csv = ROOT / smooth_csv

    qa_tables = {
        morph: load_qa_lookup(MORPHOLOGY_INPUTS.get(morph), smooth_csv, morph)
        for morph in args.morphologies
    }

    rows = []
    spectra_rows = []
    for morph in args.morphologies:
        cfg = set_morphology(base_cfg, morph)
        qa = qa_tables[morph]
        print(f"\n=== morphology={morph} ===", flush=True)
        for icase, (bg, group, flux) in enumerate(iter_cases(cfg, args.limit_cases), start=1):
            ctx0 = load_background_npz(
                bg["path"],
                bg_id=bg.get("id"),
                product=bg.get("product", "unknown"),
                processing_history=bg.get("processing_history", "unspecified"),
            )
            group_id = group.get("id", group.get("name", "baseline"))
            ctx = ctx0 if group.get("use_native_baseline", False) else with_baseline(
                ctx0, _baseline_vector(group, ctx0.baseline_enu_m), group_id=group_id
            )
            qrow = qa[
                (qa["bg_id"] == ctx.bg_id)
                & (qa["baseline_group"] == group_id)
                & np.isclose(qa["S_ref_jy"].astype(float), flux)
            ]
            if len(qrow) != 1:
                raise ValueError(f"Missing QA row for {morph} {ctx.bg_id} {group_id} {flux}")
            q = qrow.iloc[0]

            sat_vis, _track, sat_report = build_starlink_visibility(ctx, cfg, s_ref_jy=flux)
            taper_cfg = cfg.get("metrics", {}).get("window", {})
            taper = make_taper(len(ctx.freqs_hz), taper_cfg.get("taper", "blackman_harris"), taper_cfg)
            win = window_mask(ctx, cfg)
            delays = delay_axis_s(ctx.freqs_hz)
            kpar, kperp, cosmo = k_axes(ctx.freqs_hz, float(np.linalg.norm(ctx.baseline_enu_m)))
            p_bg = cross_delay_power(ctx.vis_tf, ctx.weights_tf, taper)

            for inj_mode in args.injection_modes:
                dirty = dirty_for_mode(ctx.vis_tf, sat_vis, inj_mode)
                p_dirty = cross_delay_power(dirty, ctx.weights_tf, taper)
                bias_obs = p_dirty - p_bg

                null_biases = []
                for _ in range(args.n_null):
                    sat_null = phase_randomized_sat(sat_vis, rng, args.phase_mode)
                    dirty_null = dirty_for_mode(ctx.vis_tf, sat_null, inj_mode)
                    p_null = cross_delay_power(dirty_null, ctx.weights_tf, taper)
                    null_biases.append(p_null - p_bg)
                null_bias = np.asarray(null_biases, dtype=float)
                bias_win = bias_obs[win]
                z = robust_z(bias_obs, null_bias)
                z_win = z[win]
                null_win = null_bias[:, win]
                obs_max_bias = float(np.nanmax(bias_win)) if np.any(np.isfinite(bias_win)) else float("nan")
                obs_abs_integrated = float(np.nansum(np.abs(bias_win)))
                null_max_bias = np.nanmax(null_win, axis=1)
                null_abs_integrated = np.nansum(np.abs(null_win), axis=1)
                z_trial_max = robust_z_scalar(obs_max_bias, null_max_bias)
                z_trial_absint = robust_z_scalar(obs_abs_integrated, null_abs_integrated)
                abs_bias_sum = float(np.nansum(np.abs(bias_win)))
                signed_bias_sum = float(np.nansum(bias_win))
                bg_abs_sum = float(np.nansum(np.abs(p_bg[win])))
                bin_max_z = float(np.nanmax(z_win)) if np.any(np.isfinite(z_win)) else float("nan")
                bin_z95 = float(np.nanpercentile(z_win, 95)) if np.any(np.isfinite(z_win)) else float("nan")

                row = {
                    "morphology": morph,
                    "injection_mode": inj_mode,
                    "bg_id": ctx.bg_id,
                    "baseline_group": group_id,
                    "baseline_length_m": float(np.linalg.norm(ctx.baseline_enu_m)),
                    "S_ref_jy": flux,
                    "qa_delta_null_db": float(q["qa_delta_null_db"]),
                    "qa_null_positive": bool(q["qa_null_positive"]),
                    "qa_S_win_db": float(q["qa_S_win_db"]),
                    "qa_null_p95_db": float(q["qa_null_p95_db"]),
                    "Z_PS_max": z_trial_max,
                    "Z_PS_abs_integrated": z_trial_absint,
                    "Z_PS_bin_max": bin_max_z,
                    "Z_PS_bin_p95": bin_z95,
                    "PS_gt_1": bool(z_trial_max > 1.0),
                    "PS_gt_2": bool(z_trial_max > 2.0),
                    "PS_gt_3": bool(z_trial_max > 3.0),
                    "window_abs_bias_sum": abs_bias_sum,
                    "window_signed_bias_sum": signed_bias_sum,
                    "window_bg_abs_sum": bg_abs_sum,
                    "relative_abs_bias": abs_bias_sum / max(bg_abs_sum, 1e-30),
                    "eta_tau": sat_report.get("eta_tau", float("nan")),
                    "margin_to_window_ns": sat_report.get("margin_to_window_ns", float("nan")),
                    "kperp": kperp,
                    **cosmo,
                }
                rows.append(row)

                for tau_s, kp, is_win, b, nmed, zz in zip(delays, kpar, win, bias_obs, np.nanmedian(null_bias, axis=0), z):
                    if is_win:
                        spectra_rows.append({
                            "morphology": morph,
                            "injection_mode": inj_mode,
                            "bg_id": ctx.bg_id,
                            "baseline_group": group_id,
                            "S_ref_jy": flux,
                            "tau_ns": float(tau_s * 1e9),
                            "kpar": float(kp),
                            "kperp": kperp,
                            "bias": float(b),
                            "null_median_bias": float(nmed),
                            "Z_PS": float(zz),
                        })
            print(f"  {icase:02d} {ctx.bg_id} {group_id} S={flux:g} Jy", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "bandpower_propagation_detail.csv", index=False)
    pd.DataFrame(spectra_rows).to_csv(out_dir / "bandpower_window_bins.csv", index=False)

    summary = (
        detail.groupby(["morphology", "injection_mode"])
        .agg(
            rows=("Z_PS_max", "size"),
            qa_positive=("qa_null_positive", "sum"),
            ps_gt1=("PS_gt_1", "sum"),
            ps_gt2=("PS_gt_2", "sum"),
            ps_gt3=("PS_gt_3", "sum"),
            median_Z_PS_max=("Z_PS_max", "median"),
            p95_Z_PS_max=("Z_PS_max", lambda x: float(np.nanpercentile(x, 95))),
            median_relative_abs_bias=("relative_abs_bias", "median"),
            p95_relative_abs_bias=("relative_abs_bias", lambda x: float(np.nanpercentile(x, 95))),
        )
        .reset_index()
    )
    summary["qa_positive_rate"] = summary["qa_positive"] / summary["rows"]
    summary["Pr_Zgt1"] = summary["ps_gt1"] / summary["rows"]
    summary["Pr_Zgt2"] = summary["ps_gt2"] / summary["rows"]
    summary["Pr_Zgt3"] = summary["ps_gt3"] / summary["rows"]
    summary.to_csv(out_dir / "bandpower_propagation_summary.csv", index=False)

    cond_rows = []
    for (morph, inj_mode), sub in detail.groupby(["morphology", "injection_mode"]):
        for qa_state, qsub in sub.groupby("qa_null_positive"):
            for threshold in [1, 2, 3]:
                cond_rows.append({
                    "morphology": morph,
                    "injection_mode": inj_mode,
                    "qa_null_positive": bool(qa_state),
                    "threshold": f"Z_PS>{threshold}",
                    "n": len(qsub),
                    "count": int((qsub["Z_PS_max"] > threshold).sum()),
                    "probability": float((qsub["Z_PS_max"] > threshold).mean()) if len(qsub) else float("nan"),
                })
    cond = pd.DataFrame(cond_rows)
    cond.to_csv(out_dir / "bandpower_conditional_false_excess.csv", index=False)

    report = ["# Matched-Null Exceedance To Bandpower Propagation", ""]
    for _, r in summary.iterrows():
        report.extend([
            f"## {r['morphology']} / {r['injection_mode']}",
            "",
            f"- rows: {int(r['rows'])}",
            f"- matched-null positive: {int(r['qa_positive'])}/{int(r['rows'])} ({r['qa_positive_rate']:.3f})",
            f"- Pr(Z_PS,max > 1): {r['Pr_Zgt1']:.3f}",
            f"- Pr(Z_PS,max > 2): {r['Pr_Zgt2']:.3f}",
            f"- Pr(Z_PS,max > 3): {r['Pr_Zgt3']:.3f}",
            f"- median Z_PS,max: {r['median_Z_PS_max']:.3f}",
            f"- p95 Z_PS,max: {r['p95_Z_PS_max']:.3f}",
            "",
        ])
    (out_dir / "bandpower_propagation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote propagation outputs to {out_dir}")


if __name__ == "__main__":
    main()
