#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import h5py
from skyfield.api import wgs84

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.io_background import BackgroundContext
from pathb.metrics import window_mask
from pathb.pipeline import C_M_PER_S, delay_axis_s, make_taper
from pathb.satellite import SatelliteRecord, build_visibility_for_sat, skyfield_time_from_ctx
from scripts.revision_matched_null_to_bandpower import (
    cross_delay_power,
    dirty_for_mode,
    k_axes,
    robust_z,
    robust_z_scalar,
)
from skyfield.api import EarthSatellite, load


BEAMS = {
    "frozen_polybeam": {
        "mode": "hera_poly",
        "strict_polybeam": True,
        "fixed_eval_freq_hz": 150_000_000.0,
    },
    "full_polybeam": {"mode": "hera_poly", "strict_polybeam": True},
}

MORPHOLOGIES = {
    "smooth": "smooth",
    "lines": "literature_lofar_lines",
    "khz_comb": "literature_lofar_khz_comb",
}


def to_local_path(value: str | Path) -> Path:
    path = Path(str(value))
    text = str(value)
    if platform.system() != "Windows" and len(text) >= 3 and text[1:3] in {":\\", ":/"}:
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    if path.is_absolute():
        return path
    return ROOT / path


def safe_token(text: Any) -> str:
    return str(text).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "p")


def load_tle_fast(path: Path, max_records: int | None = None) -> tuple[list[SatelliteRecord], dict]:
    ts = load.timescale()
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    recs: list[SatelliteRecord] = []
    i = 0
    while i < len(lines):
        name = ""
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            line1, line2 = lines[i], lines[i + 1]
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue
        norad = line1[2:7].strip()
        epoch = line1[18:32].strip()
        sat_name = name or f"SAT-{norad}-E{epoch}"
        recs.append(SatelliteRecord(EarthSatellite(line1, line2, sat_name, ts), norad, epoch, sat_name))
        if max_records is not None and len(recs) >= int(max_records):
            break
    return recs, {
        "tle_records_loaded_fast": len(recs),
        "tle_selection": "first_records_in_file_for_coverage_grid",
    }


def polarization_index(pol_array: np.ndarray, pol: str = "ee") -> tuple[int, str]:
    preferred = {"ee": -5, "xx": -5, "nn": -6, "yy": -6}
    code = preferred.get(pol, -5)
    pol_array = np.asarray(pol_array, dtype=int)
    if code in set(pol_array.tolist()):
        return int(np.where(pol_array == code)[0][0]), pol
    return 0, str(pol_array[0])


def baseline_geometry_from_native_npz(baseline_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native = ROOT / "examples" / "backgrounds_jan2026_real" / f"native_{baseline_id}.npz"
    if not native.exists():
        raise FileNotFoundError(f"Missing native geometry NPZ for {baseline_id}: {native}")
    data = np.load(native, allow_pickle=True)
    return (
        np.asarray(data["baseline_enu_m"], dtype=float),
        np.asarray(data["ant1_enu_m"], dtype=float),
        np.asarray(data["ant2_enu_m"], dtype=float),
    )


def pte_ge(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    finite = np.isfinite(null)
    if not np.isfinite(obs) or not np.any(finite):
        return float("nan")
    return float((1 + np.sum(null[finite] >= obs)) / (1 + np.sum(finite)))


def configure_case(base_cfg: dict[str, Any], beam: str, morphology: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("beam", {}).update(BEAMS[beam])
    em = cfg.setdefault("starlink", {}).setdefault("emission_model", {})
    em["spectral_morphology"] = MORPHOLOGIES[morphology]
    if morphology in {"lines", "khz_comb"}:
        em["highres_channel_average"] = True
    return cfg


def read_uvh5_bin(row: pd.Series) -> BackgroundContext:
    path = to_local_path(row["source_uvh5"])
    ant1 = int(row["ant1"])
    ant2 = int(row["ant2"])
    pol = str(row.get("pol", "ee"))
    start = int(row["t_start_index"])
    n_time = int(row["n_time"])

    with h5py.File(path, "r") as h5:
        pol_idx, pol = polarization_index(h5["Header/polarization_array"][()], pol)
        data = np.asarray(h5["Data/visdata"][start : start + n_time, :, pol_idx])
        flags = np.asarray(h5["Data/flags"][start : start + n_time, :, pol_idx])
        nsamples = np.asarray(h5["Data/nsamples"][start : start + n_time, :, pol_idx])
        freqs_hz = np.asarray(h5["Header/freq_array"][()], dtype=float).reshape(-1)
        times_orig = np.asarray(h5["Header/time_array"][start : start + n_time], dtype=float)
    times_jd = float(row["lst_start"]) + (times_orig - times_orig[0])
    baseline_enu, ant1_enu, ant2_enu = baseline_geometry_from_native_npz(str(row["baseline_id"]))

    weights = (~flags).astype(float) * np.clip(nsamples.astype(float), 0.0, None)
    if np.nanmax(weights) > 0:
        weights = weights / np.nanmax(weights)
    finite = np.isfinite(data)
    weights = weights * finite.astype(float)
    data = np.where(finite, data, 0.0 + 0.0j)

    return BackgroundContext(
        vis_tf=np.asarray(data, dtype=complex),
        freqs_hz=freqs_hz,
        times_jd=np.asarray(times_jd, dtype=float),
        baseline_enu_m=np.asarray(baseline_enu, dtype=float),
        weights_tf=np.asarray(weights, dtype=float),
        ant1_enu_m=ant1_enu,
        ant2_enu_m=ant2_enu,
        source_path=str(path),
        bg_id=f"{row['baseline_id']}_lst{int(row['lst_bin_id']):03d}",
        product="uvh5_lst_bin",
        processing_history="10min LST bin crop from native HERA baseline UVH5",
        flag_fraction=float(np.mean(weights <= 0.0)),
        metadata={
            "baseline_id": row["baseline_id"],
            "lst_bin_id": int(row["lst_bin_id"]),
            "lst_stratum": row["lst_stratum"],
            "time_scale": "utc",
        },
        time_scale="utc",
    )


def rank_visible_sats(recs, ctx: BackgroundContext, cfg: dict[str, Any], alt_visible_deg: float) -> pd.DataFrame:
    site = cfg["site"]
    observer = wgs84.latlon(
        float(site["lat_deg"]),
        float(site["lon_deg"]),
        elevation_m=float(site.get("elev_m", 0.0)),
    )
    t = skyfield_time_from_ctx(ctx)
    rows = []
    for rec in recs:
        try:
            alt = np.asarray((rec.sat - observer).at(t).altaz()[0].degrees, dtype=float)
        except Exception:
            continue
        if not np.any(np.isfinite(alt)):
            continue
        peak = float(np.nanmax(alt))
        if peak < alt_visible_deg:
            continue
        rows.append(
            {
                "norad_id": rec.norad_id,
                "sat_name": rec.name,
                "epoch": rec.epoch,
                "peak_alt_deg": peak,
                "mean_alt_deg": float(np.nanmean(alt)),
                "n_time_visible": int(np.sum(alt >= alt_visible_deg)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["norad_id", "sat_name", "epoch", "peak_alt_deg", "mean_alt_deg", "n_time_visible"])
    return pd.DataFrame(rows).sort_values("peak_alt_deg", ascending=False).reset_index(drop=True)


def independent_global_phase_sum(sat_stack: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(sat_stack.shape[0],)))
    out = np.zeros_like(sat_stack[0], dtype=complex)
    for sat, phase in zip(sat_stack, phases, strict=False):
        out += sat * phase
    return out


def compute_metrics(
    ctx: BackgroundContext,
    cfg: dict[str, Any],
    sat_vis_list: list[np.ndarray],
    n_null: int,
    seed: int,
    injection_mode: str,
    null_stats_path: Path,
) -> dict[str, Any]:
    sat_stack = np.stack(sat_vis_list, axis=0)
    sat_obs = np.sum(sat_stack, axis=0)
    taper_cfg = cfg.get("metrics", {}).get("window", {})
    taper = make_taper(len(ctx.freqs_hz), taper_cfg.get("taper", "blackman_harris"), taper_cfg)
    win = window_mask(ctx, cfg)
    delays = delay_axis_s(ctx.freqs_hz)
    kpar, kperp, cosmo = k_axes(ctx.freqs_hz, float(np.linalg.norm(ctx.baseline_enu_m)))

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

    z = robust_z(bias_obs, null_bias)
    z_win = z[win]
    bias_win = bias_obs[win]
    null_win = null_bias[:, win]
    obs_max_bias = float(np.nanmax(bias_win)) if np.any(np.isfinite(bias_win)) else float("nan")
    obs_abs_integrated = float(np.nansum(np.abs(bias_win)))
    null_max_bias = np.nanmax(null_win, axis=1)
    null_abs_integrated = np.nansum(np.abs(null_win), axis=1)
    z_trial_max = robust_z_scalar(obs_max_bias, null_max_bias)
    z_trial_absint = robust_z_scalar(obs_abs_integrated, null_abs_integrated)
    pte_global_max = pte_ge(obs_max_bias, null_max_bias)
    pte_global_absint = pte_ge(obs_abs_integrated, null_abs_integrated)
    abs_bias_sum = float(np.nansum(np.abs(bias_win)))
    signed_bias_sum = float(np.nansum(bias_win))
    bg_abs_sum = float(np.nansum(np.abs(p_bg[win])))
    null_mad_win = float(np.nanmedian(np.abs(null_max_bias - np.nanmedian(null_max_bias))))

    null_stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        null_stats_path,
        obs_max_bias=obs_max_bias,
        obs_abs_integrated=obs_abs_integrated,
        null_max_bias=null_max_bias,
        null_abs_integrated=null_abs_integrated,
        z_trial_max=z_trial_max,
        z_trial_absint=z_trial_absint,
        pte_global_max=pte_global_max,
        pte_global_absint=pte_global_absint,
    )

    return {
        "Z_PS_max": z_trial_max,
        "Z_PS_abs_integrated": z_trial_absint,
        "PTE_global_max": pte_global_max,
        "PTE_global_absint": pte_global_absint,
        "obs_max_bias": obs_max_bias,
        "obs_abs_integrated": obs_abs_integrated,
        "null_max_bias_p50": float(np.nanpercentile(null_max_bias, 50)),
        "null_max_bias_p95": float(np.nanpercentile(null_max_bias, 95)),
        "null_max_bias_p99": float(np.nanpercentile(null_max_bias, 99)),
        "null_abs_integrated_p50": float(np.nanpercentile(null_abs_integrated, 50)),
        "null_abs_integrated_p95": float(np.nanpercentile(null_abs_integrated, 95)),
        "null_abs_integrated_p99": float(np.nanpercentile(null_abs_integrated, 99)),
        "null_mad_win": null_mad_win,
        "Z_PS_bin_max": float(np.nanmax(z_win)) if np.any(np.isfinite(z_win)) else float("nan"),
        "Z_PS_bin_p95": float(np.nanpercentile(z_win, 95)) if np.any(np.isfinite(z_win)) else float("nan"),
        "window_abs_bias_sum": abs_bias_sum,
        "window_signed_bias_sum": signed_bias_sum,
        "window_bg_abs_sum": bg_abs_sum,
        "relative_abs_bias": abs_bias_sum / max(bg_abs_sum, 1e-30),
        "n_window_bins": int(np.sum(win)),
        "kperp": kperp,
        "kpar_min_window": float(np.nanmin(kpar[win])),
        "kpar_max_window": float(np.nanmax(kpar[win])),
        "tau_ns_min_window": float(np.nanmin(delays[win]) * 1e9),
        "tau_ns_max_window": float(np.nanmax(delays[win]) * 1e9),
        "null_global_stats_path": str(null_stats_path),
        **cosmo,
    }


def case_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["baseline_id"],
        int(row["lst_bin_id"]),
        row["lst_stratum"],
        row["beam_model"],
        row["morphology"],
        float(row["flux_jy"]),
        row["multiplicity"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--config", default="configs/coverage_robustness.yaml")
    ap.add_argument("--pathb-config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--out", default="outputs/coverage_robustness_trials.csv")
    ap.add_argument("--null-dir", default="outputs/coverage_null_global_stats")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument(
        "--cells-csv",
        default=None,
        help="Optional CSV with baseline_id,lst_stratum,lst_bin_id to run only selected cells.",
    )
    args = ap.parse_args()

    t0 = time.time()
    cfg_run = yaml.safe_load(to_local_path(args.config).read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path(args.pathb_config))
    sel = pd.read_csv(to_local_path(args.selection))
    if args.cells_csv:
        cells = pd.read_csv(to_local_path(args.cells_csv))[["baseline_id", "lst_stratum", "lst_bin_id"]].drop_duplicates()
        sel = sel.merge(cells, on=["baseline_id", "lst_stratum", "lst_bin_id"], how="inner")
    out = to_local_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    null_dir = to_local_path(args.null_dir)

    tle_path = to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026_LEO_only.tle"))
    recs, tle_meta = load_tle_fast(tle_path, max_records=int(cfg_run.get("max_tle_records", 1200)))
    rec_map = {r.norad_id: r for r in recs}
    print(f"[tle] unique records={len(recs)}", flush=True)

    rows: list[dict[str, Any]] = []
    done: set[tuple[Any, ...]] = set()
    if args.resume and out.exists():
        old = pd.read_csv(out)
        rows = old.to_dict("records")
        done = {case_key(r) for r in rows}
        print(f"[resume] loaded {len(rows)} existing trials", flush=True)

    planned = (
        len(sel)
        * len(cfg_run["beam_models"])
        * len(cfg_run["morphologies"])
        * len(cfg_run["flux_jy"])
        * len(cfg_run["multiplicity"])
    )
    print(f"[plan] selected bins={len(sel)} planned trials={planned}", flush=True)

    contexts: dict[tuple[str, int], BackgroundContext] = {}
    rankings: dict[tuple[str, int], pd.DataFrame] = {}
    completed_this_run = 0
    injection_mode = "coherent_ab"
    bias_floors = [float(x) for x in cfg_run.get("bias_floors", [])]

    for _, srow in sel.iterrows():
        ckey = (str(srow["baseline_id"]), int(srow["lst_bin_id"]))
        if ckey not in contexts:
            print(f"[bin] loading baseline={ckey[0]} lst_bin={ckey[1]}", flush=True)
            contexts[ckey] = read_uvh5_bin(srow)
        ctx = contexts[ckey]
        if ckey not in rankings:
            ranked = rank_visible_sats(recs, ctx, base_cfg, float(cfg_run.get("alt_visible_deg", 70.0)))
            rankings[ckey] = ranked
            print(f"[rank] {ckey[0]} lst={ckey[1]} visible={len(ranked)}", flush=True)
        ranked = rankings[ckey]
        if "n_sat_visible_any_bin" in srow and int(srow["n_sat_visible_any_bin"]) != int(len(ranked)):
            print(
                f"[warn] metadata/ranking mismatch baseline={ckey[0]} lst={ckey[1]} "
                f"meta_any={int(srow['n_sat_visible_any_bin'])} ranked={len(ranked)}",
                flush=True,
            )
        if len(ranked) == 0:
            print(f"[skip] no visible satellites for {ckey}", flush=True)
            continue

        for beam in cfg_run["beam_models"]:
            for morphology in cfg_run["morphologies"]:
                cfg = configure_case(base_cfg, beam, morphology)
                for flux in cfg_run["flux_jy"]:
                    sat_vis_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
                    for multiplicity in cfg_run["multiplicity"]:
                        n_sat_target = 1 if multiplicity == "single" else int(cfg_run.get("max_multi_satellites", 12))
                        chosen = ranked.head(max(1, min(n_sat_target, len(ranked))))
                        row_stub = {
                            "baseline_id": srow["baseline_id"],
                            "baseline_length_m": float(srow["baseline_length_m"]),
                            "baseline_class": srow["baseline_class"],
                            "lst_stratum": srow["lst_stratum"],
                            "lst_bin_id": int(srow["lst_bin_id"]),
                            "lst_start": float(srow["lst_start"]),
                            "lst_end": float(srow["lst_end"]),
                            "beam_model": beam,
                            "morphology": morphology,
                            "flux_jy": float(flux),
                            "multiplicity": multiplicity,
                            "injection_mode": injection_mode,
                        }
                        if case_key(row_stub) in done:
                            continue
                        if args.max_trials is not None and completed_this_run >= args.max_trials:
                            pd.DataFrame(rows).to_csv(out, index=False)
                            print(f"[stop] max-trials reached; wrote {out}", flush=True)
                            return

                        sat_vis_list = []
                        reports = []
                        for norad in chosen["norad_id"].astype(str).tolist():
                            cache_key = f"{norad}|{beam}|{morphology}|{float(flux):g}"
                            if cache_key not in sat_vis_cache:
                                vis, _track, report = build_visibility_for_sat(rec_map[norad], ctx, cfg, s_ref_jy=float(flux))
                                sat_vis_cache[cache_key] = (vis, report)
                            vis, report = sat_vis_cache[cache_key]
                            sat_vis_list.append(vis)
                            reports.append(report)

                        seed = int(args.seed + len(rows) * 17 + int(srow["lst_bin_id"]))
                        null_path = null_dir / (
                            f"{safe_token(srow['baseline_id'])}_lst{int(srow['lst_bin_id']):03d}_"
                            f"{safe_token(srow['lst_stratum'])}_{beam}_{morphology}_S{float(flux):g}_{multiplicity}.npz"
                        )
                        metrics = compute_metrics(
                            ctx,
                            cfg,
                            sat_vis_list,
                            int(cfg_run.get("n_null", 100)),
                            seed,
                            injection_mode,
                            null_path,
                        )
                        out_row = {
                            **row_stub,
                            "n_sat_visible": int(srow["n_sat_visible"]),
                            "n_sat_ranked_visible": int(len(ranked)),
                            "n_injected": int(len(chosen)),
                            "selected_norad_ids": ";".join(chosen["norad_id"].astype(str).tolist()),
                            "selected_peak_alts_deg": ";".join(f"{x:.3f}" for x in chosen["peak_alt_deg"].to_numpy(float)),
                            "flag_fraction": float(srow["flag_fraction"]),
                            "pre_risk_score": float(srow["pre_risk_score"]),
                            "metadata_beam_weighted_sat_exposure": float(srow["beam_weighted_sat_exposure"]),
                            "metadata_n_sat_visible_center": int(srow.get("n_sat_visible_center", srow["n_sat_visible"])),
                            "metadata_n_sat_visible_any_bin": int(srow.get("n_sat_visible_any_bin", srow["n_sat_visible"])),
                            "metadata_n_sat_visible_peak_bin": int(srow.get("n_sat_visible_peak_bin", srow["n_sat_visible"])),
                            "metadata_n_sat_peak_visible": int(srow.get("n_sat_peak_visible", srow.get("n_sat_visible_peak_bin", srow["n_sat_visible"]))),
                            "metadata_beam_weighted_sat_exposure_center": float(srow.get("beam_weighted_sat_exposure_center", srow["beam_weighted_sat_exposure"])),
                            "metadata_beam_weighted_sat_exposure_bin_mean": float(srow.get("beam_weighted_sat_exposure_bin_mean", srow["beam_weighted_sat_exposure"])),
                            "metadata_max_sat_beam_response_center": float(srow.get("max_sat_beam_response_center", srow["max_sat_beam_response"])),
                            "metadata_max_sat_beam_response_bin": float(srow.get("max_sat_beam_response_bin", srow["max_sat_beam_response"])),
                            "metadata_pre_risk_score_bin": float(srow.get("pre_risk_score_bin", srow["pre_risk_score"])),
                            "metadata_null_mad_win_proxy": float(srow["null_mad_win_proxy"]),
                            "sat_peak_abs_jy_sum": float(np.sum([r["peak_abs_jy"] for r in reports])),
                            "sat_peak_abs_jy_max": float(np.max([r["peak_abs_jy"] for r in reports])),
                            "sat_mean_abs_jy_sum": float(np.sum([r["mean_abs_jy"] for r in reports])),
                            "zero_coupling": bool(np.sum([r["peak_abs_jy"] for r in reports]) <= 1e-12),
                            "seed": seed,
                            "n_null": int(cfg_run.get("n_null", 100)),
                            **metrics,
                        }
                        out_row["PS_gt_2"] = bool(out_row["Z_PS_max"] > 2.0)
                        out_row["PS_gt_3"] = bool(out_row["Z_PS_max"] > 3.0)
                        out_row["flag_exploratory"] = bool(
                            (out_row["Z_PS_max"] > float(cfg_run["z_thresholds"]["exploratory"]))
                            or (out_row["PTE_global_max"] < float(cfg_run["pte_thresholds"]["exploratory"]))
                        )
                        out_row["candidate_statistical"] = bool(
                            out_row["PTE_global_max"] < float(cfg_run["pte_thresholds"]["strict"])
                        )
                        for floor in bias_floors:
                            col = f"candidate_floor_{floor:.0e}".replace("-", "m")
                            out_row[col] = bool(out_row["candidate_statistical"] and out_row["relative_abs_bias"] > floor)
                        rows.append(out_row)
                        completed_this_run += 1
                        pd.DataFrame(rows).to_csv(out, index=False)
                        print(
                            f"[{len(rows):04d}/{planned}] base={srow['baseline_id']} lst={int(srow['lst_bin_id'])} "
                            f"{srow['lst_stratum']} beam={beam} morph={morphology} S={float(flux):g} "
                            f"mult={multiplicity} n={len(chosen)} Z={out_row['Z_PS_max']:.3f} "
                            f"PTE={out_row['PTE_global_max']:.3f} rel={out_row['relative_abs_bias']:.3e}",
                            flush=True,
                        )

    pd.DataFrame(rows).to_csv(out, index=False)
    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "selection": str(to_local_path(args.selection)),
        "config": str(to_local_path(args.config)),
        "pathb_config": str(to_local_path(args.pathb_config)),
        "tle_meta": tle_meta,
        "planned_trials": planned,
        "written_trials": len(rows),
        "pte_definition": "PTE=(1 + count(null_global_stat >= obs_global_stat)) / (N_null + 1)",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out} rows={len(rows)} elapsed={meta['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
