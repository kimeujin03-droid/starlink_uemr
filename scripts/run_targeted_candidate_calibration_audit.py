#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.io_background import BackgroundContext
from pathb.satellite import build_visibility_for_sat
from scripts.run_coverage_grid import configure_case, load_tle_fast, read_uvh5_bin, to_local_path
from scripts.run_coverage_tail_resolution_check import compute_tail_case
from scripts.run_full_catalog_physical_candidate_audit import classify_case


DEFAULT_SIGMAS = [0.0, 1e-4, 1e-3, 1e-2]
DEFAULT_ELL_NU_MHZ = [0.5, 1.0, 5.0]
DEFAULT_MODELS = ["white", "smooth"]


def make_rng(seed: int, case_idx: int, model_idx: int, sigma_idx: int, ell_idx: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(seed), int(case_idx), int(model_idx), int(sigma_idx), int(ell_idx)])
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-audit", default="outputs/full_catalog_physical_candidate_audit.csv")
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials_all_tle_full.csv")
    ap.add_argument("--config", default="configs/coverage_robustness_all_tle.yaml")
    ap.add_argument("--pathb-config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--out", default="outputs/targeted_candidate_calibration_audit.csv")
    ap.add_argument("--summary-out", default="outputs/targeted_candidate_calibration_audit_summary.csv")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--sigmas", nargs="*", type=float, default=DEFAULT_SIGMAS)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--ell-nu-mhz", nargs="*", type=float, default=DEFAULT_ELL_NU_MHZ)
    ap.add_argument("--doppler-mode", default="linear")
    args = ap.parse_args()

    cfg_run = yaml.safe_load(to_local_path(args.config).read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path(args.pathb_config))
    candidate_audit = pd.read_csv(to_local_path(args.candidate_audit))
    selection = pd.read_csv(to_local_path(args.selection))
    trials = pd.read_csv(to_local_path(args.trials))

    recs, _tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026.txt")),
        max_records=int(cfg_run.get("max_tle_records", 6364)),
    )
    rec_map = {r.norad_id: r for r in recs}

    rows = []
    for case_idx, row in candidate_audit.iterrows():
        baseline_id = str(row["baseline_id"])
        lst_stratum = str(row["lst_stratum"])
        lst_bin_id = int(row["lst_bin_id"])
        morphology = str(row["morphology"])
        flux = float(row["flux_jy"])
        multiplicity = str(row["multiplicity"])
        beam = str(row["beam"])
        paired_beam = str(row["paired_beam"])
        primary_seed = int(row["primary_seed"])
        paired_seed = int(row["paired_seed"])
        primary_norads = [x for x in str(row["primary_selected_norad_ids"]).split(";") if x]
        paired_norads = [x for x in str(row["paired_selected_norad_ids"]).split(";") if x]

        srows = selection[
            (selection["baseline_id"].astype(str) == baseline_id)
            & (selection["lst_stratum"].astype(str) == lst_stratum)
            & (selection["lst_bin_id"].astype(int) == lst_bin_id)
        ]
        if len(srows) != 1:
            raise ValueError(f"Expected one selection row for {(baseline_id, lst_stratum, lst_bin_id)}, found {len(srows)}")
        ctx0 = read_uvh5_bin(srows.iloc[0])

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
            raise ValueError(f"Expected one primary and one paired trial for {baseline_id}/{lst_stratum}/{lst_bin_id}")
        primary_trial = primary_trial.iloc[0]
        paired_trial = paired_trial.iloc[0]

        for model_idx, model in enumerate(args.models):
            ell_values = [None] if model == "white" else list(args.ell_nu_mhz)
            for sigma_idx, sigma_cal in enumerate(args.sigmas):
                for ell_idx, ell_nu_mhz in enumerate(ell_values):
                    rng = make_rng(args.seed, case_idx, model_idx, sigma_idx, ell_idx)
                    residual = build_cal_residual(
                        shape=ctx0.vis_tf.shape,
                        sigma_cal=float(sigma_cal),
                        model=model,
                        rng=rng,
                        ell_nu_mhz=ell_nu_mhz,
                        freq_hz=ctx0.freqs_hz,
                    )
                    ctx = apply_residual(ctx0, residual)

                    primary_cfg = configure_case(base_cfg, beam, morphology)
                    paired_cfg = configure_case(base_cfg, paired_beam, morphology)
                    primary_cfg.setdefault("starlink", {}).setdefault("emission_model", {})["doppler_mode"] = args.doppler_mode
                    paired_cfg.setdefault("starlink", {}).setdefault("emission_model", {})["doppler_mode"] = args.doppler_mode

                    def build_vis_list(cfg, norads):
                        out = []
                        for norad in norads:
                            vis, _track, _report = build_visibility_for_sat(rec_map[str(norad)], ctx, cfg, s_ref_jy=flux)
                            out.append(vis)
                        return out

                    primary_profile = compute_tail_case(
                        ctx,
                        primary_cfg,
                        build_vis_list(primary_cfg, primary_norads),
                        int(args.n_null),
                        primary_seed,
                        str(primary_trial.get("injection_mode", "coherent_ab")),
                    )
                    paired_profile = compute_tail_case(
                        ctx,
                        paired_cfg,
                        build_vis_list(paired_cfg, paired_norads),
                        int(args.n_null),
                        paired_seed,
                        str(paired_trial.get("injection_mode", "coherent_ab")),
                    )
                    final_class = classify_case(primary_profile, paired_profile)
                    rows.append(
                        {
                            "case": int(row["case"]),
                            "baseline_id": baseline_id,
                            "lst_stratum": lst_stratum,
                            "lst_bin_id": lst_bin_id,
                            "beam": beam,
                            "paired_beam": paired_beam,
                            "morphology": morphology,
                            "flux_jy": flux,
                            "multiplicity": multiplicity,
                            "residual_model": model,
                            "sigma_cal": float(sigma_cal),
                            "ell_nu_mhz": float(ell_nu_mhz) if ell_nu_mhz is not None else None,
                            "doppler_mode": args.doppler_mode,
                            "PTE_max_1000": float(primary_profile["PTE_global_max"]),
                            "B_rel": float(primary_profile["relative_abs_bias"]),
                            "PTE_absint_1000": float(primary_profile["PTE_global_absint"]),
                            "paired_PTE_max_1000": float(paired_profile["PTE_global_max"]),
                            "paired_B_rel": float(paired_profile["relative_abs_bias"]),
                            "paired_PTE_absint_1000": float(paired_profile["PTE_global_absint"]),
                            "final_class": final_class,
                            "primary_seed": primary_seed,
                            "paired_seed": paired_seed,
                            "seed": int(args.seed),
                        }
                    )
                    print(
                        f"case={int(row['case'])} model={model} sigma={sigma_cal:g} ell={ell_nu_mhz if ell_nu_mhz is not None else 'na'} "
                        f"class={final_class} PTE={float(primary_profile['PTE_global_max']):.5f}/"
                        f"{float(primary_profile['PTE_global_absint']):.5f}",
                        flush=True,
                    )

    out = to_local_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["residual_model", "sigma_cal", "ell_nu_mhz", "final_class"], dropna=False)
        .size()
        .reset_index(name="n_cases")
        .sort_values(["residual_model", "sigma_cal", "ell_nu_mhz", "final_class"])
    )
    summary.to_csv(to_local_path(args.summary_out), index=False)
    print(summary.to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
