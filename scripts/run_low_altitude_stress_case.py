#!/usr/bin/env python3
"""Low-altitude (<30 deg peak elevation) satellite-pass stress test.

Every trial in the main coverage grid (`configs/coverage_robustness*.yaml`)
only injects satellites with peak elevation >= 70 deg (`alt_visible_deg: 70.0`),
i.e. near-zenith passes. Near-horizon geometry is exactly the regime that maps
most directly onto the tau_horizon / window-boundary risk described by
`pathb.satellite.window_geometry_metrics` (eta_tau, horizon_proximity_bin),
so it is the riskiest geometry the staged QA hierarchy has never actually been
exercised against. This script reuses the same day's background contexts
already selected for the full-catalog physical-candidate audit (no new
background generation) and injects only satellites with 0 < peak_alt_deg < 30
during the 10-minute window, then runs the same paired frozen/full PolyBeam,
N_null=1000 staged classification used elsewhere in this package.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from skyfield.api import wgs84

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.config import read_config
from pathb.satellite import build_visibility_for_sat, skyfield_time_from_ctx
from scripts.run_coverage_grid import configure_case, load_tle_fast, read_uvh5_bin, to_local_path
from scripts.run_coverage_tail_resolution_check import compute_tail_case
from scripts.run_full_catalog_physical_candidate_audit import classify_case

N_NULL = 1000
LOW_ALT_MIN_DEG = 0.0
LOW_ALT_MAX_DEG = 30.0
MAX_MULTI_SATELLITES = 12
FLUX_JY = 1000.0
MULTIPLICITY = "multi"
SEED = 20270708

# Reuse the exact same background cells as the full-catalog physical-candidate
# audit (`outputs/full_catalog_physical_candidate_audit.csv`), one per major
# spectral-morphology family already flagged as sensitive in that audit.
CASES = [
    {"baseline_id": "82_0", "lst_stratum": "typical", "lst_bin_id": 18, "morphology": "khz_comb"},
    {"baseline_id": "0_1", "lst_stratum": "typical", "lst_bin_id": 44, "morphology": "smooth"},
]


def rank_low_alt_sats(recs, ctx, cfg, alt_min_deg: float, alt_max_deg: float) -> pd.DataFrame:
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
        if not (alt_min_deg < peak < alt_max_deg):
            continue
        rows.append({
            "norad_id": rec.norad_id,
            "sat_name": rec.name,
            "epoch": rec.epoch,
            "peak_alt_deg": peak,
            "mean_alt_deg": float(np.nanmean(alt)),
            "n_time_visible": int(np.sum(alt > 0)),
        })
    if not rows:
        return pd.DataFrame(columns=["norad_id", "sat_name", "epoch", "peak_alt_deg", "mean_alt_deg", "n_time_visible"])
    # Descending: closest to the 30 deg edge first (strongest still-low-elevation pass).
    return pd.DataFrame(rows).sort_values("peak_alt_deg", ascending=False).reset_index(drop=True)


def main() -> None:
    cfg_run = yaml.safe_load(to_local_path("configs/coverage_robustness_all_tle.yaml").read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path("configs/pathB_jan2026_main.yaml"))
    selection = pd.read_csv(to_local_path("outputs/lst_bin_selection.csv"))

    recs, tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026.txt")),
        max_records=int(cfg_run.get("max_tle_records", 6364)),
    )
    rec_map = {r.norad_id: r for r in recs}

    rows = []
    for case_idx, case in enumerate(CASES, start=1):
        sel = selection[
            (selection["baseline_id"].astype(str) == str(case["baseline_id"]))
            & (selection["lst_stratum"].astype(str) == str(case["lst_stratum"]))
            & (selection["lst_bin_id"].astype(int) == int(case["lst_bin_id"]))
        ]
        if len(sel) != 1:
            raise ValueError(f"Expected one selection row for {case}")
        ctx = read_uvh5_bin(sel.iloc[0])

        low_alt = rank_low_alt_sats(recs, ctx, base_cfg, LOW_ALT_MIN_DEG, LOW_ALT_MAX_DEG)
        print(f"[case {case_idx}] {case['baseline_id']}/{case['lst_stratum']}/{case['lst_bin_id']} "
              f"low-alt (0,30) deg candidates={len(low_alt)}", flush=True)
        if len(low_alt) == 0:
            raise RuntimeError(f"No low-altitude satellites found for {case}")

        chosen = low_alt.head(max(1, min(MAX_MULTI_SATELLITES, len(low_alt))))
        morphology = case["morphology"]

        profiles = {}
        horizon_reports = {}
        for beam in ["frozen_polybeam", "full_polybeam"]:
            cfg = configure_case(base_cfg, beam, morphology)
            sat_vis_list = []
            reports = []
            for norad in chosen["norad_id"].astype(str).tolist():
                vis, _track, report = build_visibility_for_sat(rec_map[norad], ctx, cfg, s_ref_jy=FLUX_JY)
                sat_vis_list.append(vis)
                reports.append(report)
            horizon_reports[beam] = reports
            profiles[beam] = compute_tail_case(ctx, cfg, sat_vis_list, N_NULL, SEED, "coherent_ab")

        final_class = classify_case(profiles["frozen_polybeam"], profiles["full_polybeam"])

        eta_taus = [r["eta_tau"] for r in horizon_reports["frozen_polybeam"]]
        horizon_bins = [r["horizon_proximity_bin"] for r in horizon_reports["frozen_polybeam"]]

        row = {
            "case": case_idx,
            "baseline_id": case["baseline_id"],
            "lst_stratum": case["lst_stratum"],
            "lst_bin_id": case["lst_bin_id"],
            "morphology": morphology,
            "flux_jy": FLUX_JY,
            "multiplicity": MULTIPLICITY,
            "n_injected": len(chosen),
            "peak_alt_deg_min": float(chosen["peak_alt_deg"].min()),
            "peak_alt_deg_max": float(chosen["peak_alt_deg"].max()),
            "selected_norad_ids": ";".join(chosen["norad_id"].astype(str).tolist()),
            "selected_peak_alts_deg": ";".join(f"{x:.3f}" for x in chosen["peak_alt_deg"].to_numpy(float)),
            "eta_tau_min": float(np.min(eta_taus)),
            "eta_tau_max": float(np.max(eta_taus)),
            "horizon_proximity_bins": ";".join(sorted(set(horizon_bins))),
            "N_null": N_NULL,
            "seed": SEED,
            "frozen_PTE_max": profiles["frozen_polybeam"]["PTE_global_max"],
            "frozen_PTE_absint": profiles["frozen_polybeam"]["PTE_global_absint"],
            "frozen_B_rel": profiles["frozen_polybeam"]["relative_abs_bias"],
            "full_PTE_max": profiles["full_polybeam"]["PTE_global_max"],
            "full_PTE_absint": profiles["full_polybeam"]["PTE_global_absint"],
            "full_B_rel": profiles["full_polybeam"]["relative_abs_bias"],
            "final_class": final_class,
        }
        rows.append(row)
        print(
            f"[case {case_idx}] {morphology} n_sat={len(chosen)} alt=[{row['peak_alt_deg_min']:.1f},{row['peak_alt_deg_max']:.1f}]deg "
            f"eta_tau=[{row['eta_tau_min']:.2f},{row['eta_tau_max']:.2f}] "
            f"frozen PTE_max={row['frozen_PTE_max']:.4f} absint={row['frozen_PTE_absint']:.4f} B_rel={row['frozen_B_rel']:.3e} | "
            f"full PTE_max={row['full_PTE_max']:.4f} absint={row['full_PTE_absint']:.4f} B_rel={row['full_B_rel']:.3e} "
            f"class={final_class}",
            flush=True,
        )

    out = to_local_path("outputs/low_altitude_stress_case.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)

    meta = {
        "description": "Low-altitude (<30 deg peak elevation) satellite-pass stress test, paired frozen/full PolyBeam at N_null=1000",
        "tle_meta": tle_meta,
        "low_alt_band_deg": [LOW_ALT_MIN_DEG, LOW_ALT_MAX_DEG],
        "n_cases": len(df_out),
        "n_beam_robust": int((df_out["final_class"] == "beam-robust contamination candidate").sum()),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    print(df_out[["case", "morphology", "frozen_PTE_max", "frozen_PTE_absint", "full_PTE_max", "full_PTE_absint", "final_class"]].to_string(index=False))


if __name__ == "__main__":
    main()
