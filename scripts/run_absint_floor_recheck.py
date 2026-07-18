#!/usr/bin/env python3
"""Re-evaluate absint PTE floor-hit cases with N_null=1000.

Coverage grid ran N_null=100 → PTE resolution floor is 1/101 = 0.0099.
Three cases hit this floor: 11_10/bin16/quiet/frozen_polybeam
  - smooth/30Jy/multi     (B_rel=3.9e-5, below 1e-3 gate)
  - smooth/1000Jy/multi   (B_rel=1.30e-3, above 1e-3 gate)
  - khz_comb/1000Jy/multi (B_rel=1.15e-3, above 1e-3 gate)

Tail refinement used (PTE_max<=0.03 OR Z>3) AND B_rel>1e-3 → missed these
because their PTE_max~0.7. This script fills the gap.
"""
from __future__ import annotations

import json
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
    to_local_path,
)
from scripts.run_coverage_tail_resolution_check import compute_tail_case


# All 3 floor-hit cases (include B_rel<1e-3 for completeness in reporting)
FLOOR_HIT_CASES = [
    # (beam, morphology, flux_jy, multiplicity)
    ("frozen_polybeam", "smooth",    30.0,   "multi"),  # B_rel=3.9e-5
    ("frozen_polybeam", "smooth",   1000.0,  "multi"),  # B_rel=1.30e-3
    ("frozen_polybeam", "khz_comb", 1000.0,  "multi"),  # B_rel=1.15e-3
]

BASELINE_ID = "11_10"
LST_STRATUM = "quiet"
LST_BIN_ID  = 16
N_NULL      = 1000


def main() -> None:
    cfg_run = yaml.safe_load(
        to_local_path("configs/coverage_robustness.yaml").read_text(encoding="utf-8")
    )
    base_cfg = read_config(to_local_path("configs/pathB_jan2026_main.yaml"))

    sel = pd.read_csv(to_local_path("outputs/lst_bin_selection.csv"))
    trials = pd.read_csv(to_local_path("outputs/coverage_robustness_trials.csv"))

    # --- load background context ---
    srows = sel[
        (sel["baseline_id"].astype(str) == BASELINE_ID)
        & (sel["lst_stratum"].astype(str) == LST_STRATUM)
        & (sel["lst_bin_id"].astype(int) == LST_BIN_ID)
    ]
    if len(srows) != 1:
        raise ValueError(f"Expected 1 selection row for {BASELINE_ID}/{LST_STRATUM}/{LST_BIN_ID}, got {len(srows)}")
    ctx = read_uvh5_bin(srows.iloc[0])

    # --- load TLE and rank visible satellites ---
    recs, tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026_LEO_only.tle")),
        max_records=int(cfg_run.get("max_tle_records", 1200)),
    )
    rec_map = {r.norad_id: r for r in recs}
    ranked = rank_visible_sats(
        recs, ctx, base_cfg, float(cfg_run.get("alt_visible_deg", 70.0))
    )
    print(f"[rank] {BASELINE_ID} lst_bin={LST_BIN_ID} visible={len(ranked)}", flush=True)
    if len(ranked) == 0:
        raise RuntimeError("No visible satellites")

    # pre-cache satellite visibilities
    vis_cache: dict[tuple, np.ndarray] = {}

    rows = []
    for beam, morphology, flux, multiplicity in FLOOR_HIT_CASES:
        old = trials[
            (trials["baseline_id"].astype(str) == BASELINE_ID)
            & (trials["lst_stratum"].astype(str) == LST_STRATUM)
            & (trials["lst_bin_id"].astype(int) == LST_BIN_ID)
            & (trials["beam_model"].astype(str) == beam)
            & (trials["morphology"].astype(str) == morphology)
            & (np.isclose(trials["flux_jy"].astype(float), flux))
            & (trials["multiplicity"].astype(str) == multiplicity)
        ]
        if len(old) != 1:
            raise ValueError(
                f"Expected 1 old trial for ({beam},{morphology},{flux},{multiplicity}), got {len(old)}"
            )
        old_row = old.iloc[0]

        n_sat_target = 1 if multiplicity == "single" else int(cfg_run.get("max_multi_satellites", 12))
        chosen = ranked.head(max(1, min(n_sat_target, len(ranked))))
        cfg = configure_case(base_cfg, beam, morphology)

        sat_vis_list = []
        for norad in chosen["norad_id"].astype(str).tolist():
            key = (norad, beam, morphology, flux)
            if key not in vis_cache:
                vis, _track, _report = build_visibility_for_sat(
                    rec_map[norad], ctx, cfg, s_ref_jy=float(flux)
                )
                vis_cache[key] = vis
            sat_vis_list.append(vis_cache[key])

        profile = compute_tail_case(
            ctx,
            cfg,
            sat_vis_list,
            N_NULL,
            int(old_row["seed"]),
            str(old_row.get("injection_mode", "coherent_ab")),
        )

        pte_abs_old = float(old_row["PTE_global_absint"])
        pte_abs_new = float(profile["PTE_global_absint"])
        pte_max_new = float(profile["PTE_global_max"])
        b_rel       = float(profile["relative_abs_bias"])

        row = {
            "baseline_id":            BASELINE_ID,
            "lst_stratum":            LST_STRATUM,
            "lst_bin_id":             LST_BIN_ID,
            "beam_model":             beam,
            "morphology":             morphology,
            "flux_jy":                flux,
            "multiplicity":           multiplicity,
            "N_null":                 N_NULL,
            "PTE_global_max_old":     float(old_row["PTE_global_max"]),
            "PTE_global_max_new":     pte_max_new,
            "PTE_global_absint_old":  pte_abs_old,
            "PTE_global_absint_new":  pte_abs_new,
            "relative_abs_bias":      b_rel,
            "n_null_exceed_absint":   int(profile["n_null_exceed_absint"]),
            "n_null_exceed_max":      int(profile["n_null_exceed_max"]),
            "window_abs_bias_sum":    float(profile["window_abs_bias_sum"]),
            "window_bg_abs_sum":      float(profile["window_bg_abs_sum"]),
            "Z_PS_max_new":           float(profile["Z_PS_max"]),
            # gate flags
            "above_brel_1e3":         bool(b_rel > 1e-3),
            "retained_absint_001":    bool(pte_abs_new < 0.01),
            "retained_max_001":       bool(pte_max_new < 0.01),
        }
        rows.append(row)

        print(
            f"[{beam}/{morphology}/{flux:g}Jy/{multiplicity}] "
            f"absint {pte_abs_old:.5f}(N=100) -> {pte_abs_new:.5f}(N=1000) "
            f"max_new={pte_max_new:.5f} B_rel={b_rel:.3e} "
            f"n_exceed_absint={row['n_null_exceed_absint']}/{N_NULL}",
            flush=True,
        )

    out = to_local_path("outputs/absint_floor_recheck.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)

    meta = {
        "description": "N=1000 re-evaluation of absint PTE floor-hit cases",
        "tle_meta": tle_meta,
        "baseline_id": BASELINE_ID,
        "lst_stratum": LST_STRATUM,
        "lst_bin_id": LST_BIN_ID,
        "n_null": N_NULL,
        "floor_hit_value_N100": round(1 / 101, 10),
        "above_brel_1e3_retained_absint": int(
            df_out[df_out["above_brel_1e3"]]["retained_absint_001"].sum()
        ),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n=== Summary ===")
    for _, r in df_out.iterrows():
        gate = "ABOVE 1e-3" if r["above_brel_1e3"] else "below 1e-3"
        print(
            f"  {r['morphology']}/{r['flux_jy']:g}Jy/{r['multiplicity']} [{gate}]"
            f"  absint_new={r['PTE_global_absint_new']:.4f}"
            f"  retained_absint={r['retained_absint_001']}"
        )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
