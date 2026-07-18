#!/usr/bin/env python3
"""N=1000 re-evaluation of the full_polybeam side of the two first1200 absint
floor-hit candidates.

`outputs/absint_floor_recheck.csv` already reruns the frozen_polybeam side of
these two cases at N_null=1000. The paired full_polybeam comparison that
determines the final "0 beam-robust candidates" verdict for this catalog was
only ever evaluated at N_null=100 (see `outputs/coverage_robustness_trials.csv`,
PTE_global_absint = 0.029703 and 0.019802 for the two cases). At N_null=100 the
empirical PTE floor is 1/101 = 0.0099, which is too coarse to treat those
values as a settled non-detection. This script closes that gap by rerunning
the full_polybeam side at N_null=1000 with the same injection geometry, seed,
and satellite selection as the original N=100 trial, then applies the same
paired-beam classification rule used for the full-catalog audit.
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
from scripts.run_full_catalog_physical_candidate_audit import classify_case

BASELINE_ID = "11_10"
LST_STRATUM = "quiet"
LST_BIN_ID = 16
N_NULL = 1000

# (beam, morphology, flux_jy, multiplicity) pairs: frozen side (already at N=1000
# in absint_floor_recheck.csv) paired against the full_polybeam side rerun here.
CASES = [
    ("smooth", 1000.0, "multi"),
    ("khz_comb", 1000.0, "multi"),
]


def main() -> None:
    cfg_run = yaml.safe_load(to_local_path("configs/coverage_robustness.yaml").read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path("configs/pathB_jan2026_main.yaml"))

    sel = pd.read_csv(to_local_path("outputs/lst_bin_selection.csv"))
    trials = pd.read_csv(to_local_path("outputs/coverage_robustness_trials.csv"))
    frozen_n1000 = pd.read_csv(to_local_path("outputs/absint_floor_recheck.csv"))

    srows = sel[
        (sel["baseline_id"].astype(str) == BASELINE_ID)
        & (sel["lst_stratum"].astype(str) == LST_STRATUM)
        & (sel["lst_bin_id"].astype(int) == LST_BIN_ID)
    ]
    if len(srows) != 1:
        raise ValueError(f"Expected 1 selection row, got {len(srows)}")
    ctx = read_uvh5_bin(srows.iloc[0])

    recs, tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026_LEO_only.tle")),
        max_records=int(cfg_run.get("max_tle_records", 1200)),
    )
    rec_map = {r.norad_id: r for r in recs}
    ranked = rank_visible_sats(recs, ctx, base_cfg, float(cfg_run.get("alt_visible_deg", 70.0)))
    print(f"[rank] {BASELINE_ID} lst_bin={LST_BIN_ID} visible={len(ranked)}", flush=True)
    if len(ranked) == 0:
        raise RuntimeError("No visible satellites")

    rows = []
    for morphology, flux, multiplicity in CASES:
        frozen_row = frozen_n1000[
            (frozen_n1000["beam_model"] == "frozen_polybeam")
            & (frozen_n1000["morphology"] == morphology)
            & (np.isclose(frozen_n1000["flux_jy"].astype(float), flux))
            & (frozen_n1000["multiplicity"] == multiplicity)
        ]
        if len(frozen_row) != 1:
            raise ValueError(f"Expected 1 frozen N=1000 row for {(morphology, flux, multiplicity)}, got {len(frozen_row)}")
        frozen_row = frozen_row.iloc[0]
        frozen_profile = {
            "PTE_global_max": float(frozen_row["PTE_global_max_new"]),
            "PTE_global_absint": float(frozen_row["PTE_global_absint_new"]),
            "relative_abs_bias": float(frozen_row["relative_abs_bias"]),
        }

        old_full = trials[
            (trials["baseline_id"].astype(str) == BASELINE_ID)
            & (trials["lst_stratum"].astype(str) == LST_STRATUM)
            & (trials["lst_bin_id"].astype(int) == LST_BIN_ID)
            & (trials["beam_model"] == "full_polybeam")
            & (trials["morphology"] == morphology)
            & (np.isclose(trials["flux_jy"].astype(float), flux))
            & (trials["multiplicity"] == multiplicity)
        ]
        if len(old_full) != 1:
            raise ValueError(f"Expected 1 old full_polybeam N=100 trial for {(morphology, flux, multiplicity)}, got {len(old_full)}")
        old_full = old_full.iloc[0]

        n_sat_target = 1 if multiplicity == "single" else int(cfg_run.get("max_multi_satellites", 12))
        chosen = ranked.head(max(1, min(n_sat_target, len(ranked))))
        selected_norads = ";".join(chosen["norad_id"].astype(str).tolist())
        if selected_norads != str(old_full["selected_norad_ids"]):
            raise ValueError(
                f"Satellite selection mismatch for {(morphology, flux, multiplicity)}: "
                f"rerun={selected_norads} old={old_full['selected_norad_ids']}"
            )

        cfg = configure_case(base_cfg, "full_polybeam", morphology)
        sat_vis_list = []
        for norad in chosen["norad_id"].astype(str).tolist():
            vis, _track, _report = build_visibility_for_sat(rec_map[norad], ctx, cfg, s_ref_jy=float(flux))
            sat_vis_list.append(vis)

        full_profile = compute_tail_case(
            ctx,
            cfg,
            sat_vis_list,
            N_NULL,
            int(old_full["seed"]),
            str(old_full.get("injection_mode", "coherent_ab")),
        )

        final_class = classify_case(frozen_profile, full_profile)

        row = {
            "baseline_id": BASELINE_ID,
            "lst_stratum": LST_STRATUM,
            "lst_bin_id": LST_BIN_ID,
            "morphology": morphology,
            "flux_jy": float(flux),
            "multiplicity": multiplicity,
            "N_null": N_NULL,
            "frozen_PTE_max_1000": frozen_profile["PTE_global_max"],
            "frozen_PTE_absint_1000": frozen_profile["PTE_global_absint"],
            "frozen_B_rel": frozen_profile["relative_abs_bias"],
            "full_PTE_max_100_old": float(old_full["PTE_global_max"]),
            "full_PTE_absint_100_old": float(old_full["PTE_global_absint"]),
            "full_PTE_max_1000_new": float(full_profile["PTE_global_max"]),
            "full_PTE_absint_1000_new": float(full_profile["PTE_global_absint"]),
            "full_B_rel_1000_new": float(full_profile["relative_abs_bias"]),
            "final_class": final_class,
            "seed": int(old_full["seed"]),
            "selected_norad_ids": selected_norads,
        }
        rows.append(row)
        print(
            f"[{morphology}/{flux:g}Jy/{multiplicity}] full_polybeam absint "
            f"{float(old_full['PTE_global_absint']):.5f}(N=100) -> {full_profile['PTE_global_absint']:.5f}(N=1000) "
            f"max_new={full_profile['PTE_global_max']:.5f} B_rel={full_profile['relative_abs_bias']:.3e} "
            f"class={final_class}",
            flush=True,
        )

    out = to_local_path("outputs/polybeam_pair_n1000_recheck.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)

    meta = {
        "description": "N=1000 re-evaluation of the full_polybeam side of the 2 first1200 absint floor-hit candidates",
        "tle_meta": tle_meta,
        "baseline_id": BASELINE_ID,
        "lst_stratum": LST_STRATUM,
        "lst_bin_id": LST_BIN_ID,
        "n_null": N_NULL,
        "n_beam_robust": int((df_out["final_class"] == "beam-robust contamination candidate").sum()),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    print(df_out[["morphology", "full_PTE_max_1000_new", "full_PTE_absint_1000_new", "full_B_rel_1000_new", "final_class"]].to_string(index=False))


if __name__ == "__main__":
    main()
