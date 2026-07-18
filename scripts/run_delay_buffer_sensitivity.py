#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
    compute_metrics,
    configure_case,
    load_tle_fast,
    rank_visible_sats,
    read_uvh5_bin,
    to_local_path,
)


def choose_cases(trials: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    trials = trials.copy()
    trials["Z_PS_max_num"] = pd.to_numeric(trials["Z_PS_max"], errors="coerce")
    chosen = trials[
        (trials.get("candidate_statistical", False).astype(str) == "True")
        | (trials.get("flag_exploratory", False).astype(str) == "True")
        | (trials.get("PS_gt_3", False).astype(str) == "True")
    ].copy()
    if len(chosen) < max_cases:
        extra = trials.sort_values("Z_PS_max_num", ascending=False).head(max_cases)
        chosen = pd.concat([chosen, extra], ignore_index=True)
    key_cols = ["baseline_id", "lst_stratum", "lst_bin_id", "beam_model", "morphology", "flux_jy", "multiplicity"]
    return chosen.drop_duplicates(key_cols).sort_values("Z_PS_max_num", ascending=False).head(max_cases)


def set_buffer(cfg: dict, buffer_ns: float) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("metrics", {}).setdefault("window", {})["buffer_ns"] = float(buffer_ns)
    out.setdefault("pipeline", {}).setdefault("delay_filter", {})["buffer_ns"] = float(buffer_ns)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials.csv")
    ap.add_argument("--config", default="configs/coverage_robustness.yaml")
    ap.add_argument("--pathb-config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--out", default="outputs/delay_buffer_sensitivity.csv")
    ap.add_argument("--summary-out", default="outputs/delay_buffer_sensitivity_summary.csv")
    ap.add_argument("--buffers-ns", nargs="*", type=float, default=[0.0, 50.0, 100.0, 150.0, 200.0])
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--max-cases", type=int, default=6)
    ap.add_argument("--seed-offset", type=int, default=910000)
    args = ap.parse_args()

    cfg_run = yaml.safe_load(to_local_path(args.config).read_text(encoding="utf-8"))
    base_cfg = read_config(to_local_path(args.pathb_config))
    sel = pd.read_csv(to_local_path(args.selection))
    trials = pd.read_csv(to_local_path(args.trials))
    cases = choose_cases(trials, args.max_cases)

    recs, _tle_meta = load_tle_fast(
        to_local_path(cfg_run.get("tle_path", "tle/starlink_jan2026_LEO_only.tle")),
        max_records=int(cfg_run.get("max_tle_records", 1200)),
    )
    rec_map = {r.norad_id: r for r in recs}
    rows = []
    vis_cache = {}
    ctx_cache = {}
    rank_cache = {}

    for cidx, case in enumerate(cases.itertuples(index=False), start=1):
        cell_key = (str(case.baseline_id), str(case.lst_stratum), int(case.lst_bin_id))
        if cell_key not in ctx_cache:
            srow = sel[
                (sel["baseline_id"].astype(str) == cell_key[0])
                & (sel["lst_stratum"].astype(str) == cell_key[1])
                & (sel["lst_bin_id"].astype(int) == cell_key[2])
            ]
            if len(srow) != 1:
                raise ValueError(f"Expected one selection row for {cell_key}, found {len(srow)}")
            ctx_cache[cell_key] = read_uvh5_bin(srow.iloc[0])
            rank_cache[cell_key] = rank_visible_sats(
                recs,
                ctx_cache[cell_key],
                base_cfg,
                float(cfg_run.get("alt_visible_deg", 70.0)),
            )
        ctx = ctx_cache[cell_key]
        ranked = rank_cache[cell_key]
        n_sat_target = 1 if str(case.multiplicity) == "single" else int(cfg_run.get("max_multi_satellites", 12))
        chosen = ranked.head(max(1, min(n_sat_target, len(ranked))))
        if len(chosen) == 0:
            continue

        cfg_case_base = configure_case(base_cfg, str(case.beam_model), str(case.morphology))
        sat_vis_list = []
        for norad in chosen["norad_id"].astype(str).tolist():
            vkey = (cell_key, str(case.beam_model), str(case.morphology), float(case.flux_jy), norad)
            if vkey not in vis_cache:
                vis, _track, _report = build_visibility_for_sat(
                    rec_map[norad], ctx, cfg_case_base, s_ref_jy=float(case.flux_jy)
                )
                vis_cache[vkey] = vis
            sat_vis_list.append(vis_cache[vkey])

        for bidx, buffer_ns in enumerate(args.buffers_ns):
            cfg = set_buffer(cfg_case_base, float(buffer_ns))
            seed = int(getattr(case, "seed", args.seed_offset + 100 * cidx))
            metrics = compute_metrics(
                ctx,
                cfg,
                sat_vis_list,
                int(args.n_null),
                seed + args.seed_offset,
                str(getattr(case, "injection_mode", "coherent_ab")),
                to_local_path("outputs/delay_buffer_null_stats")
                / f"{cell_key[0]}_lst{cell_key[2]:03d}_{case.beam_model}_{case.morphology}_S{float(case.flux_jy):g}_{case.multiplicity}_buf{buffer_ns:g}.npz",
            )
            rows.append(
                {
                    "case_rank": cidx,
                    "baseline_id": cell_key[0],
                    "lst_stratum": cell_key[1],
                    "lst_bin_id": cell_key[2],
                    "beam_model": str(case.beam_model),
                    "morphology": str(case.morphology),
                    "flux_jy": float(case.flux_jy),
                    "multiplicity": str(case.multiplicity),
                    "buffer_ns": float(buffer_ns),
                    "n_null": int(args.n_null),
                    "Z_PS_max": metrics["Z_PS_max"],
                    "PTE_global_max": metrics["PTE_global_max"],
                    "PTE_global_absint": metrics["PTE_global_absint"],
                    "relative_abs_bias": metrics["relative_abs_bias"],
                    "n_window_bins": metrics["n_window_bins"],
                    "tau_ns_min_window": metrics["tau_ns_min_window"],
                    "tau_ns_max_window": metrics["tau_ns_max_window"],
                    "candidate_statistical": bool(metrics["PTE_global_max"] < 0.01),
                    "candidate_physical_1e2": bool(metrics["PTE_global_max"] < 0.01 and metrics["relative_abs_bias"] > 1e-2),
                }
            )
        print(f"[{cidx}/{len(cases)}] {cell_key} {case.beam_model}/{case.morphology}", flush=True)

    df = pd.DataFrame(rows)
    out = to_local_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = (
        df.groupby("buffer_ns")
        .agg(
            n_cases=("case_rank", "nunique"),
            n_rows=("case_rank", "size"),
            n_candidate_statistical=("candidate_statistical", "sum"),
            n_candidate_physical_1e2=("candidate_physical_1e2", "sum"),
            median_Z_PS_max=("Z_PS_max", "median"),
            min_PTE_global_max=("PTE_global_max", "min"),
            median_relative_abs_bias=("relative_abs_bias", "median"),
            median_n_window_bins=("n_window_bins", "median"),
        )
        .reset_index()
    )
    summary.to_csv(to_local_path(args.summary_out), index=False)
    meta = {
        "n_null": args.n_null,
        "buffers_ns": args.buffers_ns,
        "max_cases": args.max_cases,
        "case_selection": "candidate_statistical OR exploratory OR PS_gt_3, then highest Z_PS_max",
        "scope": "representative-case sensitivity, not full 648-row grid rerun",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(f"saved {to_local_path(args.summary_out)}")


if __name__ == "__main__":
    main()
