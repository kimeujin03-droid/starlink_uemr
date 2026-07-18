#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials.csv")
    ap.add_argument("--out-cell", default="outputs/background_sensitivity_by_cell.csv")
    ap.add_argument("--out-stratum", default="outputs/background_sensitivity_by_lst_stratum.csv")
    ap.add_argument("--out-summary", default="outputs/background_sensitivity_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.trials)
    for col in ["Z_PS_max", "PTE_global_max", "PTE_global_absint", "relative_abs_bias"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["candidate_statistical_bool"] = df["candidate_statistical"].astype(str).eq("True")
    df["candidate_physical_1e2_bool"] = df["candidate_statistical_bool"] & (df["relative_abs_bias"] > 1e-2)

    cell = (
        df.groupby(["baseline_id", "baseline_class", "lst_stratum", "lst_bin_id"])
        .agg(
            rows=("Z_PS_max", "size"),
            median_Z_PS_max=("Z_PS_max", "median"),
            p95_Z_PS_max=("Z_PS_max", lambda x: float(np.nanquantile(x, 0.95))),
            min_PTE_global_max=("PTE_global_max", "min"),
            median_PTE_global_absint=("PTE_global_absint", "median"),
            median_relative_abs_bias=("relative_abs_bias", "median"),
            p95_relative_abs_bias=("relative_abs_bias", lambda x: float(np.nanquantile(x, 0.95))),
            n_candidate_statistical=("candidate_statistical_bool", "sum"),
            n_candidate_physical_1e2=("candidate_physical_1e2_bool", "sum"),
        )
        .reset_index()
    )
    cell.to_csv(ROOT / args.out_cell, index=False)

    stratum = (
        cell.groupby(["baseline_class", "lst_stratum"])
        .agg(
            n_cells=("lst_bin_id", "size"),
            median_cell_Z_PS=("median_Z_PS_max", "median"),
            max_cell_p95_Z_PS=("p95_Z_PS_max", "max"),
            min_cell_PTE=("min_PTE_global_max", "min"),
            median_cell_Brel=("median_relative_abs_bias", "median"),
            max_cell_Brel_p95=("p95_relative_abs_bias", "max"),
            total_candidate_statistical=("n_candidate_statistical", "sum"),
            total_candidate_physical_1e2=("n_candidate_physical_1e2", "sum"),
        )
        .reset_index()
    )
    stratum.to_csv(ROOT / args.out_stratum, index=False)

    summary = pd.DataFrame(
        [
            {
                "n_rows": int(df.shape[0]),
                "n_cells": int(cell.shape[0]),
                "cell_median_Z_PS_range_min": float(cell["median_Z_PS_max"].min()),
                "cell_median_Z_PS_range_max": float(cell["median_Z_PS_max"].max()),
                "cell_p95_Z_PS_max": float(cell["p95_Z_PS_max"].max()),
                "cell_min_PTE_global_max": float(cell["min_PTE_global_max"].min()),
                "cell_median_Brel_range_min": float(cell["median_relative_abs_bias"].min()),
                "cell_median_Brel_range_max": float(cell["median_relative_abs_bias"].max()),
                "cell_p95_Brel_max": float(cell["p95_relative_abs_bias"].max()),
                "n_cells_with_statistical_candidate": int((cell["n_candidate_statistical"] > 0).sum()),
                "n_statistical_candidates_total": int(cell["n_candidate_statistical"].sum()),
                "n_physical_1e2_candidates_total": int(cell["n_candidate_physical_1e2"].sum()),
            }
        ]
    )
    summary.to_csv(ROOT / args.out_summary, index=False)
    (ROOT / Path(args.out_summary).with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "scope": "Post-hoc summary of existing 648-row coverage grid; no new injections.",
                "grouping": "baseline_id, baseline_class, lst_stratum, lst_bin_id",
                "physical_floor": "relative_abs_bias > 1e-2 combined with candidate_statistical",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {ROOT / args.out_cell}")
    print(f"saved {ROOT / args.out_stratum}")
    print(f"saved {ROOT / args.out_summary}")


if __name__ == "__main__":
    main()
