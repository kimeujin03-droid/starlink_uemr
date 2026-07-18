#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


KEY = ["baseline_id", "lst_stratum", "lst_bin_id", "morphology", "flux_jy", "multiplicity"]


def main() -> None:
    trials = pd.read_csv(ROOT / "coverage_robustness_trials_fourflux.csv")
    out_dir = ROOT / "fourflux_summary"
    out_dir.mkdir(exist_ok=True)

    floors = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    rows = []
    strict = trials["PTE_global_max"].astype(float) < 0.01
    absint_strict = trials["PTE_global_absint"].astype(float) < 0.01
    rel = trials["relative_abs_bias"].astype(float)
    for floor in floors:
        rows.append(
            {
                "B_floor": floor,
                "n_rows": len(trials),
                "n_strict_local": int(strict.sum()),
                "n_local_physical": int((strict & (rel > floor)).sum()),
                "n_strict_absint": int(absint_strict.sum()),
                "n_absint_physical": int((absint_strict & (rel > floor)).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "coverage_candidate_counts_by_floor_fourflux.csv", index=False)

    piv = trials.pivot_table(
        index=KEY,
        columns="beam_model",
        values=["PTE_global_max", "PTE_global_absint", "relative_abs_bias", "Z_PS_max"],
        aggfunc="first",
    )
    piv.columns = [f"{metric}__{beam}" for metric, beam in piv.columns]
    pairs = piv.reset_index()
    pairs["candidate_statistical__frozen_polybeam"] = pairs["PTE_global_max__frozen_polybeam"] < 0.01
    pairs["candidate_statistical__full_polybeam"] = pairs["PTE_global_max__full_polybeam"] < 0.01
    pairs["absint_strict__frozen_polybeam"] = pairs["PTE_global_absint__frozen_polybeam"] < 0.01
    pairs["absint_strict__full_polybeam"] = pairs["PTE_global_absint__full_polybeam"] < 0.01
    pairs["local_physical_1e3__frozen_polybeam"] = (
        pairs["candidate_statistical__frozen_polybeam"] & (pairs["relative_abs_bias__frozen_polybeam"] > 1e-3)
    )
    pairs["local_physical_1e3__full_polybeam"] = (
        pairs["candidate_statistical__full_polybeam"] & (pairs["relative_abs_bias__full_polybeam"] > 1e-3)
    )
    pairs["absint_physical_1e3__frozen_polybeam"] = (
        pairs["absint_strict__frozen_polybeam"] & (pairs["relative_abs_bias__frozen_polybeam"] > 1e-3)
    )
    pairs["absint_physical_1e3__full_polybeam"] = (
        pairs["absint_strict__full_polybeam"] & (pairs["relative_abs_bias__full_polybeam"] > 1e-3)
    )
    pairs["beam_robust_local_physical_1e3"] = (
        pairs["local_physical_1e3__frozen_polybeam"] & pairs["local_physical_1e3__full_polybeam"]
    )
    pairs["frozen_only_local_physical_1e3"] = (
        pairs["local_physical_1e3__frozen_polybeam"] & ~pairs["local_physical_1e3__full_polybeam"]
    )
    pairs["full_only_local_physical_1e3"] = (
        ~pairs["local_physical_1e3__frozen_polybeam"] & pairs["local_physical_1e3__full_polybeam"]
    )
    pairs["beam_robust_absint_physical_1e3"] = (
        pairs["absint_physical_1e3__frozen_polybeam"] & pairs["absint_physical_1e3__full_polybeam"]
    )
    pairs["frozen_only_absint_physical_1e3"] = (
        pairs["absint_physical_1e3__frozen_polybeam"] & ~pairs["absint_physical_1e3__full_polybeam"]
    )
    pairs["full_only_absint_physical_1e3"] = (
        ~pairs["absint_physical_1e3__frozen_polybeam"] & pairs["absint_physical_1e3__full_polybeam"]
    )
    pairs["delta_PTE_global_max"] = pairs["PTE_global_max__full_polybeam"] - pairs["PTE_global_max__frozen_polybeam"]
    pairs["delta_PTE_global_absint"] = pairs["PTE_global_absint__full_polybeam"] - pairs["PTE_global_absint__frozen_polybeam"]
    pairs["delta_relative_abs_bias"] = pairs["relative_abs_bias__full_polybeam"] - pairs["relative_abs_bias__frozen_polybeam"]
    pairs.to_csv(out_dir / "polybeam_pair_audit_fourflux.csv", index=False)

    local_tail = trials[((trials["PTE_global_max"] <= 0.03) | (trials["Z_PS_max"] > 3.0)) & (rel > 1e-3)].copy()
    absint_tail = trials[(trials["PTE_global_absint"] <= 0.03) & (rel > 1e-3)].copy()
    local_tail.to_csv(out_dir / "coverage_tail_candidates_local_fourflux.csv", index=False)
    absint_tail.to_csv(out_dir / "coverage_tail_candidates_absint_fourflux.csv", index=False)

    by_flux = (
        trials.groupby("flux_jy")
        .agg(
            rows=("PTE_global_max", "size"),
            n_strict_local=("PTE_global_max", lambda x: int((x < 0.01).sum())),
            n_strict_absint=("PTE_global_absint", lambda x: int((x < 0.01).sum())),
            median_Brel=("relative_abs_bias", "median"),
            max_Brel=("relative_abs_bias", "max"),
            min_PTE_global_max=("PTE_global_max", "min"),
            min_PTE_global_absint=("PTE_global_absint", "min"),
        )
        .reset_index()
    )
    by_flux.to_csv(out_dir / "coverage_summary_by_flux_fourflux.csv", index=False)

    summary = {
        "n_rows": len(trials),
        "n_pairs": len(pairs),
        "n_strict_local": int(strict.sum()),
        "n_strict_absint": int(absint_strict.sum()),
        "n_local_physical_1e3": int((strict & (rel > 1e-3)).sum()),
        "n_absint_physical_1e3": int((absint_strict & (rel > 1e-3)).sum()),
        "n_beam_robust_local_physical_1e3": int(pairs["beam_robust_local_physical_1e3"].sum()),
        "n_frozen_only_local_physical_1e3": int(pairs["frozen_only_local_physical_1e3"].sum()),
        "n_full_only_local_physical_1e3": int(pairs["full_only_local_physical_1e3"].sum()),
        "n_beam_robust_absint_physical_1e3": int(pairs["beam_robust_absint_physical_1e3"].sum()),
        "n_frozen_only_absint_physical_1e3": int(pairs["frozen_only_absint_physical_1e3"].sum()),
        "n_full_only_absint_physical_1e3": int(pairs["full_only_absint_physical_1e3"].sum()),
        "n_local_tail_candidates_for_n1000": len(local_tail),
        "n_absint_tail_candidates_for_n1000": len(absint_tail),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "fourflux_topline_summary.csv", index=False)

    md = ["# Four-Flux Coverage Run Summary", "", "Run date: 2026-07-14", ""]
    md.append("## Scope")
    md.append("")
    md.append(f"- Rows: {summary['n_rows']}")
    md.append(f"- Frozen/full paired cases: {summary['n_pairs']}")
    md.append("- Flux tiers: 10, 30, 300, 1000 Jy")
    md.append("- Null draws per row: 100")
    md.append("")
    md.append("## Top-Line Counts")
    md.append("")
    md.append("| metric | count |")
    md.append("| --- | ---: |")
    for key, val in summary.items():
        md.append(f"| `{key}` | {val} |")
    md.append("")
    md.append("## Outputs")
    md.append("")
    for name in [
        "coverage_candidate_counts_by_floor_fourflux.csv",
        "coverage_summary_by_flux_fourflux.csv",
        "polybeam_pair_audit_fourflux.csv",
        "coverage_tail_candidates_local_fourflux.csv",
        "coverage_tail_candidates_absint_fourflux.csv",
        "fourflux_topline_summary.csv",
    ]:
        md.append(f"- `fourflux_summary/{name}`")
    (ROOT / "2026-07-14_fourflux_coverage_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"saved summaries to {out_dir}")


if __name__ == "__main__":
    main()
