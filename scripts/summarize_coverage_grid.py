#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_neglog10(x: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(float)
    vals = np.clip(vals, np.nextafter(0.0, 1.0), 1.0)
    return -np.log10(vals)


def write_tables(df: pd.DataFrame, out_dir: Path, floors: list[float]) -> None:
    rows = []
    candidate_stat = df["candidate_statistical"].astype(bool)
    rel_bias = pd.to_numeric(df["relative_abs_bias"], errors="coerce")
    for floor in floors:
        physical = candidate_stat & (rel_bias > float(floor))
        rows.append(
            {
                "relative_abs_bias_floor": floor,
                "n_trials": len(df),
                "n_candidate_statistical": int(df["candidate_statistical"].sum()),
                "n_candidate_physical": int(physical.sum()),
                "Pr_candidate_physical": float(physical.mean()) if len(df) else float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "coverage_candidate_counts_by_floor.csv", index=False)

    by_cell = (
        df.groupby(["baseline_id", "baseline_class", "lst_stratum", "lst_bin_id"])
        .agg(
            rows=("Z_PS_max", "size"),
            max_Z_PS_max=("Z_PS_max", "max"),
            median_Z_PS_max=("Z_PS_max", "median"),
            min_PTE_global_max=("PTE_global_max", "min"),
            max_relative_abs_bias=("relative_abs_bias", "max"),
            n_Zgt3=("PS_gt_3", "sum"),
            n_candidate_statistical=("candidate_statistical", "sum"),
            n_exploratory=("flag_exploratory", "sum"),
        )
        .reset_index()
    )
    by_cell.to_csv(out_dir / "coverage_summary_by_baseline_lst.csv", index=False)

    factor_rows = []
    for factor in ["beam_model", "morphology", "flux_jy", "multiplicity", "baseline_class", "lst_stratum"]:
        sub = (
            df.groupby(factor)
            .agg(
                rows=("Z_PS_max", "size"),
                median_Z_PS_max=("Z_PS_max", "median"),
                p95_Z_PS_max=("Z_PS_max", lambda x: float(np.nanpercentile(x, 95))),
                min_PTE_global_max=("PTE_global_max", "min"),
                median_relative_abs_bias=("relative_abs_bias", "median"),
                n_Zgt3=("PS_gt_3", "sum"),
                n_candidate_statistical=("candidate_statistical", "sum"),
            )
            .reset_index()
            .rename(columns={factor: "level"})
        )
        sub.insert(0, "factor", factor)
        factor_rows.append(sub)
    pd.concat(factor_rows, ignore_index=True).to_csv(out_dir / "coverage_summary_by_factor.csv", index=False)

    try:
        import statsmodels.formula.api as smf

        fit = smf.ols(
            "Z_PS_max ~ C(beam_model) + C(morphology) + C(flux_jy) + C(multiplicity) + "
            "C(baseline_class) + C(lst_stratum) + C(beam_model):C(morphology) + "
            "C(beam_model):C(lst_stratum)",
            data=df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Z_PS_max"]),
        ).fit()
        pd.DataFrame(
            {
                "term": fit.params.index,
                "coef": fit.params.to_numpy(),
                "pvalue": fit.pvalues.to_numpy(),
                "stderr": fit.bse.to_numpy(),
            }
        ).to_csv(out_dir / "coverage_anova_results.csv", index=False)
    except Exception as exc:
        (out_dir / "coverage_anova_results.txt").write_text(f"ANOVA/regression skipped: {exc}\n", encoding="utf-8")


def make_figures(df: pd.DataFrame, metadata: pd.DataFrame | None, selection: pd.DataFrame | None, fig_dir: Path, floors: list[float]) -> None:
    if metadata is not None and selection is not None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        base_order = sorted(metadata["baseline_id"].astype(str).unique())
        base_map = {b: i for i, b in enumerate(base_order)}
        x = metadata["lst_bin_id"].to_numpy(float)
        y = metadata["baseline_id"].astype(str).map(base_map).to_numpy(float)
        sc = ax.scatter(x, y, c=np.log10(metadata["pre_risk_score"].clip(lower=1e-30)), s=12, cmap="viridis", alpha=0.7)
        markers = {"quiet": "o", "typical": "s", "stress": "^"}
        for name, sub in selection.groupby("lst_stratum"):
            ax.scatter(
                sub["lst_bin_id"],
                sub["baseline_id"].astype(str).map(base_map),
                marker=markers.get(name, "x"),
                s=70,
                edgecolor="black",
                facecolor="none",
                linewidth=1.2,
                label=name,
            )
        ax.set_yticks(range(len(base_order)))
        ax.set_yticklabels(base_order)
        ax.set_xlabel("10-min LST bin")
        ax.set_ylabel("baseline")
        ax.set_title("R1 LST-bin pre-risk selection")
        fig.colorbar(sc, ax=ax, label="log10(pre_risk_score)")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "R1_lst_selection_map.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(df["Z_PS_max"], safe_neglog10(df["PTE_global_max"]), c=np.log10(df["relative_abs_bias"].clip(lower=1e-30)), s=22, cmap="magma", alpha=0.8)
    ax.axvline(3.0, color="0.25", linestyle="--", linewidth=1)
    ax.axhline(-np.log10(0.05), color="0.5", linestyle=":", linewidth=1)
    ax.axhline(-np.log10(0.01), color="0.25", linestyle="--", linewidth=1)
    ax.set_xlabel("Z_PS,max")
    ax.set_ylabel("-log10(PTE_global,max)")
    ax.set_title("R2 robust-z vs global PTE")
    fig.colorbar(sc, ax=ax, label="log10(relative_abs_bias)")
    fig.tight_layout()
    fig.savefig(fig_dir / "R2_z_vs_pte_global.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.log10(df["relative_abs_bias"].clip(lower=1e-30))
    y = safe_neglog10(df["PTE_global_max"])
    ax.scatter(x, y, c=df["Z_PS_max"], s=22, cmap="coolwarm", alpha=0.8)
    for floor in floors:
        ax.axvline(np.log10(floor), color="0.65", linestyle=":", linewidth=0.9)
    ax.axhline(-np.log10(0.01), color="0.25", linestyle="--", linewidth=1)
    ax.set_xlabel("log10(relative_abs_bias)")
    ax.set_ylabel("-log10(PTE_global,max)")
    ax.set_title("R3 physical-bias floor sensitivity")
    fig.tight_layout()
    fig.savefig(fig_dir / "R3_bias_floor_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(
        np.log10(df["null_mad_win"].clip(lower=1e-300)),
        df["Z_PS_max"],
        c=np.log10(df["relative_abs_bias"].clip(lower=1e-30)),
        s=22,
        cmap="viridis",
        alpha=0.8,
    )
    ax.axhline(3.0, color="0.25", linestyle="--", linewidth=1)
    ax.set_xlabel("log10(null_mad_win)")
    ax.set_ylabel("Z_PS,max")
    ax.set_title("R4 null-MAD diagnostic")
    fig.colorbar(sc, ax=ax, label="log10(relative_abs_bias)")
    fig.tight_layout()
    fig.savefig(fig_dir / "R4_null_mad_diagnostic.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials.csv")
    ap.add_argument("--metadata", default="outputs/lst_bin_metadata.csv")
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--fig-dir", default="figures/coverage_robustness")
    ap.add_argument("--bias-floors", nargs="*", type=float, default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    args = ap.parse_args()

    df = pd.read_csv(args.trials)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)
    metadata = pd.read_csv(args.metadata) if Path(args.metadata).exists() else None
    selection = pd.read_csv(args.selection) if Path(args.selection).exists() else None
    write_tables(df, out_dir, args.bias_floors)
    make_figures(df, metadata, selection, fig_dir, args.bias_floors)
    print(f"saved coverage summaries to {out_dir} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
