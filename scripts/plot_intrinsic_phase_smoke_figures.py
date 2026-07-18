#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "intrinsic_phase_smoke5_figures"


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def plot_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    ax.set_axis_off()
    boxes = [
        ("HERA-like background\n+ TLE-based UEMR injection", 0.50, 0.93, 0.46, 0.08),
        ("QA operator\nweighted delay transform + window mask", 0.50, 0.78, 0.44, 0.08),
        ("Local branch L\nPTE_global,max < 0.01", 0.27, 0.58, 0.30, 0.08),
        ("Integrated branch I\nPTE_global,absint < 0.01", 0.73, 0.58, 0.34, 0.08),
        ("Physical-amplitude gate\nB_rel > B_floor", 0.50, 0.40, 0.36, 0.08),
        ("Paired beam robustness\nfrozen vs full chromatic beam", 0.50, 0.27, 0.40, 0.08),
        ("Calibration-residual stability audit", 0.50, 0.15, 0.40, 0.07),
        ("Classification and reporting", 0.50, 0.05, 0.34, 0.06),
    ]
    for text, x, y, w, h in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                facecolor="#f7f7f7",
                edgecolor="#333333",
                linewidth=1.2,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=10)

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"),
        )

    arrow(0.50, 0.89, 0.50, 0.82)
    arrow(0.50, 0.74, 0.27, 0.63)
    arrow(0.50, 0.74, 0.73, 0.63)
    arrow(0.27, 0.54, 0.50, 0.44)
    arrow(0.73, 0.54, 0.50, 0.44)
    arrow(0.50, 0.36, 0.50, 0.31)
    arrow(0.50, 0.23, 0.50, 0.185)
    arrow(0.50, 0.115, 0.50, 0.08)
    ax.text(
        0.50,
        0.685,
        "Local and integrated branches are diagnostically parallel,\n"
        "but final integrated-contamination interpretation requires\n"
        "significance in the integrated branch.",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    ax.text(
        0.50,
        0.005,
        "Output classes: exploratory QA flag | local physical candidate | "
        "beam-sensitive integrated candidate | beam-robust integrated candidate",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    savefig(fig, "F1_parallel_hierarchy_smoke5")


def plot_decision_plane(trials: pd.DataFrame, coherent: pd.DataFrame) -> None:
    focus = trials[
        (trials["baseline_id"].astype(str) == "11_10")
        & (trials["lst_bin_id"].astype(int) == 16)
        & (trials["multiplicity"].astype(str) == "multi")
        & (trials["morphology"].isin(["smooth", "lines", "khz_comb"]))
    ].copy()
    coh = coherent[
        (coherent["baseline_id"].astype(str) == "11_10")
        & (coherent["lst_bin_id"].astype(int) == 16)
        & (coherent["multiplicity"].astype(str) == "multi")
        & (coherent["morphology"].isin(["smooth", "lines", "khz_comb"]))
    ].copy()
    morphs = ["smooth", "lines", "khz_comb"]
    colors = {"frozen_polybeam": "#2f6f9f", "full_polybeam": "#b4473c"}
    markers = {"frozen_polybeam": "o", "full_polybeam": "s"}
    labels = {"frozen_polybeam": "frozen", "full_polybeam": "full"}

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    for ax, morph in zip(axes, morphs, strict=True):
        sub = focus[focus["morphology"] == morph]
        for seed, grp in sub.groupby("phase_seed"):
            if len(grp) >= 2:
                grp = grp.sort_values("beam_model")
                ax.plot(
                    np.log10(grp["relative_abs_bias"]),
                    -np.log10(grp["pte_global_absint"]),
                    color="#bbbbbb",
                    lw=0.8,
                    zorder=1,
                )
        for beam in ["frozen_polybeam", "full_polybeam"]:
            g = sub[sub["beam_model"] == beam]
            ax.scatter(
                np.log10(g["relative_abs_bias"]),
                -np.log10(g["pte_global_absint"]),
                s=32,
                marker=markers[beam],
                facecolor=colors[beam],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
                label=labels[beam],
                zorder=2,
            )
            cg = coh[(coh["morphology"] == morph) & (coh["beam_model"] == beam)]
            if not cg.empty:
                ax.scatter(
                    np.log10(cg["relative_abs_bias"]),
                    -np.log10(cg["PTE_global_absint_new"]),
                    s=150,
                    marker=markers[beam],
                    facecolor="none",
                    edgecolor=colors[beam],
                    linewidth=2.0,
                    zorder=3,
                )
        ax.axhline(2.0, color="#333333", lw=1.0, ls="--")
        ax.axvline(-3.0, color="#333333", lw=1.0, ls=":")
        ax.axvline(-2.0, color="#777777", lw=1.0, ls=":")
        ax.set_title(morph)
        ax.set_xlabel("log10(B_rel)")
        ax.grid(True, color="#eeeeee", linewidth=0.8)
    axes[0].set_ylabel("-log10(PTE_global,absint)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Smoke5 intrinsic-phase beam decision plane", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    savefig(fig, "F3_phase_beam_decision_plane_smoke5")


def plot_class_counts(counts: pd.DataFrame) -> None:
    counts = counts.copy()
    counts["label"] = counts["paired_case_id"].str.replace("11_10_lst016_quiet_", "", regex=False)
    counts["label"] = counts["label"].str.replace("4_196_lst053_typical_", "4_196 ", regex=False)
    pivot = counts.pivot_table(
        index=["label", "beam_model"],
        columns="final_class",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    classes = ["no strict excess", "integrated physical", "statistical QA flag", "local physical", "local+integrated physical"]
    colors = {
        "no strict excess": "#d9d9d9",
        "integrated physical": "#b4473c",
        "statistical QA flag": "#e6a23c",
        "local physical": "#2f6f9f",
        "local+integrated physical": "#4b8f5b",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))
    for cls in classes:
        vals = pivot[cls].to_numpy(dtype=float) if cls in pivot.columns else np.zeros(len(pivot))
        ax.barh(y, vals, left=left, color=colors[cls], edgecolor="white", height=0.72, label=cls)
        left += vals
    labels = [f"{idx[0]}\n{idx[1].replace('_polybeam', '')}" for idx in pivot.index]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("phase seeds")
    ax.set_xlim(0, max(5, float(left.max())) + 0.2)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.grid(True, axis="x", color="#eeeeee", linewidth=0.8)
    ax.set_title("Smoke5 class frequencies")
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    savefig(fig, "F4_phase_class_frequencies_smoke5")


def plot_beam_delta(pairs: pd.DataFrame) -> None:
    pairs = pairs.copy()
    order = sorted(pairs["paired_case_id"].unique())
    data = [pairs.loc[pairs["paired_case_id"] == key, "delta_beam_logpte"].to_numpy(dtype=float) for key in order]
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.boxplot(data, vert=True, patch_artist=True, showfliers=True)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full_like(vals, i, dtype=float), vals, s=24, color="#555555", alpha=0.75, zorder=3)
    ax.axhline(0.0, color="#333333", lw=1.0, ls="--")
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(
        [x.replace("11_10_lst016_quiet_", "").replace("4_196_lst053_typical_", "4_196 ") for x in order],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("log10(PTE_full / PTE_frozen)")
    ax.set_title("Smoke5 paired beam PTE difference")
    ax.grid(True, axis="y", color="#eeeeee", linewidth=0.8)
    fig.tight_layout()
    savefig(fig, "F3B_paired_beam_delta_smoke5")


def main() -> None:
    trials = pd.read_csv(ROOT / "intrinsic_phase_multiseed_trials_smoke5.csv")
    pairs = pd.read_csv(ROOT / "intrinsic_phase_beam_pairs_smoke5.csv")
    counts = pd.read_csv(ROOT / "intrinsic_phase_class_counts_smoke5.csv")
    coherent = pd.read_csv(ROOT / "coverage_absint_tail_refined_pte003_brel1e3.csv")
    plot_hierarchy()
    plot_decision_plane(trials, coherent)
    plot_beam_delta(pairs)
    plot_class_counts(counts)
    print(f"saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
