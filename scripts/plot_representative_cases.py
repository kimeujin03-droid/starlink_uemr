#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="outputs/coverage_robustness_trials.csv")
    ap.add_argument("--buffer", default="outputs/delay_buffer_sensitivity.csv")
    ap.add_argument("--fig-dir", default="figures/coverage_robustness/representative_cases")
    args = ap.parse_args()

    trials = pd.read_csv(ROOT / args.trials)
    for col in ["Z_PS_max", "PTE_global_max", "PTE_global_absint", "relative_abs_bias"]:
        trials[col] = pd.to_numeric(trials[col], errors="coerce")
    trials["candidate_statistical_bool"] = trials["candidate_statistical"].astype(str).eq("True")
    top = trials.sort_values("Z_PS_max", ascending=False).head(12).copy()
    top["case_label"] = (
        top["baseline_id"].astype(str)
        + " "
        + top["lst_stratum"].astype(str)
        + "/"
        + top["lst_bin_id"].astype(str)
        + "\n"
        + top["beam_model"].astype(str)
        + " "
        + top["morphology"].astype(str)
        + " S"
        + top["flux_jy"].astype(str)
        + " "
        + top["multiplicity"].astype(str)
    )

    fig_dir = ROOT / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = top["candidate_statistical_bool"].map({True: "#b22222", False: "#4c78a8"})
    ax.barh(range(len(top)), top["Z_PS_max"], color=colors)
    ax.axvline(3.0, color="black", linestyle="--", linewidth=1.0, label="Z=3")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["case_label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Z_PS,max")
    ax.set_title("Representative high-local-excursion coverage rows")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "representative_top_zps_cases.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    sc = ax.scatter(
        trials["PTE_global_max"],
        trials["relative_abs_bias"],
        c=trials["Z_PS_max"],
        cmap="viridis",
        s=28,
        alpha=0.75,
        edgecolors="none",
    )
    ax.axvline(0.01, color="black", linestyle="--", linewidth=1.0, label="PTE=0.01")
    ax.axhline(1e-2, color="#b22222", linestyle="--", linewidth=1.0, label="Brel=1e-2")
    ax.set_yscale("log")
    ax.set_xlabel("PTE_global,max")
    ax.set_ylabel("relative_abs_bias")
    ax.set_title("Representative QA gate plane")
    ax.legend(frameon=False)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Z_PS,max")
    fig.tight_layout()
    fig.savefig(fig_dir / "representative_gate_plane.png", dpi=180)
    plt.close(fig)

    buffer_path = ROOT / args.buffer
    if buffer_path.exists():
        buf = pd.read_csv(buffer_path)
        fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
        ax1.plot(buf["buffer_ns"], buf["Z_PS_max"], marker="o", color="#4c78a8", label="Z_PS,max")
        ax1.axhline(3.0, color="#4c78a8", linestyle=":", linewidth=1.0)
        ax1.set_xlabel("delay-window buffer (ns)")
        ax1.set_ylabel("Z_PS,max", color="#4c78a8")
        ax1.tick_params(axis="y", labelcolor="#4c78a8")
        ax2 = ax1.twinx()
        ax2.plot(buf["buffer_ns"], buf["PTE_global_max"], marker="s", color="#b22222", label="PTE_global,max")
        ax2.axhline(0.01, color="#b22222", linestyle=":", linewidth=1.0)
        ax2.set_ylabel("PTE_global,max", color="#b22222")
        ax2.tick_params(axis="y", labelcolor="#b22222")
        ax1.set_title("Representative delay-buffer sensitivity case")
        fig.tight_layout()
        fig.savefig(fig_dir / "representative_delay_buffer_sensitivity.png", dpi=180)
        plt.close(fig)

    manifest = pd.DataFrame(
        [
            {"figure": "representative_top_zps_cases.png", "source": args.trials},
            {"figure": "representative_gate_plane.png", "source": args.trials},
            {"figure": "representative_delay_buffer_sensitivity.png", "source": args.buffer if buffer_path.exists() else ""},
        ]
    )
    manifest.to_csv(fig_dir / "manifest.csv", index=False)
    print(f"saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
