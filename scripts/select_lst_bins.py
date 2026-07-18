#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def robust_z(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    med = vals.median()
    mad = (vals - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else vals.std()
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (vals - med) / scale


def choose_distinct(candidates: pd.DataFrame, used: set[int]) -> pd.Series:
    for _, row in candidates.iterrows():
        bid = int(row["lst_bin_id"])
        if bid not in used:
            used.add(bid)
            return row
    row = candidates.iloc[0]
    used.add(int(row["lst_bin_id"]))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="outputs/lst_bin_metadata.csv")
    ap.add_argument("--out", default="outputs/lst_bin_selection.csv")
    args = ap.parse_args()
    meta = pd.read_csv(args.metadata)
    if "n_sat_peak_visible" not in meta and "n_sat_visible_peak_bin" in meta:
        meta["n_sat_peak_visible"] = meta["n_sat_visible_peak_bin"]
    if "pre_risk_score_bin" not in meta and "pre_risk_score" in meta:
        meta["pre_risk_score_bin"] = meta["pre_risk_score"]
    rows = []
    for baseline_id, g in meta.groupby("baseline_id"):
        g = g.copy().reset_index(drop=True)
        used: set[int] = set()
        flag_med = g["flag_fraction"].median()
        exp_col = "beam_weighted_sat_exposure_bin_mean" if "beam_weighted_sat_exposure_bin_mean" in g else "beam_weighted_sat_exposure"
        risk_col = "pre_risk_score_bin" if "pre_risk_score_bin" in g else "pre_risk_score"
        exp = g[exp_col].astype(float)
        q20 = exp.quantile(0.2)
        quiet_candidates = g[exp <= q20].copy()
        quiet_candidates["score"] = (quiet_candidates["flag_fraction"] - flag_med).abs()
        quiet = choose_distinct(quiet_candidates.sort_values("score"), used)
        rows.append({**quiet.to_dict(), "lst_stratum": "quiet", "selection_rule": "lowest 20pct exposure; flag closest to median"})

        zE = robust_z(g[exp_col]).abs()
        zF = robust_z(g["flag_fraction"]).abs()
        zM = robust_z(g["null_mad_win_proxy"]).abs()
        typical_candidates = g.copy()
        typical_candidates["score"] = zE + zF + zM
        typical = choose_distinct(typical_candidates.sort_values("score"), used)
        rows.append({**typical.to_dict(), "lst_stratum": "typical", "selection_rule": "minimum robust distance to median exposure/flag/MAD"})

        stress_candidates = g.sort_values(risk_col, ascending=False)
        stress = choose_distinct(stress_candidates, used)
        rows.append({**stress.to_dict(), "lst_stratum": "stress", "selection_rule": "maximum pre_risk_score not already selected"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sel = pd.DataFrame(rows)
    cols = [
        "baseline_id", "baseline_length_m", "baseline_class", "lst_stratum", "lst_bin_id",
        "lst_start", "lst_end", "flag_fraction", "n_sat_visible",
        "n_sat_visible_center", "n_sat_visible_any_bin", "n_sat_visible_peak_bin", "n_sat_peak_visible",
        "beam_weighted_sat_exposure_center", "beam_weighted_sat_exposure_bin_mean",
        "beam_weighted_sat_exposure", "max_sat_beam_response", "bg_window_power_proxy",
        "max_sat_beam_response_center", "max_sat_beam_response_bin",
        "null_mad_win_proxy", "pre_risk_score", "pre_risk_score_bin", "selection_rule",
        "source_uvh5", "pol", "ant1", "ant2", "t_start_index", "n_time",
        "baseline_enu_e_m", "baseline_enu_n_m", "baseline_enu_u_m",
    ]
    sel[cols].to_csv(out, index=False)
    print(f"saved {out} rows={len(sel)}")


if __name__ == "__main__":
    main()
