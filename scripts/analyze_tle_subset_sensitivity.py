#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time
from astropy.utils import iers
from skyfield.api import EarthSatellite, load

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.satellite import SatelliteRecord
from scripts.build_lst_metadata import satellite_exposures


def load_tle_records(path: Path) -> list[SatelliteRecord]:
    ts = load.timescale()
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    recs: list[SatelliteRecord] = []
    i = 0
    seen: set[tuple[str, str]] = set()
    while i < len(lines):
        name = ""
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            line1, line2 = lines[i], lines[i + 1]
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue
        norad = line1[2:7].strip()
        epoch = line1[18:32].strip()
        key = (norad, epoch)
        if key in seen:
            continue
        seen.add(key)
        sat_name = name or f"SAT-{norad}-E{epoch}"
        recs.append(SatelliteRecord(EarthSatellite(line1, line2, sat_name, ts), norad, epoch, sat_name))
    return recs


def make_bin_times(selection: pd.DataFrame, n_time: int) -> tuple[list[np.ndarray], np.ndarray]:
    bin_times: list[np.ndarray] = []
    centers = []
    for row in selection.itertuples(index=False):
        start = float(row.lst_start)
        end = float(row.lst_end)
        times = np.linspace(start, end, int(n_time), dtype=float)
        bin_times.append(times)
        centers.append(float(np.median(times)))
    return bin_times, np.asarray(centers, dtype=float)


def summarize_subset(
    label: str,
    recs: list[SatelliteRecord],
    selection: pd.DataFrame,
    bin_times: list[np.ndarray],
    centers: np.ndarray,
    site: dict,
    alt_visible_deg: float,
) -> pd.DataFrame:
    meta = satellite_exposures(recs, bin_times, centers, site, alt_visible_deg)
    out = selection[
        ["baseline_id", "baseline_class", "lst_stratum", "lst_bin_id", "lst_start", "lst_end"]
    ].copy()
    out.insert(0, "subset_label", label)
    out["n_tle_records"] = len(recs)
    for key in [
        "n_sat_visible_center",
        "n_sat_visible_any_bin",
        "n_sat_visible_peak_bin",
        "n_sat_peak_visible",
        "beam_weighted_sat_exposure_center",
        "beam_weighted_sat_exposure_bin_mean",
        "max_sat_beam_response_center",
        "max_sat_beam_response_bin",
    ]:
        out[key] = [m[key] for m in meta]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tle", default="tle/starlink_jan2026_LEO_only.tle")
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--config", default="configs/coverage_robustness.yaml")
    ap.add_argument("--out", default="outputs/tle_subset_sensitivity.csv")
    ap.add_argument("--summary-out", default="outputs/tle_subset_sensitivity_summary.csv")
    ap.add_argument("--subset-size", type=int, default=1200)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--n-time", type=int, default=62)
    ap.add_argument("--alt-visible-deg", type=float, default=70.0)
    args = ap.parse_args()

    iers.conf.auto_download = False
    tle_path = ROOT / args.tle
    selection = pd.read_csv(ROOT / args.selection)
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if "site" not in cfg:
        main_cfg = yaml.safe_load((ROOT / "configs" / "pathB_jan2026_main.yaml").read_text(encoding="utf-8"))
        cfg["site"] = main_cfg["site"]
    recs_all = load_tle_records(tle_path)
    if len(recs_all) < args.subset_size:
        raise ValueError(f"Only {len(recs_all)} TLE records available; requested {args.subset_size}")

    bin_times, centers_utc = make_bin_times(selection, args.n_time)

    rng = np.random.default_rng(args.seed)
    blocks = [
        summarize_subset(
            "first1200",
            recs_all[: args.subset_size],
            selection,
            bin_times,
            centers_utc,
            cfg["site"],
            args.alt_visible_deg,
        ),
        summarize_subset(
            "all_available",
            recs_all,
            selection,
            bin_times,
            centers_utc,
            cfg["site"],
            args.alt_visible_deg,
        ),
    ]
    all_idx = np.arange(len(recs_all))
    for i in range(args.n_random):
        idx = np.sort(rng.choice(all_idx, size=args.subset_size, replace=False))
        blocks.append(
            summarize_subset(
                f"random1200_seed{args.seed + i}",
                [recs_all[j] for j in idx],
                selection,
                bin_times,
                centers_utc,
                cfg["site"],
                args.alt_visible_deg,
            )
        )

    detail = pd.concat(blocks, ignore_index=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out, index=False)

    summary = (
        detail.groupby("subset_label")
        .agg(
            n_tle_records=("n_tle_records", "first"),
            n_cells=("lst_bin_id", "size"),
            mean_visible_any=("n_sat_visible_any_bin", "mean"),
            max_visible_any=("n_sat_visible_any_bin", "max"),
            mean_visible_peak=("n_sat_visible_peak_bin", "mean"),
            max_visible_peak=("n_sat_visible_peak_bin", "max"),
            mean_exposure=("beam_weighted_sat_exposure_bin_mean", "mean"),
            p95_exposure=("beam_weighted_sat_exposure_bin_mean", lambda x: float(np.quantile(x, 0.95))),
            max_exposure=("beam_weighted_sat_exposure_bin_mean", "max"),
            mean_max_beam=("max_sat_beam_response_bin", "mean"),
            max_max_beam=("max_sat_beam_response_bin", "max"),
        )
        .reset_index()
    )
    first = summary.loc[summary["subset_label"] == "first1200"].iloc[0]
    for col in ["mean_visible_any", "mean_exposure", "p95_exposure", "max_exposure"]:
        summary[f"delta_vs_first1200_{col}"] = summary[col].astype(float) - float(first[col])
    summary.to_csv(ROOT / args.summary_out, index=False)

    meta = {
        "tle": args.tle,
        "selection": args.selection,
        "config": args.config,
        "n_tle_records_all": len(recs_all),
        "subset_size": args.subset_size,
        "n_random": args.n_random,
        "seed": args.seed,
        "n_time_per_bin": args.n_time,
        "alt_visible_deg": args.alt_visible_deg,
        "time_note": "Per-bin times are reconstructed from lst_start/lst_end in outputs/lst_bin_selection.csv.",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(f"saved {ROOT / args.summary_out}")


if __name__ == "__main__":
    main()
