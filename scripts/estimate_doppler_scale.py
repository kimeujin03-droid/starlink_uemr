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
from skyfield.api import EarthSatellite, load, wgs84

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.pipeline import C_M_PER_S
from pathb.satellite import SatelliteRecord


def load_tle_records(path: Path, max_records: int | None = None) -> list[SatelliteRecord]:
    ts = load.timescale()
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    recs: list[SatelliteRecord] = []
    i = 0
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
        sat_name = name or f"SAT-{norad}-E{epoch}"
        recs.append(SatelliteRecord(EarthSatellite(line1, line2, sat_name, ts), norad, epoch, sat_name))
        if max_records is not None and len(recs) >= int(max_records):
            break
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/lst_bin_selection.csv")
    ap.add_argument("--config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--tle", default="tle/starlink_jan2026_LEO_only.tle")
    ap.add_argument("--out", default="outputs/doppler_scale_selected_cells.csv")
    ap.add_argument("--summary-out", default="outputs/doppler_scale_summary.csv")
    ap.add_argument("--max-tle-records", type=int, default=1200)
    ap.add_argument("--n-time", type=int, default=62)
    ap.add_argument("--alt-visible-deg", type=float, default=70.0)
    ap.add_argument("--max-satellites-per-cell", type=int, default=12)
    ap.add_argument("--freq-mhz", type=float, default=150.0)
    args = ap.parse_args()

    iers.conf.auto_download = False
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    site = cfg["site"]
    observer = wgs84.latlon(
        float(site["lat_deg"]),
        float(site["lon_deg"]),
        elevation_m=float(site.get("elev_m", 0.0)),
    )
    recs = load_tle_records(ROOT / args.tle, max_records=args.max_tle_records)
    selection = pd.read_csv(ROOT / args.selection)
    ts = load.timescale()

    rows = []
    for srow in selection.itertuples(index=False):
        times_jd_utc = np.linspace(float(srow.lst_start), float(srow.lst_end), args.n_time, dtype=float)
        times_tt = Time(times_jd_utc, format="jd", scale="utc").tt.jd
        t = ts.tt_jd(times_tt)
        time_sec = (times_jd_utc - times_jd_utc[0]) * 86400.0
        ranked = []
        for rec in recs:
            try:
                alt, _az, dist = (rec.sat - observer).at(t).altaz()
            except Exception:
                continue
            alt_deg = np.asarray(alt.degrees, dtype=float)
            if not np.any(np.isfinite(alt_deg)):
                continue
            peak_alt = float(np.nanmax(alt_deg))
            if peak_alt < args.alt_visible_deg:
                continue
            range_m = np.asarray(dist.km, dtype=float) * 1e3
            range_rate = np.gradient(range_m, time_sec) if len(time_sec) > 1 else np.zeros_like(range_m)
            doppler_hz = -range_rate / C_M_PER_S * (args.freq_mhz * 1e6)
            visible = np.isfinite(alt_deg) & (alt_deg >= args.alt_visible_deg)
            if not np.any(visible):
                continue
            ranked.append(
                {
                    "rec": rec,
                    "peak_alt_deg": peak_alt,
                    "mean_alt_visible_deg": float(np.nanmean(alt_deg[visible])),
                    "n_time_visible": int(np.sum(visible)),
                    "doppler_min_hz": float(np.nanmin(doppler_hz[visible])),
                    "doppler_max_hz": float(np.nanmax(doppler_hz[visible])),
                    "doppler_abs_max_hz": float(np.nanmax(np.abs(doppler_hz[visible]))),
                    "doppler_span_hz": float(np.nanmax(doppler_hz[visible]) - np.nanmin(doppler_hz[visible])),
                }
            )
        ranked = sorted(ranked, key=lambda x: x["peak_alt_deg"], reverse=True)
        for rank, item in enumerate(ranked[: args.max_satellites_per_cell], start=1):
            rec = item.pop("rec")
            rows.append(
                {
                    "baseline_id": srow.baseline_id,
                    "baseline_class": srow.baseline_class,
                    "lst_stratum": srow.lst_stratum,
                    "lst_bin_id": int(srow.lst_bin_id),
                    "sat_rank": rank,
                    "norad_id": rec.norad_id,
                    "sat_name": rec.name,
                    "epoch": rec.epoch,
                    "freq_mhz": args.freq_mhz,
                    **item,
                    "doppler_abs_max_khz": item["doppler_abs_max_hz"] / 1e3,
                    "doppler_span_khz": item["doppler_span_hz"] / 1e3,
                    "fraction_of_48p8khz_spacing": item["doppler_abs_max_hz"] / 48_800.0,
                    "fraction_of_12p2khz_linewidth": item["doppler_abs_max_hz"] / 12_200.0,
                }
            )
    df = pd.DataFrame(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = pd.DataFrame(
        [
            {
                "n_cells": int(selection.shape[0]),
                "n_satellite_cell_rows": int(df.shape[0]),
                "median_abs_doppler_khz": float(df["doppler_abs_max_khz"].median()),
                "p95_abs_doppler_khz": float(df["doppler_abs_max_khz"].quantile(0.95)),
                "max_abs_doppler_khz": float(df["doppler_abs_max_khz"].max()),
                "median_doppler_span_khz": float(df["doppler_span_khz"].median()),
                "p95_doppler_span_khz": float(df["doppler_span_khz"].quantile(0.95)),
                "max_doppler_span_khz": float(df["doppler_span_khz"].max()),
                "median_fraction_of_48p8khz_spacing": float(df["fraction_of_48p8khz_spacing"].median()),
                "p95_fraction_of_48p8khz_spacing": float(df["fraction_of_48p8khz_spacing"].quantile(0.95)),
                "median_fraction_of_12p2khz_linewidth": float(df["fraction_of_12p2khz_linewidth"].median()),
                "p95_fraction_of_12p2khz_linewidth": float(df["fraction_of_12p2khz_linewidth"].quantile(0.95)),
            }
        ]
    )
    summary.to_csv(ROOT / args.summary_out, index=False)
    meta = {
        "tle": args.tle,
        "max_tle_records": args.max_tle_records,
        "selection": args.selection,
        "n_time": args.n_time,
        "alt_visible_deg": args.alt_visible_deg,
        "max_satellites_per_cell": args.max_satellites_per_cell,
        "freq_mhz": args.freq_mhz,
        "note": "Doppler shift is estimated from topocentric range-rate over visible samples; no Doppler-shifted injection is run.",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(f"saved {ROOT / args.summary_out}")


if __name__ == "__main__":
    main()
