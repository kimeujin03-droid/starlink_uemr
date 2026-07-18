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
from pathb.satellite import SatelliteRecord, altaz_to_enu_m


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
    ap.add_argument("--out", default="outputs/nearfield_delay_validation.csv")
    ap.add_argument("--summary-out", default="outputs/nearfield_delay_validation_summary.csv")
    ap.add_argument("--max-tle-records", type=int, default=1200)
    ap.add_argument("--n-time", type=int, default=62)
    ap.add_argument("--alt-visible-deg", type=float, default=70.0)
    ap.add_argument("--max-satellites-per-cell", type=int, default=12)
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
        baseline = np.array(
            [float(srow.baseline_enu_e_m), float(srow.baseline_enu_n_m), float(srow.baseline_enu_u_m)],
            dtype=float,
        )
        bl_len = float(np.linalg.norm(baseline))
        times_jd_utc = np.linspace(float(srow.lst_start), float(srow.lst_end), args.n_time, dtype=float)
        times_tt = Time(times_jd_utc, format="jd", scale="utc").tt.jd
        t = ts.tt_jd(times_tt)
        ranked = []
        for rec in recs:
            try:
                alt, az, dist = (rec.sat - observer).at(t).altaz()
            except Exception:
                continue
            alt_deg = np.asarray(alt.degrees, dtype=float)
            visible = np.isfinite(alt_deg) & (alt_deg >= args.alt_visible_deg)
            if not np.any(visible):
                continue
            az_deg = np.asarray(az.degrees, dtype=float)
            range_km = np.asarray(dist.km, dtype=float)
            sat_enu = altaz_to_enu_m(alt_deg, az_deg, range_km)
            unit = sat_enu / np.linalg.norm(sat_enu, axis=1)[:, None]
            r1 = np.linalg.norm(sat_enu, axis=1)
            r2 = np.linalg.norm(sat_enu - baseline[None, :], axis=1)
            tau_near = (r2 - r1) / C_M_PER_S
            tau_plane = -np.sum(baseline[None, :] * unit, axis=1) / C_M_PER_S
            err_ns = (tau_near - tau_plane) * 1e9
            tau_near_ns = tau_near * 1e9
            tau_plane_ns = tau_plane * 1e9
            peak_alt = float(np.nanmax(alt_deg[visible]))
            ranked.append(
                {
                    "rec": rec,
                    "peak_alt_deg": peak_alt,
                    "n_time_visible": int(np.sum(visible)),
                    "baseline_length_m": bl_len,
                    "max_abs_tau_near_ns": float(np.nanmax(np.abs(tau_near_ns[visible]))),
                    "max_abs_tau_plane_ns": float(np.nanmax(np.abs(tau_plane_ns[visible]))),
                    "max_abs_near_minus_plane_ns": float(np.nanmax(np.abs(err_ns[visible]))),
                    "median_abs_near_minus_plane_ns": float(np.nanmedian(np.abs(err_ns[visible]))),
                    "max_fraction_of_horizon": float(
                        np.nanmax(np.abs(err_ns[visible])) / max((bl_len / C_M_PER_S) * 1e9, 1e-30)
                    ),
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
                    **item,
                }
            )

    df = pd.DataFrame(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = (
        df.groupby("baseline_class")
        .agg(
            n_satellite_cell_rows=("norad_id", "size"),
            median_error_ns=("max_abs_near_minus_plane_ns", "median"),
            p95_error_ns=("max_abs_near_minus_plane_ns", lambda x: float(np.quantile(x, 0.95))),
            max_error_ns=("max_abs_near_minus_plane_ns", "max"),
            median_fraction_of_horizon=("max_fraction_of_horizon", "median"),
            max_fraction_of_horizon=("max_fraction_of_horizon", "max"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "baseline_class": "all",
                "n_satellite_cell_rows": int(df.shape[0]),
                "median_error_ns": float(df["max_abs_near_minus_plane_ns"].median()),
                "p95_error_ns": float(df["max_abs_near_minus_plane_ns"].quantile(0.95)),
                "max_error_ns": float(df["max_abs_near_minus_plane_ns"].max()),
                "median_fraction_of_horizon": float(df["max_fraction_of_horizon"].median()),
                "max_fraction_of_horizon": float(df["max_fraction_of_horizon"].max()),
            }
        ]
    )
    pd.concat([summary, overall], ignore_index=True).to_csv(ROOT / args.summary_out, index=False)
    meta = {
        "tle": args.tle,
        "max_tle_records": args.max_tle_records,
        "selection": args.selection,
        "n_time": args.n_time,
        "alt_visible_deg": args.alt_visible_deg,
        "max_satellites_per_cell": args.max_satellites_per_cell,
        "plane_wave_formula": "tau_plane = -dot(baseline_enu, unit_sat_enu) / c",
        "near_field_formula": "tau_near = (|sat_enu - baseline_enu| - |sat_enu|) / c",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(f"saved {ROOT / args.summary_out}")


if __name__ == "__main__":
    main()
