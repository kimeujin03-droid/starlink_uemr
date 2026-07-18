#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from skyfield.api import wgs84

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.pipeline import C_M_PER_S, delay_axis_s, make_taper, weighted_delay_transform
from pathb.satellite import SatelliteRecord

try:
    from astropy.time import Time
    from astropy.utils import iers
except ImportError as exc:
    raise ImportError("astropy is required") from exc

from skyfield.api import EarthSatellite, load


def safe_id(name: str) -> str:
    return name.replace(".", "p").replace("-", "m")


def baseline_class(length_m: float) -> str:
    if length_m < 50:
        return "short"
    if length_m < 170:
        return "mid"
    return "long"


def read_pair(path: Path) -> tuple[int, int]:
    m = re.search(r"baseline\.(\d+)_(\d+)\.sum", path.name)
    if not m:
        raise ValueError(f"Cannot parse baseline pair from {path.name}")
    return int(m.group(1)), int(m.group(2))


def load_tle_fast(path: Path, max_records: int | None = None) -> tuple[list[SatelliteRecord], dict]:
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
    return recs, {
        "tle_records_loaded_fast": len(recs),
        "tle_selection": "first_records_in_file_for_pre_injection_metadata",
    }


def polarization_index(pol_array: np.ndarray, pol: str = "ee") -> tuple[int, str]:
    # HERA east-west dipoles use pyuvdata linear-pol integers. In these files,
    # -5 maps to xx/ee and is the same product used by the existing NPZ exports.
    preferred = {"ee": -5, "xx": -5, "nn": -6, "yy": -6}
    code = preferred.get(pol, -5)
    pol_array = np.asarray(pol_array, dtype=int)
    if code in set(pol_array.tolist()):
        return int(np.where(pol_array == code)[0][0]), pol
    return 0, str(pol_array[0])


def baseline_geometry_from_native_npz(baseline_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native = ROOT / "examples" / "backgrounds_jan2026_real" / f"native_{baseline_id}.npz"
    if not native.exists():
        raise FileNotFoundError(f"Missing native geometry NPZ for {baseline_id}: {native}")
    data = np.load(native, allow_pickle=True)
    return (
        np.asarray(data["baseline_enu_m"], dtype=float),
        np.asarray(data["ant1_enu_m"], dtype=float),
        np.asarray(data["ant2_enu_m"], dtype=float),
    )


def background_power_proxy(vis, weights, freqs_hz, baseline_enu_m) -> tuple[float, float]:
    taper = make_taper(len(freqs_hz), "blackman_harris", {})
    delays = delay_axis_s(freqs_hz)
    horizon_ns = np.linalg.norm(baseline_enu_m) / C_M_PER_S * 1e9
    mask = np.abs(delays) >= (horizon_ns + 100.0) * 1e-9
    dly = weighted_delay_transform(vis, weights, taper)[:, mask]
    power = np.abs(dly) ** 2
    bg_sum = float(np.nansum(power))
    per_time = np.nansum(power, axis=1)
    mad = float(np.nanmedian(np.abs(per_time - np.nanmedian(per_time))))
    return bg_sum, mad


def satellite_exposures(
    recs,
    bin_times_jd_utc: list[np.ndarray],
    bin_centers_jd_utc: np.ndarray,
    site,
    alt_visible_deg: float,
) -> list[dict[str, float | int]]:
    ts = __import__("skyfield.api").api.load.timescale()
    # This is pre-injection stratification metadata, not a science statistic.
    # Keep the old center-time proxy for provenance, but use the full bin
    # time samples for the stratification columns consumed downstream.
    bin_centers_jd_utc = np.asarray(bin_centers_jd_utc, dtype=float)
    n_bins = len(bin_times_jd_utc)
    flat_times = np.concatenate([np.asarray(x, dtype=float) for x in bin_times_jd_utc])
    bin_slices = []
    start = 0
    for times in bin_times_jd_utc:
        stop = start + len(times)
        bin_slices.append(slice(start, stop))
        start = stop
    iers.conf.auto_download = False
    t_center = ts.tt_jd(Time(bin_centers_jd_utc, format="jd", scale="utc").tt.jd)
    t_full = ts.tt_jd(Time(flat_times, format="jd", scale="utc").tt.jd)
    observer = wgs84.latlon(
        float(site["lat_deg"]),
        float(site["lon_deg"]),
        elevation_m=float(site.get("elev_m", 0.0)),
    )
    exposure_center = np.zeros(n_bins, dtype=float)
    max_b_center = np.zeros(n_bins, dtype=float)
    n_visible_center = np.zeros(n_bins, dtype=int)
    exposure_bin_mean = np.zeros(n_bins, dtype=float)
    max_b_bin = np.zeros(n_bins, dtype=float)
    n_visible_any_bin = np.zeros(n_bins, dtype=int)
    n_visible_peak_bin = np.zeros(n_bins, dtype=int)
    fwhm = 10.0
    sigma = fwhm / np.sqrt(8.0 * np.log(2.0))
    for rec in recs:
        try:
            alt_center = np.asarray((rec.sat - observer).at(t_center).altaz()[0].degrees, dtype=float)
            alt_full = np.asarray((rec.sat - observer).at(t_full).altaz()[0].degrees, dtype=float)
        except Exception:
            continue
        valid_center = np.isfinite(alt_center) & (alt_center >= alt_visible_deg)
        if np.any(valid_center):
            za_center = np.maximum(0.0, 90.0 - alt_center[valid_center])
            b_center = np.exp(-0.5 * (za_center / sigma) ** 2)
            exposure_center[valid_center] += b_center
            max_b_center[valid_center] = np.maximum(max_b_center[valid_center], b_center)
            n_visible_center[valid_center] += 1
        for ibin, sl in enumerate(bin_slices):
            alt = alt_full[sl]
            finite = np.isfinite(alt)
            if not np.any(finite):
                continue
            visible = finite & (alt >= alt_visible_deg)
            if not np.any(visible):
                continue
            za = np.maximum(0.0, 90.0 - alt[visible])
            beam = np.exp(-0.5 * (za / sigma) ** 2)
            n_visible_any_bin[ibin] += 1
            if float(np.nanmax(alt)) >= alt_visible_deg:
                n_visible_peak_bin[ibin] += 1
            exposure_bin_mean[ibin] += float(np.nanmean(beam))
            max_b_bin[ibin] = max(max_b_bin[ibin], float(np.nanmax(beam)))
    return [
        {
            "n_sat_visible_center": int(n_visible_center[i]),
            "n_sat_visible_any_bin": int(n_visible_any_bin[i]),
            "n_sat_visible_peak_bin": int(n_visible_peak_bin[i]),
            "n_sat_peak_visible": int(n_visible_peak_bin[i]),
            "beam_weighted_sat_exposure_center": float(exposure_center[i]),
            "beam_weighted_sat_exposure_bin_mean": float(exposure_bin_mean[i]),
            "max_sat_beam_response_center": float(max_b_center[i]),
            "max_sat_beam_response_bin": float(max_b_bin[i]),
        }
        for i in range(n_bins)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-data", default=".")
    ap.add_argument("--baselines", nargs="*", default=None)
    ap.add_argument("--bin-minutes", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/lst_bin_metadata.csv")
    ap.add_argument("--config", default="configs/pathB_jan2026_main.yaml")
    ap.add_argument("--tle", default="tle/starlink_jan2026_LEO_only.tle")
    ap.add_argument("--target-utc", default="2026-01-01T00:00:00")
    ap.add_argument("--alt-visible-deg", type=float, default=70.0)
    ap.add_argument("--max-satellites", type=int, default=1200)
    args = ap.parse_args()

    import yaml

    print("[start] build_lst_metadata", flush=True)
    iers.conf.auto_download = False
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    data_dir = Path(args.input_data)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("zen.LST.baseline.*.sum.uvh5"))
    if args.baselines:
        keep = set(args.baselines)
        files = [p for p in files if f"{read_pair(p)[0]}_{read_pair(p)[1]}" in keep]
    if not files:
        raise FileNotFoundError(f"No baseline UVH5 files found in {data_dir}")

    print("[time] resolving target UTC", flush=True)
    target_jd_utc0 = float(Time(args.target_utc, format="isot", scale="utc").utc.jd)
    print("[tle] loading fast subset", flush=True)
    recs, tle_meta = load_tle_fast(
        ROOT / args.tle,
        max_records=args.max_satellites,
    )
    print(f"[tle] loaded {len(recs)} records", flush=True)
    rows = []
    for path in files:
        ant1, ant2 = read_pair(path)
        baseline_id = f"{ant1}_{ant2}"
        print(f"[read] {path.name}", flush=True)
        with h5py.File(path, "r") as h5:
            pol_idx, pol = polarization_index(h5["Header/polarization_array"][()], "ee")
            data = np.asarray(h5["Data/visdata"][:, :, pol_idx])
            flags = np.asarray(h5["Data/flags"][:, :, pol_idx])
            nsamples = np.asarray(h5["Data/nsamples"][:, :, pol_idx])
            freqs_hz = np.asarray(h5["Header/freq_array"][()], dtype=float).reshape(-1)
            times_orig = np.asarray(h5["Header/time_array"][()], dtype=float).reshape(-1)
        baseline_enu_m, _ant1_enu, _ant2_enu = baseline_geometry_from_native_npz(baseline_id)
        bl_len = float(np.linalg.norm(baseline_enu_m))
        weights = (~flags).astype(float) * np.clip(nsamples.astype(float), 0.0, None)
        if np.nanmax(weights) > 0:
            weights = weights / np.nanmax(weights)
        dt_sec = float(np.nanmedian(np.diff(times_orig)) * 86400.0)
        n_per_bin = max(1, int(round(args.bin_minutes * 60.0 / dt_sec)))
        n_bins = len(times_orig) // n_per_bin
        bin_centers = []
        bin_times = []
        for ibin in range(n_bins):
            sl = slice(ibin * n_per_bin, (ibin + 1) * n_per_bin)
            jd_start = target_jd_utc0 + (ibin * args.bin_minutes) / 1440.0
            times_new = jd_start + (times_orig[sl] - times_orig[sl][0])
            bin_times.append(times_new)
            bin_centers.append(float(np.nanmedian(times_new)))
        sat_meta = satellite_exposures(recs, bin_times, np.asarray(bin_centers), cfg["site"], args.alt_visible_deg)
        for ibin in range(n_bins):
            sl = slice(ibin * n_per_bin, (ibin + 1) * n_per_bin)
            # Remap each contiguous LST bin to Jan 1 UTC while preserving within-bin cadence.
            times_new = bin_times[ibin]
            flag_frac = float(np.mean(weights[sl] <= 0.0))
            bg_power, mad_proxy = background_power_proxy(data[sl], weights[sl], freqs_hz, baseline_enu_m)
            sat = sat_meta[ibin]
            nvis = int(sat["n_sat_visible_peak_bin"])
            exposure = float(sat["beam_weighted_sat_exposure_bin_mean"])
            max_b = float(sat["max_sat_beam_response_bin"])
            pre_score = exposure / (mad_proxy + 1e-12)
            rows.append(
                {
                    "baseline_id": baseline_id,
                    "ant1": ant1,
                    "ant2": ant2,
                    "source_uvh5": str(path),
                    "pol": pol,
                    "baseline_length_m": bl_len,
                    "baseline_class": baseline_class(bl_len),
                    "baseline_enu_e_m": float(baseline_enu_m[0]),
                    "baseline_enu_n_m": float(baseline_enu_m[1]),
                    "baseline_enu_u_m": float(baseline_enu_m[2]),
                    "lst_bin_id": int(ibin),
                    "t_start_index": int(ibin * n_per_bin),
                    "n_time": int(n_per_bin),
                    "lst_start": float(times_new[0]),
                    "lst_end": float(times_new[-1]),
                    "flag_fraction": flag_frac,
                    **sat,
                    # Legacy downstream columns are now full-bin aliases rather
                    # than center-time proxies.
                    "n_sat_visible": int(nvis),
                    "beam_weighted_sat_exposure": exposure,
                    "max_sat_beam_response": max_b,
                    "pre_risk_score_bin": pre_score,
                    "bg_window_power_proxy": bg_power,
                    "null_mad_win_proxy": mad_proxy,
                    "pre_risk_score": pre_score,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = {"tle_meta": tle_meta, "bin_minutes": args.bin_minutes, "n_rows": len(df)}
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out} rows={len(df)}")


if __name__ == "__main__":
    main()
