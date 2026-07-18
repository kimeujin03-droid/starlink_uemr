from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from skyfield.api import EarthSatellite, load, wgs84

from .io_background import BackgroundContext

try:
    from scipy.special import j1 as _bessel_j1
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

C_M_PER_S = 299_792_458.0
_SPECTRAL_TEMPLATE_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, Dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Time-scale helper (1순위)
# ---------------------------------------------------------------------------

def skyfield_time_from_ctx(ctx: "BackgroundContext"):
    """Convert ctx.times_jd to a Skyfield Time object, respecting ctx.time_scale.

    ctx.time_scale == "utc" (default): converts UTC JD → TT JD via astropy.
    ctx.time_scale == "tt"           : passes JD directly as TT (no conversion).

    LEO satellite positions shift ~7 km per minute; a 1-minute UTC/TT confusion
    (~64 s in 2026) would misplace a Starlink by ~8 km, invalidating near-field
    delay calculations. This helper makes the conversion explicit and auditable.
    """
    ts = load.timescale()
    scale = str(getattr(ctx, "time_scale", (ctx.metadata or {}).get("time_scale", "utc"))).lower()
    if scale == "tt":
        return ts.tt_jd(ctx.times_jd)
    if scale == "utc":
        try:
            from astropy.time import Time as _ATime
            from astropy.utils import iers as _iers
        except ImportError as exc:
            raise ImportError(
                "times_jd is marked as UTC JD; astropy is required for UTC→TT conversion. "
                "Install with: pip install astropy"
            ) from exc
        _iers.conf.auto_download = False
        return ts.tt_jd(_ATime(ctx.times_jd, format="jd", scale="utc").tt.jd)
    raise ValueError(f"Unsupported time_scale: {scale!r}. Use 'utc' or 'tt'.")


@dataclass
class SatelliteRecord:
    sat: EarthSatellite
    norad_id: str
    epoch: str
    name: str


def _tle_field(line: str, start: int, stop: int) -> str:
    return line[start:stop].strip() if len(line) >= start else ""


def _norad(line1: str) -> str:
    return _tle_field(line1, 2, 7)


def _epoch(line1: str) -> str:
    return _tle_field(line1, 18, 32)


def load_tle_one_epoch_per_norad(path: str | Path, target_jd: Optional[float] = None, max_scan: Optional[int] = None) -> Tuple[List[SatelliteRecord], Dict[str, Any]]:
    """Load TLE records and keep one epoch per physical NORAD ID.

    This is treated as input provenance control, not as a scientific result.
    """
    ts = load.timescale()
    lines = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    recs: List[Dict[str, Any]] = []
    total = 0
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
        total += 1
        norad = _norad(line1)
        epoch = _epoch(line1)
        sat_name = name or f"SAT-{norad}-E{epoch}"
        sat = EarthSatellite(line1, line2, sat_name, ts)
        fallback_identity = f"NO_NORAD_{epoch}_{sat_name}_{line1[:68].strip()}_{line2[:68].strip()}"
        recs.append({"sat": sat, "norad_id": norad, "epoch": epoch, "name": sat_name, "epoch_jd": float(sat.epoch.tt), "fallback_identity": fallback_identity})
    grouped: Dict[str, Dict[str, Any]] = {}
    for rec in recs:
        key = rec["norad_id"] if rec["norad_id"] else rec["fallback_identity"]
        old = grouped.get(key)
        if old is None:
            grouped[key] = rec
        else:
            if target_jd is None:
                take = rec["epoch_jd"] > old["epoch_jd"]
            else:
                take = abs(rec["epoch_jd"] - target_jd) < abs(old["epoch_jd"] - target_jd)
            if take:
                grouped[key] = rec
    out = [SatelliteRecord(r["sat"], r["norad_id"], r["epoch"], r["name"]) for r in grouped.values()]
    out = sorted(out, key=lambda r: (r.norad_id, r.epoch))
    if max_scan:
        out = out[:int(max_scan)]
    meta = {"tle_records_total": total, "tle_unique_norad_or_fallback_identity": len(grouped), "tle_history_records_collapsed": max(len(recs) - len(grouped), 0), "tle_selection": "nearest_epoch_to_window" if target_jd is not None else "latest_epoch_per_norad"}
    return out, meta


def select_satellite(tle_path: str | Path, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[SatelliteRecord, Dict[str, Any]]:
    site = cfg["site"]
    scfg = cfg.get("starlink", {})
    target_jd = float(np.nanmedian(ctx.times_jd))
    sats, meta = load_tle_one_epoch_per_norad(tle_path, target_jd=target_jd, max_scan=scfg.get("max_scan_satellites", 2000))
    if not sats:
        raise ValueError(f"No TLE records in {tle_path}")
    name = scfg.get("satellite_name")
    if name:
        for rec in sats:
            if name.upper() in rec.name.upper() or name == rec.norad_id:
                return rec, {**meta, "selection": "requested_name_or_norad", "satellite_name": rec.name, "norad_id": rec.norad_id}
        raise ValueError(f"Requested satellite not found: {name}")

    t = skyfield_time_from_ctx(ctx)
    observer = wgs84.latlon(float(site["lat_deg"]), float(site["lon_deg"]), elevation_m=float(site.get("elev_m", 0.0)))
    amin = float(scfg.get("peak_alt_min_deg", 25.0))
    amax = float(scfg.get("peak_alt_max_deg", 85.0))
    rows = []
    for idx, rec in enumerate(sats):
        try:
            alt = (rec.sat - observer).at(t).altaz()[0].degrees
            peak = float(np.nanmax(alt))
            imax = int(np.nanargmax(alt))
            if amin <= peak <= amax:
                rows.append((abs(peak - 0.5 * (amin + amax)), idx, rec, peak, float(ctx.times_jd[imax])))
        except Exception:
            continue
    if not rows:
        rows = []
        for idx, rec in enumerate(sats):
            try:
                alt = (rec.sat - observer).at(t).altaz()[0].degrees
                rows.append((-float(np.nanmax(alt)), idx, rec, float(np.nanmax(alt)), float(ctx.times_jd[int(np.nanargmax(alt))])))
            except Exception:
                continue
        if not rows:
            raise ValueError("No usable satellite track in TLE set.")
    rows.sort(key=lambda x: x[0])
    _, idx, rec, peak, peak_jd = rows[0]
    return rec, {**meta, "selection": "peak_altitude_scan", "satellite_name": rec.name, "norad_id": rec.norad_id, "peak_alt_deg": peak, "peak_jd": peak_jd}


def altaz_to_enu_m(alt_deg: np.ndarray, az_deg: np.ndarray, distance_km: np.ndarray) -> np.ndarray:
    alt = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)
    r_m = distance_km * 1e3
    return np.column_stack([r_m * np.cos(alt) * np.sin(az), r_m * np.cos(alt) * np.cos(az), r_m * np.sin(alt)])


def compute_nearfield_track(sat: EarthSatellite, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    site = cfg["site"]
    t = skyfield_time_from_ctx(ctx)
    observer = wgs84.latlon(float(site["lat_deg"]), float(site["lon_deg"]), elevation_m=float(site.get("elev_m", 0.0)))
    app = (sat - observer).at(t)
    alt, az, dist = app.altaz()
    alt_deg = np.asarray(alt.degrees, dtype=float)
    az_deg = np.asarray(az.degrees, dtype=float)
    range_km = np.asarray(dist.km, dtype=float)
    sat_enu = altaz_to_enu_m(alt_deg, az_deg, range_km)
    r1 = np.linalg.norm(sat_enu - ctx.ant1_enu_m[None, :], axis=1)
    r2 = np.linalg.norm(sat_enu - ctx.ant2_enu_m[None, :], axis=1)
    tau_s = (r2 - r1) / C_M_PER_S
    time_sec = (ctx.times_jd - ctx.times_jd[0]) * 86400.0
    tau_dot = np.gradient(tau_s, time_sec) if len(time_sec) > 1 else np.zeros_like(tau_s)
    range_rate_m_s = np.gradient(range_km * 1e3, time_sec) if len(time_sec) > 1 else np.zeros_like(range_km)
    dnu = float(np.median(np.diff(ctx.freqs_hz))) if len(ctx.freqs_hz) > 1 else 1.0
    dt = float(np.median(np.diff(ctx.times_jd)) * 86400.0) if len(ctx.times_jd) > 1 else float(cfg.get("time_frequency", {}).get("dt_sec", 10.0))
    fringe_rate_hz = tau_dot[:, None] * ctx.freqs_hz[None, :]
    sinc_time = np.sinc(fringe_rate_hz * dt)
    sinc_freq = np.sinc(tau_s[:, None] * dnu)
    attenuation = np.abs(sinc_time * sinc_freq)
    track = pd.DataFrame({"jd": ctx.times_jd, "time_sec": time_sec, "alt_deg": alt_deg, "az_deg": az_deg, "range_km": range_km, "range_rate_m_s": range_rate_m_s, "tau_s": tau_s, "tau_dot_s_per_s": tau_dot})
    arrays = {"tau_s": tau_s, "fringe_rate_hz": fringe_rate_hz, "attenuation_tf": attenuation, "sat_enu_m": sat_enu}
    return track, arrays


def apply_beam_sensitivity(power: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply optional beam floor and sidelobe boost for sensitivity analysis.

    Config keys (all under beam:):
      power_floor_db   : float or null — minimum beam power in dB (e.g. -60)
      sidelobe_boost_db: float (default 0) — multiply low-power regions by this boost
      sidelobe_threshold: float (default 0.01) — power < threshold gets boosted
      clip_to_unit     : bool (default true) — clip output to [0, 1]

    Returns (power_out, meta_dict). Use meta to populate report for paper defence.
    """
    bcfg = cfg.get("beam", {})
    out = np.array(power, dtype=float)
    meta: Dict[str, Any] = {}

    floor_db = bcfg.get("power_floor_db", None)
    if floor_db is not None:
        floor_lin = 10.0 ** (float(floor_db) / 10.0)
        out = np.maximum(out, floor_lin)
        meta["power_floor_db"] = float(floor_db)
        meta["power_floor_linear"] = float(floor_lin)

    boost_db = float(bcfg.get("sidelobe_boost_db", 0.0))
    if boost_db != 0.0:
        boost_lin = 10.0 ** (boost_db / 10.0)
        threshold = float(bcfg.get("sidelobe_threshold", 1e-2))
        mask = out < threshold
        out = out.copy()
        out[mask] *= boost_lin
        meta["sidelobe_boost_db"] = boost_db
        meta["sidelobe_boost_fraction"] = float(np.mean(mask))

    clip = bool(bcfg.get("clip_to_unit", True))
    if clip:
        clipped = float(np.mean(out > 1.0))
        out = np.clip(out, 0.0, 1.0)
        meta["clip_to_unit"] = True
        meta["clipped_fraction_after_sensitivity"] = clipped
    else:
        out = np.clip(out, 0.0, None)
        meta["clip_to_unit"] = False

    return out, meta


def gaussian_beam(track: pd.DataFrame, freqs_hz: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Gaussian main-lobe beam (no sidelobes). Dispatches by mode."""
    bcfg = cfg.get("beam", {})
    mode = str(bcfg.get("mode", "gaussian")).lower()
    if mode == "none":
        return np.ones((len(track), len(freqs_hz))), {"mode": "none"}
    if mode == "airy":
        return airy_beam(track, freqs_hz, cfg)
    if mode == "hera_poly":
        return hera_polybeam(track, freqs_hz, cfg)
    fwhm_deg_ref = float(bcfg.get("fwhm_deg_ref", 10.0))
    freq_ref_hz = float(bcfg.get("freq_ref_hz", 150e6))
    za_deg = 90.0 - track["alt_deg"].to_numpy()[:, None]
    fwhm = fwhm_deg_ref * (freq_ref_hz / freqs_hz[None, :])
    sigma = fwhm / np.sqrt(8.0 * np.log(2.0))
    power = np.exp(-0.5 * (za_deg / sigma) ** 2)
    power_out, sens_meta = apply_beam_sensitivity(power, cfg)
    meta = {"mode": "gaussian_power", "fwhm_deg_ref": fwhm_deg_ref, "freq_ref_hz": freq_ref_hz}
    meta.update(sens_meta)
    return power_out, meta


def airy_beam(track: pd.DataFrame, freqs_hz: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Airy disk power beam for a circular aperture of diameter D.

    B(θ, ν) = [2 J₁(u) / u]²,  u = π D sin(θ) / λ(ν)

    This is the physically correct far-field diffraction pattern for a uniformly
    illuminated circular aperture (HERA dish, D=14 m). Unlike the Gaussian
    approximation, it includes the full sidelobe structure:
      - First null at θ ≈ 1.22 λ/D  (~10° at 150 MHz for D=14 m)
      - First sidelobe peak at ≈ −17.6 dB
      - Higher sidelobes decay as ~u^{-3/2} but remain non-negligible

    The sidelobes allow off-zenith sources to contribute to the visibility even
    at large za, contrasting with the Gaussian model's exponential suppression.
    """
    if not _SCIPY_OK:
        raise ImportError("scipy is required for airy_beam: pip install scipy")
    bcfg = cfg.get("beam", {})
    dish_diameter_m = float(bcfg.get("dish_diameter_m", 14.0))
    za_deg = 90.0 - track["alt_deg"].to_numpy()[:, None]   # (T, 1)
    za_rad = np.deg2rad(za_deg)
    lambdas = C_M_PER_S / freqs_hz[None, :]                # (1, F)
    u = np.pi * dish_diameter_m * np.sin(za_rad) / lambdas # (T, F)
    with np.errstate(invalid="ignore", divide="ignore"):
        jinc = np.where(np.abs(u) < 1e-10, 1.0, 2.0 * _bessel_j1(u) / u)
    power = jinc ** 2
    power_out, sens_meta = apply_beam_sensitivity(power, cfg)
    meta = {
        "mode": "airy_disk_power",
        "dish_diameter_m": dish_diameter_m,
        "first_null_deg_150MHz": float(np.degrees(np.arcsin(1.22 * C_M_PER_S / (150e6 * dish_diameter_m)))),
        "first_sidelobe_db": -17.6,
    }
    meta.update(sens_meta)
    return power_out, meta


def hera_polybeam(track: pd.DataFrame, freqs_hz: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """HERA H2C CST-fit PolyBeam via hera_sim.

    Loads the Chebyshev polynomial fit (HERA_H2C_BEAM_POLY.npy) from hera_sim's
    data directory and evaluates the azimuthally-symmetric power beam B(za, nu)
    across the full track × frequency grid.

    strict_polybeam (default True): raise ImportError if hera_sim is unavailable.
      Set to False only for development; paper runs must use strict=True.
    """
    bcfg = cfg.get("beam", {})
    strict = bool(bcfg.get("strict_polybeam", True))
    fixed_eval_freq_hz = bcfg.get("fixed_eval_freq_hz")

    try:
        from hera_sim import beams as _hera_beams
        import os as _os
        _data_path = _os.path.join(_os.path.dirname(_hera_beams.__file__), "data")
        _coeffs = np.load(_os.path.join(_data_path, "HERA_H2C_BEAM_POLY.npy"), allow_pickle=True)
        _beam = _hera_beams.PolyBeam(
            beam_coeffs=_coeffs, spectral_index=-1.6755, ref_freq=150e6
        )
    except (ImportError, FileNotFoundError) as _e:
        if strict:
            raise ImportError(
                f"hera_poly beam requested but hera_sim PolyBeam is unavailable: {_e}. "
                "Set beam.strict_polybeam: false to fall back to airy_beam (not for paper runs)."
            ) from _e
        import warnings
        warnings.warn(f"hera_sim PolyBeam unavailable ({_e}); falling back to airy_beam")
        power, meta = airy_beam(track, freqs_hz, cfg)
        meta["fallback_from_hera_poly"] = True
        meta["fallback_reason"] = repr(_e)
        return power, meta

    za_deg = 90.0 - track["alt_deg"].to_numpy()   # (T,)
    az_deg = track["az_deg"].to_numpy()            # (T,)
    za_rad = np.clip(np.deg2rad(za_deg), 0.0, np.pi / 2.0)
    az_rad = np.deg2rad(az_deg)

    eval_freqs_hz = np.asarray(freqs_hz, dtype=float)
    if fixed_eval_freq_hz is not None:
        eval_freqs_hz = np.full_like(eval_freqs_hz, float(fixed_eval_freq_hz), dtype=float)

    efield = np.asarray(_beam.efield_eval(
        az_array=az_rad, za_array=za_rad, freq_array=eval_freqs_hz
    ))

    expected_shape = (2, 2, len(freqs_hz), len(track))
    if efield.shape != expected_shape:
        raise ValueError(
            f"Unexpected PolyBeam efield shape: got {efield.shape}, "
            f"expected {expected_shape}. "
            "If hera_sim API changed, add an explicit axis mapping here — "
            "do NOT silently average axes."
        )

    # Power = sum |E|^2 over pol axes (0,1); shape (F, T)
    raw_power_ft = np.sum(np.abs(efield) ** 2, axis=(0, 1))

    # Zenith normalization
    efield_zen = np.asarray(_beam.efield_eval(
        az_array=np.array([0.0]), za_array=np.array([0.0]), freq_array=eval_freqs_hz
    ))  # (2, 2, F, 1)
    zen_power = np.sum(np.abs(efield_zen[:, :, :, 0]) ** 2, axis=(0, 1))  # (F,)
    zen_power = np.where(zen_power < 1e-30, 1.0, zen_power)

    power = (raw_power_ft / zen_power[:, None]).T   # (T, F)

    # Raw power diagnostics (before floor/clip)
    raw_arr = np.asarray(power, dtype=float)
    raw_valid = raw_arr[np.isfinite(raw_arr)]
    nan_frac = float(np.mean(~np.isfinite(raw_arr)))

    # Reference dB at za~30° and za~60° at 150 MHz
    mid_freq_idx = np.argmin(np.abs(freqs_hz - 150e6))
    za30_idx = np.argmin(np.abs(za_deg - 30.0)) if len(za_deg) > 0 else 0
    za60_idx = np.argmin(np.abs(za_deg - 60.0)) if len(za_deg) > 0 else 0
    _p30 = float(power[za30_idx, mid_freq_idx]) if power.size > 0 else 0.0
    _p60 = float(power[za60_idx, mid_freq_idx]) if power.size > 0 else 0.0

    power_out, sens_meta = apply_beam_sensitivity(power, cfg)

    meta = {
        "mode": "hera_h2c_polybeam_frozenfreq" if fixed_eval_freq_hz is not None else "hera_h2c_polybeam",
        "beam_file": "HERA_H2C_BEAM_POLY.npy",
        "spectral_index": -1.6755,
        "ref_freq_mhz": 150.0,
        "strict_polybeam": strict,
        "fixed_eval_freq_mhz": (float(fixed_eval_freq_hz) / 1e6) if fixed_eval_freq_hz is not None else None,
        "za30_150mhz_db": float(10 * np.log10(max(_p30, 1e-30))),
        "za60_150mhz_db": float(10 * np.log10(max(_p60, 1e-30))),
        "raw_power_min": float(np.nanmin(raw_valid)) if len(raw_valid) else float("nan"),
        "raw_power_max": float(np.nanmax(raw_valid)) if len(raw_valid) else float("nan"),
        "raw_power_p95": float(np.nanpercentile(raw_valid, 95)) if len(raw_valid) else float("nan"),
        "raw_power_p99": float(np.nanpercentile(raw_valid, 99)) if len(raw_valid) else float("nan"),
        "raw_power_nan_fraction": nan_frac,
    }
    meta.update(sens_meta)
    return power_out, meta


def literature_uemr_spectrum(freqs_hz: np.ndarray, cfg: Dict[str, Any], channel_width_hz: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Literature-anchored Starlink UEMR spectral template (Bassa 2024, Di Vruno 2023).

    Constructs a morphologically plausible (NOT a proprietary waveform reconstruction):
    - Two broadband downlink windows at ~120 MHz and ~161 MHz
    - Narrowband comb lines at configurable MHz positions
    - Optional Bassa v2-mini comb spacings (48.8 / 65.0 / 97.5 kHz)
    - Subchannel line dilution: effective peak = peak × (intrinsic_width / channel_width)
      when the line is narrower than the channel (unresolved line → flux diluted over channel)

    Output is max-normalized to [0, 1]; scale by reference_flux_jy in the caller.
    """
    em = cfg.get("starlink", {}).get("emission_model", {})
    if channel_width_hz is None:
        channel_width_hz = float(np.median(np.diff(freqs_hz))) if len(freqs_hz) > 1 else 1.0

    spec = np.zeros_like(freqs_hz, dtype=float)
    components: List[Dict[str, Any]] = []

    def top_hat(center_mhz: float, width_mhz: float, amp: float, edge_hz: float = 0.4e6) -> np.ndarray:
        x = np.abs(freqs_hz - center_mhz * 1e6)
        return amp / (1.0 + np.exp((x - 0.5 * width_mhz * 1e6) / edge_hz))

    # Broadband downlink windows (Bassa 2024 HBA morphology)
    window_flux = float(em.get("bassa_hba_window_flux_jy", 30.0))
    for ctr_mhz, wid_mhz in [(120.0, 8.0), (161.0, 8.0)]:
        contrib = top_hat(ctr_mhz, wid_mhz, window_flux)
        spec += contrib
        components.append({"type": "broadband_window", "center_mhz": ctr_mhz, "width_mhz": wid_mhz, "amp_jy": window_flux})

    # Narrowband comb lines with subchannel dilution
    dilute = bool(em.get("dilute_subchannel_lines", True))
    intrinsic_width_hz = float(em.get("intrinsic_line_width_hz", 12_200.0))  # ~12.2 kHz per Bassa
    render_width_hz = max(intrinsic_width_hz, 0.5 * channel_width_hz)
    narrowband_peak = float(em.get("narrowband_peak_flux_jy", 50.0))

    for mhz in em.get("narrowband_lines_mhz", [125.0, 135.0, 143.05, 150.0, 175.0]):
        effective_peak = narrowband_peak * (min(1.0, intrinsic_width_hz / max(channel_width_hz, 1.0)) if dilute else 1.0)
        spec += effective_peak * np.exp(-0.5 * ((freqs_hz - float(mhz) * 1e6) / render_width_hz) ** 2)
        components.append({"type": "narrowband_line", "center_mhz": float(mhz), "effective_peak_jy": float(effective_peak), "diluted": dilute})

    # Optional Bassa v2-mini comb spacings (157–165 MHz region)
    comb_spacings_khz = em.get("bassa_v2mini_comb_spacings_khz", [])
    comb_peak = float(em.get("bassa_comb_peak_flux_jy", 25.0))
    for spacing_khz in comb_spacings_khz:
        comb_freqs = np.arange(157.0e6, 165.0e6, float(spacing_khz) * 1e3)
        effective_peak = comb_peak * (min(1.0, intrinsic_width_hz / max(channel_width_hz, 1.0)) if dilute else 1.0)
        for f_comb in comb_freqs:
            spec += effective_peak * np.exp(-0.5 * ((freqs_hz - f_comb) / render_width_hz) ** 2)
        components.append({"type": "bassa_v2mini_comb", "spacing_khz": float(spacing_khz), "n_tones": int(len(comb_freqs))})

    if np.nanmax(spec) <= 0:
        spec = np.ones_like(freqs_hz, dtype=float)
    spec = spec / float(np.nanmax(spec))

    # Spectral morphology modifier
    spectral_morphology = str(em.get("spectral_morphology", "smooth")).lower()
    if spectral_morphology == "comb":
        # Frequency-domain comb: creates delay-domain copies at ±Δτ_comb
        # Δτ_comb > τ_hor injects power into EoR window for all baseline groups
        delta_tau_s = float(em.get("comb_delay_ns", 800.0)) * 1e-9
        mod = 0.5 * (1.0 + np.cos(2.0 * np.pi * delta_tau_s * freqs_hz))
        spec = spec * mod
        spec = np.clip(spec, 0.0, None)
        peak = float(np.nanmax(spec))
        if peak > 0:
            spec = spec / peak
        components.append({"type": "spectral_comb_modulation", "comb_delay_ns": float(delta_tau_s * 1e9)})

    return spec, {
        "emission_model_type": "literature_parameterized_uemr_template_not_proprietary_waveform",
        "spectral_morphology": spectral_morphology,
        "channel_width_hz": float(channel_width_hz),
        "dilute_subchannel_lines": dilute,
        "intrinsic_line_width_hz": float(intrinsic_width_hz),
        "render_width_hz": float(render_width_hz),
        "components": components,
        "normalization": "max-normalized; scaled by reference_flux_jy × range_att × beam × smearing",
        "not_proprietary_waveform": True,
    }


def literature_uemr_spectrum_v2(freqs_hz: np.ndarray, cfg: Dict[str, Any], channel_width_hz: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Starlink-like UEMR template with stress-test and literature modes separated."""
    em = cfg.get("starlink", {}).get("emission_model", {})
    if channel_width_hz is None:
        channel_width_hz = float(np.median(np.diff(freqs_hz))) if len(freqs_hz) > 1 else 1.0

    morphology = str(em.get("spectral_morphology", "smooth")).lower()
    ripple_aliases = {"comb", "ripple_stress", "controlled_ripple_stress", "mhz_ripple_stress"}
    literature_line_modes = {"literature_lofar_lines", "lofar_gen1_narrowband_lines"}
    literature_khz_modes = {
        "literature_lofar_khz_comb",
        "lofar_gen2_khz_comb",
        "literature_lofar_lines_khz_comb",
        "lofar_lines_khz_comb",
    }
    use_literature_lines = morphology in literature_line_modes or morphology in literature_khz_modes
    use_literature_khz_comb = morphology in literature_khz_modes
    include_lines = bool(em.get("include_narrowband_lines", morphology != "broadband_only"))
    include_khz_comb = bool(em.get("include_literature_khz_comb", use_literature_khz_comb))

    intrinsic_width_hz = float(em.get("intrinsic_line_width_hz", 12_200.0))
    highres_df_hz = float(em.get("highres_df_khz", 1.0)) * 1e3
    channel_average = bool(em.get("highres_channel_average", False) or use_literature_lines or include_khz_comb)
    components: List[Dict[str, Any]] = []

    def evaluate(grid_hz: np.ndarray, effective_channel_width_hz: float) -> np.ndarray:
        spec_local = np.zeros_like(grid_hz, dtype=float)

        def top_hat(center_mhz: float, width_mhz: float, amp: float, edge_hz: float = 0.4e6) -> np.ndarray:
            x = np.abs(grid_hz - center_mhz * 1e6)
            return amp / (1.0 + np.exp((x - 0.5 * width_mhz * 1e6) / edge_hz))

        window_flux = float(em.get("bassa_hba_window_flux_jy", 30.0))
        for ctr_mhz, width_mhz in em.get("broadband_windows_mhz", [(120.0, 8.0), (161.0, 8.0)]):
            spec_local += top_hat(float(ctr_mhz), float(width_mhz), window_flux)

        dilute = bool(em.get("dilute_subchannel_lines", True))
        render_width_hz = max(intrinsic_width_hz, float(em.get("minimum_render_width_hz", highres_df_hz)))
        line_peak = float(em.get("narrowband_peak_flux_jy", 50.0))
        default_lines = [125.0, 135.0, 150.0, 175.0] if use_literature_lines else [125.0, 135.0, 143.05, 150.0, 175.0]
        line_mhz = list(em.get("narrowband_lines_mhz", default_lines)) if include_lines else []
        if bool(em.get("include_reflection_control_line", False)) and 143.05 not in [float(x) for x in line_mhz]:
            line_mhz.append(143.05)
        effective_line_peak = line_peak * (min(1.0, intrinsic_width_hz / max(effective_channel_width_hz, 1.0)) if dilute else 1.0)
        for mhz in line_mhz:
            spec_local += effective_line_peak * np.exp(-0.5 * ((grid_hz - float(mhz) * 1e6) / render_width_hz) ** 2)

        spacing_khz = em.get(
            "literature_comb_spacings_khz",
            em.get("bassa_v2mini_comb_spacings_khz", [48.8, 50.0, 65.0, 97.5, 150.0, 220.0] if include_khz_comb else []),
        )
        comb_range_mhz = em.get("literature_comb_range_mhz", [157.0, 165.0])
        comb_peak = float(em.get("bassa_comb_peak_flux_jy", 25.0))
        effective_comb_peak = comb_peak * (min(1.0, intrinsic_width_hz / max(effective_channel_width_hz, 1.0)) if dilute else 1.0)
        if include_khz_comb:
            for spacing in spacing_khz:
                comb_freqs = np.arange(float(comb_range_mhz[0]) * 1e6, float(comb_range_mhz[1]) * 1e6, float(spacing) * 1e3)
                for f_comb in comb_freqs:
                    spec_local += effective_comb_peak * np.exp(-0.5 * ((grid_hz - f_comb) / render_width_hz) ** 2)
        return spec_local

    if channel_average:
        spec = np.zeros_like(freqs_hz, dtype=float)
        half = 0.5 * channel_width_hz
        for i, center_hz in enumerate(freqs_hz):
            grid = np.arange(center_hz - half, center_hz + half, highres_df_hz)
            if len(grid) == 0:
                grid = np.array([center_hz], dtype=float)
            spec[i] = float(np.nanmean(evaluate(grid, channel_width_hz)))
        components.append({"type": "highres_channel_average", "df_hz": highres_df_hz, "channel_width_hz": float(channel_width_hz)})
    else:
        spec = evaluate(freqs_hz, channel_width_hz)

    components.append({"type": "broadband_windows", "centers_mhz": [120.0, 161.0], "width_mhz": 8.0})
    if include_lines:
        components.append({"type": "lofar_narrowband_lines", "line_width_hz": intrinsic_width_hz, "unresolved_at_channel_scale": bool(intrinsic_width_hz < channel_width_hz)})
    if include_khz_comb:
        components.append({"type": "lofar_khz_comb", "spacing_khz": em.get("literature_comb_spacings_khz", em.get("bassa_v2mini_comb_spacings_khz", [48.8, 50.0, 65.0, 97.5, 150.0, 220.0])), "not_mhz_ripple": True})

    if morphology in ripple_aliases:
        ripple_delay_ns = float(em.get("ripple_delay_ns", em.get("comb_delay_ns", 800.0)))
        delay_s = ripple_delay_ns * 1e-9
        mod = 0.5 * (1.0 + np.cos(2.0 * np.pi * delay_s * freqs_hz))
        spec = np.clip(spec * mod, 0.0, None)
        components.append({
            "type": "controlled_mhz_ripple_stress_test",
            "ripple_delay_ns": ripple_delay_ns,
            "ripple_spacing_mhz": float(1e3 / ripple_delay_ns),
            "not_literature_starlink_comb_spacing": True,
        })

    if np.nanmax(spec) <= 0:
        spec = np.ones_like(freqs_hz, dtype=float)
    spec = spec / float(np.nanmax(spec))

    morphology_class = "smooth_or_broadband"
    if morphology in ripple_aliases:
        morphology_class = "controlled_eor_window_stress_test"
    elif use_literature_lines or include_khz_comb:
        morphology_class = "literature_anchored_lofar_uemr"

    return spec, {
        "emission_model_type": "literature_parameterized_uemr_template_not_proprietary_waveform",
        "spectral_morphology": morphology,
        "morphology_class": morphology_class,
        "channel_width_hz": float(channel_width_hz),
        "highres_channel_average": channel_average,
        "highres_df_hz": highres_df_hz if channel_average else None,
        "intrinsic_line_width_hz": float(intrinsic_width_hz),
        "components": components,
        "normalization": "max-normalized; scaled by reference_flux_jy x range_att x beam x smearing",
        "not_proprietary_waveform": True,
    }


def spectral_template(freqs_hz: np.ndarray, cfg: Dict[str, Any], channel_width_hz: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dispatcher: CSV → literature model (in that priority)."""
    em = cfg.get("starlink", {}).get("emission_model", {})
    csv_path = em.get("spectral_template_csv")
    if csv_path:
        df = pd.read_csv(csv_path)
        f = df["freq_hz"].to_numpy(float) if "freq_hz" in df else df["freq_mhz"].to_numpy(float) * 1e6
        col = "relative_amplitude" if "relative_amplitude" in df else "relative_power" if "relative_power" in df else "flux_jy"
        y = np.clip(df[col].to_numpy(float), 0.0, None)
        spec = np.interp(freqs_hz, f[np.argsort(f)], y[np.argsort(f)], left=0.0, right=0.0)
        spec = spec / max(float(np.nanmax(spec)), 1e-30)
        return spec, {"mode": "csv", "path": str(csv_path), "column": col}
    if channel_width_hz is None:
        channel_width_hz = float(np.median(np.diff(freqs_hz))) if len(freqs_hz) > 1 else 1.0
    cache_key = (
        len(freqs_hz),
        float(freqs_hz[0]) if len(freqs_hz) else 0.0,
        float(freqs_hz[-1]) if len(freqs_hz) else 0.0,
        float(channel_width_hz),
        str(em.get("spectral_morphology", "smooth")).lower(),
        tuple(float(x) for x in em.get("narrowband_lines_mhz", [])),
        bool(em.get("include_reflection_control_line", False)),
        tuple(float(x) for x in em.get("literature_comb_spacings_khz", em.get("bassa_v2mini_comb_spacings_khz", []))),
        tuple(float(x) for x in em.get("literature_comb_range_mhz", [157.0, 165.0])),
        float(em.get("intrinsic_line_width_hz", 12_200.0)),
        float(em.get("highres_df_khz", 1.0)),
        bool(em.get("highres_channel_average", False)),
        float(em.get("comb_delay_ns", em.get("ripple_delay_ns", 800.0))),
    )
    cached = _SPECTRAL_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        spec_cached, meta_cached = cached
        meta = dict(meta_cached)
        meta["cache_hit"] = True
        return spec_cached.copy(), meta
    spec, meta = literature_uemr_spectrum_v2(freqs_hz, cfg, channel_width_hz=channel_width_hz)
    meta = dict(meta)
    meta["cache_hit"] = False
    _SPECTRAL_TEMPLATE_CACHE[cache_key] = (spec.copy(), meta)
    return spec, meta


def doppler_shift_spectrum(
    freqs_hz: np.ndarray,
    spec: np.ndarray,
    range_rate_m_s: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply an optional Doppler shift to a frequency-domain spectral template.

    The template is interpreted in the observer frame. For positive range rate
    (receding source), the observed frequency is redshifted, so the emitted
    spectrum is sampled at a slightly higher frequency.

    Supported modes under `starlink.emission_model.doppler_mode`:
      - none / off / false: no Doppler shift
      - constant: use the median range rate across the track
      - linear / track / per_time: use the per-time range rate samples
    """
    em = cfg.get("starlink", {}).get("emission_model", {})
    mode = str(em.get("doppler_mode", "none")).strip().lower()
    if mode in {"", "none", "off", "false", "0"}:
        return np.asarray(spec, dtype=float)[None, :], {"doppler_mode": "none", "doppler_applied": False}

    rates = np.asarray(range_rate_m_s, dtype=float).reshape(-1)
    if mode == "constant":
        finite = np.isfinite(rates)
        ref_rate = float(np.nanmedian(rates[finite])) if np.any(finite) else 0.0
        rates = np.full_like(rates, ref_rate, dtype=float)
    elif mode in {"linear", "track", "per_time"}:
        pass
    else:
        raise ValueError(f"Unsupported doppler_mode: {mode!r}")

    freqs = np.asarray(freqs_hz, dtype=float)
    spec = np.asarray(spec, dtype=float)
    out = np.empty((len(rates), len(freqs)), dtype=float)
    c = C_M_PER_S
    for i, rr in enumerate(rates):
        if not np.isfinite(rr):
            rr = 0.0
        scale = 1.0 - rr / c
        if abs(scale) < 1e-12:
            scale = 1e-12
        emit_freqs = freqs / scale
        out[i] = np.interp(emit_freqs, freqs, spec, left=0.0, right=0.0)

    meta = {
        "doppler_mode": mode,
        "doppler_applied": True,
        "doppler_rate_min_m_s": float(np.nanmin(rates)) if np.any(np.isfinite(rates)) else float("nan"),
        "doppler_rate_max_m_s": float(np.nanmax(rates)) if np.any(np.isfinite(rates)) else float("nan"),
        "doppler_rate_median_m_s": float(np.nanmedian(rates)) if np.any(np.isfinite(rates)) else float("nan"),
        "doppler_rate_span_m_s": float(np.nanmax(rates) - np.nanmin(rates)) if np.any(np.isfinite(rates)) else float("nan"),
    }
    return out, meta


def _window_buffer_ns(cfg: Dict[str, Any]) -> float:
    return float(
        cfg.get("metrics", {}).get("window", {}).get(
            "buffer_ns",
            cfg.get("pipeline", {}).get("delay_filter", {}).get("buffer_ns", 100.0),
        )
    )


def classify_horizon_proximity(eta_tau: float) -> str:
    """Classify horizon proximity. This is not an EoR-window risk label."""
    if not np.isfinite(eta_tau):
        return "unknown"
    if eta_tau < 0.4:
        return "inner_wedge"
    if eta_tau < 0.8:
        return "mid_wedge"
    return "near_horizon"


def window_geometry_metrics(tau_s: np.ndarray, baseline_enu_m: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Geometry diagnostics relative to the buffered EoR-window boundary.

    eta_tau is only a horizon-proximity descriptor. The window mask starts at
    tau_hor + buffer_ns, so margin_to_window_ns is the signed distance to the
    actual window boundary used by the metric.
    """
    tau_hor_ns = float(np.linalg.norm(baseline_enu_m) / C_M_PER_S * 1e9)
    max_abs_tau_sat_ns = float(np.nanmax(np.abs(tau_s)) * 1e9)
    buffer_ns = _window_buffer_ns(cfg)
    tau_window_ns = tau_hor_ns + buffer_ns
    eta_tau = max_abs_tau_sat_ns / max(tau_hor_ns, 1e-30)
    margin_to_window_ns = tau_window_ns - max_abs_tau_sat_ns
    return {
        "tau_horizon_ns": tau_hor_ns,
        "tau_hor_ns": tau_hor_ns,
        "window_buffer_ns": buffer_ns,
        "tau_window_ns": tau_window_ns,
        "max_abs_tau_sat_ns": max_abs_tau_sat_ns,
        "margin_to_window_ns": margin_to_window_ns,
        "geometry_reaches_window": bool(margin_to_window_ns <= 0.0),
        "eta_tau": float(eta_tau),
        "horizon_proximity_bin": classify_horizon_proximity(float(eta_tau)),
    }


def _eta_tau(tau_s: np.ndarray, baseline_enu_m: np.ndarray) -> float:
    """η_τ = max_t|τ_sat(t)| / τ_hor; < 1 = inside wedge, > 1 = window-side risk."""
    tau_hor = np.linalg.norm(baseline_enu_m) / C_M_PER_S
    tau_max = float(np.nanmax(np.abs(tau_s)))
    return tau_max / max(tau_hor, 1e-30)


def build_starlink_visibility(ctx: BackgroundContext, cfg: Dict[str, Any], s_ref_jy: Optional[float] = None) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    scfg = cfg.get("starlink", {})
    rec, sel_meta = select_satellite(scfg["tle_path"], ctx, cfg)
    track, geom = compute_nearfield_track(rec.sat, ctx, cfg)
    dnu = float(np.median(np.diff(ctx.freqs_hz))) if len(ctx.freqs_hz) > 1 else 1.0
    beam_tf, beam_meta = gaussian_beam(track, ctx.freqs_hz, cfg)
    spec, spec_meta = spectral_template(ctx.freqs_hz, cfg, channel_width_hz=dnu)
    spec_tf, doppler_meta = doppler_shift_spectrum(ctx.freqs_hz, spec, track["range_rate_m_s"].to_numpy(float), cfg)
    s_ref = float(s_ref_jy if s_ref_jy is not None else scfg.get("reference_flux_jy", 100.0))
    r_ref = float(scfg.get("reference_range_km", 550.0))
    mode = scfg.get("range_attenuation_mode", "flux_density_1_over_r2")
    r = np.clip(track["range_km"].to_numpy(float), 1e-6, None)
    if mode == "none_observed_apparent_flux":
        range_att = np.ones_like(r)
    elif mode == "field_amplitude_1_over_r":
        range_att = r_ref / r
    else:
        range_att = (r_ref / r) ** 2
    phase = np.exp(-2j * np.pi * geom["tau_s"][:, None] * ctx.freqs_hz[None, :])
    amp = s_ref * range_att[:, None] * geom["attenuation_tf"] * beam_tf * spec_tf

    # Bursty morphology: periodic on/off envelope that spreads fringe-rate sidelobes
    em_cfg = scfg.get("emission_model", {})
    if str(em_cfg.get("spectral_morphology", "smooth")).lower() == "bursty":
        time_sec_burst = (ctx.times_jd - ctx.times_jd[0]) * 86400.0
        burst_period_s = float(em_cfg.get("burst_period_s", 30.0))
        duty_cycle = float(em_cfg.get("burst_duty_cycle", 0.5))
        burst_env = ((time_sec_burst % burst_period_s) < (duty_cycle * burst_period_s)).astype(float)
        amp = amp * burst_env[:, None]

    vis = amp * phase
    eta = _eta_tau(geom["tau_s"], ctx.baseline_enu_m)
    geom_metrics = window_geometry_metrics(geom["tau_s"], ctx.baseline_enu_m, cfg)
    time_scale_in = getattr(ctx, "time_scale", (ctx.metadata or {}).get("time_scale", "utc"))
    report = {
        "selected_satellite": sel_meta,
        "time_scale_input": time_scale_in,
        "time_scale_skyfield": "TT",
        "time_conversion": "UTC JD → TT JD via astropy" if time_scale_in == "utc" else "direct TT JD",
        "reference_flux_jy": s_ref,
        "reference_range_km": r_ref,
        "range_attenuation_mode": mode,
        "baseline_length_m": float(np.linalg.norm(ctx.baseline_enu_m)),
        "tau_horizon_ns": float(np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9),
        "beam": beam_meta,
        "beam_mode": beam_meta.get("mode"),
        "beam_peak": float(np.nanmax(beam_tf)),
        "beam_median": float(np.nanmedian(beam_tf)),
        "beam_nonzero_fraction": float(np.mean(beam_tf > 0)),
        "attenuation_min": float(np.nanmin(geom["attenuation_tf"])),
        "attenuation_median": float(np.nanmedian(geom["attenuation_tf"])),
        "attenuation_max": float(np.nanmax(geom["attenuation_tf"])),
        "max_abs_fringe_rate_Hz": float(np.nanmax(np.abs(geom["fringe_rate_hz"]))),
        "spectral_template": spec_meta,
        "doppler": doppler_meta,
        "peak_abs_jy": float(np.nanmax(np.abs(vis))),
        "mean_abs_jy": float(np.nanmean(np.abs(vis))),
        "tau_min_ns": float(np.nanmin(geom["tau_s"]) * 1e9),
        "tau_max_ns": float(np.nanmax(geom["tau_s"]) * 1e9),
        "eta_tau": eta,
        **geom_metrics,
    }
    return vis.astype(complex), track, report


def build_synthetic_visibility(
    ctx: BackgroundContext,
    cfg: Dict[str, Any],
    za_peak_deg: float,
    az_transit_deg: float = 90.0,
    height_km: float = 550.0,
    s_ref_jy: Optional[float] = None,
    beam_mode_override: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Inject a synthetic satellite at a prescribed peak zenith angle.

    The satellite traces a smooth arc: azimuth sweeps ±15° around az_transit_deg,
    while zenith angle varies as za(t) = sqrt(za_peak_deg^2 + (delta_az*t)^2),
    approximating a great-circle transit with minimum za = za_peak_deg at t=0.
    This isolates the geometric effect of za on delay-horizon proximity (η_τ).

    beam_mode_override: "gaussian" | "airy" | "hera_poly" | "none" — overrides cfg beam mode.
    """
    nt = len(ctx.times_jd)
    t_norm = np.linspace(-1.0, 1.0, nt)  # −1 at start, +1 at end

    # Az sweeps ±15° (≈ typical 9.8-min transit angular width for Starlink LEO)
    az_half_sweep_deg = 15.0
    az_deg = az_transit_deg + t_norm * az_half_sweep_deg
    # za traces a smooth arc through za_peak at t=0
    za_deg = np.sqrt(za_peak_deg**2 + (t_norm * az_half_sweep_deg) ** 2)
    za_deg = np.clip(za_deg, 0.1, 89.9)

    alt_deg = 90.0 - za_deg
    range_km = height_km / np.cos(np.deg2rad(za_deg))

    sat_enu = altaz_to_enu_m(alt_deg, az_deg, range_km)

    r1 = np.linalg.norm(sat_enu - ctx.ant1_enu_m[None, :], axis=1)
    r2 = np.linalg.norm(sat_enu - ctx.ant2_enu_m[None, :], axis=1)
    tau_s = (r2 - r1) / C_M_PER_S

    time_sec = (ctx.times_jd - ctx.times_jd[0]) * 86400.0
    tau_dot = np.gradient(tau_s, time_sec) if len(time_sec) > 1 else np.zeros_like(tau_s)

    dnu = float(np.median(np.diff(ctx.freqs_hz))) if len(ctx.freqs_hz) > 1 else 1.0
    dt = float(np.median(np.diff(ctx.times_jd)) * 86400.0) if len(ctx.times_jd) > 1 else 9.66
    fringe_rate_hz = tau_dot[:, None] * ctx.freqs_hz[None, :]
    sinc_time = np.sinc(fringe_rate_hz * dt)
    sinc_freq = np.sinc(tau_s[:, None] * dnu)
    attenuation = np.abs(sinc_time * sinc_freq)

    track = pd.DataFrame({
        "jd": ctx.times_jd,
        "time_sec": time_sec,
        "alt_deg": alt_deg,
        "az_deg": az_deg,
        "range_km": range_km,
        "range_rate_m_s": np.gradient(range_km * 1e3, time_sec) if len(time_sec) > 1 else np.zeros(nt),
        "tau_s": tau_s,
        "tau_dot_s_per_s": tau_dot,
    })

    # Beam response
    bcfg_override = dict(cfg.get("beam", {}))
    if beam_mode_override is not None:
        bcfg_override["mode"] = beam_mode_override
    cfg_beam = {**cfg, "beam": bcfg_override}
    dnu = float(np.median(np.diff(ctx.freqs_hz))) if len(ctx.freqs_hz) > 1 else 1.0
    beam_tf, beam_meta = gaussian_beam(track, ctx.freqs_hz, cfg_beam)
    spec, spec_meta = spectral_template(ctx.freqs_hz, cfg, channel_width_hz=dnu)
    spec_tf, doppler_meta = doppler_shift_spectrum(ctx.freqs_hz, spec, track["range_rate_m_s"].to_numpy(float), cfg)

    scfg = cfg.get("starlink", {})
    s_ref = float(s_ref_jy if s_ref_jy is not None else scfg.get("reference_flux_jy", 100.0))
    r_ref = float(scfg.get("reference_range_km", 550.0))
    range_att = (r_ref / np.clip(range_km, 1e-6, None)) ** 2

    phase = np.exp(-2j * np.pi * tau_s[:, None] * ctx.freqs_hz[None, :])
    amp = s_ref * range_att[:, None] * attenuation * beam_tf * spec_tf

    # Bursty morphology
    em_cfg = cfg.get("starlink", {}).get("emission_model", {})
    if str(em_cfg.get("spectral_morphology", "smooth")).lower() == "bursty":
        burst_period_s = float(em_cfg.get("burst_period_s", 30.0))
        duty_cycle = float(em_cfg.get("burst_duty_cycle", 0.5))
        burst_env = ((time_sec % burst_period_s) < (duty_cycle * burst_period_s)).astype(float)
        amp = amp * burst_env[:, None]

    vis = amp * phase

    eta = _eta_tau(tau_s, ctx.baseline_enu_m)
    geom_metrics = window_geometry_metrics(tau_s, ctx.baseline_enu_m, cfg)
    report = {
        "selected_satellite": "synthetic",
        "za_peak_deg": float(za_peak_deg),
        "az_transit_deg": float(az_transit_deg),
        "height_km": float(height_km),
        "beam_mode": bcfg_override.get("mode", "gaussian"),
        "reference_flux_jy": s_ref,
        "baseline_length_m": float(np.linalg.norm(ctx.baseline_enu_m)),
        "tau_horizon_ns": float(np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9),
        "beam": beam_meta,
        "beam_peak": float(np.nanmax(beam_tf)),
        "beam_median": float(np.nanmedian(beam_tf)),
        "attenuation_min": float(np.nanmin(attenuation)),
        "attenuation_median": float(np.nanmedian(attenuation)),
        "attenuation_max": float(np.nanmax(attenuation)),
        "max_abs_fringe_rate_Hz": float(np.nanmax(np.abs(fringe_rate_hz))),
        "spectral_template": spec_meta,
        "doppler": doppler_meta,
        "peak_abs_jy": float(np.nanmax(np.abs(vis))),
        "mean_abs_jy": float(np.nanmean(np.abs(vis))),
        "tau_min_ns": float(np.nanmin(tau_s) * 1e9),
        "tau_max_ns": float(np.nanmax(tau_s) * 1e9),
        "eta_tau": eta,
        **geom_metrics,
    }
    return vis.astype(complex), track, report


# ---------------------------------------------------------------------------
# Phase 3B helpers: real TLE ensemble with stratified sampling
# ---------------------------------------------------------------------------

def sample_unique_norad_ids(tle_path: str | Path, n_sample: int, seed: int = 42) -> List[str]:
    """Read TLE file, collect unique NORAD IDs, and return a random sample of n_sample."""
    lines = [x.strip() for x in Path(tle_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    seen: List[str] = []
    norad_set: set = set()
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            norad = _norad(lines[i])
            if norad and norad not in norad_set:
                norad_set.add(norad)
                seen.append(norad)
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            norad = _norad(lines[i + 1])
            if norad and norad not in norad_set:
                norad_set.add(norad)
                seen.append(norad)
            i += 3
        else:
            i += 1
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seen), size=min(n_sample, len(seen)), replace=False)
    return [seen[j] for j in sorted(idx)]


def load_tle_for_norads(tle_path: str | Path, norads: List[str], target_jd: Optional[float] = None) -> List[SatelliteRecord]:
    """Load TLE records for a specific set of NORAD IDs."""
    ts = load.timescale()
    norads_set = set(str(n) for n in norads)
    lines = [x.strip() for x in Path(tle_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    grouped: Dict[str, Dict[str, Any]] = {}
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
        norad = _norad(line1)
        if norad not in norads_set:
            continue
        epoch = _epoch(line1)
        sat_name = name or f"SAT-{norad}-E{epoch}"
        try:
            sat = EarthSatellite(line1, line2, sat_name, ts)
        except Exception:
            continue
        epoch_jd = float(sat.epoch.tt)
        old = grouped.get(norad)
        if old is None:
            grouped[norad] = {"sat": sat, "norad_id": norad, "epoch": epoch, "name": sat_name, "epoch_jd": epoch_jd}
        else:
            if target_jd is None:
                take = epoch_jd > old["epoch_jd"]
            else:
                take = abs(epoch_jd - target_jd) < abs(old["epoch_jd"] - target_jd)
            if take:
                grouped[norad] = {"sat": sat, "norad_id": norad, "epoch": epoch, "name": sat_name, "epoch_jd": epoch_jd}
    return [SatelliteRecord(r["sat"], r["norad_id"], r["epoch"], r["name"]) for r in grouped.values()]


def compute_eta_tau_for_sat(rec: SatelliteRecord, ctx: BackgroundContext, cfg: Dict[str, Any]) -> float:
    """Fast η_τ computation for screening — no beam or spectral template."""
    try:
        track, geom = compute_nearfield_track(rec.sat, ctx, cfg)
        return _eta_tau(geom["tau_s"], ctx.baseline_enu_m)
    except Exception:
        return float("nan")


def build_visibility_for_sat(
    rec: SatelliteRecord,
    ctx: BackgroundContext,
    cfg: Dict[str, Any],
    s_ref_jy: Optional[float] = None,
    beam_mode_override: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Like build_starlink_visibility but uses a pre-selected SatelliteRecord.

    Bypasses satellite selection, allowing the caller to specify exactly which
    satellite to inject. Used for Phase 3B ensemble experiments.
    """
    track, geom = compute_nearfield_track(rec.sat, ctx, cfg)

    cfg_beam = {**cfg, "beam": {**cfg.get("beam", {})}}
    if beam_mode_override is not None:
        cfg_beam["beam"]["mode"] = beam_mode_override

    dnu = float(np.median(np.diff(ctx.freqs_hz))) if len(ctx.freqs_hz) > 1 else 1.0
    beam_tf, beam_meta = gaussian_beam(track, ctx.freqs_hz, cfg_beam)
    spec, spec_meta = spectral_template(ctx.freqs_hz, cfg, channel_width_hz=dnu)
    spec_tf, doppler_meta = doppler_shift_spectrum(ctx.freqs_hz, spec, track["range_rate_m_s"].to_numpy(float), cfg)

    scfg = cfg.get("starlink", {})
    s_ref = float(s_ref_jy if s_ref_jy is not None else scfg.get("reference_flux_jy", 100.0))
    r_ref = float(scfg.get("reference_range_km", 550.0))
    mode = scfg.get("range_attenuation_mode", "flux_density_1_over_r2")
    r = np.clip(track["range_km"].to_numpy(float), 1e-6, None)
    if mode == "none_observed_apparent_flux":
        range_att = np.ones_like(r)
    elif mode == "field_amplitude_1_over_r":
        range_att = r_ref / r
    else:
        range_att = (r_ref / r) ** 2

    phase = np.exp(-2j * np.pi * geom["tau_s"][:, None] * ctx.freqs_hz[None, :])
    amp = s_ref * range_att[:, None] * geom["attenuation_tf"] * beam_tf * spec_tf
    vis = amp * phase
    eta = _eta_tau(geom["tau_s"], ctx.baseline_enu_m)
    geom_metrics = window_geometry_metrics(geom["tau_s"], ctx.baseline_enu_m, cfg)
    time_scale_in = getattr(ctx, "time_scale", (ctx.metadata or {}).get("time_scale", "utc"))

    sel_meta = {
        "satellite_name": rec.name,
        "norad_id": rec.norad_id,
        "peak_alt_deg": float(np.nanmax(track["alt_deg"])),
        "selection": "pre_selected_ensemble",
    }
    report = {
        "selected_satellite": sel_meta,
        "time_scale_input": time_scale_in,
        "time_scale_skyfield": "TT",
        "reference_flux_jy": s_ref,
        "reference_range_km": r_ref,
        "range_attenuation_mode": mode,
        "baseline_length_m": float(np.linalg.norm(ctx.baseline_enu_m)),
        "tau_horizon_ns": float(np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9),
        "beam": beam_meta,
        "beam_mode": beam_meta.get("mode"),
        "beam_peak": float(np.nanmax(beam_tf)),
        "beam_median": float(np.nanmedian(beam_tf)),
        "beam_nonzero_fraction": float(np.mean(beam_tf > 0)),
        "attenuation_min": float(np.nanmin(geom["attenuation_tf"])),
        "attenuation_median": float(np.nanmedian(geom["attenuation_tf"])),
        "attenuation_max": float(np.nanmax(geom["attenuation_tf"])),
        "max_abs_fringe_rate_Hz": float(np.nanmax(np.abs(geom["fringe_rate_hz"]))),
        "spectral_template": spec_meta,
        "doppler": doppler_meta,
        "peak_abs_jy": float(np.nanmax(np.abs(vis))),
        "mean_abs_jy": float(np.nanmean(np.abs(vis))),
        "tau_min_ns": float(np.nanmin(geom["tau_s"]) * 1e9),
        "tau_max_ns": float(np.nanmax(geom["tau_s"]) * 1e9),
        "eta_tau": eta,
        **geom_metrics,
    }
    return vis.astype(complex), track, report
