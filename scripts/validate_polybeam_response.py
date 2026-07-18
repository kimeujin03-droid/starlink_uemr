#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.satellite import hera_polybeam


def synthetic_track(za_deg: np.ndarray, az_deg: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "jd": np.arange(len(za_deg), dtype=float),
            "time_sec": np.arange(len(za_deg), dtype=float),
            "alt_deg": 90.0 - za_deg,
            "az_deg": np.full(len(za_deg), float(az_deg)),
            "range_km": np.full(len(za_deg), 550.0),
            "range_rate_m_s": np.zeros(len(za_deg), dtype=float),
            "tau_s": np.zeros(len(za_deg), dtype=float),
            "tau_dot_s_per_s": np.zeros(len(za_deg), dtype=float),
        }
    )


def summarize_response(label: str, power: np.ndarray, za_deg: np.ndarray, freqs_hz: np.ndarray) -> list[dict]:
    rows = []
    for za in [0, 5, 10, 20, 30, 45, 60, 75, 90]:
        zi = int(np.argmin(np.abs(za_deg - za)))
        vals = np.asarray(power[zi], dtype=float)
        rows.append(
            {
                "beam_model": label,
                "za_deg": float(za_deg[zi]),
                "min_power": float(np.nanmin(vals)),
                "median_power": float(np.nanmedian(vals)),
                "max_power": float(np.nanmax(vals)),
                "power_110mhz": float(vals[int(np.argmin(np.abs(freqs_hz - 110e6)))]),
                "power_150mhz": float(vals[int(np.argmin(np.abs(freqs_hz - 150e6)))]),
                "power_190mhz": float(vals[int(np.argmin(np.abs(freqs_hz - 190e6)))]),
                "finite_fraction": float(np.mean(np.isfinite(vals))),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/polybeam_validation_summary.csv")
    ap.add_argument("--freq-min-mhz", type=float, default=110.0)
    ap.add_argument("--freq-max-mhz", type=float, default=190.0)
    ap.add_argument("--n-freq", type=int, default=219)
    ap.add_argument("--n-za", type=int, default=181)
    args = ap.parse_args()

    freqs_hz = np.linspace(args.freq_min_mhz * 1e6, args.freq_max_mhz * 1e6, args.n_freq)
    za_deg = np.linspace(0.0, 90.0, args.n_za)
    track = synthetic_track(za_deg)

    cfg_full = {"beam": {"mode": "hera_poly", "strict_polybeam": True, "clip_to_unit": True}}
    cfg_frozen = {
        "beam": {
            "mode": "hera_poly",
            "strict_polybeam": True,
            "clip_to_unit": True,
            "fixed_eval_freq_hz": 150_000_000.0,
        }
    }
    full, full_meta = hera_polybeam(track, freqs_hz, cfg_full)
    frozen, frozen_meta = hera_polybeam(track, freqs_hz, cfg_frozen)

    rows = summarize_response("full_polybeam", full, za_deg, freqs_hz)
    rows.extend(summarize_response("frozen_polybeam_150mhz", frozen, za_deg, freqs_hz))
    df = pd.DataFrame(rows)
    df["abs_full_minus_frozen_at_same_row"] = np.nan
    for za in df["za_deg"].unique():
        mask_full = (df["za_deg"] == za) & (df["beam_model"] == "full_polybeam")
        mask_frozen = (df["za_deg"] == za) & (df["beam_model"] == "frozen_polybeam_150mhz")
        if mask_full.any() and mask_frozen.any():
            diff = abs(
                float(df.loc[mask_full, "power_150mhz"].iloc[0])
                - float(df.loc[mask_frozen, "power_150mhz"].iloc[0])
            )
            df.loc[mask_full | mask_frozen, "abs_full_minus_frozen_at_same_row"] = diff

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    diagnostics = {
        "freq_min_mhz": args.freq_min_mhz,
        "freq_max_mhz": args.freq_max_mhz,
        "n_freq": args.n_freq,
        "n_za": args.n_za,
        "full_meta": full_meta,
        "frozen_meta": frozen_meta,
        "full_shape": list(full.shape),
        "frozen_shape": list(frozen.shape),
        "full_nan_fraction": float(np.mean(~np.isfinite(full))),
        "frozen_nan_fraction": float(np.mean(~np.isfinite(frozen))),
        "full_min": float(np.nanmin(full)),
        "full_max": float(np.nanmax(full)),
        "frozen_min": float(np.nanmin(frozen)),
        "frozen_max": float(np.nanmax(frozen)),
        "full_fraction_gt_one": float(np.mean(full > 1.0)),
        "frozen_fraction_gt_one": float(np.mean(frozen > 1.0)),
        "normalization_note": "pathb.satellite.hera_polybeam normalizes each frequency by zenith power before optional clipping.",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(f"saved {out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
