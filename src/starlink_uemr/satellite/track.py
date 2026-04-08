import numpy as np
import pandas as pd


def compute_topocentric_track(satellite, observer, times):
    difference = satellite - observer
    topocentric = difference.at(times)

    alt, az, distance = topocentric.altaz()
    range_rate = topocentric.frame_latlon_and_rates(observer)[5]

    return {
        "alt_deg": alt.degrees,
        "az_deg": az.degrees,
        "range_km": distance.km,
        "range_rate_km_s": range_rate.km_per_s,
        "times": times,
    }


def track_to_dataframe(track_dict):
    times = track_dict["times"]
    jd = np.array(times.tt)

    alt_deg = np.asarray(track_dict["alt_deg"])
    az_deg = np.asarray(track_dict["az_deg"])
    range_km = np.asarray(track_dict["range_km"])
    range_rate_km_s = np.asarray(track_dict["range_rate_km_s"])

    alt_rad = np.deg2rad(alt_deg)
    az_rad = np.deg2rad(az_deg)

    # ENU unit vector
    ux = np.cos(alt_rad) * np.sin(az_rad)  # east
    uy = np.cos(alt_rad) * np.cos(az_rad)  # north
    uz = np.sin(alt_rad)                   # up

    df = pd.DataFrame({
        "jd": jd,
        "alt_deg": alt_deg,
        "az_deg": az_deg,
        "range_km": range_km,
        "range_rate_km_s": range_rate_km_s,
        "ux": ux,
        "uy": uy,
        "uz": uz,
    })
    return df


def filter_above_horizon(track_df, min_alt_deg: float = 0.0):
    return track_df.loc[track_df["alt_deg"] >= min_alt_deg].reset_index(drop=True)
