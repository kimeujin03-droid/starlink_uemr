import numpy as np

C_M_PER_S = 299792458.0


def delay_series_seconds(track_df, baseline_enu_m):
    bx, by, bz = baseline_enu_m
    sdotb = (
        bx * track_df["ux"].to_numpy()
        + by * track_df["uy"].to_numpy()
        + bz * track_df["uz"].to_numpy()
    )
    tau_s = sdotb / C_M_PER_S
    return tau_s


def delay_rate_series_seconds_per_second(tau_s, dt_sec: float):
    return np.gradient(tau_s, dt_sec)


def fringe_washing_attenuation(freqs_hz, tau_dot_s_per_s, dt_sec: float):
    freqs_hz = np.asarray(freqs_hz)
    tau_dot_s_per_s = np.asarray(tau_dot_s_per_s)

    x = tau_dot_s_per_s[:, None] * freqs_hz[None, :] * dt_sec
    attenuation = np.abs(np.sinc(x))
    return attenuation


def geometry_only_summary(track_df, baseline_enu_m, freqs_hz, dt_sec: float):
    tau_s = delay_series_seconds(track_df, baseline_enu_m)
    tau_dot = delay_rate_series_seconds_per_second(tau_s, dt_sec)
    attenuation = fringe_washing_attenuation(freqs_hz, tau_dot, dt_sec)

    return {
        "tau_s": tau_s,
        "tau_dot_s_per_s": tau_dot,
        "attenuation": attenuation,
        "freqs_hz": freqs_hz,
        "jd": track_df["jd"].to_numpy(),
        "alt_deg": track_df["alt_deg"].to_numpy(),
        "az_deg": track_df["az_deg"].to_numpy(),
    }
