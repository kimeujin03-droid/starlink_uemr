import matplotlib.pyplot as plt


def plot_alt_track(track_df):
    plt.figure(figsize=(8, 4))
    plt.plot(track_df["jd"], track_df["alt_deg"])
    plt.xlabel("JD")
    plt.ylabel("Altitude [deg]")
    plt.title("Satellite Altitude Track")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_range_rate(track_df):
    plt.figure(figsize=(8, 4))
    plt.plot(track_df["jd"], track_df["range_rate_km_s"])
    plt.xlabel("JD")
    plt.ylabel("Range rate [km/s]")
    plt.title("Satellite Range Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_attenuation_heatmap(summary):
    import numpy as np

    att = summary["attenuation"]
    freqs_mhz = summary["freqs_hz"] / 1e6

    plt.figure(figsize=(8, 5))
    plt.imshow(
        att,
        aspect="auto",
        origin="lower",
        extent=[freqs_mhz.min(), freqs_mhz.max(), 0, att.shape[0]]
    )
    plt.colorbar(label="|sinc attenuation|")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time index")
    plt.title("Fringe-washing attenuation")
    plt.tight_layout()
    plt.show()
