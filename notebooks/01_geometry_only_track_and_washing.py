"""
geometry_only_track_and_washing.py
===================================
Starlink satellite visibility + interferometric smearing (geometry-only).
 
물리 모델 정리
--------------
기하학적 지연:   τ(t) = b·ŝ(t) / c          [s]
주파수 스미어:   sinc(τ · Δν)                 [무차원]
시간 스미어:     sinc(f_fringe · Δt)          [무차원]
프린지율:        f_fringe = (ν/c) · b · (dŝ/dt)  [Hz]
 
numpy.sinc(x) = sin(πx)/(πx) → 인수에 π 불필요
"""
 
import pickle
import sys
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skyfield.api import load
 
sys.path.append(str(Path("../src").resolve()))
 
from starlink_uemr.config import load_yaml
from starlink_uemr.geometry.washing import geometry_only_summary
from starlink_uemr.satellite.tle import build_timescale, load_satellite_from_tle, make_observer
from starlink_uemr.satellite.track import (
    compute_topocentric_track,
    filter_above_horizon,
    track_to_dataframe,
)
from starlink_uemr.viz.tracks import plot_alt_track, plot_attenuation_heatmap, plot_range_rate
 
C_M_S = 299_792_458.0  # 광속 [m/s]
 
# ─────────────────────────────────────────────
# 1. 설정 로드
# ─────────────────────────────────────────────
# %%
cfg = load_yaml("../configs/runs/exp00_geometry_only.yaml")
 
ts = build_timescale()
start_jd = float(cfg["time"]["start_jd"])
n_times   = int(cfg["time"]["n_times"])
dt_sec    = float(cfg["time"]["dt_sec"])
 
# TLE epoch와 30일 이상 차이나면 SGP4 오차 급증 → epoch으로 보정
_sat0 = load.tle_file(cfg["tle_path"])[0]
tle_epoch_jd = float(_sat0.model.jdsatepoch + _sat0.model.jdsatepochF)
if abs(start_jd - tle_epoch_jd) > 30:
    print(f"[WARN] start_jd={start_jd:.5f} → TLE epoch={tle_epoch_jd:.5f} 으로 보정")
    start_jd = tle_epoch_jd
 
observer = make_observer(
    cfg["site"]["lat_deg"],
    cfg["site"]["lon_deg"],
    cfg["site"]["elev_m"],
)
 
start_hz = float(cfg["freq"]["start_hz"])
stop_hz  = float(cfg["freq"]["stop_hz"])
n_freqs  = int(cfg["freq"]["n_freqs"])
freqs_hz = np.linspace(start_hz, stop_hz, n_freqs)
 
# ─────────────────────────────────────────────
# 2. 최적 Starlink 위성 탐색 (2단계 스캔 + 캐시)
# ─────────────────────────────────────────────
# %%
def _batch_alt_max(satellites, observer, ts, start_jd, hours, dt_sec):
    """
    전체 위성을 한 번에 벡터화하여 최대 고도와 그 시각을 반환.
    루프 대신 (N_sat, N_time) 배열 연산 → 대폭 빠름.
 
    Returns
    -------
    pd.DataFrame : name, alt_max_deg, jd_at_alt_max
    """
    n = int(hours * 3600 / dt_sec)
    jd_arr = start_jd + np.arange(n) * dt_sec / 86400.0
    times  = ts.tt_jd(jd_arr)
 
    rows = []
    for sat in satellites:
        try:
            alt_deg = (sat - observer).at(times).altaz()[0].degrees
            idx = int(np.argmax(alt_deg))
            rows.append(dict(name=sat.name,
                             alt_max_deg=float(alt_deg[idx]),
                             jd_at_alt_max=float(jd_arr[idx])))
        except Exception:
            rows.append(dict(name=sat.name, alt_max_deg=np.nan, jd_at_alt_max=np.nan))
    return pd.DataFrame(rows)
 
 
def find_best_visible_satellite(
    tle_path, observer, ts, start_jd,
    keyword="STARLINK",
    coarse_hours=6,  coarse_dt_sec=120.0,
    refine_hours=12, refine_dt_sec=30.0,   # ← 60→30으로 줄여 정확도↑
    top_k=30,
):
    sats = load.tle_file(tle_path)
    cands = [s for s in sats if keyword.upper() in s.name.upper()]
    print(f"후보 위성: {len(cands)}개")
 
    # 1단계: 거친 스캔
    coarse = _batch_alt_max(cands, observer, ts, start_jd,
                            coarse_hours, coarse_dt_sec
                            ).sort_values("alt_max_deg", ascending=False)
 
    top_names = set(coarse.head(top_k)["name"])
    top_cands = [s for s in cands if s.name in top_names]
 
    # 2단계: 정밀 스캔
    refined = _batch_alt_max(top_cands, observer, ts, start_jd,
                             refine_hours, refine_dt_sec
                             ).sort_values("alt_max_deg", ascending=False)
 
    rest = coarse[~coarse["name"].isin(refined["name"])]
    return pd.concat([refined, rest], ignore_index=True)
 
 
# ─── 캐시 ───────────────────────────────────
# %%
CACHE = Path("../data/processed/caches/vis_df_cache.pkl")
CACHE.parent.mkdir(parents=True, exist_ok=True)
 
if CACHE.exists():
    with open(CACHE, "rb") as f:
        vis_df = pickle.load(f)
    print(f"[cache] 로드: {CACHE}")
else:
    vis_df = find_best_visible_satellite(
        tle_path=cfg["tle_path"], observer=observer, ts=ts, start_jd=start_jd,
    )
    with open(CACHE, "wb") as f:
        pickle.dump(vis_df, f)
    print(f"[cache] 저장: {CACHE}")
 
print("Best 10:\n", vis_df.head(10).to_string(index=False))
 
# ─────────────────────────────────────────────
# 3. 최선 위성으로 Pass 재구성
# ─────────────────────────────────────────────
# %%
best_row     = vis_df.iloc[0]
best_sat_name = best_row["name"]
peak_jd       = float(best_row["jd_at_alt_max"])
print(f"선택: {best_sat_name}  (peak alt ~{best_row['alt_max_deg']:.2f}°)")
 
all_sats = load.tle_file(cfg["tle_path"])
sat_best = next((s for s in all_sats if s.name == best_sat_name), None)
if sat_best is None:
    raise ValueError(f"TLE에서 위성을 찾지 못함: {best_sat_name}")
 
# Peak 중심으로 관측창 설정
window_sec    = n_times * dt_sec
start_jd_pass = peak_jd - (window_sec / 2.0) / 86400.0
jd_pass       = start_jd_pass + np.arange(n_times) * dt_sec / 86400.0
times_pass    = ts.tt_jd(jd_pass)
 
track_raw   = compute_topocentric_track(sat_best, observer, times_pass)
track_all   = track_to_dataframe(track_raw).sort_values("jd").reset_index(drop=True)
track_vis   = filter_above_horizon(track_all, min_alt_deg=0.0).reset_index(drop=True)
 
assert track_all["jd"].is_monotonic_increasing, "jd가 단조증가가 아님"
print(f"전체 {len(track_all)}행 | 지평선 위 {len(track_vis)}행")
print(f"고도: {track_vis['alt_deg'].min():.2f}° ~ {track_vis['alt_deg'].max():.2f}°")
print(f"거리: {track_vis['range_km'].min():.1f} ~ {track_vis['range_km'].max():.1f} km")
 
# 물리적 sanity: LEO 위성 거리 ~500–2000 km
assert track_vis["range_km"].max() < 10_000, \
    "[warn] range > 10 000 km → TLE epoch / 시간창 확인 필요"
print("[ok] range 스케일 정상")
 
plot_alt_track(track_all)
plot_range_rate(track_all)
 
# ─────────────────────────────────────────────
# 4. 기하 감쇠 (주파수 스미어링만)
# ─────────────────────────────────────────────
# %%
BASELINES = {
    "short_14.6m": (14.6, 0.0, 0.0),
    "mid_50m":     (50.0, 0.0, 0.0),
    "long_150m":   (150.0, 0.0, 0.0),
}
 
summaries: dict[str, dict] = {}
for name, bl in BASELINES.items():
    summaries[name] = geometry_only_summary(
        track_df=track_vis,
        baseline_enu_m=bl,
        freqs_hz=freqs_hz,
        dt_sec=dt_sec,
    )
 
# ─────────────────────────────────────────────
# 5. 시간 스미어링 (LOS 기반 정확한 프린지율)
# ─────────────────────────────────────────────
# %%
def _unit_los(pos_enu: np.ndarray) -> np.ndarray:
    """ENU 위치벡터 → 단위 LOS 벡터  (Ntime, 3)"""
    norms = np.linalg.norm(pos_enu, axis=1, keepdims=True)
    return pos_enu / np.clip(norms, 1e-12, None)
 
 
def time_smearing_sinc(
    baseline_enu_m: np.ndarray,
    pos_enu_m: np.ndarray,
    time_sec: np.ndarray,
    freqs_hz: np.ndarray,
    dt_int_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    시간 스미어링 sinc 계산.
 
    물리 공식
    ---------
    τ = b·ŝ / c
    f_fringe = dτ/dt · ν = (ν/c) · b · (dŝ/dt)
    sinc_time = sinc(f_fringe · Δt)   [numpy 정규화 sinc]
 
    Parameters
    ----------
    pos_enu_m : (Ntime, 3)  observer → satellite [m]
    time_sec  : (Ntime,)
    freqs_hz  : (Nfreq,)
    dt_int_sec: 적분(dump) 시간 [s]
 
    Returns
    -------
    fringe_rate : (Ntime, Nfreq) [Hz]
    sinc_time   : (Ntime, Nfreq)
    """
    s_hat = _unit_los(pos_enu_m)                        # (Ntime, 3)
 
    # dŝ/dt : 각 성분을 np.gradient로 수치 미분
    ds_dt = np.gradient(s_hat, time_sec, axis=0)        # (Ntime, 3)
 
    # b · (dŝ/dt) : scalar projection  (Ntime,)
    proj = ds_dt @ np.asarray(baseline_enu_m, dtype=float)
 
    # f_fringe = (ν/c) · proj : (Ntime, Nfreq)
    fringe_rate = (proj[:, None] * freqs_hz[None, :]) / C_M_S
 
    # numpy sinc : sinc(x) = sin(πx)/(πx)  → 인수 = f_fringe * Δt
    sinc_t = np.sinc(fringe_rate * dt_int_sec)
    return fringe_rate, sinc_t
 
 
def freq_smearing_sinc(tau_s: np.ndarray, channel_width_hz: float) -> np.ndarray:
    """
    주파수 스미어링 sinc.
 
    sinc_freq[t, f] = sinc(τ(t) · Δν)
 
    Parameters
    ----------
    tau_s          : (Ntime,)  기하학적 지연 [s]
    channel_width_hz: 채널 폭 [Hz]
    """
    return np.sinc(tau_s[:, None] * channel_width_hz)   # (Ntime, 1) broadcast OK
 
 
def full_smearing(
    summary: dict,
    baseline_enu_m,
    track_df: pd.DataFrame,
    freqs_hz: np.ndarray,
    dt_sec: float,
) -> np.ndarray:
    """
    전체 감쇠 = |sinc_freq| * |sinc_time|  (Ntime, Nfreq)
 
    두 항은 서로 독립적(분리 가능)이므로 element-wise 곱이 물리적으로 정확.
    """
    pos_enu = np.column_stack([
        track_df["ux"] * track_df["range_km"] * 1e3,
        track_df["uy"] * track_df["range_km"] * 1e3,
        track_df["uz"] * track_df["range_km"] * 1e3,
    ])
    time_sec = (track_df["jd"].to_numpy() - track_df["jd"].to_numpy()[0]) * 86400.0
 
    _, sinc_t = time_smearing_sinc(baseline_enu_m, pos_enu, time_sec, freqs_hz, dt_sec)
 
    dnu = float(np.median(np.diff(freqs_hz))) if len(freqs_hz) > 1 else 0.0
    sinc_f = freq_smearing_sinc(summary["tau_s"], dnu)
 
    return np.abs(sinc_f) * np.abs(sinc_t)   # 물리량은 크기만 의미 있음
 
 
# 전 baseline에 대해 계산
attenuation_full: dict[str, np.ndarray] = {}
for name, bl in BASELINES.items():
    attenuation_full[name] = full_smearing(
        summaries[name], bl, track_vis, freqs_hz, dt_sec
    )
    a = attenuation_full[name]
    print(f"{name:15s}: attenuation [{a.min():.4f}, {a.max():.4f}]")
 
# ─────────────────────────────────────────────
# 6. HERA-like 빔 적용
# ─────────────────────────────────────────────
# %%
from hera_sim.beams import PolyBeam
 
beam = PolyBeam.like_fagnoni19()
 
az_rad = np.deg2rad(track_vis["az_deg"].to_numpy())
za_rad = np.pi / 2.0 - np.deg2rad(track_vis["alt_deg"].to_numpy())   # zenith angle
 
ef_raw = np.asarray(beam.efield_eval(az_array=az_rad, za_array=za_rad, freq_array=freqs_hz))
 
 
def efield_to_power(ef: np.ndarray, n_time: int, n_freq: int) -> np.ndarray:
    """
    efield_eval 출력 → (Ntime, Nfreq) 전력 빔.
 
    efield 형태가 라이브러리 버전마다 다르므로 time/freq 축을 안전하게 탐색.
    두 축이 같은 크기면 ValueError를 명시적으로 발생시켜 암묵적 오류 방지.
    """
    power = np.abs(ef) ** 2
 
    shape = power.shape
    if n_time == n_freq:
        raise ValueError(
            f"n_time == n_freq == {n_time}: 축 구분 불가. "
            "관측 시간 또는 주파수 샘플 수를 다르게 설정하세요."
        )
 
    t_axes = [i for i, s in enumerate(shape) if s == n_time]
    f_axes = [i for i, s in enumerate(shape) if s == n_freq]
 
    if not t_axes:
        raise ValueError(f"time 축(길이 {n_time})을 찾지 못함. shape={shape}")
    if not f_axes:
        raise ValueError(f"freq 축(길이 {n_freq})을 찾지 못함. shape={shape}")
 
    out = np.moveaxis(power, [t_axes[-1], f_axes[0]], [0, 1])
    if out.ndim > 2:
        out = out.mean(axis=tuple(range(2, out.ndim)))  # 편광 등 나머지 평균
    return out
 
 
beam_tf = efield_to_power(ef_raw, len(az_rad), len(freqs_hz))
print(f"beam_tf shape: {beam_tf.shape}  min/max: {beam_tf.min():.3e}/{beam_tf.max():.3e}")
 
# ─────────────────────────────────────────────
# 7. 위성 가시성 (Satellite-only visibility)
# ─────────────────────────────────────────────
# %%
def satellite_visibility(
    summary: dict,
    beam_power_tf: np.ndarray,
    attenuation: np.ndarray,
    freqs_hz: np.ndarray,
    amp: float = 1.0,
) -> np.ndarray:
    """
    V_sat(t, ν) = S(ν) · A_beam(t,ν) · A_smear(t,ν) · exp(-2πi ν τ(t))
 
    - S(ν)         : 위성 스펙트럼 (여기서는 flat)
    - A_beam       : 빔 전력 응답
    - A_smear      : 전체 스미어링 감쇠 (주파수 + 시간)
    - exp(-2πiντ)  : 기하학적 위상
    """
    tau_s  = summary["tau_s"]                                    # (Ntime,)
    phase  = np.exp(-2j * np.pi * tau_s[:, None] * freqs_hz[None, :])
 
    return amp * beam_power_tf * attenuation * phase              # (Ntime, Nfreq)
 
 
sat_vis: dict[str, np.ndarray] = {}
for name in BASELINES:
    sat_vis[name] = satellite_visibility(
        summaries[name], beam_tf, attenuation_full[name], freqs_hz
    )
 
# ─────────────────────────────────────────────
# 8. 시각화
# ─────────────────────────────────────────────
# %%
def _plot_tf(data_dict: dict[str, np.ndarray], title: str,
             cbar_label: str, transform=None, freq_mhz=None):
    n = len(data_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]
 
    vals = [transform(v) if transform else v for v in data_dict.values()]
    vmin, vmax = min(v.min() for v in vals), max(v.max() for v in vals)
 
    extent = [freq_mhz[0], freq_mhz[-1], 0, vals[0].shape[0]] if freq_mhz is not None else None
 
    for ax, (name, val) in zip(axes, zip(data_dict, vals)):
        im = ax.imshow(val, aspect="auto", origin="lower",
                       extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Frequency [MHz]" if freq_mhz is not None else "Freq index")
        ax.set_ylabel("Time index")
 
    fig.colorbar(im, ax=axes, shrink=0.95, label=cbar_label)
    fig.suptitle(title)
    plt.show()
 
 
fmhz = freqs_hz / 1e6
 
_plot_tf(attenuation_full,
         "전체 스미어링 감쇠 (주파수 + 시간)",
         "|sinc_freq · sinc_time|", freq_mhz=fmhz)
 
_plot_tf(sat_vis,
         "위성 가시성 |V_sat|",
         "|V_sat|", transform=np.abs, freq_mhz=fmhz)
 
# 시간 방향 평균 진폭
plt.figure(figsize=(8, 4))
for name, vis in sat_vis.items():
    plt.plot(np.mean(np.abs(vis), axis=1), label=name)
plt.xlabel("Time index")
plt.ylabel("Mean |V_sat| over frequency")
plt.title("Satellite-only visibility amplitude (HERA-like beam)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
