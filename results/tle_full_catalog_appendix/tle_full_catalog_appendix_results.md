# Full TLE Catalog Appendix Result

Date: 2026-06-22

Inputs:
- TLE catalog: `tle/starlink_jan2026_LEO_only.tle`
- LST/bin selection: `outputs/lst_bin_selection.csv`
- Configuration: `configs/coverage_robustness.yaml`
- Visibility criterion: altitude >= 70 deg
- Time sampling: 16 samples per 10-minute bin
- Grid size: 27 baseline-LST cells

Generated files:
- `tle_subset_sensitivity.csv`
- `tle_subset_sensitivity_summary.csv`
- `tle_subset_sensitivity.meta.json`

## B-floor Anchor

HERA Phase I reported a 95% upper limit of `Delta^2_21 <= (30.76 mK)^2` at `k = 0.192 h Mpc^-1` and `z = 7.9` (Abdurashidova et al. 2022; arXiv: https://arxiv.org/abs/2108.02263). If the operational relative bias gate `B_floor = 10^-2` is only used as a scale anchor against that power, it corresponds to an rms temperature scale of `sqrt(10^-2) * 30.76 mK = 3.08 mK`; this is a comparison anchor, not a HERA systematic-error allocation.

## Appendix Table

The full 6364-record catalog increases satellite occupancy relative to the deterministic first-1200 subset, but this check is an exposure/coverage sensitivity diagnostic only; it does not rerun the full null-calibrated window-power grid.

| TLE subset | Records | Cells | Mean visible satellites | Max visible satellites | Mean beam-weighted exposure | p95 beam-weighted exposure | Max beam-weighted exposure | Mean max beam | Max max beam |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first1200 | 1200 | 27 | 2.815 | 4 | 0.114 | 0.506 | 0.916 | 0.094 | 0.535 |
| full catalog | 6364 | 27 | 18.778 | 26 | 1.950 | 3.511 | 3.870 | 0.748 | 0.972 |

Relative to first1200, the full catalog changes the mean visible-satellite count by `+15.963`, the mean beam-weighted exposure by `+1.836`, the p95 exposure by `+3.005`, and the maximum exposure by `+2.954`.

## Random-1200 Context

The five random 1200-record subsets give mean visible counts of 2.48--4.33 and mean beam-weighted exposures of 0.156--0.426. Thus the deterministic first1200 subset is within the random-subset occupancy range for visible counts but near the low end of beam-weighted exposure, while the full catalog is much higher because it contains all available Starlink-like records.

## Ready-to-Paste LaTeX

```tex
\begin{table}[t]
\centering
\caption{Sensitivity of the selected baseline--LST cells to the TLE catalog size. The comparison uses the same 27 baseline--LST cells, 16 time samples per 10-minute bin, and an altitude threshold of $70^\circ$. The quantities are exposure/coverage diagnostics only and do not replace the null-calibrated window-power tests.}
\label{tab:tle_full_catalog_appendix}
\begin{tabular}{lrrrrrrrr}
\hline
TLE subset & Records & Cells & Mean vis. & Max vis. & Mean exp. & p95 exp. & Max exp. & Max beam \\
\hline
first1200 & 1200 & 27 & 2.815 & 4 & 0.114 & 0.506 & 0.916 & 0.535 \\
full catalog & 6364 & 27 & 18.778 & 26 & 1.950 & 3.511 & 3.870 & 0.972 \\
\hline
\end{tabular}
\end{table}
```

Suggested appendix sentence:

```tex
As a physical scale reference for the reporting gate, the HERA Phase I limit $\Delta^2_{21}\le(30.76\,{\rm mK})^2$ at $k=0.192\,h\,{\rm Mpc}^{-1}$ and $z=7.9$ implies that a purely relative $B_{\rm floor}=10^{-2}$ corresponds to an rms scale of $0.1\times30.76=3.08\,{\rm mK}$ when anchored to that power; this comparison is only a scale anchor and is not a HERA systematic-error budget.
```
