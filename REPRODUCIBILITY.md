# Reproducibility Notes

This repository is a manuscript-level reproducibility archive, not a full raw-data release. It supports numerical inspection and figure regeneration from archived CSV outputs. A complete raw-data rerun requires external HERA-like UVH5 backgrounds, native geometry NPZ files, and TLE products that are not redistributed here.

## What Is Included

Included:

- Source scripts for the active coverage/PTE robustness analysis
- The active YAML configuration
- Compact summary CSV files used for manuscript tables and figures
- Top-level coverage robustness figures
- A small `pathb/` package with shared delay-spectrum, null, satellite, and pipeline helpers

Not included:

- Raw HERA `uvh5` visibility files
- Large raw/background HDF5 products
- External native geometry files such as `examples/backgrounds_jan2026_real/native_<baseline_id>.npz`
- External TLE catalogs such as `tle/starlink_jan2026_LEO_only.tle`
- Cached null ensembles such as `outputs/coverage_null_global_stats/*.npz`
- Local virtual environments
- Exploratory or smoke-test runs

The archived CSV outputs are therefore the source of record for manuscript-level numerical inspection. Full reruns require access to the external visibility products and TLE inputs used by the local pipeline.

## Randomness And Null Policy

The active coverage configuration is `configs/coverage_robustness.yaml`.

- Coarse coverage grid: `N_null = 100`
- Near-threshold tail refinement: `N_null = 1000`
- Coarse-grid seeds are stored per row in `outputs/coverage_robustness_trials.csv`
- Trial metadata is stored in `outputs/coverage_robustness_trials.meta.json`

The row-level null diagnostics include both local maximum statistics and integrated absolute-bias diagnostics:

- `Z_PS_max`
- `PTE_global_max`
- `PTE_global_absint`
- `relative_abs_bias`

The reporting-level integrated-bias gate is `B_floor = 1e-2`, summarized in `outputs/coverage_candidate_counts_by_floor_extended.csv`.

## Minimal Inspection Workflow

Inspect the coarse-grid results:

```bash
python scripts/summarize_coverage_grid.py \
  --trials outputs/coverage_robustness_trials.csv \
  --out-dir outputs
```

Inspect the refined near-threshold summary:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv("outputs/coverage_tail_refined_near_threshold_summary.csv"))
print(pd.read_csv("outputs/coverage_candidate_counts_by_floor_extended.csv"))
PY
```

Expected high-level values:

- `coverage_robustness_trials.csv`: 648 rows
- `coverage_tail_refined_near_threshold_summary.csv`: 4 refined cases
- `n_refined_PTE_lt_001 = 0`
- `n_refined_beam_robust = 0`
- `n_full_polybeam_strict = 0`
- `B_floor = 1e-2` physical candidates in coarse grid: 0/648

## Full Rerun Order

A full rerun requires external visibility inputs, native geometry NPZ files, and TLE products. The following commands are not expected to run out-of-the-box from this archive. Use the archived TLE file from the original experiment rather than downloading a current catalog, because later orbital-element downloads can shift pass timing. With the raw inputs available at the paths configured locally, the intended order is:

```bash
python scripts/build_lst_metadata.py
python scripts/select_lst_bins.py
python scripts/run_coverage_grid.py --config configs/coverage_robustness.yaml
python scripts/summarize_coverage_grid.py --trials outputs/coverage_robustness_trials.csv --out-dir outputs
python scripts/run_coverage_tail_refined_near_threshold.py \
  --config configs/coverage_robustness.yaml \
  --trials outputs/coverage_robustness_trials.csv \
  --n-null 1000
```

The default full rerun is intentionally not advertised as a one-command reproduction because the raw HERA-like backgrounds are not redistributed in this repository.

## Active Versus Archived Materials

The active coverage/PTE analysis uses the `configs/coverage_robustness.yaml`, `scripts/run_coverage_*`, `outputs/coverage_*`, and `figures/coverage_robustness/` files.

Older morphology matched-null manuscript material has been moved to `archive/old_manuscripts/`. Older Path B morphology/beam release results, figures, and scripts have been moved to `archive/old_pathb_release/`. They are retained only to avoid silently deleting provenance and should not be cited as the active submitted analysis.
