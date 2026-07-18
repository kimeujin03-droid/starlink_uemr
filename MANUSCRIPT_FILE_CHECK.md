# Manuscript/File Check

This folder is arranged as the GitHub upload root for the active manuscript release.

## Paper Dependencies

Checked against `paper/paper.tex`.

| Dependency type | Status |
| --- | --- |
| Manuscript source | `paper/paper.tex` present |
| Main bibliography | `starlink_uemr_refs.bib` present |
| Statistical bibliography additions | `starlink_uemr_stat_refs_additions.bib` present |
| Citation keys | 15 cited keys, 0 missing |
| Included figures | 2 included figures, 0 missing |

Included manuscript figures:

- `figures/coverage_robustness/R2_z_vs_pte_global.png`
- `figures/coverage_robustness/R3_bias_floor_sensitivity.png`

## Numerical Claim Check

| Manuscript claim | Source file | Checked value |
| --- | --- | --- |
| Controlled grid rows | `results/revision_bandpower_propagation_v3_pte/bandpower_propagation_detail.csv` | 600 |
| Controlled `PTE_global,max < 0.01` candidates | same | 0 |
| Controlled minimum `PTE_global,max` | same | 0.0198019801980198 |
| Coverage grid rows | `outputs/coverage_robustness_trials.csv` | 648 |
| Coverage exploratory flags | same | 16 |
| Coarse `PTE_global,max < 0.01` trials | same | 1 |
| Coarse `PTE_global,max < 0.01` and `B_rel > 1e-3` trials | same | 0 |
| Nominal expected false positives at alpha=0.01 | same | 6.48 |
| Selected baseline/LST cells | `outputs/lst_bin_selection.csv` | 27 |
| Selection items changed by full-bin metadata correction | old vs corrected selection | 15 |
| Tail-refined cases after corrected selection | `outputs/coverage_tail_refined_near_threshold.csv` | 4 |
| Tail-refined strict local-max candidates | same | 0 |
| Full-polybeam strict candidates | `outputs/coverage_tail_refined_near_threshold_summary.csv` | 0 |
| Beam-robust strict candidates | same | 0 |
| Frozen-only strict candidates | same | 0 |
| Metadata audit columns in coverage CSV | `outputs/coverage_robustness_trials.csv` | present |

## Note On Bias-Floor Tables

`outputs/coverage_candidate_counts_by_floor_extended.csv` summarizes the corrected coarse 100-null coverage grid. It reports 1 statistical candidate at very low bias floors, but 0 candidates at `B_floor >= 1e-3`. The manuscript's final strict-candidate counts use `outputs/coverage_tail_refined_near_threshold_summary.csv`, where the 1000-null refinement gives 0 strict local-max candidates.

## Compile Note

Compile from the repository root:

```bash
lualatex -interaction=nonstopmode paper/paper.tex
bibtex paper
lualatex -interaction=nonstopmode paper/paper.tex
lualatex -interaction=nonstopmode paper/paper.tex
```
