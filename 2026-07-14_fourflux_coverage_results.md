# Four-Flux Coverage Run Summary

Run date: 2026-07-14

## Scope

- Rows: 1296
- Frozen/full paired cases: 648
- Flux tiers: 10, 30, 300, 1000 Jy
- Null draws per row: 100

## Top-Line Counts

| metric | count |
| --- | ---: |
| `n_rows` | 1296 |
| `n_pairs` | 648 |
| `n_strict_local` | 30 |
| `n_strict_absint` | 74 |
| `n_local_physical_1e3` | 11 |
| `n_absint_physical_1e3` | 57 |
| `n_beam_robust_local_physical_1e3` | 4 |
| `n_frozen_only_local_physical_1e3` | 1 |
| `n_full_only_local_physical_1e3` | 2 |
| `n_beam_robust_absint_physical_1e3` | 10 |
| `n_frozen_only_absint_physical_1e3` | 27 |
| `n_full_only_absint_physical_1e3` | 10 |
| `n_local_tail_candidates_for_n1000` | 60 |
| `n_absint_tail_candidates_for_n1000` | 126 |

## Outputs

- `fourflux_summary/coverage_candidate_counts_by_floor_fourflux.csv`
- `fourflux_summary/coverage_summary_by_flux_fourflux.csv`
- `fourflux_summary/polybeam_pair_audit_fourflux.csv`
- `fourflux_summary/coverage_tail_candidates_local_fourflux.csv`
- `fourflux_summary/coverage_tail_candidates_absint_fourflux.csv`
- `fourflux_summary/fourflux_topline_summary.csv`
