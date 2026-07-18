# Intrinsic Phase Multi-Seed Audit Results

Run date: 2026-07-13

## Scope

- Input case table: `coverage_absint_tail_refined_pte003_brel1e3.csv`
- Phase seeds: 5
- Null draws per row/phase: 1000
- Trial rows: 50
- Paired frozen/full phase comparisons: 25
- Phase seed base: `2026071300`

## Top-Line Counts

| metric | count |
| --- | ---: |
| strict integrated rows | 1 |
| strict integrated + `B_rel > 1e-3` rows | 1 |
| paired beam-robust integrated + `B_rel > 1e-3` phase cases | 0 |
| frozen-only integrated + `B_rel > 1e-3` phase cases | 0 |
| full-only integrated + `B_rel > 1e-3` phase cases | 1 |

No row passed the strict local branch in this smoke test.

The only strict integrated + physical row was:

| paired case | beam | phase seed | `PTE_global,absint` | `B_rel` | beam pairing |
| --- | --- | ---: | ---: | ---: | --- |
| `11_10_lst016_quiet_smooth_S1000_single` | full PolyBeam | 2026071303 | 0.003996 | 0.003024 | full-only |

This row is not one of the three coherent-default multi-satellite integrated
tail rows and is not beam robust. Its paired frozen-beam row at the same phase
seed has `PTE_global,absint = 0.096903` and `B_rel = 0.000891`.

## Main Multi-Satellite Tail Cases

The three coherent-default frozen-beam integrated rows did not reproduce a
strict integrated excess in any of the five random intrinsic phase seeds.

| morphology | beam | strict integrated | integrated + `B_rel>1e-3` | note |
| --- | --- | ---: | ---: | --- |
| smooth multi | frozen | 0/5 | 0/5 | `B_rel>1e-3` in 5/5, but PTE not strict |
| lines multi | frozen | 0/5 | 0/5 | `B_rel>1e-3` in 5/5, but PTE not strict |
| kHz-comb multi | frozen | 0/5 | 0/5 | `B_rel>1e-3` in 3/5, but PTE not strict |
| smooth multi | full | 0/5 | 0/5 | `B_rel>1e-3` in 5/5, but PTE not strict |
| lines multi | full | 0/5 | 0/5 | `B_rel>1e-3` in 5/5, but PTE not strict |
| kHz-comb multi | full | 0/5 | 0/5 | `B_rel>1e-3` in 5/5, but PTE not strict |

For these three morphology-matched multi-satellite cases, the coherent-default
`T_absint`, `-log10(PTE_absint)`, and `B_rel` all sit at the 100th percentile
relative to the five sampled random intrinsic phases, for both frozen and full
beam rows. With only five phase seeds this is a smoke-test diagnostic, not a
rate estimate.

## Beam Pair Summary

Across the 25 paired frozen/full phase comparisons:

- beam-robust integrated physical cases: 0
- frozen-only integrated physical cases: 0
- full-only integrated physical cases: 1

The median paired beam difference,

```text
Delta_beam = log10(PTE_absint_full / PTE_absint_frozen)
```

was small for most paired cases, but individual phase seeds showed large
movement. For example, the smooth multi case at phase seed 2026071303 moved
from frozen `PTE_absint = 0.855145` to full `PTE_absint = 0.073926`, still not
strict. The single smooth full-only hit at phase seed 2026071303 moved from
frozen `PTE_absint = 0.096903` to full `PTE_absint = 0.003996`.

## Outputs

- `intrinsic_phase_multiseed_trials_smoke5.csv`
- `intrinsic_phase_multiseed_summary_smoke5.csv`
- `intrinsic_phase_beam_pairs_smoke5.csv`
- `intrinsic_phase_class_counts_smoke5.csv`
- `intrinsic_phase_coherent_percentiles_smoke5.csv`

## Interpretation

The smoke test confirms that the new multi-seed pipeline runs end-to-end and
produces the intended trial, summary, paired-beam, class-count, and coherent
percentile tables. In the five sampled random intrinsic phases, the original
coherent-default multi-satellite frozen-beam integrated excesses do not recur.
The one strict integrated physical hit is full-beam-only, single-satellite, and
not beam robust. Because `M=5` is only a smoke test, these counts should not be
used as final occurrence-rate estimates.
