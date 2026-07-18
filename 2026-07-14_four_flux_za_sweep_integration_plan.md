# Four-Flux And Zenith-Angle Sweep Integration Plan

작성일: 2026-07-14

## 1. 결정 사항

다음 full-catalog coverage run에서는 다음 네 개의 주입 플럭스 계층을 사용한다.

```text
S_ref = {10, 30, 300, 1000} Jy
```

이 설계는 문헌 규모 anchor 두 개와 deliberate stress tier 두 개를 분리하기 위한 것이다.

- `10 Jy`: first-generation broadband UEMR 문헌 규모의 high-end anchor
- `30 Jy`: v2-Mini broadband UEMR 문헌 규모의 lower/high anchor
- `300 Jy`: QA operator dynamic-range stress tier
- `1000 Jy`: failure-boundary stress tier

기존 648-row 결과는 2-flux grid 결과로만 유지한다. 이를 four-flux 결과로 재명명하거나 재해석하지 않는다.

## 2. Grid 크기 변화

기존 coverage grid:

```text
27 baseline-LST cells
x 2 beam models
x 3 morphologies
x 2 flux tiers
x 2 multiplicity states
= 648 rows
```

새 four-flux coverage grid:

```text
27 baseline-LST cells
x 2 beam models
x 3 morphologies
x 4 flux tiers
x 2 multiplicity states
= 1296 rows
```

따라서 다음 산출물은 모두 새로 재생성해야 한다.

- candidate counts
- tail refinement target list
- `N_null=1000` tail refinement
- paired-beam audit
- multiplicity/bootstrap summary
- Type II ANOVA / descriptive variance decomposition
- candidate-level calibration audit
- candidate-level Doppler or phase follow-ups, if triggered

## 3. Replacement Paragraph: Flux Tiers

원고의 flux tier 설명은 다음 문단으로 교체한다.

```text
S_ref is not a calibrated prediction of the HERA-apparent flux of a particular
Starlink transit, but an order-of-magnitude stress-scaling parameter. Published
LOFAR measurements place first-generation broadband UEMR at roughly 0.1--10 Jy
and second-generation v2-Mini broadband UEMR at roughly 2--100 Jy in two 8-MHz
windows near 120 and 161 MHz. We therefore use 10 and 30 Jy as
literature-scale low/high anchors. The 300 and 1000 Jy levels are deliberate
stress tiers used to probe the dynamic range and failure boundary of the QA
operator; they are not occurrence-rate or calibrated-flux claims. Beam
coupling, slant range, polarization, off-axis response, and emission
directionality are absorbed into the interpretation of S_ref as a stress
coordinate rather than a direct observable.
```

## 4. Important Compatibility Note

기존 648-row 결과와 새 four-flux 결과를 섞지 않는다.

명시할 문장:

```text
The existing 648-row coverage result used two flux tiers and is retained as a
historical two-flux analysis. The four-flux design expands the paired coverage
grid to 1296 rows, so candidate counts, tail refinement, paired-beam audits,
multiplicity summaries, ANOVA tables, and candidate-level follow-up audits are
regenerated rather than inherited from the 648-row run.
```

## 5. Config 변경 사항

coverage run config의 flux grid는 다음과 같아야 한다.

```yaml
flux_jy:
  - 10
  - 30
  - 300
  - 1000
```

PathB-style experiment config의 flux grid도 다음으로 맞춘다.

```yaml
experiment:
  flux_grid_jy: [10, 30, 300, 1000]
```

주의:

- `100 Jy`는 이번 four-flux integration design에서 제외한다.
- 기존 `30, 1000 Jy` 산출물을 four-flux 결과로 확장 해석하지 않는다.
- `300 Jy`는 문헌 anchor가 아니라 stress tier다.

## 6. ANOVA 업데이트

기존 ANOVA 문구와 스크립트는 2-level flux grid를 전제로 했다.

새 설계에서는 다음 원칙을 적용한다.

- `flux_jy`는 main Type II ANOVA에서 categorical factor로 유지한다.
- 4개 flux level이 있으므로 “2개 점이라 dose-response shape를 식별할 수 없다”는 기존 설명은 제거한다.
- 보조적으로 `log10(B_rel) ~ log10(flux_jy)` slope를 descriptive scaling diagnostic으로 보고할 수 있다.
- slope는 candidate gate의 근거가 아니라 flux scaling sanity check로만 사용한다.

권장 문구:

```text
Flux is modeled as a categorical factor in the Type II variance decomposition
to avoid making the candidate hierarchy depend on a specific dose-response
shape. A separate log-flux slope diagnostic is reported only as a descriptive
scaling check.
```

## 7. Auxiliary ZA-Sweep Subsection

원고에는 다음 보조 분석 subsection을 추가한다.

### Synthetic Zenith-Angle Sensitivity Analysis

```text
To isolate geometric sensitivity from the heterogeneous TLE ensemble, we
perform an auxiliary synthetic-orbit sweep at azimuth 90 degrees and altitude
550 km. The peak zenith angle is varied over 5, 15, 30, 45, 60, 70, and
78 degrees. Three representative east-west baselines (14.6, 140.4, and
207.3 m), five background realizations, and two beam conditions are crossed.
The primary beam condition is the full-chromatic HERA H2C CST PolyBeam; B=1 is
retained only as a geometry-only diagnostic comparator and is not used as a
physical candidate gate. Holding S_ref=300 Jy fixed isolates the angular
response, yielding 5 x 3 x 7 x 2 = 210 cases. Every case is evaluated with the
same PTE_global,max, PTE_global,absint, and B_rel hierarchy used in the main
analysis.
```

## 8. ZA-Sweep Compatibility Note

반드시 포함할 주의 문구:

```text
The older 210-case zenith-angle sweep used a null-p95 excess metric rather
than the present PTE/B_rel candidate hierarchy. Its design can be reused, but
its numerical exceedance counts should not be pasted into the current paper
without rerunning the cases under the current operator and null definition.
```

## 9. ZA-Sweep Design Matrix

```text
background realizations: 5
baselines: 3
  - 14.6 m
  - 140.4 m
  - 207.3 m
peak zenith angles: 7
  - 5 deg
  - 15 deg
  - 30 deg
  - 45 deg
  - 60 deg
  - 70 deg
  - 78 deg
beam conditions: 2
  - full-chromatic HERA H2C CST PolyBeam
  - B=1 geometry-only diagnostic comparator
S_ref: 300 Jy
total cases: 5 x 3 x 7 x 2 = 210
```

## 10. Required Rerun Order

권장 실행 순서:

1. coverage config를 four-flux grid로 고정한다.
2. full-catalog coverage grid를 1296 rows로 재실행한다.
3. factor summary와 candidate count table을 새로 만든다.
4. paired frozen/full beam audit을 새 1296-row table에서 다시 계산한다.
5. `PTE_global,max` 및 `PTE_global,absint` 기준 tail candidates를 새로 선택한다.
6. 선택된 candidates를 `N_null=1000`으로 tail refinement한다.
7. `B_rel` floor별 candidate table을 다시 계산한다.
8. Type II ANOVA를 새 1296-row table에서 다시 실행한다.
9. 새 candidate set에 대해서 calibration residual audit을 다시 수행한다.
10. ZA-sweep 210 cases를 현재 PTE/B_rel hierarchy로 재실행한다.

## 11. Output File Naming

기존 648-row 산출물과 섞이지 않도록 four-flux prefix를 붙인다.

권장 파일명:

```text
outputs/coverage_robustness_trials_fourflux.csv
outputs/coverage_robustness_trials_fourflux.meta.json
outputs/coverage_summary_by_factor_fourflux.csv
outputs/coverage_candidate_counts_by_floor_fourflux.csv
outputs/polybeam_pair_audit_fourflux.csv
outputs/coverage_tail_refined_fourflux.csv
outputs/coverage_tail_refined_fourflux_summary.csv
outputs/coverage_anova_typeII_fourflux.csv
outputs/coverage_anova_typeII_fourflux.meta.json
outputs/targeted_candidate_calibration_audit_fourflux.csv

outputs/za_sweep_current_operator.csv
outputs/za_sweep_current_operator_summary.csv
outputs/za_sweep_current_operator.meta.json
```

## 12. Interpretation Rules

새 four-flux 결과가 나오기 전에는 다음처럼 제한해서 쓴다.

```text
The two-flux 648-row analysis is a completed historical coverage audit. The
four-flux grid is a new integration run designed to separate literature-scale
anchors from deliberate stress tiers; all downstream candidate counts and
follow-up audits are regenerated for that design.
```

ZA-sweep에 대해서는 다음처럼 쓴다.

```text
The synthetic zenith-angle sweep is an auxiliary geometry-isolation analysis.
It is not a replacement for the TLE-based full-catalog coverage grid, and the
B=1 branch is a geometry-only comparator rather than a physical candidate gate.
```
