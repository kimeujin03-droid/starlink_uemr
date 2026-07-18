# Intrinsic Phase Multi-Seed Audit And Figure Plan

작성일: 2026-07-13

## 1. 목적

이 추가 실험의 핵심 질문은 다음과 같다.

> coherent-default 다중 위성 합산에서 나타난 window-integrated 초과가 특정 상대 위상 선택의 우연한 결과인가, 아니면 위성별 고유 위상을 여러 번 바꾸어도 반복되는 반응인가? 또한 그 반응은 frozen PolyBeam에서 full chromatic PolyBeam으로 바꾸어도 유지되는가?

현재 결과는 다음 상태다.

- 확장된 `PTE_global,absint` 꼬리 선택 규칙으로 고른 coarse-grid seed row: 7개
- frozen/full PolyBeam 짝을 완성한 최종 재평가 행: 10개
- `N_null = 1000` coherent-default 재평가에서 strict integrated + `B_rel > 1e-3`: 3개
- 위 3개는 모두 `frozen_polybeam` 전용이고, paired `full_polybeam`에서는 strict integrated gate를 통과하지 않음
- 단일 random intrinsic phase 실현에서는 strict local, strict integrated, beam-robust integrated 후보가 모두 0개

따라서 새 계산은 기존 단일 random intrinsic phase check를 `M = 100`개의 위성별 고유 위상 실현으로 확장하는 것이다. 판정 계층 도식, 대표 지연 프로파일, 빔-위상 비교 그림은 이 감사 결과와 기존 산출물을 재사용해 만든다.

## 2. 고정 분석 집합

새 사례를 다시 고르지 않는다. 기존 확장 적분 꼬리에서 선택된 10개 paired-beam 재평가 행을 그대로 사용한다.

기준 파일:

- coherent-default 10-row 결과: `coverage_absint_tail_refined_pte003_brel1e3.csv`
- 단일 random intrinsic phase check: `coverage_absint_tail_refined_pte003_brel1e3_random_phase.csv`
- 기존 요약 문서: `2026-07-09_absint_tail_phase_audit.md`
- 관련 실행 스크립트: `scripts/run_coverage_tail_refined_near_threshold.py`

분석 집합의 고정 조건:

- baseline, LST bin, morphology, flux, multiplicity를 기존 10개 행과 동일하게 유지
- `N_null = 1000`
- `frozen_polybeam`과 `full_polybeam`을 모두 평가
- 같은 physical case의 두 beam row는 같은 위성 목록, 같은 phase seed, 같은 null seed policy, 같은 background realization을 사용

## 3. 고유 위상 모형

위상 시드 `m`에서 위성 `s`의 intrinsic phase를 독립적으로 추출한다.

```text
theta_s^(m) ~ Uniform(0, 2*pi)
```

주입 visibility는 다음 형태다.

```text
V_sat^(m)(t, nu)
  = sum_s A_s(t, nu) exp[-2*pi*i*nu*tau_s(t) + i*theta_s^(m)]
```

`theta_s^(m)`는 한 injection realization 안에서 시간과 주파수에 대해 고정한다.

위상 배정은 위성 순서에 의존하지 않도록 위성 ID에 연결한다.

```text
theta_s^(m) = phase_prng(global_phase_seed = m, satellite_ID = s)
```

같은 phase seed 안에서는 반드시 다음 항목을 paired로 유지한다.

- frozen beam과 full beam에 동일한 `theta_s^(m)` 사용
- 동일한 satellite ID 목록 사용
- 동일한 background realization 사용
- 동일한 time-frequency support 사용
- matched-null 생성에 동일한 null seed stream 사용

이 짝지음이 유지되어야 frozen/full 차이를 beam operator 차이로 해석할 수 있다.

## 4. 대응 null

각 intrinsic phase realization 주변에서 matched null을 새로 만든다. coherent-default에서 생성한 null ensemble은 재사용하지 않는다.

```text
V_sat,null^(m,r)
  = sum_s A_s(t, nu)
      exp[-2*pi*i*nu*tau_s(t) + i*theta_s^(m) + i*phi_s^(m,r)]

phi_s^(m,r) ~ Uniform(0, 2*pi)
```

여기서 `m`은 intrinsic phase seed, `r`은 null draw index다.

## 5. 실행 규모

권장 본 실행:

```text
10 paired rows x 100 intrinsic phase seeds x 1000 nulls
= 1,000,000 matched-null evaluations
```

실행 순서:

1. 기존 10개 paired rows와 설정을 고정한다.
2. coherent-default 결과가 `coverage_absint_tail_refined_pte003_brel1e3.csv`와 일치하는지 재현한다.
3. `M = 5`, `N_null = 1000` smoke test를 수행한다.
4. `M = 100`, `N_null = 1000` 본 실행을 수행한다.
5. phase별 요약표와 beam mismatch rate를 계산한다.
6. 사전 정의된 규칙으로 대표 delay-profile 사례를 선택한다.
7. Figure 1-3 및 summary table을 생성한다.
8. 전체 phase seed 결과와 seed policy를 공개 산출물로 남긴다.

## 6. Trial-Level 저장 열

주 출력 파일:

```text
outputs/intrinsic_phase_multiseed_trials.csv
```

권장 열:

```text
case_id
paired_case_id
phase_seed
phase_condition
satellite_count
satellite_ids
beam_model
morphology
flux_jy
multiplicity
baseline_id
baseline_length_m
lst_stratum
lst_bin_id
N_null
pte_global_max
pte_global_absint
relative_abs_bias
t_max
t_absint
z_ps_max
n_null_exceed_max
n_null_exceed_absint
strict_local
strict_integrated
physical_1e3
physical_1e2
integrated_physical_1e3
integrated_physical_1e2
margin_integrated
margin_local
margin_bias_1e3
margin_bias_1e2
final_class
intrinsic_phases_rad
null_seed_base
```

권장 margin 정의:

```text
margin_integrated = -log10(PTE_global_absint) - 2
margin_local      = -log10(PTE_global_max) - 2
margin_bias_1e3   = log10(B_rel / 1e-3)
margin_bias_1e2   = log10(B_rel / 1e-2)
```

## 7. Summary 산출물

권장 파일:

```text
outputs/intrinsic_phase_multiseed_summary.csv
outputs/intrinsic_phase_beam_pairs.csv
outputs/intrinsic_phase_class_counts.csv
outputs/intrinsic_phase_coherent_percentiles.csv
```

`intrinsic_phase_multiseed_summary.csv`는 `case_id x beam_model` 단위로 다음을 저장한다.

- `n_phase_seeds`
- `n_strict_local`
- `n_strict_integrated`
- `n_physical_1e3`
- `n_physical_1e2`
- `n_integrated_physical_1e3`
- `n_integrated_physical_1e2`
- 각 비율의 Wilson 또는 Clopper-Pearson 95% confidence interval
- coherent-default의 `T_absint`, `PTE_absint`, `B_rel` percentile

`intrinsic_phase_beam_pairs.csv`는 같은 physical case와 같은 phase seed의 frozen/full row를 한 줄로 묶는다.

필수 열:

```text
paired_case_id
phase_seed
morphology
flux_jy
multiplicity
baseline_id
lst_bin_id
pte_absint_frozen
pte_absint_full
brel_frozen
brel_full
strict_integrated_frozen
strict_integrated_full
integrated_physical_1e3_frozen
integrated_physical_1e3_full
beam_robust_integrated_1e3
frozen_only_integrated_1e3
full_only_integrated_1e3
delta_beam_logpte
```

여기서

```text
delta_beam_logpte = log10(PTE_absint_full / PTE_absint_frozen)
```

양수이면 frozen beam이 더 극단적인 integrated PTE를 갖는다.

## 8. 주요 평가량

### 8.1 위상별 strict integrated rate

각 case와 beam에서 다음을 계산한다.

```text
pi_I(1e-3)
  = mean_m[ PTE_global_absint^(m) < 0.01 and B_rel^(m) > 1e-3 ]

pi_I(1e-2)
  = mean_m[ PTE_global_absint^(m) < 0.01 and B_rel^(m) > 1e-2 ]
```

### 8.2 Beam-robust integrated rate

같은 phase seed에서 두 beam이 동시에 strict integrated + physical gate를 통과한 비율을 계산한다.

```text
pi_beam
  = mean_m[ I_frozen^(m) = 1 and I_full^(m) = 1 ]
```

이 값이 이번 추가 실험의 핵심 결과다.

### 8.3 Beam mismatch rate

```text
pi_mismatch
  = mean_m[ I_frozen^(m) != I_full^(m) ]
```

다음 두 방향을 따로 저장한다.

- frozen-only integrated physical excess
- full-only integrated physical excess

### 8.4 Coherent-default의 위상 분포 내 위치

coherent-default의 `T_absint`, `PTE_global_absint`, `B_rel`이 random intrinsic phase 100개 중 어느 percentile에 위치하는지 계산한다.

보고 예시:

```text
The coherent-default T_absint lies at the 98th percentile of the sampled
intrinsic-phase distribution for the frozen smooth multi-satellite case.
```

이 값은 coherent-default를 일반적 물리 예측이 아니라 결맞음 상한 stress configuration으로 해석하는 근거가 된다.

## 9. Figure 1: 판정 계층 도식

출력 파일:

```text
figures/F1_parallel_hierarchy.pdf
```

도식 구조:

```text
HERA-like background + TLE-based UEMR injection
                         |
                         v
                   QA operator
        weighted delay transform + window mask
                         |
          +--------------+--------------+
          |                             |
   Local branch L                Integrated branch I
 PTE_global,max                  PTE_global,absint
      < 0.01                          < 0.01
          |                             |
          +--------------+--------------+
                         |
              Physical-amplitude gate
                B_rel > B_floor
                         |
                         v
         Paired beam robustness assessment
           frozen vs full chromatic beam
                         |
                         v
         Calibration-residual stability audit
                         |
                         v
             Classification and reporting
```

도식에 반드시 넣을 문장:

```text
Local and integrated branches are diagnostically parallel, but final
integrated-contamination interpretation requires significance in the
integrated branch.
```

최종 class는 다음 네 갈래로 분리한다.

- exploratory QA flag
- local-branch physical candidate
- beam-sensitive integrated or composite candidate
- beam-robust integrated contamination candidate

## 10. Figure 2: 대표 지연 프로파일

출력 파일:

```text
figures/F2_representative_delay_profiles.pdf
```

목적은 다음 세 현상을 숫자가 아니라 delay-space profile로 보여 주는 것이다.

- local spike만 큰 경우
- 넓은 delay 영역에 약한 integrated excess가 분산된 경우
- beam model을 바꾸면 excess가 사라지는 경우

사례 선택은 사전에 고정한다.

### Case A: local-only significant case

catalog-complete 조건의 `(L+, I-)` physical candidate 중 `PTE_global,max`가 가장 작은 행을 선택한다.

현재 후보군에서는 kHz-comb single-satellite local candidate가 대표 예시가 될 수 있다. 최종 선택은 `outputs/full_catalog_physical_candidate_audit.csv`에서 규칙으로 재계산한다.

### Case B: integrated-only significant case

coherent-default 확장 꼬리 10-row 결과 중 `PTE_global,absint`가 가장 작은 frozen-beam 행을 선택한다.

현재 기준 파일에서는 다음 세 행이 후보군이다.

- frozen smooth multi: `PTE_global,absint = 0.005994`, `B_rel = 0.001295`
- frozen lines multi: `PTE_global,absint = 0.007992`, `B_rel = 0.001295`
- frozen khz_comb multi: `PTE_global,absint = 0.007992`, `B_rel = 0.001154`

따라서 현재 deterministic tie-break가 없으면 frozen smooth multi가 우선 선택된다.

### Case C: beam-sensitive case

paired beam comparison에서 final grade 또는 integrated physical gate가 불일치한 사례 중 다음 값이 가장 큰 행을 선택한다.

```text
abs(log10(PTE_absint_frozen) - log10(PTE_absint_full))
```

### 패널 구성

3행 x 2열:

```text
                         signed delay profile     cumulative absolute bias
Local-only                       A1                         A2
Integrated-only                  B1                         B2
Beam-sensitive                   C1                         C2
```

왼쪽 패널:

```text
Delta P_norm(tau)
  = (|V_inj_tilde(tau)|^2 - |V_bg_tilde(tau)|^2)
    / sum_{tau in W_win} |V_bg_tilde(tau)|^2
```

오른쪽 패널:

```text
C(tau)
  = sum_{tau_win <= |tau'| <= |tau|} |Delta P(tau')|
    / sum_{tau' in W_win} |V_bg_tilde(tau')|^2
```

표시 요소:

- horizon delay `+/- tau_hor`
- window boundary `+/- tau_win`
- observed injection profile
- null median
- null 68% and 95% bands
- 필요 시 signed profile y-axis는 `symlog`

## 11. Figure 3: Phase-Beam Decision Plane

출력 파일:

```text
figures/F3_phase_beam_decision_plane.pdf
figures/F4_phase_class_frequencies.pdf
```

분석 대상은 coherent-default에서 strict integrated excess가 있었던 세 morphology다.

- smooth multi
- lines multi
- khz_comb multi

모두 같은 physical cell이다.

- baseline: `11_10`
- LST stratum: `quiet`
- LST bin: `16`
- flux: `1000 Jy`
- multiplicity: `multi`
- beams: frozen and full PolyBeam

### Panel A: PTE-bias decision plane

x축:

```text
log10(B_rel)
```

y축:

```text
-log10(PTE_global,absint)
```

Threshold lines:

- horizontal: `-log10(0.01) = 2`
- vertical: `log10(1e-3) = -3`
- vertical: `log10(1e-2) = -2`

오른쪽 위 영역이 statistical gate와 physical-amplitude gate를 동시에 통과하는 영역이다.

Encoding:

- small points: random intrinsic phase seeds
- large marker: coherent-default
- marker shape: frozen vs full beam
- facets: smooth, lines, khz_comb
- same phase seed의 frozen/full points는 가는 선으로 연결

### Panel B: paired beam difference

각 phase seed에서 다음을 계산한다.

```text
Delta_beam = log10(PTE_absint_full / PTE_absint_frozen)
```

해석:

- `Delta_beam = 0`: 두 beam의 integrated PTE가 같음
- `Delta_beam > 0`: frozen beam이 더 극단적
- `Delta_beam < 0`: full chromatic beam이 더 극단적

형태학별 boxplot 또는 violin plot으로 표시한다.

### Panel C: class frequencies

형태학별 100개 phase realization에서 다음 class를 누적 막대로 표시한다.

- neither strict
- local-only strict
- integrated-only strict
- both local and integrated strict
- frozen-only integrated physical
- full-only integrated physical
- beam-robust integrated physical

발표용으로는 Panel C를 별도 `F4_phase_class_frequencies.pdf`로 분리해도 된다.

## 12. 본문 Table

출력 파일:

```text
outputs/intrinsic_phase_multiseed_summary.csv
```

논문용 table skeleton:

```latex
\begin{table*}
\centering
\caption{Multi-satellite intrinsic-phase 100-seed audit.}
\begin{tabular}{llrrrrrr}
\toprule
Case & Beam &
Phase seeds &
Strict local &
Strict integrated &
$B_{\rm rel}>10^{-3}$ &
$B_{\rm rel}>10^{-2}$ &
Beam-robust integrated \\
\midrule
Smooth broadband & Frozen/full & 100 & ... & ... & ... & ... & ... \\
Narrowband lines & Frozen/full & 100 & ... & ... & ... & ... & ... \\
kHz comb & Frozen/full & 100 & ... & ... & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table*}
```

## 13. 해석 규칙

### 결과 1: 100개 phase seed에서 beam-robust integrated candidate가 0개

사용 문장:

```text
Across 100 independent per-satellite intrinsic-phase realizations, no strict
integrated candidate was retained simultaneously under the paired frozen and
full chromatic beam models. The coherent-default integrated excesses are
therefore interpreted as phase-sensitive stress responses under the sampled
phase model, not as repeatedly reproduced beam-robust contamination candidates.
```

금지할 표현:

- impossible
- phase-independent absence
- ruled out for all phases

### 결과 2: 일부 phase에서 frozen-only strict integrated candidate가 재현됨

사용 문장:

```text
Some random intrinsic-phase realizations reproduce a strict integrated excess
under the frozen beam, but the excess is not retained under the full chromatic
beam. Candidate classification is therefore controlled by the interaction
between phase realization and beam approximation.
```

### 결과 3: 일부 phase에서 beam-robust integrated candidate가 발견됨

이 경우 해당 phase seed는 positive follow-up target으로 승격한다.

후속 작업:

- `N_null = 10000` tail refinement
- calibration-residual stability audit
- representative delay profile
- satellite leave-one-out contribution test
- partial-coherence time model

## 14. 공개 저장소 부록 산출물

부록 또는 공개 저장소에 다음을 남긴다.

- 10개 행 전체의 100-seed trial table
- 모든 phase별 delay profile 또는 profile array
- phase seed 목록
- satellite ID별 `theta_s`
- null seed policy
- coherent-default percentile table
- Wilson 또는 Clopper-Pearson confidence intervals
- full class transition table

권장 최종 파일 구조:

```text
outputs/
  intrinsic_phase_multiseed_trials.csv
  intrinsic_phase_multiseed_summary.csv
  intrinsic_phase_beam_pairs.csv
  intrinsic_phase_class_counts.csv
  intrinsic_phase_coherent_percentiles.csv

figures/
  F1_parallel_hierarchy.pdf
  F2_representative_delay_profiles.pdf
  F3_phase_beam_decision_plane.pdf
  F4_phase_class_frequencies.pdf
```

## 15. 구현 메모

현재 `scripts/run_coverage_tail_refined_near_threshold.py`는 `--intrinsic-phase-mode random_per_satellite`와 `--intrinsic-phase-seed`를 받아 단일 phase realization을 만들 수 있다. 100-seed 감사에는 다음 확장이 필요하다.

1. 10개 paired rows를 한 번 선택하고 고정한다.
2. `phase_seed` loop를 추가한다.
3. 같은 physical case의 frozen/full row가 동일한 satellite-ID keyed phase map을 공유하게 한다.
4. 각 phase seed마다 matched null을 새로 생성하되, paired beam 비교가 가능하도록 null seed stream을 deterministic하게 묶는다.
5. delay-profile plotting에 필요한 observed profile과 null quantile을 선택 case에 대해 저장한다.

기존 단일 phase check는 row index에 seed offset을 더하는 방식이므로, 100-seed paired-beam 감사에서는 위성 ID 기반 phase map으로 바꾸는 것이 더 안전하다. 그래야 satellite ordering 변경이 결과를 바꾸지 않는다.
