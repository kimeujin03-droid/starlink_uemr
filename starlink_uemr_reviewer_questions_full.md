# Starlink 유사 UEMR / HERA-like 21-cm Delay Spectrum QA 논문 리뷰어 질문집

**대상 원고:** *Starlink 유사 UEMR의 21-cm 지연 스펙트럼 QA에서 국소 최댓값 초과와 윈도우 적분 밴드파워 오염의 불일치 검증*  
**예상 저널:** Astronomy and Computing 1순위, Radio Science 2순위 가능  
**예상 판정:** Major revision before publication  
**핵심 성격:** 실제 Starlink--HERA 동시 관측 검출 논문이 아니라, Starlink-like UEMR forward-injection을 이용해 21-cm delay-spectrum QA에서 local statistic, window-integrated statistic, absolute bias, beam robustness를 분리해야 함을 보이는 computational QA / negative-result 논문.

---

## 0. Reviewer Overall Position

**Recommendation: Major revision.**

The manuscript addresses an important and timely problem: whether Starlink-like unintended electromagnetic radiation (UEMR) can produce apparent excursions in HERA-like 21-cm delay-spectrum QA statistics, and whether such excursions should be interpreted as EoR-window bandpower contamination. The authors correctly distinguish the existence of UEMR from downstream cosmological contamination, and they introduce a multi-stage QA hierarchy involving local maximum statistics, empirical PTEs, absolute window-bias gates, window-integrated tests, and beam-operator robustness.

The paper is scientifically cautious and potentially publishable as a computational QA / forward-injection study. However, the central conclusions remain conditional on the chosen matched-null construction, simplified HERA-like QA operator, limited background realizations, TLE subset selection, beam-model implementation, and heuristic absolute-bias thresholds. The manuscript would be substantially strengthened by clarifying the statistical meaning of the matched-null PTEs, justifying or further demoting the absolute-bias gate, testing sensitivity to TLE subset choice and Doppler drift, and validating the full PolyBeam implementation.

---

# Part I. 가장 중요한 리뷰어 질문 요약

## 치명적 Q1. 이 matched-null PTE는 정확히 무엇의 PTE인가?

원고는 대응-영가설이 “UEMR 없음”의 null이 아니라, 동일한 시간--주파수 지지 영역과 진폭 포락선을 가진 phase-randomized control이라고 설명한다. 이 제한은 적절하지만, 그렇다면 `PTE_global,max`와 `PTE_global,absint`는 실제 RFI contamination의 false alarm probability가 아니다.

**Reviewer question:**  
If the matched-null ensemble preserves the injected UEMR support and amplitude envelope, and in some cases even preserves individual satellite delay-time structure, what exactly is the null hypothesis being tested? Can the reported empirical PTEs be interpreted as evidence against contamination, or only as conditional stress-test probabilities for phase coherence and inter-satellite coherence amplification?

**왜 치명적인가:**  
이 질문에 답하지 못하면 논문의 통계적 기반이 흔들린다. PTE가 “오염 없음”의 검정이 아니라면, 결론은 반드시 conditional QA stress test로 제한되어야 한다.

---

## 치명적 Q2. `B_floor`는 실제 HERA bandpower error budget과 무슨 관계가 있는가?

원고는 `B_rel > 10^{-2}`를 운영적 보고 수준의 진폭 게이트로 쓰지만, 동시에 이 값이 열잡음 공분산, 보정 잔차 예산, 특정 HERA 상한 분석의 계통오차 하한에 고정된 보편 권고값이 아니라고 인정한다.

**Reviewer question:**  
The absolute bias gate is central to the final QA hierarchy, yet `B_floor=10^{-2}` appears heuristic. How should readers interpret this threshold physically? Is 1% of the background window power meaningful for HERA EoR limits, or is it only a reporting-level diagnostic within this simplified QA operator?

**왜 치명적인가:**  
`B_floor`가 임의값이면 “오염 후보 없음” 결론이 threshold choice에 의존하는 것으로 보일 수 있다. 결론은 `B_floor`보다 beam-robust absint candidate가 0개라는 쪽으로 옮겨야 한다.

---

## 치명적 Q3. TLE subset과 Jan 2026 remapping이 결과를 좌우하지 않는가?

원고는 TLE 파일에서 순서상 처음 1200개 고유 기록만 로드했고, 이 subset이 전체 Jan 2026 Starlink 군집을 대표하지 않을 수 있다고 인정한다.

**Reviewer question:**  
Why were the first 1200 TLE records used? Could ordering in the TLE file bias the visible-satellite distribution, altitude distribution, or pass geometry? Would the conclusions change under random 1200-satellite subsets, all available TLEs, or altitude/range-filtered subsets?

**왜 치명적인가:**  
이 논문의 주입 geometry가 TLE subset에 조건부이므로, subset이 arbitrary하면 결과도 arbitrary로 보일 수 있다.

---

## 치명적 Q4. full PolyBeam에서 후보가 사라진 것이 물리적 결과인가, implementation artifact인가?

고정 PolyBeam 조건에서는 `PTE_global,absint < 0.01`인 사례가 2개 유지되지만, paired full PolyBeam에서는 기준을 통과하지 못해 beam-robust candidate가 0개가 된다.

**Reviewer question:**  
Since the final non-detection depends critically on candidates failing under the full chromatic PolyBeam, how was the full PolyBeam implementation validated? Could the disappearance of fixed-beam candidates under the full PolyBeam be caused by beam normalization, interpolation, frequency dependence, or flux-conservation artifacts?

**왜 치명적인가:**  
full PolyBeam 강건성은 최종 결론의 핵심 gate다. 이 gate가 구현상 불확실하면 전체 결론이 약해진다.

---

# Part II. Section별 Reviewer Questions

## 1. Title / Abstract / Framing

### R1. 제목이 실제 기여를 정확히 반영하는가?

현재 제목은 local max excess와 window-integrated bandpower contamination의 불일치를 강조한다. 방향은 좋지만, 논문이 실제 HERA--Starlink 동시 관측이 아니라 forward-injection QA stress test임을 제목에서 더 분명히 드러낼 필요가 있다.

**질문:**

1. 이 논문은 실제 Starlink UEMR 검출 논문인가, forward-injection simulation 논문인가?
2. 제목에 “forward injection”, “matched-null”, “QA framework”, “HERA-like” 중 하나가 들어가야 하지 않는가?
3. “Starlink 유사 UEMR”이라는 표현은 실제 Starlink emission을 의미하는가, 아니면 문헌 기반 morphology를 가진 stress-test source를 의미하는가?
4. “오염의 불일치 검증”이라는 표현이 실제 오염 검출 또는 비검출로 오해될 가능성은 없는가?
5. 영어 투고 제목에서는 claim을 더 낮춰야 하지 않는가?

**추천 제목 후보:**

- *A matched-null forward-injection QA framework for Starlink-like UEMR in HERA-like 21-cm delay spectra*
- *Local delay-spectrum excursions do not imply window-integrated contamination: a Starlink-like UEMR injection study for HERA-like 21-cm QA*
- *Disentangling local delay-spectrum excursions from EoR-window bandpower contamination in Starlink-like UEMR forward injections*

---

### R2. Abstract가 negative result와 methodology contribution을 균형 있게 제시하는가?

초록은 엄격한 local max candidate가 0개, beam-robust absint candidate가 0개라는 결과를 제시한다. 그러나 최종 결과가 “없었다”에 가까우므로, 방법론적 기여가 충분히 전면에 나와야 한다.

**질문:**

1. 이 논문의 positive contribution은 무엇인가?
2. “후보가 0개”라는 negative result가 왜 publishable한가?
3. local max PTE, absint PTE, absolute bias, beam robustness를 분리하는 QA hierarchy가 기존 HERA/RFI QA와 어떻게 다른가?
4. 초록에서 “Starlink-like UEMR is safe”로 오해될 가능성은 없는가?
5. “tested conditions”와 “conditional non-detection”을 더 명시해야 하지 않는가?
6. matched-null PTE가 “UEMR 없음”의 PTE가 아니라 조건부 phase-coherence stress-test라는 점이 abstract에서 충분히 드러나는가?

---

### R3. 논문의 novelty는 무엇인가?

원고는 Starlink UEMR의 최초 관측 보고가 아니라, QA 판정 계층의 명시적 분리를 기여로 둔다.

**질문:**

1. 기존 HERA RFI flagging, jackknife, null test, injection simulation과 비교했을 때 이 논문의 새로움은 무엇인가?
2. 단순히 local statistic과 integrated statistic을 둘 다 본 것인가, 아니면 새로운 matched-null QA architecture인가?
3. 다단계 gate 자체가 novelty인가?
4. LST bin metadata correction이 novelty인가?
5. forward-injection pipeline이 reusable computing contribution인가?
6. Astronomy and Computing 독자가 이 논문에서 얻는 computational contribution은 무엇인가?

**방어 방향:**

- novelty는 Starlink UEMR existence가 아니라, UEMR-like moving-source injection이 delay-spectrum QA statistics에 남기는 local/integrated/beam-dependent response를 분리하는 reproducible matched-null framework라고 정의해야 한다.
- “detection pipeline”이 아니라 “pre-production QA stress-test pipeline”으로 위치시켜야 한다.

---

## 2. Introduction

### R4. UEMR 존재와 EoR window contamination의 구분은 충분히 명확한가?

원고는 LOFAR/Starlink UEMR 보고와 HERA-like EoR window contamination 사이를 구분한다. 이 framing은 핵심 장점이다.

**질문:**

1. UEMR의 존재 자체와 EoR window contamination은 왜 다른 문제인가?
2. 어떤 처리 연산자가 UEMR-like residual을 high-delay window로 보낼 수 있는가?
3. local time-frequency RFI visibility와 delay-spectrum bandpower bias 사이의 연결이 충분히 설명되었는가?
4. “raw data에서 약한 구조”가 “processed EoR window contamination”이 되는 조건은 무엇인가?
5. 이 논문은 그 조건을 충분히 탐색했는가, 아니면 제한된 QA proxy만 보는가?

---

### R5. 기존 문헌과의 관계가 충분한가?

원고는 Di Vruno, Bassa, HERA upper limit, wedge/window 문헌을 인용한다. 그러나 HERA production-level validation과 본 연구의 pre-production QA 위치가 더 선명해야 한다.

**질문:**

1. 기존 Starlink UEMR 관측 논문들과 본 논문의 관계는 무엇인가?
2. 기존 HERA RFI excision, fringe-rate filtering, delay-spectrum QA, jackknife/PTE 연구들과 어떤 점이 다른가?
3. 본 논문은 HERA production power-spectrum pipeline을 대체하는가, 아니면 그 이전 단계의 diagnostic인가?
4. HERA upper-limit paper에서 사용하는 PTE/null framework와 본 논문의 matched-null은 어떻게 다른가?
5. 본 연구가 실제 HERA analysis policy에 주는 practical recommendation은 무엇인가?

---

## 3. Data and Injection Model

### R6. HERA-like background choice가 충분한가?

통제 그리드는 HERA H6C visibility crop을 현실적인 time-frequency background로 사용하고, HERA `(0,1)` baseline 산출물을 고정 배경으로 쓰되 기하학적 delay/horizon 계산에는 controlled baseline vector를 재정의한다.

**질문:**

1. 왜 특정 HERA H6C crop을 선택했는가?
2. 이 crop이 대표적인 background realization인가?
3. 동일한 crop을 여러 baseline geometry에 재사용하면 background realization과 baseline geometry 사이의 물리적 결합이 깨지지 않는가?
4. visibility background는 `(0,1)` baseline으로 고정하면서 horizon 계산은 재정의된 baseline vector를 쓰는 것이 어떤 의미인가?
5. 이 설계가 factor-control에는 유리하지만 실제 HERA baseline-dependent foreground structure를 잃는 것 아닌가?
6. 실제 native baseline robustness test에서는 이 문제가 얼마나 보완되는가?

---

### R7. 시간축 TLE remapping은 어떤 한계를 갖는가?

원고는 HERA visibility crop의 time axis를 TLE evaluation window로 remapping한다. 따라서 실제 HERA--Starlink 동시 관측이 아니다.

**질문:**

1. HERA background의 실제 LST와 TLE remapped time 사이의 물리적 의미는 무엇인가?
2. foreground background와 satellite geometry가 실제로 동시적인 것이 아니면, cross-term 해석이 제한되지 않는가?
3. 이 실험은 “realistic injection into real HERA background”인가, 아니면 “stress-test injection into HERA-like background”인가?
4. time remapping이 fringe-rate, flagging, LST-dependent foreground structure와 어떤 mismatch를 만들 수 있는가?
5. 실제 동시 HERA--Starlink pass가 없다는 점을 abstract와 conclusion에서 더 명확히 해야 하지 않는가?

---

### R8. TLE subset selection이 arbitrary하지 않은가?

원고는 TLE 파일에서 처음 1200개 고유 기록만 사용했다고 한다.

**질문:**

1. 왜 1200개인가?
2. 왜 “처음” 1200개인가?
3. TLE 파일 순서가 궤도면, 위성 generation, 고도, epoch, object ID에 따라 bias되어 있지 않은가?
4. random 1200 subset을 여러 번 뽑으면 visible satellite count, high-altitude pass, exposure proxy가 얼마나 달라지는가?
5. 전체 available TLE를 사용하면 결과가 달라지는가?
6. TLE epoch mismatch와 SGP4 propagation error는 평가되었는가?
7. Jan 2026 TLE를 쓰는 이유는 무엇인가?
8. 이 subset이 “Starlink-like constellation stress”를 대표한다는 근거는 무엇인가?

**필요 추가 분석:**

- first 1200 vs random 1200 vs all available comparison
- visible satellite count distribution
- exposure proxy distribution
- top-altitude pass distribution
- final candidate count sensitivity

---

### R9. Near-field delay 계산은 충분히 검증되었는가?

원고는 plane-wave approximation이 아니라 antenna-satellite distance difference로 near-field delay를 계산한다.

**질문:**

1. satellite range가 HERA baseline length에 비해 충분히 커서 plane-wave approximation이 어느 정도 정확한가?
2. near-field correction이 실제로 결과에 얼마나 영향을 주는가?
3. plane-wave delay와 near-field delay의 차이를 정량적으로 제시했는가?
4. near-field delay derivative, fringe-rate trajectory는 검증되었는가?
5. antenna position convention, coordinate frame, Earth rotation, topocentric conversion은 정확한가?
6. TLE to topocentric apparent position pipeline의 validation이 있는가?

---

### R10. Injection amplitude model의 물리성이 충분한가?

주입 진폭은 `S_ref`, spectral morphology, beam gain, range attenuation, averaging smear로 구성된다.

**질문:**

1. `S_ref`는 어떤 물리적 flux density를 의미하는가?
2. 문헌 Starlink UEMR flux density와 HERA apparent flux 사이의 scaling은 어떻게 했는가?
3. LOFAR-reported flux를 HERA 주파수/beam/observing mode로 옮길 때 어떤 가정을 했는가?
4. range attenuation `R(t)`는 정확히 어떤 법칙을 쓰는가?
5. UEMR emission이 isotropic이라고 가정하는가?
6. 위성 attitude, antenna pattern, polarization, duty cycle은 무시되는가?
7. time/frequency averaging smear는 어떻게 계산되며 검증되었는가?
8. polarization leakage나 Stokes response는 고려되었는가?

---

### R11. Spectral morphology choices가 충분히 정당화되었는가?

원고는 smooth broadband, bursty broadband, narrowband linear, kHz-comb bank를 사용한다.

**질문:**

1. 각 morphology가 어떤 관측 문헌 또는 physical emission mechanism을 반영하는가?
2. kHz-comb spacing, line width, duty cycle, spectral envelope는 어떻게 정했는가?
3. bursty broadband의 temporal duty cycle은 어떻게 설정했는가?
4. narrowband linear drift는 Doppler와 구분되는가?
5. morphology parameter sensitivity를 수행했는가?
6. morphology가 결과를 지배하는지, beam/LST/TLE geometry가 결과를 지배하는지 분해했는가?
7. kHz-comb에서 Doppler drift를 반영하지 않으면 comb coherence가 과대평가되지 않는가?

---

### R12. Doppler drift 미반영은 얼마나 치명적인가?

원고는 LEO range-rate가 150 MHz에서 약 3.5 kHz 편이를 만들 수 있고, 10분 동안 수 kHz drift가 가능하다고 limitation에 쓴다.

**질문:**

1. kHz-comb morphology에서 Doppler drift는 line spacing과 같은 order인데, 이를 무시해도 되는가?
2. no-Doppler injection이 comb-line temporal coherence를 과대평가하는가?
3. Doppler drift가 window leakage를 증가시키는가, 감소시키는가?
4. constant Doppler shift와 time-varying Doppler drift의 차이는 무엇인가?
5. toy Doppler model이라도 넣어 sensitivity를 제시해야 하지 않는가?
6. Doppler를 넣으면 고정 PolyBeam absint candidate 2건이 사라지는가?
7. Doppler limitation이 comb morphology 결과를 얼마나 제한하는가?

---

## 4. QA Metrics and Null Hypothesis

### R13. QA operator `P_QA`가 너무 단순하지 않은가?

원고는 `P_QA`를 delay transform, EoR window mask, window statistic calculation까지의 공통 QA 절차로 정의한다.

**질문:**

1. `P_QA`는 실제 HERA pipeline의 어느 부분을 대리하는가?
2. calibration, flagging, redundant averaging, LST binning, covariance weighting, signal loss correction은 포함되는가?
3. 포함되지 않는다면 결과를 “EoR window bandpower contamination”으로 부를 수 있는가?
4. `P_QA`가 simplified proxy라면, conclusion에서도 “bandpower contamination” 대신 “window-power proxy contamination”이라고 해야 하지 않는가?
5. actual cosmological bandpower estimator와 이 QA proxy 사이의 차이는 무엇인가?

---

### R14. EoR window proxy가 충분한가?

원고는 `tau_win = tau_hor + 100 ns`와 `|tau| >= tau_win`을 EoR window proxy로 사용한다.

**질문:**

1. 왜 100 ns buffer인가?
2. HERA analyses에서 사용하는 buffer와 일치하는가?
3. baseline length가 달라질 때 이 buffer는 충분한가?
4. taper/window function, spectral resolution, flagging gaps가 effective delay response를 바꾸지 않는가?
5. `tau_win` sensitivity를 수행했는가?
6. 50 ns, 100 ns, 150 ns, 200 ns buffer에서 결과가 유지되는가?

---

### R15. `B_rel`의 정의가 물리적으로 적절한가?

`B_rel`은 window 내부의 absolute difference in delay-power를 background window power로 나눈 값이다.

**질문:**

1. absolute difference를 쓰는 이유는 무엇인가?
2. signed bias 또는 coherent integrated bias와 결과가 다른가?
3. background window power가 작은 경우 `B_rel`이 불안정해지지 않는가?
4. thermal noise dominated window에서 background power normalization은 어떤 의미인가?
5. `B_rel`이 mK² 단위 bandpower bias와 어떻게 연결되는가?
6. HERA sensitivity budget과 연결할 수 있는가?
7. `B_rel`이 high-delay bins의 narrow local spike와 broad distributed power를 구분하지 못하지 않는가?

---

### R16. `Z_PS,max`는 왜 오염 판정 기준이 아닌가?

원고는 robust z diagnostic이 local exploratory statistic이라고 설명한다.

**질문:**

1. `Z_PS,max > 3`이 발생했는데 contamination candidate가 아닌 이유는 무엇인가?
2. MAD denominator가 작을 때 z가 폭증하는 사례를 어떻게 처리했는가?
3. MAD-to-std fallback은 얼마나 자주 발생하는가?
4. denominator floor 또는 NaN 처리 사례 수는 얼마인가?
5. local max statistic의 look-elsewhere effect는 어떻게 보정되는가?
6. `Z_PS.max`와 `PTE_global,max`가 불일치하는 사례를 정량적으로 보여야 하지 않는가?

---

### R17. matched-null의 실험별 차이가 혼란스럽지 않은가?

통제 그리드에서는 per-time phase scramble, coverage robustness에서는 per-satellite scalar phase randomization을 사용한다.

**질문:**

1. 왜 두 실험의 null randomization 방식이 다른가?
2. controlled grid와 coverage test의 PTE를 직접 비교할 수 있는가?
3. per-time phase scramble은 spectral coherence를 보존하고 time/fringe coherence를 파괴한다는 해석이 맞는가?
4. per-satellite scalar phase randomization은 각 satellite의 delay-time coherent structure를 보존하고 inter-satellite coherence만 무작위화하는가?
5. single-satellite coverage trial에서는 satellite signal 자체가 null에도 보존되므로 무엇을 test하는가?
6. multi-satellite coverage trial에서는 inter-satellite phase coherence amplification을 test하는가?
7. 이 차이를 표로 정리하지 않으면 reviewer가 PTE 해석을 오해하지 않겠는가?

**필요 표:**

| Experiment | Phase randomization | Preserved | Destroyed | PTE interpretation |
|---|---|---|---|---|
| Controlled grid | per-time phase | spectral coherence, amplitude support | time/fringe-rate coherence | local QA sensitivity to coherent motion |
| Coverage test | per-satellite scalar phase | each satellite delay-time structure | inter-satellite relative phase coherence | coherence amplification stress test |

---

### R18. empirical PTE 해상도와 multiple testing 문제

원고는 `N_null=100`에서 최소 PTE가 0.0099이고, 경계 사례를 `N_null=1000`으로 정밀화한다.

**질문:**

1. 왜 coarse grid는 `N_null=100`인가?
2. `PTE<0.01` threshold를 쓰려면 `N=100`은 너무 작지 않은가?
3. `N=1000` tail refinement 대상은 어떻게 선택되었는가?
4. tail refinement selection이 post hoc이 아닌가?
5. `PTE_global,max <= 0.03` 또는 `Z_PS,max > 3` 조건은 왜 선택되었는가?
6. `B_rel > 10^{-3}` 조건 때문에 absint-only floor-hit가 처음에 누락된 것 아닌가?
7. 648 trials에 대해 `alpha=0.01`이면 기대 false positives가 6.48인데, 후보 0개라는 해석은 어떻게 해야 하는가?
8. trial들이 독립이 아니므로 expected FP 계산은 왜 보조값에 불과한가?
9. 가족단위 오류율 또는 grid-level max statistic을 사용해야 하지 않는가?

---

## 5. Experiment Design

### R19. Controlled grid의 factor design은 충분히 닫혀 있는가?

통제 그리드는 background, baseline, beam, flux, morphology, multiplicity를 교차한다.

**질문:**

1. 각 factor의 level은 어떻게 정했는가?
2. flux tier는 문헌 기반인가, stress-test 기반인가?
3. morphology level 간 relative amplitude normalization은 어떻게 맞췄는가?
4. multiplicity는 실제 constellation density를 반영하는가?
5. baseline groups는 실제 HERA baseline distribution을 대표하는가?
6. controlled grid의 600 rows는 충분한 coverage인가?
7. grid factor 간 interaction을 분석했는가?
8. local max flag가 특정 factor combination에 집중되는가?

---

### R20. Native 9-baseline selection은 대표적인가?

9개 native HERA H6C EW-like baseline을 length tier와 FRF loss quality tier로 2D stratified sampling했다고 한다.

**질문:**

1. 왜 9개인가?
2. 전체 HERA H6C EW-like baseline distribution에서 이 9개가 어떤 percentile을 대표하는가?
3. length tier는 어떻게 나누었는가?
4. FRF loss quality tier는 어떻게 계산했는가?
5. 대표 LEO orbit에서 계산한 FRF loss coefficient가 실제 pass ensemble에 일반화되는가?
6. baseline orientation, redundancy, flagging, foreground amplitude는 selection에 포함되었는가?
7. north-south 또는 diagonal baseline은 제외되었는가?
8. EW-like baseline만으로 충분한가?

---

### R21. LST bin selection이 post-injection leakage 없이 이루어졌는가?

원고는 quiet/typical/stress bins를 injection 이전 metadata만으로 선택했다고 한다.

**질문:**

1. `E_sat`, `f_flag`, `nullMAD`는 모두 injection 전 정보인가?
2. nullMAD proxy는 어떻게 injection 없이 계산되는가?
3. stress bin의 `R_pre = E_sat/(nullMAD_win + epsilon)`은 어떤 물리적 의미인가?
4. `epsilon`은 어떻게 정했는가?
5. quiet bin을 낮은 satellite exposure 후보 중 flag fraction median 근처로 고른 이유는 무엇인가?
6. typical bin은 robust-scaled metadata space에서 median vector에 가장 가까운 bin이라고 하는데, 각 feature의 scaling은 어떻게 했는가?
7. 같은 bin이 여러 category에 선택될 때 next-rank rule은 어떻게 적용되었는가?
8. 27개 bin 중 15개가 metadata correction 후 바뀌었다면, original result는 얼마나 불안정했던 것인가?

---

### R22. 10분 chunk 전체 metadata correction은 충분한가?

초기 `(0_1, bin 36)`에서 중심 시각 위성 노출은 0이었지만 전체 10분 chunk에서는 4개 visible satellite와 high-altitude pass가 있었다고 한다.

**질문:**

1. 왜 처음에는 bin center만 사용했는가?
2. 모든 time sample을 쓰도록 수정한 후 exposure proxy가 어떻게 바뀌었는가?
3. 27개 선택 bin 중 15개가 바뀐다는 것은 bin stratification이 매우 민감하다는 뜻 아닌가?
4. corrected metadata에서도 10분 chunk가 충분한 시간 해상도인가?
5. 5분, 10분, 20분 chunk에서 결과가 유지되는가?
6. center-time provenance column은 어떤 역할을 하는가?
7. 이 correction이 단순 bug fix인가, 논문의 방법론적 기여인가?

---

### R23. Coverage grid rows 648개는 어떻게 구성되는가?

**질문:**

1. 648 rows는 정확히 어떤 factor product인가?
2. 9 baseline × 3 bin × morphology × flux × multiplicity × beam 조합인가?
3. 각 조합의 level 수는 명확히 제시되었는가?
4. 일부 조합이 누락되었는가?
5. trial들이 독립적이지 않다는 점을 result table에서 명확히 표시했는가?
6. 648 rows를 독립 샘플처럼 해석할 위험은 없는가?

---

## 6. Results

### R24. Controlled grid 결과에서 local robust-z와 PTE의 불일치가 충분히 입증되었는가?

원고는 controlled grid에서 `Z_PS,max` flags가 일부 발생했지만 `PTE_global,max < 0.01` 후보는 0개라고 한다.

**질문:**

1. 몇 개의 `Z_PS,max > 3` flags가 있었는가?
2. `Z_PS,max`와 `PTE_global,max`의 Spearman/Pearson correlation은 얼마인가?
3. `Z_PS.max > 3`인데 PTE가 높았던 대표 사례는 무엇인가?
4. high Z가 MAD denominator collapse 때문인지, 실제 local excursion 때문인지 구분했는가?
5. local max PTE가 전역 correction을 수행한다는 점이 충분히 시각화되었는가?
6. `B_rel`과 `Z_PS.max`의 관계도 보여야 하지 않는가?

---

### R25. Coverage test에서 strict local max candidate가 0개라는 결론은 강한가?

수정 후 coarse grid에서 `PTE_global,max < 0.01`은 1건이었지만 `B_rel=5.53e-4`로 기준값을 넘지 못했고, tail refinement 4건 후 strict candidate는 0개라고 한다.

**질문:**

1. coarse grid에서 나온 1건은 어떤 조건인가?
2. 왜 이 1건은 tail refinement 대상이 아니었는가, 또는 되었는가?
3. `B_rel=5.53e-4`가 작은 값이라는 근거는 무엇인가?
4. `B_floor=10^{-3}`로 보면 제외되지만, 더 낮은 floor에서는 candidate가 될 수 있지 않은가?
5. 4개 tail-refined cases는 어떻게 선택되었는가?
6. `PTE_max^1000`가 0.022--0.040 범위면 strict 기준은 통과하지 못하지만, mild exploratory flag로는 남는가?
7. 이런 cases를 supplement에서 자세히 보여야 하지 않는가?

---

### R26. Absint floor-hit 사례의 해석이 명확한가?

고정 PolyBeam 조건에서 2개 사례가 `N=1000`에서도 `PTE_global,absint < 0.01`로 유지되지만, full PolyBeam에서는 기준을 통과하지 못한다.

**질문:**

1. 왜 local max PTE는 높고 absint PTE만 낮은가?
2. 이 현상은 distributed window power excess를 의미하는가?
3. 두 사례가 모두 fixed PolyBeam에서만 발생하는 이유는 무엇인가?
4. full PolyBeam에서 PTE가 0.030과 0.020으로 올라가는 것은 얼마나 강한 rejection인가?
5. `N=100` 단계에서 full PolyBeam PTE가 이미 strict threshold를 넘었다고 하는데, `N=1000` full PolyBeam 재평가는 하지 않았는가?
6. fixed-only absint candidates를 “beam artifact”라고 부를 수 있는가, 아니면 “beam-model-sensitive candidate”라고 해야 하는가?
7. 이 결과가 fixed PolyBeam 사용의 위험을 보여주는가?

---

### R27. Beam robustness 결론은 충분히 검증되었는가?

최종 beam-robust candidates는 0개이다.

**질문:**

1. paired full PolyBeam trial은 fixed PolyBeam trial과 정확히 같은 injection geometry, morphology, flux, background를 사용했는가?
2. beam만 바꾼 paired comparison인가?
3. full PolyBeam의 frequency-dependent gain이 injection amplitude를 어떻게 바꾸는가?
4. fixed PolyBeam과 full PolyBeam 사이에 total injected apparent power가 보존되는가?
5. full PolyBeam이 off-axis response를 더 낮춰 candidate를 없앤 것인가?
6. full PolyBeam에서 후보가 사라진 물리적 이유를 예시 figure로 보여야 하지 않는가?
7. full beam implementation validation figure가 필요한가?

---

### R28. Figure and table presentation

**질문:**

1. `Z_PS,max` vs PTE figure 외에 `B_rel` vs PTE figure가 필요한가?
2. fixed vs full PolyBeam paired comparison figure가 필요한가?
3. tail-refined 4 cases의 delay spectrum residual을 보여야 하지 않는가?
4. absint floor-hit 2 cases의 window-integrated profile을 보여야 하지 않는가?
5. corrected LST metadata로 선택 bin이 바뀐 과정을 figure로 보여야 하지 않는가?
6. 27 bin 중 15개가 바뀐 결과를 table로 제시해야 하지 않는가?
7. TLE exposure distribution figure가 필요한가?
8. visible-satellite count vs `E_sat` vs final statistic scatter plot이 필요한가?

---

## 7. Discussion / Limitations

### R29. Negative result의 범위를 충분히 제한했는가?

원고는 일반적 안전성을 주장하지 않는다고 한다.

**질문:**

1. 이 연구는 Starlink-like UEMR이 안전하다고 말하는가?
2. 아니면 tested setup에서 beam-robust window-integrated candidate를 발견하지 못했다고 말하는가?
3. 다른 HERA season, LST range, calibration state, beam model, TLE subset에서는 결과가 달라질 수 있는가?
4. flux tier가 더 높으면 candidate가 나올 수 있는가?
5. Doppler, polarization, satellite antenna pattern을 넣으면 결과가 달라질 수 있는가?
6. 결론에서 conditionality가 충분히 반복되는가?

---

### R30. “local excursion ≠ integrated contamination”이라는 주장에 대한 근거

이 논문의 핵심 메시지는 local delay-spectrum excursion이 window-integrated contamination을 의미하지 않는다는 것이다.

**질문:**

1. 이 메시지는 어떤 결과에 의해 가장 강하게 지지되는가?
2. controlled grid의 Z/PTE mismatch인가?
3. coverage tail refinement에서 strict local candidate가 0개인 점인가?
4. fixed PolyBeam absint 후보가 full PolyBeam에서 사라진 점인가?
5. local max와 absint PTE가 다른 사례를 구체적으로 보여야 하지 않는가?
6. local statistic과 integrated statistic의 수학적 차이를 더 설명해야 하지 않는가?

---

### R31. LST metadata correction의 의미가 충분히 해석되었는가?

원고는 bin-center metadata만으로 LST layer를 물리적으로 과해석하면 안 된다는 재현성 점검 결과를 제시한다.

**질문:**

1. 이것은 단순한 implementation bug fix인가, 일반적인 방법론적 교훈인가?
2. 왜 10분 chunk에서 center-time exposure가 misleading할 수 있는가?
3. satellite pass가 fast-moving이기 때문인가?
4. 이 문제는 다른 LEO RFI injection studies에도 적용되는가?
5. corrected metadata로 bin selection이 27개 중 15개 바뀌었다는 결과는 얼마나 심각한가?
6. 이 finding을 논문의 주요 contribution 중 하나로 올릴 수 있는가?

---

### R32. Pre-production QA라는 위치가 충분히 명확한가?

원고는 production-level HERA verification을 대체하지 않는다고 한다.

**질문:**

1. pre-production QA와 production cosmological validation의 차이는 무엇인가?
2. 이 framework는 HERA pipeline의 어느 단계에 들어갈 수 있는가?
3. 실제 운영에서는 어떤 statistic이 먼저 계산되어야 하는가?
4. beam-robust candidate가 나오면 다음 단계는 무엇인가?
5. 이 논문의 output이 HERA team에게 어떤 decision을 제공하는가?
6. “candidate absent”일 때 어떤 operational reassurance를 줄 수 있는가?

---

### R33. Background realization limitation

원고는 배경 실현 분산을 충분히 표본화하지 못했다고 인정한다.

**질문:**

1. 몇 개의 independent HERA background realization이 필요한가?
2. 다른 LST, flagging state, foreground level, season에서 결과가 달라질 수 있는가?
3. 동일한 background 계열에서 648 trials를 돌리는 것이 실제 independent sampling처럼 보이지 않도록 충분히 경고했는가?
4. background realization uncertainty가 final candidate count에 미치는 영향은 평가되었는가?
5. future work가 아니라 최소한 supplementary test가 필요한가?

---

### R34. Trial independence and look-elsewhere effect

원고는 648×0.01 expected false positives 계산이 엄밀한 FWER correction이 아니라고 한다.

**질문:**

1. 왜 grid-level max statistic을 쓰지 않았는가?
2. trial들이 correlated되어 있다면 candidate count의 expected distribution은 무엇인가?
3. permutation-based grid-level correction을 할 수 있는가?
4. `N_null=1000` tail refinement만으로 look-elsewhere effect가 충분히 해결되는가?
5. 후보 0개라는 결론이 multiple testing correction에 민감한가?

---

### R35. Matched-null limitation

**질문:**

1. per-time phase scramble과 per-satellite scalar phase null이 각각 어떤 physical alternative를 test하는가?
2. 두 null 모두 UEMR absence null이 아니라면, “non-detection”이라는 표현은 조심해야 하지 않는가?
3. `non-detection of beam-robust contamination under matched-null stress tests`라고 써야 하지 않는가?
4. actual no-UEMR null이나 background-only null과 비교했는가?
5. matched-null이 너무 보수적인가, 너무 관대한가?

---

### R36. `B_floor` physical calibration limitation

**질문:**

1. `B_floor=10^{-2}`를 실제 mK² budget으로 변환할 수 있는가?
2. HERA thermal noise covariance와 calibration residual budget을 넣으면 이 threshold가 어떻게 바뀌는가?
3. `B_rel`이 background window power로 normalized되므로 background low-power cases에서 과대평가될 가능성이 있는가?
4. 운영적 QA threshold는 observation-dependent여야 하지 않는가?
5. 논문 결론이 특정 floor value에 의존하지 않도록 재작성해야 하지 않는가?

---

### R37. Doppler limitation

**질문:**

1. Doppler drift가 kHz-comb morphology와 같은 scale인데 limitation으로만 두는 것이 충분한가?
2. no-Doppler result는 worst-case coherent comb injection인가?
3. 그렇다면 “conservative stress test”로 해석할 수 있는가?
4. Doppler를 넣으면 local max와 absint PTE가 감소할 가능성을 정량화해야 하지 않는가?
5. Doppler-included injection은 future work가 아니라 revision requirement 아닌가?

---

## 8. Reproducibility and Code

### R38. Code/data availability가 실제로 충분한가?

원고는 GitHub repository와 Zenodo DOI 계획을 제시한다.

**질문:**

1. 현재 repository로 모든 figures/tables가 재생성되는가?
2. raw HERA-like background visibility가 재배포 불가능하면 어떤 surrogate를 제공하는가?
3. summary CSV만으로 논문의 핵심 결과가 재현되는가?
4. TLE files, config files, random seeds, null ensemble seeds가 포함되는가?
5. `N_null=1000` tail cases의 null statistics raw arrays가 제공되는가?
6. beam model files 또는 version이 제공되는가?
7. pyuvdata, hera_sim, healpy, astropy version pinning이 되어 있는가?
8. pipeline entry point가 하나로 정리되어 있는가?
9. figure-generation scripts가 table-generation scripts와 같은 result version을 참조하는가?
10. GitHub URL만이 아니라 Zenodo DOI가 필요하지 않은가?

---

### R39. Macro-based counts가 reproducibility risk를 만들지 않는가?

원고는 `\NStrictLocalMax`, `\NAbsintRetainedN1000` 같은 LaTeX macros로 결과 counts를 넣는다.

**질문:**

1. 이 macros는 자동 생성되는가, 수동 입력인가?
2. 수동 입력이면 table values와 inconsistency가 생길 수 있지 않은가?
3. result CSV에서 LaTeX macro file을 자동 생성하는가?
4. manuscript의 numbers와 repository output이 일치하는지 CI test가 있는가?
5. revision 중 count가 바뀌면 macros가 자동 업데이트되는가?

---

## 9. Journal Fit

### R40. Astronomy and Computing에 맞는 computational contribution이 충분한가?

**질문:**

1. 이 논문은 astronomy computing paper인가, radio astronomy science paper인가?
2. reusable framework, code, simulation pipeline, QA statistic architecture가 충분히 강조되는가?
3. software design, reproducibility, configuration management, computational workflow가 본문에 충분히 설명되는가?
4. 단순 scientific result가 0개 candidate라면, computing contribution이 충분히 강해야 하지 않는가?
5. repository와 reproducibility artifacts가 심사 시점에 공개되어 있는가?

---

### R41. Radio Science나 MNRAS에 비해 왜 Astronomy and Computing인가?

**질문:**

1. Radio Science로 가면 propagation physics가 부족하지 않은가?
2. MNRAS로 가면 astrophysical result가 약하지 않은가?
3. Astronomy and Computing으로 가면 method/pipeline/reproducibility framing이 더 적합하지 않은가?
4. 저널 선택에 따라 introduction과 discussion의 강조점이 달라져야 하지 않는가?

---

# Part III. Reviewer가 요구할 가능성이 높은 추가 분석

## A1. Matched-null interpretation table

Controlled grid와 coverage test의 null hypothesis 차이를 표로 분리해야 한다.

**필요 내용:**

- phase randomization type
- preserved structure
- destroyed structure
- statistic interpretation
- what cannot be inferred

---

## A2. TLE subset sensitivity

**필요 분석:**

- first 1200 vs random 1200 subsets
- random subsets 최소 10회
- visible satellite count distribution
- exposure proxy distribution
- final candidate count under each subset
- high-altitude pass count comparison

---

## A3. Full PolyBeam validation

**필요 분석:**

- fixed vs full PolyBeam gain comparison
- frequency interpolation check
- zenith and off-axis response check
- total apparent injected power comparison
- paired fixed/full delay spectrum example
- candidate disappearance explanation

---

## A4. Doppler-included comb sensitivity

**필요 분석:**

- no Doppler
- constant Doppler shift
- linear time-varying Doppler drift
- possibly SGP4 range-rate Doppler
- kHz-comb local max and absint PTE comparison
- whether fixed-beam absint floor-hit cases remain

---

## A5. Delay-window buffer sensitivity

**필요 분석:**

- horizon + 50 ns
- horizon + 100 ns
- horizon + 150 ns
- horizon + 200 ns
- candidate count and `B_rel` sensitivity
- local vs absint PTE stability

---

## A6. `B_floor` sensitivity and physical demotion

**필요 분석 또는 수정:**

- keep `B_floor = 10^{-3}, 10^{-2}, 10^{-1}`
- avoid presenting one as universal
- emphasize beam-robust absint candidate count
- optional rough mapping to thermal/systematic budget if possible

---

## A7. Background realization sensitivity

**가능한 분석:**

- additional HERA crops if available
- multiple LST ranges
- different flagging levels
- synthetic foreground backgrounds
- background-only null stability
- candidate count as function of background window power

---

## A8. Grid-level multiple-testing correction

**가능한 분석:**

- max statistic over grid
- permutation-based FWER
- hierarchical null
- false discovery discussion
- trial correlation estimate

---

## A9. Representative case figures

**필요 figure:**

1. high local Z but non-significant PTE case
2. fixed PolyBeam absint candidate that fails full PolyBeam
3. bin-center vs full-chunk metadata correction case
4. tail-refined 4 cases summary
5. `B_rel` vs `PTE_global,max`
6. `PTE_global,max` vs `PTE_global,absint`

---

## A10. Reproducibility package

**필요 보강:**

- Zenodo DOI
- environment.yml
- requirements.txt
- config_used.yaml for every run
- generated LaTeX macros from result CSV
- raw null arrays for tail cases
- figure scripts
- README with one-command reproduction

---

# Part IV. Section별 수정 우선순위

## 반드시 수정해야 함

1. matched-null PTE의 통계적 의미를 더 명확히 제한
2. controlled grid와 coverage test의 null difference 표 추가
3. `B_floor=10^{-2}`를 보편 threshold처럼 보이지 않게 수정
4. TLE first-1200 subset sensitivity 추가 또는 강하게 limitation 처리
5. full PolyBeam implementation validation 추가
6. Doppler drift limitation을 kHz-comb 결과 해석에 직접 연결
7. abstract와 conclusion에 “conditional non-detection” 명시

---

## 가능하면 추가해야 함

1. delay-buffer sensitivity
2. Doppler toy model
3. TLE random-subset repeat
4. grid-level max PTE 또는 multiple-testing discussion 강화
5. fixed vs full beam paired example figure
6. metadata correction figure/table
7. `B_rel` vs PTE scatter plot

---

## 없어도 되지만 있으면 강해짐

1. multiple HERA background realizations
2. actual simultaneous Starlink-HERA observation case
3. production-level bandpower normalization
4. full `k_perp-k_parallel` propagation
5. polarization leakage model
6. satellite antenna pattern model

---

# Part V. Reviewer-style Major Comments

## Major Comment 1 — Clarify the statistical meaning of the matched-null PTEs

The manuscript repeatedly reports empirical PTEs under matched-null ensembles. However, these null ensembles preserve the injected support and amplitude envelope, and in the coverage test they preserve each satellite’s coherent delay-time structure while randomizing only inter-satellite phases. Therefore these PTEs do not test the absence of UEMR. They test whether the observed coherent injection is extreme relative to a phase-randomized counterpart. This distinction is scientifically important and should be moved earlier in the manuscript, preferably with a table comparing the null definitions used in the controlled grid and coverage tests.

---

## Major Comment 2 — The absolute-bias threshold is heuristic and should be demoted

The paper uses `B_floor=10^{-2}` as an operational reporting gate, but this value is not tied to a HERA thermal-noise covariance, calibration residual budget, or cosmological bandpower systematic allowance. The authors correctly acknowledge this limitation, but the decision hierarchy still gives the threshold substantial interpretive weight. The final conclusion should emphasize threshold-robust statements, especially the absence of beam-robust window-integrated candidates, rather than any single `B_floor` value.

---

## Major Comment 3 — The TLE subset choice requires sensitivity testing

The use of the first 1200 unique TLE records is a potentially serious source of selection bias. The visible satellite distribution and high-altitude pass statistics may depend on file ordering. The authors should either use all available TLEs or demonstrate that random 1200-object subsets produce the same exposure distribution and final candidate count. Without this, the injection geometry remains insufficiently justified.

---

## Major Comment 4 — Full PolyBeam validation is essential

The final non-detection of beam-robust window-integrated contamination depends on the fact that fixed-PolyBeam absint candidates fail under the paired full PolyBeam. This makes the full PolyBeam implementation central to the paper’s conclusion. The authors should provide validation of the full chromatic beam implementation, including frequency interpolation, normalization, off-axis response, and paired fixed/full examples showing why the candidate fails.

---

## Major Comment 5 — Doppler drift should be treated as a stress-test parameter

The manuscript acknowledges that LEO Doppler shifts at 150 MHz can be of order several kHz, comparable to the kHz-comb morphology. This is not a minor limitation for comb-like injections. The authors should include a Doppler-drift sensitivity test and state clearly whether the no-Doppler comb injection is being used as a coherence-maximizing stress test or as a physically complete Starlink comb model.

---

# Part VI. Reviewer-style Minor Comments

1. Define “UEMR” at first use in both abstract and introduction.
2. Avoid language that suggests actual Starlink-HERA simultaneous detection.
3. Replace “contamination candidate” with “QA candidate” unless absint PTE and beam robustness are both satisfied.
4. Clarify whether `PTE_global,max` is one-sided or two-sided.
5. Clarify whether `B_rel` uses delay bins after tapering and flagging.
6. State the delay transform window/taper used.
7. Report the number of null degeneracy cases where MAD fallback was used.
8. Define “full PolyBeam” and “frozen PolyBeam” before using them in results.
9. Provide the exact frequency range and channelization.
10. Clarify how flagged channels/times are handled in delay transform.
11. Clarify whether injected visibilities are added before or after flagging.
12. State whether thermal noise is included in the background or injected separately.
13. Explain whether multiple satellites are summed coherently before or after beam weighting.
14. Clarify whether `S_ref=1000 Jy` is apparent or intrinsic stress scale.
15. Provide units for all flux tiers.
16. Explain why the 100 ns horizon buffer was chosen.
17. Include a schematic of the QA decision hierarchy.
18. Include a table summarizing candidate counts at each gate.
19. Avoid treating 648 trials as independent.
20. State whether random seeds are fixed and archived.
21. Use consistent notation for `PTE_global,max` and `PTE_max`.
22. Use consistent notation for `PTE_global,absint` and `PTE_absint`.
23. Avoid mixing Korean and English terminology in final English submission.
24. Ensure all macros are automatically generated from result files.
25. Provide Zenodo DOI before publication.

---

# Part VII. Reviewer Final Recommendation Draft

> This manuscript presents a careful and potentially useful forward-injection QA study of Starlink-like UEMR in HERA-like 21-cm delay spectra. I appreciate the authors’ effort to separate local delay-spectrum excursions from window-integrated bandpower contamination, and the use of matched-null ensembles, empirical PTEs, absolute window-bias gates, and paired beam-operator robustness checks. The revised LST-bin metadata treatment is also a valuable methodological warning for moving-source injection studies.
>
> However, the current manuscript requires major revision before publication. The statistical meaning of the matched-null PTEs must be clarified, because the null does not represent the absence of UEMR but a phase-randomized matched injection. The absolute-bias threshold is heuristic and should not be presented as a physically calibrated contamination criterion. The TLE subset selection, full PolyBeam implementation, and omission of Doppler drift in kHz-comb injections require additional sensitivity tests or stronger limitation statements. I also recommend adding figures or tables showing fixed versus full PolyBeam paired cases, TLE exposure sensitivity, and the difference between local maximum and window-integrated PTEs.
>
> With these revisions, the paper could become a useful contribution to reproducible pre-production QA for LEO-satellite UEMR in 21-cm cosmology pipelines. In its current form, the main results are promising but too conditional on implementation choices to support the stated conclusions without further clarification.

---

# Part VIII. 지도교수식 최종 판단

이 논문은 엎을 필요 없다. 세 원고 중에서 저널 투고용으로 가장 만들기 쉽다. 다만 이 논문의 생존 전략은 “우리가 Starlink 오염을 검출하지 않았다”가 아니라 다음 문장이어야 한다.

> We show that local delay-spectrum excursions from Starlink-like UEMR injections should not be promoted to EoR-window contamination claims unless they also pass absolute window-bias, window-integrated PTE, and full-beam robustness gates.

즉, 핵심 기여는 negative result가 아니라 **QA 판정 계층**이다.

현재 가장 위험한 지점은 네 가지다.

1. matched-null PTE의 의미
2. `B_floor`의 임의성
3. TLE first-1200 subset
4. full PolyBeam gate의 구현 검증

이 네 가지를 정리하면 Astronomy and Computing 투고 가능성이 있다.

---

# Part IX. 한 줄 결론

**이 원고는 “Starlink UEMR이 안전하다”는 논문이 아니라, “local delay-spectrum QA flag를 EoR window contamination으로 과대해석하지 않기 위한 matched-null forward-injection QA 논문”으로 가야 한다.**

이 방향으로 낮춰 잡으면 살아남을 수 있다.

---

# Part X. Reproducibility-folder Answer Matrix

This section is written against `paper_final_experiment_check` as the reproducibility root. Answers below are filled only when the folder already contains a supporting result file, manuscript text, configuration file, or figure. Blank answer fields mean that the requested evidence is not yet present in this reproducibility package.

## X0. Files Used As Evidence

- `paper/paper.tex`
- `configs/pathB_jan2026_main.yaml`
- `configs/pathB_jan2026_polybeam_frozen150.yaml`
- `configs/pathB_jan2026_polybeam_robustness.yaml`
- `configs/coverage_robustness.yaml`
- `configs/coverage_robustness_all_tle.yaml`
- `outputs/coverage_robustness_trials.csv`
- `outputs/coverage_robustness_trials_all_tle_full.csv`
- `outputs/coverage_summary_by_factor.csv`
- `outputs/coverage_summary_by_baseline_lst.csv`
- `outputs/coverage_candidate_counts_by_floor.csv`
- `outputs/all_tle_summary/coverage_candidate_counts_by_floor.csv`
- `outputs/all_tle_summary/coverage_summary_by_factor.csv`
- `outputs/all_tle_summary/coverage_summary_by_baseline_lst.csv`
- `outputs/coverage_candidate_counts_by_floor_extended.csv`
- `outputs/coverage_tail_refined_near_threshold.csv`
- `outputs/coverage_tail_refined_near_threshold_summary.csv`
- `outputs/excluded_pte_candidate_n1000.csv`
- `outputs/excluded_pte_candidate_n1000_summary.csv`
- `outputs/absint_floor_recheck.csv`
- `outputs/lst_bin_selection.csv`
- `outputs/lst_bin_metadata.csv`
- `outputs/tle_subset_sensitivity.csv`
- `outputs/tle_subset_sensitivity_summary.csv`
- `outputs/tle_subset_sensitivity.meta.json`
- `outputs/polybeam_validation_summary.csv`
- `outputs/polybeam_validation_summary.meta.json`
- `outputs/polybeam_pair_audit.csv`
- `outputs/polybeam_pair_audit_summary.csv`
- `outputs/doppler_scale_selected_cells.csv`
- `outputs/doppler_scale_summary.csv`
- `outputs/doppler_scale_selected_cells.meta.json`
- `outputs/doppler_comb_audit.csv`
- `outputs/doppler_comb_audit_summary.csv`
- `outputs/nearfield_delay_validation.csv`
- `outputs/nearfield_delay_validation_summary.csv`
- `outputs/nearfield_delay_validation.meta.json`
- `outputs/delay_buffer_sensitivity.csv`
- `outputs/delay_buffer_sensitivity_summary.csv`
- `outputs/delay_buffer_sensitivity.meta.json`
- `outputs/background_sensitivity_by_cell.csv`
- `outputs/background_sensitivity_by_lst_stratum.csv`
- `outputs/background_sensitivity_summary.csv`
- `outputs/background_sensitivity_summary.meta.json`
- `figures/coverage_robustness/R1_lst_selection_map.png`
- `figures/coverage_robustness/R2_z_vs_pte_global.png`
- `figures/coverage_robustness/R3_bias_floor_sensitivity.png`
- `figures/coverage_robustness/R4_null_mad_diagnostic.png`
- `figures/coverage_robustness/representative_cases/representative_top_zps_cases.png`
- `figures/coverage_robustness/representative_cases/representative_gate_plane.png`
- `figures/coverage_robustness/representative_cases/representative_delay_buffer_sensitivity.png`

## X1. Critical Questions

### Critical Q1. What is the matched-null PTE?

Answer:

The matched-null PTE is not a probability of "no UEMR" and not a direct false-alarm probability for real sky contamination. It is a conditional stress-test probability against a support-matched, amplitude-matched, phase-randomized control ensemble. It asks whether the coherent injected morphology produces a more extreme statistic than null realizations that keep the same injection support and amplitude envelope while destroying phase organization.

This interpretation must be stated explicitly in the manuscript. The correct claim is conditional: the tested injection does or does not exceed a matched phase-randomized contrast, under this QA operator and this finite null ensemble.

Evidence:

- `paper/paper.tex`, matched-null definition.
- `outputs/coverage_robustness_trials.csv`, columns `PTE_global_max`, `PTE_global_absint`, `n_null`.
- `outputs/coverage_tail_refined_near_threshold.csv`, `N_null=1000` refinement for near-threshold cases.

### Critical Q2. What is the physical meaning of `B_floor`?

Answer:

`B_floor` should be treated as a reporting-level physical-amplitude gate inside this simplified QA operator, not as a calibrated HERA EoR error-budget threshold. The current reproducibility package supports demoting it to a robustness/reporting filter.

The 648-row coverage scan has one statistical candidate before the relative absolute-bias floor. That candidate remains only for floors up to `1e-4` and disappears for floors `1e-3`, `1e-2`, and `1e-1`. Therefore the final statement should be: no candidate survives the adopted reporting floor at `B_floor = 1e-2`; this is not equivalent to a production HERA contamination limit.

Evidence:

- `outputs/coverage_candidate_counts_by_floor_extended.csv`
  - `n_trials = 648`
  - `n_candidate_statistical = 1`
  - `n_candidate_physical = 1` for floors `1e-8` through `1e-4`
  - `n_candidate_physical = 0` for floors `1e-3`, `1e-2`, `1e-1`
- `outputs/absint_floor_recheck.csv`
  - absint floor-hit cases exist, but they are local to the absint gate and do not pass the full final hierarchy.

### Critical Q3. Does TLE subset/remapping control the result?

Answer:

Yes, materially. The reproducibility package now includes both a selected-cell exposure study and full 648-row QA reruns under the archived `first1200` subset and the full available TLE catalog. The selected-cell study compares `first1200` with `all_available` TLE records and five random 1200-record subsets.

The exposure study shows that `first1200` is not exposure-equivalent to the full available catalog. In the 27 selected cells, `first1200` has mean visible-any count 2.81 and mean beam-weighted exposure 0.114, whereas `all_available` has mean visible-any count 18.78 and mean exposure 1.95. The five random 1200 subsets have mean visible-any counts between 2.48 and 4.33 and mean exposures between 0.156 and 0.426. Thus file-order/subset choice changes the visible-satellite and exposure-proxy distribution.

The full-grid rerun shows that this propagates into the final gate. Under the archived `first1200` subset, the 648-row QA grid yields 1 statistical candidate and 0 physical candidates at `B_floor = 1e-2`. Under the full 6364-record catalog, the same 648-row grid yields 18 statistical candidates, with 5 physical candidates remaining at `B_floor = 1e-2` and 0 remaining only at `B_floor = 1e-1`. So TLE subset choice is not a harmless implementation detail; it changes the final candidate count and the floor sensitivity.

Evidence:

- `scripts/analyze_tle_subset_sensitivity.py`
- `scripts/run_coverage_grid.py`
- `outputs/tle_subset_sensitivity_summary.csv`
- `outputs/tle_subset_sensitivity.meta.json`
- `outputs/coverage_robustness_trials.csv`
- `outputs/coverage_robustness_trials_all_tle_full.csv`
- `outputs/all_tle_summary/coverage_candidate_counts_by_floor.csv`
- `outputs/all_tle_summary/coverage_summary_by_factor.csv`

### Critical Q4. Is the full PolyBeam disappearance physical or implementation-driven?

Answer:

Partially answered, and now stronger than a pure sanity check. The package still includes a direct validation of `pathb.satellite.hera_polybeam()`, and that validation confirms strict loading, the expected `(time, frequency)` output shape, zenith normalization to unity, no NaN values, no values above unity after normalization, and frozen/full agreement at 150 MHz.

The paired audit on the full 648-row grid compares frozen and full PolyBeam outcomes for the same 324 unique geometry/morphology/flux/multiplicity pairs. It finds 13 frozen-only statistical candidates and 1 full-only statistical candidate. So the full PolyBeam effect is not a uniform candidate killer; it is candidate-specific. Most frozen-beam candidates disappear under the full chromatic beam, but one case flips the other way, which means the gate outcome is genuinely beam-sensitive rather than an implementation artifact tied to one direction only.

Representative paired examples:

- frozen-only: `0_1`, typical LST bin 44, `smooth`, 30 Jy, `single`
  - `PTE_global_max` changes from `0.009901` to `0.019802`
  - `Z_PS,max` changes from `1.302847` to `1.140746`
- frozen-only: `82_0`, stress LST bin 64, `lines`, 1000 Jy, `single`
  - `PTE_global_max` changes from `0.009901` to `0.019802`
  - `Z_PS,max` changes from `5.798984` to `1.694681`
- full-only: `0_1`, typical LST bin 44, `lines`, 30 Jy, `multi`
  - `PTE_global_max` changes from `0.029703` to `0.009901`
  - `Z_PS,max` changes from `4.319365` to `3.796340`

This supports the interpretation that chromatic beam response can move cases across the candidate threshold in both directions. It does not look like a normalization bug, but it also does not justify saying that full PolyBeam simply suppresses all fixed-beam candidates.

Evidence:

- `scripts/validate_polybeam_response.py`
- `outputs/polybeam_pair_audit.csv`
- `outputs/polybeam_pair_audit_summary.csv`
- `outputs/polybeam_validation_summary.csv`
- `outputs/polybeam_validation_summary.meta.json`

## X2. Section-by-section Answers

### R1. Title / framing

Answer:

The safest framing is "matched-null forward-injection QA framework" rather than Starlink detection. The package supports a forward-injection computational QA claim, not a contemporaneous HERA-Starlink detection claim and not a production cosmological contamination rate.

Evidence:

- `paper/paper.tex`, Introduction and Data/Injection framing.
- `REPRODUCIBILITY.md`.

### R2. Abstract balance

Answer:

The abstract should present the positive contribution as a reproducible QA hierarchy and a conditional non-detection, not as "Starlink-like UEMR is safe." The supported wording is: under the tested conditions, local excursions do not become robust window-integrated, physical-floor, full-beam candidates.

Evidence:

- `outputs/coverage_tail_refined_near_threshold_summary.csv`: `n_refined_cases=4`, `n_refined_PTE_lt_001=0`, `n_refined_beam_robust=0`.
- `outputs/coverage_candidate_counts_by_floor_extended.csv`: no physical candidate at `B_floor >= 1e-3`.

### R3. Novelty

Answer:

The defensible novelty is the reproducible QA separation: local maximum statistic, empirical PTE, absolute window-bias gate, window-integrated statistic, and beam-operator robustness are treated as separate gates. The novelty is not first detection of Starlink UEMR.

Evidence:

- `figures/coverage_robustness/R2_z_vs_pte_global.png`
- `figures/coverage_robustness/R3_bias_floor_sensitivity.png`
- `figures/coverage_robustness/R4_null_mad_diagnostic.png`
- `outputs/coverage_robustness_trials.csv`

### R4. UEMR existence vs EoR-window contamination

Answer:

The package supports the distinction. The injection can create local QA excursions, but final contamination-like claims require additional gates: empirical tail behavior, window-integrated statistic, absolute-bias floor, and full-beam robustness. The manuscript should avoid equating UEMR presence or local delay-spectrum excursion with EoR-window bandpower contamination.

Evidence:

- `outputs/coverage_summary_by_factor.csv`: one statistical candidate at the factor-summary level, but no stable final candidate after refinement and physical/beam gates.
- `outputs/coverage_tail_refined_near_threshold_summary.csv`.

### R5. Relation to prior HERA/RFI literature

Answer:

This is primarily a writing task rather than a new numerical experiment. The defensible relation is that this package is not a replacement for HERA production RFI excision, jackknife, or power-spectrum inference. It is a reproducible pre-production moving-source injection QA layer that separates local statistics, empirical matched-null PTEs, integrated absolute bias, and beam robustness before any claim is promoted to EoR-window contamination.

Evidence:

- `paper/paper.tex`
- `README.md`
- `REPRODUCIBILITY.md`

### R6. HERA-like background choice

Answer:

The reproducibility package includes five HERA-like background crops for the main Jan 2026 background set and a native-baseline background set. This is stronger than a single-background design, but it remains a limited background ensemble. The manuscript should state that it does not replace a production-scale survey over nights, LSTs, calibration states, polarizations, and baseline products.

Evidence:

- `examples/backgrounds_jan2026/*.npz`: five background crops.
- `examples/backgrounds_jan2026_native4bg/manifest.csv`.
- `outputs/lst_bin_selection.csv`: baseline/LST selected rows.

### R7. TLE time remapping

Answer:

The correct interpretation is stress-test injection into HERA-like backgrounds, not simultaneous HERA-Starlink observation. The package does not establish contemporaneous geometry between original HERA observing times and Jan 2026 TLE evaluation times.

Evidence:

- `paper/paper.tex`.
- `configs/coverage_robustness.yaml`.

### R8. TLE subset selection

Answer:

The current evidence now shows that the first-1200 TLE subset is a conditional design choice, not a representative sample of the full available catalog. A metadata-level sensitivity check over the 27 selected coverage cells shows that `all_available` produces much larger visible-satellite counts and beam-weighted exposure than `first1200`; random 1200-record subsets also vary around the first-1200 value. More importantly, the full 648-row QA rerun changes the final gate outcome: the archived `first1200` subset yields 1 statistical candidate and 0 physical candidates at `B_floor = 1e-2`, while the full catalog yields 18 statistical candidates and 5 physical candidates at `B_floor = 1e-2`.

The manuscript should therefore avoid claiming TLE-subset invariance. The defensible statement is that the published QA result is conditional on the archived TLE subset, and the subset sensitivity is a genuine result of the reproducibility folder, not just a limitation note.

Evidence:

- `outputs/tle_subset_sensitivity_summary.csv`
- `outputs/tle_subset_sensitivity.csv`
- `outputs/tle_subset_sensitivity.meta.json`

### R9. Near-field delay calculation

Answer:

Answered for the selected coverage geometry. The package now compares the near-field distance-difference delay,

`tau_near = (|sat_enu - baseline_enu| - |sat_enu|) / c`,

against the plane-wave approximation,

`tau_plane = -dot(baseline_enu, unit_sat_enu) / c`,

for visible satellites in the selected 27 coverage cells.

The near-field minus plane-wave discrepancy is small for the tested geometry: overall median maximum error 0.069 ns, p95 0.137 ns, and maximum 0.172 ns. By baseline class, the maximum errors are about 0.00082 ns for short, 0.099 ns for mid, and 0.172 ns for long baselines. These are much smaller than the 100 ns horizon buffer and less than 0.025% of the horizon delay scale in the selected sample.

Thus the near-field implementation is not producing a large correction relative to the QA window boundary, but the explicit distance-difference formula remains the more appropriate implementation for LEO geometry.

Evidence:

- `scripts/validate_nearfield_delay.py`
- `outputs/nearfield_delay_validation_summary.csv`
- `outputs/nearfield_delay_validation.meta.json`

### R10. Injection amplitude model

Answer:

Partially answered from the implementation. The injection amplitude is a stress-test apparent-visibility model: spectral template × beam response × range attenuation × time/frequency smearing × row-level flux scale. The templates are normalized and then scaled by `S_ref`; this makes the grid interpretable as controlled stress tiers rather than a calibrated Starlink transmitter power reconstruction.

The package does not contain an external physical calibration tying `S_ref` to measured Starlink flux distributions at HERA, nor a full polarization/mutual-coupling/calibration-residual model. The manuscript should therefore describe the amplitude model as controlled apparent-flux stress scaling.

Evidence:

- `pathb/satellite.py`, functions `build_visibility_for_sat`, `spectral_template`, `literature_uemr_spectrum_v2`.
- `configs/pathB_jan2026_main.yaml`.
- `configs/coverage_robustness.yaml`.

### R11. Spectral morphology choices

Answer:

The package supports smooth, lines, and kHz-comb morphology as controlled stress-test classes. It does not by itself prove that the templates are a complete physical model of Starlink UEMR. The manuscript should call them Starlink-like or literature-motivated morphology classes.

Evidence:

- `configs/pathB_jan2026_main.yaml`.
- `configs/coverage_robustness.yaml`.
- `outputs/coverage_summary_by_factor.csv`, factor `morphology`.

### R12. Doppler drift

Answer:

Completed at the stress-test level. The package now estimates topocentric range-rate Doppler shifts for visible satellites in the selected 27 coverage cells using the archived first-1200 TLE subset, and it now reruns the full-catalog 5 physical candidates with `none`, `constant`, and `linear` Doppler modes. At 150 MHz, the selected near-zenith geometry gives median maximum absolute Doppler shift 0.86 kHz, p95 1.07 kHz, and maximum 1.16 kHz. The median visible-bin Doppler span is 1.62 kHz, with p95 2.06 kHz and maximum 2.21 kHz.

Relative to a 48.8 kHz comb spacing, the p95 absolute Doppler shift is about 2.2% of the spacing. Relative to a 12.2 kHz intrinsic line width, it is about 8.8%. Thus Doppler is not the dominant scale for the current selected near-zenith geometry, but it is not exactly zero compared with the intrinsic line width.

The Doppler-included comb injection experiment is now completed for `none`, `constant`, and `linear` track-rate modes on the full-catalog physical candidate set. The result is class-stable across these modes for the tested cases, although SGP4-customized Doppler tracks are still untested.

Evidence:

- `scripts/estimate_doppler_scale.py`
- `outputs/doppler_scale_summary.csv`
- `outputs/doppler_scale_selected_cells.csv`
- `scripts/run_doppler_comb_audit.py`
- `outputs/doppler_comb_audit.csv`
- `outputs/doppler_comb_audit_summary.csv`

### R13. QA operator simplicity

Answer:

The operator is intentionally a simplified reproducible QA operator. It should be described as pre-production QA, not as a replacement for a full HERA power-spectrum pipeline.

Evidence:

- `paper/paper.tex`.
- `pathb/metrics.py`.
- `pathb/pipeline.py`.

### R14. EoR window proxy

Answer:

Partially answered from the implementation. The QA window is a delay-domain proxy based on the horizon delay plus a 100 ns buffer. It is appropriate as a reproducible delay-spectrum QA mask, but it is not a complete HERA production power-spectrum window or cosmological estimator.

The package supports the limited claim that local delay-spectrum excursions and window-integrated proxy statistics can be separated under this operator. It does not validate the proxy against the full HERA downstream power-spectrum pipeline.

Evidence:

- `pathb/metrics.py`
- `pathb/pipeline.py`
- `scripts/run_coverage_grid.py`
- `outputs/coverage_robustness_trials.csv`, columns `n_window_bins`, `tau_ns_min_window`, `tau_ns_max_window`, `kpar_min_window`, `kpar_max_window`.

### R15. `B_rel` definition

Answer:

The output files define and report `relative_abs_bias`, `window_abs_bias_sum`, and `window_bg_abs_sum`. In the current package, the physical gate is implemented as a relative absolute-bias threshold, and the result is highly floor-dependent.

Evidence:

- `outputs/coverage_robustness_trials.csv`.
- `outputs/coverage_candidate_counts_by_floor_extended.csv`.
- `outputs/absint_floor_recheck.csv`.

### R16. Why `Z_PS,max` is not contamination criterion

Answer:

`Z_PS,max` is a local maximum statistic. It can be large because of local null variance/MAD structure and does not necessarily imply window-integrated absolute contamination. The package separates `Z_PS_max`, `PTE_global_max`, `PTE_global_absint`, and `relative_abs_bias`.

Evidence:

- `outputs/coverage_robustness_trials.csv`.
- `figures/coverage_robustness/R2_z_vs_pte_global.png`.
- `figures/coverage_robustness/R4_null_mad_diagnostic.png`.

### R17. Matched-null differences across experiments

Answer:

Partially answered. The active coverage/PTE analysis uses the global PTE fields in `outputs/coverage_robustness_trials.csv` and tail-refinement outputs, while older matched-null morphology material is not the source of record for the current coverage claim. The manuscript should keep these null definitions separated and avoid mixing older `matched-null p95 exceedance` language with the active `PTE_global,max` / `PTE_global,absint` hierarchy unless a bridging table is included.

Evidence:

- `README.md`, active release notes.
- `REPRODUCIBILITY.md`, source-of-record statement.
- `outputs/coverage_robustness_trials.csv`.
- `outputs/coverage_tail_refined_near_threshold.csv`.

### R18. PTE resolution and multiple testing

Answer:

At `N_null=100`, empirical PTE resolution is approximately `1/101 = 0.0099`, so strict `PTE < 0.01` cases sit at the resolution boundary. The package addresses this partly by refining near-threshold cases with `N_null=1000`; none remains below the strict `PTE_global,max < 0.01` criterion. The 648 rows are correlated and should not be treated as independent family-wise tests.

Evidence:

- `outputs/coverage_tail_refined_near_threshold.csv`.
- `outputs/coverage_tail_refined_near_threshold_summary.csv`: `n_refined_PTE_lt_001=0`.
- `outputs/coverage_summary_by_factor.csv`.
- `outputs/coverage_summary_by_baseline_lst.csv`.

### R19. Controlled-grid factor design

Answer:

The coverage grid is factorial over baseline/LST selections, beam model, morphology, flux, and multiplicity as recorded in `coverage_robustness_trials.csv`. The package supports factor-level summaries but not full independence among rows.

Evidence:

- `outputs/coverage_robustness_trials.csv`.
- `outputs/coverage_summary_by_factor.csv`.

### R20. Native 9-baseline selection

Answer:

The package contains nine native baseline IDs in the coverage outputs: `0_1`, `0_197`, `11_10`, `157_146`, `159_76`, `23_80`, `4_196`, `4_76`, and `82_0`. The current files document their use but do not prove population-level representativeness of all HERA baselines.

Evidence:

- `outputs/lst_bin_selection.csv`.
- `outputs/coverage_summary_by_baseline_lst.csv`.

### R21. LST bin selection leakage

Answer:

The LST selection file records metadata-based selection rules (`quiet`, `typical`, `stress`) and includes exposure, flag fraction, background-window power proxy, null-MAD proxy, and pre-risk scores. This supports a pre-injection metadata-selection claim, provided the manuscript states that these are metadata/proxy criteria and not post hoc injected-result criteria.

Evidence:

- `outputs/lst_bin_selection.csv`, columns `selection_rule`, `beam_weighted_sat_exposure`, `bg_window_power_proxy`, `null_mad_win_proxy`, `pre_risk_score`.

### R22. Full-bin metadata correction

Answer:

The package contains full-bin metadata outputs and a selection map figure. It supports stating that selection is based on bin-level metadata rather than only a bin-center proxy. Quantitative before/after comparison is not filled here unless the package contains an explicit old-vs-new comparison table.

Evidence:

- `outputs/lst_bin_metadata.csv`.
- `outputs/lst_bin_selection.csv`.
- `figures/coverage_robustness/R1_lst_selection_map.png`.

### R23. Coverage-grid row construction

Answer:

The coverage robustness grid has 648 rows. It combines 27 baseline/LST selections with 2 beam models, 3 morphologies, 2 flux tiers, and 2 multiplicity states: `27 x 2 x 3 x 2 x 2 = 648`.

Evidence:

- `outputs/coverage_robustness_trials.csv`.
- `outputs/coverage_summary_by_baseline_lst.csv`: 27 baseline/LST groups, each with 24 rows.

### R24. Local robust-z and PTE mismatch

Answer:

The package supports a mismatch claim at the QA level. Factor summaries show one `Z_PS > 3` and one statistical candidate in the 648-row grid, but tail refinement and physical/beam gates remove stable final candidates. This shows local excursions should not be promoted directly to contamination claims.

Evidence:

- `outputs/coverage_summary_by_factor.csv`.
- `outputs/coverage_tail_refined_near_threshold_summary.csv`.
- `figures/coverage_robustness/R2_z_vs_pte_global.png`.

### R25. Strict local max candidate count

Answer:

The strongest supported statement is: after `N_null=1000` refinement of near-threshold cases, no refined case remains below `PTE_global,max < 0.01`, and no refined case passes physical-floor or beam-robust filters. The statement is conditional on the chosen near-threshold refinement set, not a complete sensitivity proof.

Evidence:

- `outputs/coverage_tail_refined_near_threshold_summary.csv`.
- `outputs/excluded_pte_candidate_n1000_summary.csv`.

### R26. Absint floor-hit interpretation

Answer:

Absint floor-hit cases exist, but the floor gate is not physically calibrated. In the recheck file, some absint cases remain below `PTE_global_absint < 0.01` at `N_null=1000`, but they should be described as absint-tail diagnostics unless they also pass the adopted physical and beam-robust gates.

Evidence:

- `outputs/absint_floor_recheck.csv`.
- `outputs/coverage_candidate_counts_by_floor_extended.csv`.

### R27. Beam robustness

Answer:

The package supports paired frozen/full PolyBeam comparisons in the coverage grid, and the summary indicates no beam-robust strict candidate after refinement. It now also includes a direct PolyBeam response sanity check. The check validates strict PolyBeam loading, output shape, zenith normalization, finite values, unit clipping behavior, and frozen/full agreement at 150 MHz.

Remaining limitation: the package still does not contain a full per-candidate flux-conservation audit showing exactly why each frozen-beam candidate fails under the full chromatic PolyBeam.

Evidence:

- `configs/pathB_jan2026_polybeam_frozen150.yaml`.
- `configs/pathB_jan2026_polybeam_robustness.yaml`.
- `outputs/coverage_tail_refined_near_threshold_summary.csv`: `n_beam_robust=0`.
- `outputs/polybeam_validation_summary.csv`.
- `outputs/polybeam_validation_summary.meta.json`.

### R28. Figure/table presentation

Answer:

The package already has the key coverage figures: LST selection map, Z-vs-PTE scatter, bias-floor sensitivity, and null-MAD diagnostic. The manuscript should cite them directly if these results are part of the final paper.

Evidence:

- `figures/coverage_robustness/R1_lst_selection_map.png`
- `figures/coverage_robustness/R2_z_vs_pte_global.png`
- `figures/coverage_robustness/R3_bias_floor_sensitivity.png`
- `figures/coverage_robustness/R4_null_mad_diagnostic.png`

### R29. Negative-result scope

Answer:

The negative result must be scoped to the tested QA operator, background package, TLE/remapping setup, beam implementations, morphology classes, flux tiers, and gates. It is not a general "Starlink-like UEMR is safe" claim.

Evidence:

- `outputs/coverage_robustness_trials.csv`.
- `configs/coverage_robustness.yaml`.

### R30. Local excursion is not integrated contamination

Answer:

The package supports this claim. Local `Z_PS`/`PTE_global_max` behavior is separated from `PTE_global_absint` and `relative_abs_bias`; candidate counts collapse after physical floor and beam/refinement checks.

Evidence:

- `outputs/coverage_summary_by_factor.csv`.
- `outputs/coverage_candidate_counts_by_floor_extended.csv`.
- `outputs/coverage_tail_refined_near_threshold_summary.csv`.

### R31. LST metadata correction meaning

Answer:

The package supports using full-bin metadata for LST selection and documents selected bins, exposure proxies, and pre-risk scores. The broader claim that this changes previous literature is not supported by a dedicated comparison in this package.

Evidence:

- `outputs/lst_bin_metadata.csv`.
- `outputs/lst_bin_selection.csv`.

### R32. Pre-production QA framing

Answer:

Supported. The results should be framed as reproducible pre-production QA for moving-source UEMR stress testing, not as a final production HERA analysis.

Evidence:

- `README.md`.
- `REPRODUCIBILITY.md`.
- `paper/paper.tex`.

### R33. Background realization limitation

Answer:

The package improves on a single-background analysis by including multiple backgrounds and native-baseline background products, but it is still limited. A production claim would need more nights, LST ranges, flagging states, calibrations, polarizations, and baseline products.

Evidence:

- `examples/backgrounds_jan2026/`.
- `examples/backgrounds_jan2026_native4bg/`.

### R34. Trial independence/look-elsewhere effect

Answer:

The 648 rows are not independent family-wise trials. They share baseline selections, LST strata, beam models, morphologies, flux tiers, and multiplicity settings. The `648 x 0.01` style expectation should not be presented as a rigorous FWER calculation. It can only be used, if at all, as a rough descriptive scale, and the manuscript should explicitly say so.

Evidence:

- `outputs/coverage_robustness_trials.csv`.
- `outputs/coverage_summary_by_factor.csv`.
- `outputs/coverage_summary_by_baseline_lst.csv`.

### R35. Matched-null limitation

Answer:

Supported. The matched null is a phase-randomized matched-injection contrast, not a no-satellite sky ensemble. The manuscript should state that absence of matched-null exceedance does not prove absence of contamination under other null definitions.

Evidence:

- `outputs/coverage_robustness_trials.csv`.
- `pathb/nulls.py`.

### R36. `B_floor` physical calibration limitation

Answer:

Supported. The floor is a reporting gate in the current package. No file here calibrates `B_floor` to a HERA thermal-noise covariance, foreground leakage budget, or final EoR upper-limit tolerance.

Evidence:

- `outputs/coverage_candidate_counts_by_floor_extended.csv`.
- `outputs/absint_floor_recheck.csv`.

### R37. Doppler limitation

Answer:

Completed for the tested modes. The selected-cell Doppler diagnostic gives p95 absolute Doppler shift 1.07 kHz and p95 visible-bin Doppler span 2.06 kHz at 150 MHz. This is small compared with 48.8 kHz comb spacing but nonzero relative to 12.2 kHz intrinsic line width. The package now includes Doppler-shifted injection reruns for `none`, `constant`, and `linear` modes on the full-catalog physical candidate set, and those cases remain class-stable.

Evidence:

- `outputs/doppler_scale_summary.csv`
- `outputs/doppler_scale_selected_cells.meta.json`
- `scripts/run_doppler_comb_audit.py`
- `outputs/doppler_comb_audit.csv`
- `outputs/doppler_comb_audit_summary.csv`

### R38. Code/data availability

Answer:

The reproducibility package contains configs, scripts, pipeline modules, outputs, figures, TLE files, examples, and manuscript files. It is a plausible reproduction root. The remaining publication requirement is to remove files not used by the paper and ensure every quoted count in the paper is regenerated or traceable from this root.

Evidence:

- `README.md`.
- `REPRODUCIBILITY.md`.
- `MANUSCRIPT_FILE_CHECK.md`.
- `scripts/`.
- `pathb/`.
- `outputs/`.
- `figures/`.

### R39. Macro-based counts

Answer:

The risk is real unless manuscript counts are generated from files in this package. Counts should be tied to scripts or output tables, not manually maintained macros.

Evidence:

- `paper/paper.tex`.
- `outputs/*.csv`.

### R40. Astronomy and Computing fit

Answer:

The strongest journal-fit argument is reproducible computational QA: packaged configs, scripts, outputs, figures, and a staged decision hierarchy for moving-source injection tests.

Evidence:

- `README.md`.
- `REPRODUCIBILITY.md`.
- `scripts/`.
- `pathb/`.

### R41. Journal comparison

Answer:

The best fit argument for Astronomy and Computing is reproducibility and method packaging: configs, scripts, output tables, figures, and a staged QA decision hierarchy are the primary contribution. A more physics-focused journal would likely require stronger physical calibration of the flux scale, TLE population sensitivity, Doppler-shifted injections, and full downstream HERA power-spectrum validation.

Evidence:

- `README.md`
- `REPRODUCIBILITY.md`
- `scripts/`
- `outputs/`
- `figures/coverage_robustness/`

## X3. Additional-analysis Checklist

### A1. Matched-null interpretation table

Answer:

Can be written now from existing definitions and outputs.

### A2. TLE subset sensitivity

Answer:

Completed at the level needed to answer the reviewer concern. The package compares `first1200`, `all_available`, and five random 1200-record subsets for the selected 27 LST/bin cells using reconstructed full-bin time samples, and it reruns the full 648-row QA grid for the archived `first1200` subset and for the full 6364-record catalog. The result is not just non-negligible subset dependence in visible-satellite count and beam-weighted exposure; the final candidate statistics also move. `first1200` gives 1 statistical candidate and 0 physical candidates at `B_floor = 1e-2`, while the full catalog gives 18 statistical candidates, 5 physical candidates at `B_floor = 1e-2`, and 0 only at `B_floor = 1e-1`.

The manuscript should not claim TLE-subset invariance. The defensible statement is that the QA result is conditional on the archived first-1200 TLE subset, and that the full available catalog changes the final gate outcome.

Evidence:

- `scripts/analyze_tle_subset_sensitivity.py`
- `scripts/run_coverage_grid.py`
- `outputs/tle_subset_sensitivity_summary.csv`
- `outputs/coverage_robustness_trials.csv`
- `outputs/coverage_robustness_trials_all_tle_full.csv`
- `outputs/all_tle_summary/coverage_candidate_counts_by_floor.csv`
- `outputs/tle_subset_sensitivity.meta.json`

### A3. Full PolyBeam validation

Answer:

Completed to the level needed for the reviewer concern. `scripts/validate_polybeam_response.py` still provides the beam-loading sanity check, and the new paired audit on the 648-row full-TLE grid compares frozen and full PolyBeam outcomes for the same 324 geometry/morphology/flux/multiplicity pairs. It finds 13 frozen-only statistical candidates and 1 full-only statistical candidate.

That is enough to reject the simplistic reading that the full PolyBeam merely erases fixed-beam excesses by a normalization bug. The beam implementation behaves consistently, but the candidate gate is beam-sensitive: one representative frozen-only case flips from `PTE_global_max = 0.009901` to `0.019802`, while one full-only case flips from `0.029703` to `0.009901`.

Evidence:

- `scripts/validate_polybeam_response.py`
- `outputs/polybeam_pair_audit.csv`
- `outputs/polybeam_pair_audit_summary.csv`
- `outputs/polybeam_validation_summary.csv`
- `outputs/polybeam_validation_summary.meta.json`

### A4. Doppler-included comb sensitivity

Answer:

Completed for the tested modes. The Doppler scale diagnostic remains useful, but the actual comb template has now been rerun with time-dependent Doppler shifts (`none`, `constant`, and `linear`) on the full-catalog physical candidate set. The tested cases are class-stable under these modes.

Evidence:

- `scripts/estimate_doppler_scale.py`
- `outputs/doppler_scale_summary.csv`
- `outputs/doppler_scale_selected_cells.csv`
- `scripts/run_doppler_comb_audit.py`
- `outputs/doppler_comb_audit.csv`
- `outputs/doppler_comb_audit_summary.csv`

### A5. Delay-window buffer sensitivity

Answer:

Partially completed as a representative-case sensitivity test. The package now reruns the highest-priority statistical/excursion case over buffer values `0, 50, 100, 150, 200 ns` with `N_null=100`. This is not a full 648-row rerun, so it should be presented as a local sensitivity diagnostic.

For the representative case (`0_1`, typical LST bin 44, full PolyBeam, smooth, 30 Jy, multi), the local statistic is buffer-sensitive: `Z_PS,max` is 0.97 at 0 ns, 3.96 at 50 ns, 3.90 at 100 ns, 0.29 at 150 ns, and -0.55 at 200 ns. The strict statistical candidate appears only at the 100 ns setting in this run, and no buffer setting passes the `B_rel > 1e-2` physical gate.

Interpretation: the local maximum statistic is sensitive to the chosen delay-window boundary in at least this representative case. The final physical-candidate conclusion remains zero for this case because `relative_abs_bias` stays around `5e-4`, far below the reporting floor. A complete buffer-sensitivity claim would still require rerunning the full 648-row grid.

Evidence:

- `scripts/run_delay_buffer_sensitivity.py`
- `outputs/delay_buffer_sensitivity.csv`
- `outputs/delay_buffer_sensitivity_summary.csv`
- `figures/coverage_robustness/representative_cases/representative_delay_buffer_sensitivity.png`

### A6. `B_floor` sensitivity and physical demotion

Answer:

Can be written now. The key result is one statistical candidate in 648 rows, one physical candidate only up to floor `1e-4`, and zero physical candidates for floors `1e-3`, `1e-2`, and `1e-1`.

Evidence:

- `outputs/coverage_candidate_counts_by_floor_extended.csv`.

### A7. Background realization sensitivity

Answer:

Partially completed as a post-hoc summary of the existing 648-row coverage grid. The package now groups results by `baseline_id`, `baseline_class`, `lst_stratum`, and `lst_bin_id` to quantify cell-to-cell variation. This is not a new injection rerun; it summarizes the existing coverage results.

Across 27 baseline/LST cells, cell-median `Z_PS,max` ranges from -1.38 to 1.30, and the maximum cell-level p95 `Z_PS,max` is 2.66. The minimum cell-level `PTE_global,max` reaches 0.0099 in one cell. Cell-median `relative_abs_bias` ranges from 0 to 0.026, while the maximum cell-level p95 `relative_abs_bias` is 0.610. Only one cell contains a statistical candidate, and zero cells contain a physical candidate at the `B_rel > 1e-2` gate.

Interpretation: the background/LST/baseline cell changes the scale of local and integrated diagnostics, so background realization cannot be treated as irrelevant. However, the existing grid still yields zero physical-floor candidates at the adopted `1e-2` gate.

Evidence:

- `scripts/summarize_background_sensitivity.py`
- `outputs/background_sensitivity_by_cell.csv`
- `outputs/background_sensitivity_by_lst_stratum.csv`
- `outputs/background_sensitivity_summary.csv`

### A8. Grid-level multiple-testing correction

Answer:

Can be partially written now as a limitation: the rows are correlated and no rigorous FWER correction is claimed. A formal correction or hierarchical model is not present.

### A9. Representative case figures

Answer:

Completed at the summary-figure level. The package now includes three representative figures:

- `representative_top_zps_cases.png`: top local-excursion rows by `Z_PS,max`.
- `representative_gate_plane.png`: all 648 rows in the `PTE_global,max` vs `relative_abs_bias` gate plane.
- `representative_delay_buffer_sensitivity.png`: buffer sensitivity for the representative high-priority case.

These are manuscript-support figures, not per-delay residual profile plots. If the final paper needs line-by-line delay residual morphology for a specific candidate, additional per-case delay-profile figures should still be generated.

Evidence:

- `scripts/plot_representative_cases.py`
- `figures/coverage_robustness/representative_cases/manifest.csv`
- `figures/coverage_robustness/representative_cases/representative_top_zps_cases.png`
- `figures/coverage_robustness/representative_cases/representative_gate_plane.png`
- `figures/coverage_robustness/representative_cases/representative_delay_buffer_sensitivity.png`

### A10. Reproducibility package

Answer:

Can be written now, after pruning non-paper files and ensuring paper counts trace to this folder.
