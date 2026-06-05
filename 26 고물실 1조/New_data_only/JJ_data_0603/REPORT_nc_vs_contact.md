# NC vs Contact lateral-broadening 진단 리포트
**대상:** `JJ_AFM_2.nid` (non-contact) · `JJ_AFM_4.nid` (contact)
**방법:** `analyze_nc_vs_contact.py` (단면 추출은 `measure_nid.py` 메커니즘 그대로 재현, CSV rows 2·4·5·11)

---

## 0. 증상 요약
같은 시료·같은 20 µm 영역에서, **동일한 두 물체의 height 는 사실상 같은데 NC 의 lateral 폭(FWHM)만 수백 nm 더 넓다.**

| 물체 | height (C / NC) | FWHM contact | FWHM NC | NC 초과폭 |
|---|---|---|---|---|
| tall | 96.9 / 95.3 nm | **396 nm** | **930 nm** | +534 nm |
| short | 22.8 / 22.0 nm | **439 nm** | **766 nm** | +327 nm |

(추출값은 `nid_measurements.csv` 와 ±0.5 % 일치 — 재현 검증 통과)

이상적인 NC 는 오히려 contact 보다 좁아야 하므로, 이 broadening 은 NC 의 정상 거동이 아니라 **artifact regime** 의 신호다. 원인을 단일 요인으로 보지 않고 **(1) 시료 형상 + (2) tip convolution + (3) long-range** 세 성분으로 분해해 판정한다.

---

## 1. 가설 설정 (.nid 헤더에서)

두 파일의 헤더 비교 (결정적 차이는 **굵게**):

| 항목 | JJ_AFM_2 (NC) | JJ_AFM_4 (contact) |
|---|---|---|
| Op. mode | **Dynamic Force** | **Static Force** |
| **Cantilever** | **Dyn190Al** | **Stat0.2LAuD** |
| Setpoint | 50 % (진폭) | 5 nN |
| Vibration | 165.359 kHz / 256 mV | — |
| Tip voltage | 0 V | 0 V |
| I-Gain | 1000 | 1000 |
| Image / pixels / time-line | 20 µm / 1.02k / 500 ms | 동일 |
| 환경 | Air | Air |

→ 픽셀 크기·스캔 속도·bias·환경은 **동일**해 비교 조건은 깨끗하다. 단, **두 모드가 서로 다른 프로브**(Dyn190Al ↔ Stat0.2LAuD)를 썼다는 점이 결정적이다 — tip convolution 차이가 모드 차이와 **혼재(confounded)** 한다.

이로부터 세 가설:

- **H1 (시료 사다리꼴):** top plateau 가 작다는 관찰은 tip 으로 설명되지 않는다. convex tip 은 평평한 top 폭을 보존하고 base 만 넓히므로, **작은 top 은 시료가 진짜로 사다리꼴(유한 측벽)** 이라는 증거. 두 모드 공유 형상이어야 한다.
- **H2 (tip convolution):** 서로 다른 프로브 → 유효 팁 반경 차이가 base 를 다르게 넓힌다. 단 geometric broadening 은 height 의존적: Δw ≈ 2√(2Rh).
- **H3 (long-range):** dynamic 모드만 gap 에서 vdW/정전기력을 감지한다(0 V 인가여도 CPD≠0, contact 는 기계적 접촉이라 면역). geometric 으로 안 메워지는 잔차 → long-range 폭 σ_LR. **height 의존이 약하고** top 을 둥글게 만든다.

---

## 2. 데이터 분석

### 2-1. 형상 분해와 model selection (그림 `FIT_overlay.png`)
Forward model: `apparent(x) = [trapezoid(top, sidewall, h) ⊛ tip(R)] ⊛ LR(σ)`
- contact+NC 를 **공유 참형상**으로 결합 적합. 두 경쟁 모델을 비교:
  - **M1 (tip-only):** NC = 참형상 ⊛ tip. long-range 없음.
  - **M2 (tip+LR):** NC = 참형상 ⊛ tip ⊛ Gaussian(σ_LR).

| 물체 | 추정 참형상 (top / base) | M1 NC RMSE | M2 NC RMSE | σ_LR | **ΔAIC(M1−M2)** | 판정 |
|---|---|---|---|---|---|---|
| tall | ~333 / ~531 nm | 14.8 nm | 12.2 nm | **192 nm** | **+185** | LR 필요 |
| short | ~273 / ~868 nm | 3.6 nm | 3.4 nm | **185 nm** | **+26** | LR 필요 |

- **tip-only(M1) 는 NC 를 재현하지 못한다** — 그림에서 회색 점선(M1)이 NC 데이터(빨강 점)보다 한참 좁다. LR 을 더한 M2(빨강 실선)라야 맞는다. AIC 도 두 물체 모두 M2 를 강하게 지지.
- **σ_LR ≈ 190 nm 가 두 물체에서 거의 동일**(192 vs 185) — height 96 nm 와 22 nm 에서 같은 값. long-range 의 전형적 서명.
- (뉘앙스) M2 의 Gaussian 은 NC top 을 약간 과하게 둥글린다. 실제 NC 는 "샤프한 core + 넓은 skirt" 형태로, **균일 Gaussian 보다 꼬리가 두꺼운 halo**(정전기 Coulomb 장)에 더 가깝다. 즉 LR 의 *필요성*은 robust 하나 커널 모양은 Gaussian 이 1차 근사.

### 2-2. height 의존성 — geometric 배제 (그림 `HEIGHT_DEP.png`)
NC 초과폭을 quadrature 로 정의: √(FWHM_NC² − FWHM_C²)

| 물체 | h | 초과폭 |
|---|---|---|
| tall | 96 nm | 841 nm |
| short | 22 nm | 627 nm |

- 관측 비 **tall/short = 1.34**. 순수 sphere-tip 이면 ∝√h → 예측 비 **√(96/22) = 2.07**.
- 관측이 √h 예측보다 **훨씬 작다(거의 상수)** → broadening 의 큰 부분이 **height 에 무관** = geometric tip convolution 으로 설명 불가, **long-range 성분이 지배**. 2-1 의 σ_LR ≈ const 와 정확히 일치.

### 2-3. top plateau (그림 `PROFILES.png`)
peak 의 ≥90 % 구간 폭: tall 에서 contact 156 nm → **NC 411 nm**, short 에서 176 → **333 nm**. NC 의 정점이 **둥글게 퍼진다**(plateau 보존이 아니라 rounding). convex tip dilation 은 평평한 top 을 보존하므로, 이 rounding 은 tip 이 아니라 long-range 의 작용.

### 2-4. parachuting 배제 (그림 `FB_asym.png`)
| | contact_tall | nc_tall |
|---|---|---|
| FWHM Forward / Backward | 396 / 398 | 930 / 983 |
| \|F−B\| | 1 nm | 53 nm |
| 좌/우 flank 비대칭 | 28 % | 39 % |

- NC 가 **Forward·Backward 양방향 모두** 넓다(930·983). feedback parachuting 이면 fast-scan 한쪽에만 꼬리가 생겨야 하므로, broadening 의 주원인이 아니다. NC 의 약간의 비대칭(39 %)은 부차적 기여.

---

## 3. 결론

**broadening 은 세 성분의 합이며, 지배 요인은 long-range 다.**

1. **시료는 실제 사다리꼴**(유한 측벽, 참 top ~300 nm 규모) — 두 모드 공유. "top plateau 가 작다"는 관찰의 정량 근거이며, 사용자 직관대로 **단순 box 가 아니다**. (단 top 이 0 은 아니고 ~300 nm 수준이며, 측정상 작아 보이는 156 nm 는 tip 의 corner-rounding 도 일부 포함.)
2. **tip convolution 은 기여하지만 단독으로는 NC 를 설명 못 한다.** 서로 다른 프로브(Dyn190Al↔Stat0.2LAuD)라 무시할 수 없으나, ① tip-only 모델(M1)이 NC 형상 재현 실패, ② 초과폭의 height-비의존성이 geometric 지배를 배제.
3. **long-range 가 NC 초과폭의 주원인** (1순위). 근거: ① M2≫M1 (ΔAIC +185/+26), ② σ_LR ≈ 190 nm 가 height 에 무관, ③ NC top 의 rounding, ④ "core+넓은 skirt(halo)" 프로파일, ⑤ 물리적으로 contact 는 면역(접촉)·dynamic 은 gap 에서 long-range 감지(0 V 여도 CPD≠0 → 정전기/전하 halo). 이전 논의의 **전하/CPD halo** 시나리오와 정합.
4. **parachuting 은 부차적** — 양방향 모두 broadening, 비대칭은 작음.

**한계:** 두 모드가 다른 프로브라 mode vs tip 이 원천적으로 혼재한다. LR 커널을 Gaussian 으로 근사(실제는 더 두꺼운 꼬리). short feature 는 SNR 이 낮아 fit 신뢰도 낮음(결론은 주로 tall + 두 물체 공통 height-비의존성에 근거).

### 다음 측정 권고 (오프라인 데이터로는 불가, 컨트롤 실험 필요)
- **같은 프로브로 양 모드** 측정 → tip 혼재 제거(가장 깨끗한 대조).
- **lift/setpoint sweep**(더 가까이) → 폭이 줄면 long-range 확정, 안 변하면 geometric.
- **KPFM 로 CPD nulling** 후 NC 재측정 → 폭이 붕괴하면 전하/전위 halo 확정.
- **blind tip reconstruction** → 두 팁 실제 R 측정해 H2 정량 보정.

---

### 산출 파일
- `analyze_nc_vs_contact.py` — 추출·분해·그림 일괄 실행
- `PROFILES.png` `FIT_overlay.png` `HEIGHT_DEP.png` `FB_asym.png`
