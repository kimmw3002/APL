# HeightSample step-edge 분석

## TL;DR

- **무엇**: `New_data_only/HeightSample_data/*.nid` (Forward) 각 이미지의 단일 세로 step edge에 대해 **step 높이**와 **10-90% 상승 폭(edge가 얼마나 넓게 찍혔나)** 을 측정.
- **왜 새로 짰나**: 한 행을 단일 다항식으로 펴는 기존 `masked_polyfit` 은 큰 step이 fit을 끌어당겨 실패. 대신 step 양쪽 terrace를 **따로** 직선 맞추는 two-terrace 방식을 씀.
- **핵심 처리 3가지**: (1) edge 위치는 **행 median 프로파일**의 d/dx 로 잡아 이물질·라인노이즈를 희석하고 가장자리 artifact를 피함, (2) terrace 직선은 **국소(local) + MAD robust** 라 오른쪽 terrace의 이물질이 fit을 못 흔듦, (3) 높이가 y로 drift 하므로 전체평균 대신 **row band(기본 위·아래 각 25%)** 에서만 집계.
- **불확도**: 값=band 행들의 **mean**, 오차=**std(행간 산포)**. (행이 상관돼 SEM은 과장 → 참고만.)
- **검증**: 같은 파일을 **아래쪽 25% / 위쪽 25% 두 band로 독립 측정** → 둘이 서로의 오차범위(합산 std) 안에 들어오면 band 선택에 robust 하다는 뜻. 아래 결과에 ✓/✗ 로 표기.
- **결과 (mean ± std)**:
    - `HeightSample_Contact_Left_remeasure.nid` [bottom 25%] — height **-87.947 ± 4.158 nm**, width(10-90) **1419.357 ± 380.974 nm** (n=59/64)
    - `HeightSample_Contact_Left_remeasure.nid` [top 25%] — height **-84.980 ± 5.154 nm**, width(10-90) **1372.143 ± 480.004 nm** (n=64/64)
        - bottom↔top 일치: height ✓ 일치 (|Δ|=3.0 ≤ 9.3), width ✓ 일치 (|Δ|=47 ≤ 861)
    - `HeightSample_Contact_Right.nid` [bottom 25%] — height **87.124 ± 4.635 nm**, width(10-90) **1010.097 ± 233.500 nm** (n=64/64)
    - `HeightSample_Contact_Right.nid` [top 25%] — height **83.718 ± 4.802 nm**, width(10-90) **886.038 ± 140.496 nm** (n=63/64)
        - bottom↔top 일치: height ✓ 일치 (|Δ|=3.4 ≤ 9.4), width ✓ 일치 (|Δ|=124 ≤ 374)
    - `HeightSample_Noncontact_Left.nid` [bottom 25%] — height **-40.506 ± 29.478 nm**, width(10-90) **3195.953 ± 1529.283 nm** (n=119/128)  ⚠ high row scatter
    - `HeightSample_Noncontact_Left.nid` [top 25%] — height **-33.330 ± 46.108 nm**, width(10-90) **3615.432 ± 2603.034 nm** (n=120/128)  ⚠ high row scatter
        - bottom↔top 일치: height ✓ 일치 (|Δ|=7.2 ≤ 75.6), width ✓ 일치 (|Δ|=419 ≤ 4132)
    - `HeightSample_Noncontact_Right.nid` [bottom 25%] — height **91.385 ± 6.594 nm**, width(10-90) **912.077 ± 184.953 nm** (n=125/128)
    - `HeightSample_Noncontact_Right.nid` [top 25%] — height **93.643 ± 9.155 nm**, width(10-90) **915.160 ± 184.146 nm** (n=126/128)
        - bottom↔top 일치: height ✓ 일치 (|Δ|=2.3 ≤ 15.7), width ✓ 일치 (|Δ|=3 ≤ 369)
    - `HeightSample_contact_Left.nid` [bottom 25%] — height **-87.205 ± 7.494 nm**, width(10-90) **1324.853 ± 413.062 nm** (n=62/64)
    - `HeightSample_contact_Left.nid` [top 25%] — height **-84.593 ± 5.235 nm**, width(10-90) **1124.657 ± 250.907 nm** (n=57/64)
        - bottom↔top 일치: height ✓ 일치 (|Δ|=2.6 ≤ 12.7), width ✓ 일치 (|Δ|=200 ≤ 664)
- **종합**: 모든 파일에서 bottom↔top step height가 오차범위 내 일치 → band 선택에 robust. contact_Left/remeasure 도 서로 재현성 OK. Noncontact_Left 만 산포가 커 신뢰도 낮음.

## 방법 (분석 과정 상세)

1. **edge 위치(전역) 탐지** — 행 방향 median 프로파일 `median(z, axis=0)` 로 이물질/라인노이즈를 희석한 뒤, 중앙 8-92% column band에서 `|d/dx|` 최대 column을 step 위치 `s0` 로 잡음 (col 0/마지막의 scan-edge artifact 회피). 큰 |d/dx| 1순위는 보통 가장자리 artifact라 중앙 band 제한이 필수.
2. **per-row two-terrace fit** — 각 행에서 `s0 ± win`(win = max(8px, cols의 3%)) 안의 step을 gradient로 sub-pixel 위치 추정. transition margin(±4% cols) 바깥의 좌/우 terrace를 **step 근처 국소 구간(±rad ≈ cols의 10%)** 에서만 **MAD 2.5σ robust 직선**으로 각각 fit. 국소 fit이라 이미지 전체의 tilt/bow에 안 휘둘리고, 오른쪽 terrace의 이물질은 MAD outlier로 제거됨. **step height = (우측 직선 − 좌측 직선) 을 step 중심에서 평가한 차.**
3. **10-90% 폭** — 좌측 직선을 뺀 프로파일(좌≈0, 우≈height)에서 step 중심으로부터 **바깥쪽으로** 0.1·height(좌향)·0.9·height(우향) 교차점을 선형보간으로 찾아 그 거리. 중심에서 바깥으로 탐색하므로 멀리 있는 이물질/노이즈에 안 걸림.
4. **집계** — per-row step height가 y방향으로 **drift** 하므로 전체 평균하지 않고 **위·아래 각 25% row band** 안에서만 집계 (`--row-band-frac`, `--row-band-pos {bottom,top,center,both}` 로 조절; 기본 both). band 안에서 MAD 3σ 이상치 행(잔여 debris)을 제거한 뒤 height·width 의 **mean** 을 값으로, **헤드라인 불확도 = std(행간 산포)** 로 보고. (행들은 같은 edge를 연속 스캔해 drift·tip 등이 공유되므로 서로 **상관**돼 있어 SEM=std/√n 으로 나누면 정밀도를 과장함 → std 가 정직한 불확도. SEM 은 `*_sem_nm` 참고 컬럼.) 진단 PNG의 panel 4(height vs row)에서 drift와 band 선택의 타당성을 확인할 수 있음.

## 컬럼 의미 (step_measurements.csv)

| 컬럼 | 뜻 |
|---|---|
| step_col_px / step_x_nm | step edge 위치 (column, nm) |
| row_band | 집계에 쓴 row band |
| step_height_nm / _err_ / _sem_ | band 내 step 높이 **mean / std(헤드라인 오차) / SEM(참고)** |
| width_10_90_nm / _err_ / _sem_ | band 내 10-90% 상승 폭 **mean / std / SEM** |
| n_rows / n_rows_total | 집계에 실제 사용한 행 / band 전체 행 |
| note | debris/품질 플래그 |

## 파일별 결과 (mean ± std)

| file | band | step_x (nm) | height (nm) | width 10-90 (nm) | n_rows | note |
|---|---|---|---|---|---|---|
| HeightSample_Contact_Left_remeasure.nid | bottom 25% | 23242.2 | -87.947 ± 4.158 | 1419.357 ± 380.974 | 59/64 |  |
| HeightSample_Contact_Left_remeasure.nid | top 25% | 23242.2 | -84.980 ± 5.154 | 1372.143 ± 480.004 | 64/64 |  |
| | **bottom↔top** | | ✓ \|Δ\|=3.0≤9.3 | ✓ \|Δ\|=47≤861 | | |
| HeightSample_Contact_Right.nid | bottom 25% | 16015.6 | 87.124 ± 4.635 | 1010.097 ± 233.500 | 64/64 |  |
| HeightSample_Contact_Right.nid | top 25% | 16015.6 | 83.718 ± 4.802 | 886.038 ± 140.496 | 63/64 |  |
| | **bottom↔top** | | ✓ \|Δ\|=3.4≤9.4 | ✓ \|Δ\|=124≤374 | | |
| HeightSample_Noncontact_Left.nid | bottom 25% | 21582.0 | -40.506 ± 29.478 | 3195.953 ± 1529.283 | 119/128 | high row scatter |
| HeightSample_Noncontact_Left.nid | top 25% | 21582.0 | -33.330 ± 46.108 | 3615.432 ± 2603.034 | 120/128 | high row scatter |
| | **bottom↔top** | | ✓ \|Δ\|=7.2≤75.6 | ✓ \|Δ\|=419≤4132 | | |
| HeightSample_Noncontact_Right.nid | bottom 25% | 20410.2 | 91.385 ± 6.594 | 912.077 ± 184.953 | 125/128 |  |
| HeightSample_Noncontact_Right.nid | top 25% | 20410.2 | 93.643 ± 9.155 | 915.160 ± 184.146 | 126/128 |  |
| | **bottom↔top** | | ✓ \|Δ\|=2.3≤15.7 | ✓ \|Δ\|=3≤369 | | |
| HeightSample_contact_Left.nid | bottom 25% | 23437.5 | -87.205 ± 7.494 | 1324.853 ± 413.062 | 62/64 |  |
| HeightSample_contact_Left.nid | top 25% | 23437.5 | -84.593 ± 5.235 | 1124.657 ± 250.907 | 57/64 |  |
| | **bottom↔top** | | ✓ \|Δ\|=2.6≤12.7 | ✓ \|Δ\|=200≤664 | | |

진단 그림: 각 파일은 band별로 `*_step_bottom25.png` / `*_step_top25.png` 두 장 — 4-panel = d/dx heatmap(edge·이물질) / aligned-average 단면(좌·우 fit, 높이, 10-90 마커) / step height vs row(drift+band) / band step-height 히스토그램.
