# On-Device Text-to-Image Generation: Mobile Diffusion Feasibility Report

> 측정 기기: Samsung Galaxy S23 Ultra (Snapdragon 8 Gen 2, 12GB RAM)
> 실험 완료일: 2026-03-14

---

## 0. 개요

### 0.1 문제 설정

Samsung Galaxy S26 Ultra에는 diffusion 기반 이미지 생성 기능이 탑재되어 있으며, 현재 대부분의 생성 기능은 **서버(cloud)에서 실행**된다.

**표 0.1.** Galaxy S26 Ultra 이미지 생성 기능 현황

| 기능 | 실행 위치 | 측정 latency |
|---|---|---|
| AI Eraser, Move | On-device | ~10s |
| 스타일 (화풍 변환) | Cloud | ~12–20s (원본 유지) |
| 이미지 생성 (배경화면 등) | Cloud | ~8s (1024×1024) |
| 스티커 생성 | Cloud | ~9s (720×720) |
| 스티커 세트 확장 | Cloud | ~25s (9개) |

Creative Studio의 이미지/스티커 생성 기능은 Text Encoder + UNet + VAE Decoder 구성의 Stable Diffusion 계열 모델을 기반으로 분석된다. 만약 on-device로 전환하면 서버 비용과 네트워크 의존성을 제거할 수 있으나, device의 메모리·전력·발열 자원을 직접 소비하게 된다.

**핵심 질문:**
1. SD v1.5 파이프라인을 on-device로 실행하면 현재 cloud latency(~10s)와 비교해 어느 수준인가?
2. 어떤 조건(step 수, precision, backend)에서 latency·resource 측면의 on-device feasibility가 성립하는가?
3. LCM-LoRA(few-step)는 SD v1.5 대비 mobile KPI를 얼마나 개선하는가?

### 0.2 실험 축

- **Model Variant**: SD v1.5 (EulerDiscrete scheduler, CFG 7.5) vs LCM-LoRA (DDIM-style, CFG 1.0)
- **Steps**: SD v1.5 — 20/30/50, LCM-LoRA — 4/8
- **Precision**: FP16 / W8A8 / MIXED_PR (Conv·MatMul·Gemm INT8, LayerNorm FP32)
- **Backend**: QNN HTP(NPU)
- **Perf Mode**: burst / balanced

### 0.3 측정 KPI

**표 0.2.** 측정 KPI 정의

| KPI | 설명 |
|---|---|
| E2E Latency | Tokenize ~ VAE Decode 전체 (ms, warm mean) |
| Stage Breakdown | Text Enc / UNet total / VAE Dec 비중 |
| UNet per-step | step당 추론 시간 |
| Cold Start | Session load + first inference (ms) |
| Peak Memory | NPU(HTP) + App RSS 추정 (MB) |
| Avg Power | 추론 구간 소비전력 delta (mW) |
| Thermal | 시작/종료 온도 (°C) |

**측정 방법:**
- **Latency**: `System.nanoTime()` wall-clock, warmup 2회 제외한 warm trial mean
- **Memory**: VmRSS(`/proc/self/status`) + QAI Hub profiling NPU 메모리 추정 합산
- **Power**: `BatteryManager.BATTERY_PROPERTY_CURRENT_NOW × voltage`, 시스템 전체 소비전력
- **Thermal**: `/sys/class/thermal/thermal_zone*/temp` (SoC 온도)
- **품질 평가**: LPIPS(AlexNet, Zhang et al. CVPR 2018), CLIP Score(`openai/clip-vit-base-patch16` logit scale)
- **실험 통제**: Airplane mode, 고정 brightness, trial 간 cooldown 60s (burst mode)

---

## 1. 양자화 품질 스크리닝

> 양자화 전략 수립·실행·디버깅 기술 내러티브: [`docs/model_optimization.md`](model_optimization.md)

평가 방법: FP32 ORT CPU 출력 vs 양자화 모델 출력 CosSim (`scripts/sd/eval_sd_quant_quality.py`)

**표 1.1.** Component별 양자화 품질 스크리닝 (FP32 대비 CosSim)

| Component | Variant | CosSim | RMSE | Grade | 배포 여부 |
|---|---|---|---|---|---|
| vae_encoder | W8A8 (QAI Hub) | 0.9810 | 0.163 | Marginal | 조건부 |
| text_encoder | W8A16 (AIMET) | 0.9849 | 0.178 | Marginal | 조건부 |
| vae_decoder | W8A8 (QAI Hub) | 0.9827 | 0.136 | Marginal | 조건부 |
| vae_decoder | W8A16 (AIMET) | **0.9999** | 0.005 | Excellent | ✅ |
| unet_base | MIXED_PR | **0.9968** | 0.075 | Good | ✅ |
| unet_lcm | MIXED_PR | 0.9875 | 0.143 | Marginal | 조건부 |
| unet_lcm | W8A16 (AIMET) | 0.7292 | 0.729 | Poor | ❌ |
| unet_base, lcm | INT8 QDQ full | — | — | — | ❌ compile 실패 |

**주요 발견:**
- **UNet INT8 QDQ full compile 실패**: HTP `LayerNormalization`이 full INT8 input/output 미지원 → MIXED_PR로 우회 (Conv·MatMul·Gemm만 INT8, LayerNorm FP32 유지)
- **unet_lcm W8A16**: AIMET calibration 결과 on-device 실행 시 품질 급락 (CosSim 0.7292) → 실험 전 제외
- **vae_decoder W8A16**: qai-hub-models 컴파일 모델에 `/Div+/Clip` postproc 내장 → [0,1] 출력. 앱 `normalized=true` 분기 처리됨

CosSim Marginal 등급(0.980~0.995)은 컴포넌트 단위 스크리닝 통과 후 섹션 2 end-to-end 품질 평가(LPIPS/CLIP Score)로 최종 판정.

---

## 2. Precision 비교 결과

> 목적: best precision config 선정. 고정: SD s20 / LCM s4, burst mode, 5 trials 2 warmup.

### 2.1 Steady-State Latency

**표 2.1.** Steady-State E2E & Stage Breakdown (warm mean, ms)

| ID | Config | Tokenize | TextEnc | UNet | UNet% | VAEDec | E2E |
|---|---|---|---|---|---|---|---|
| A1 | SD fp16 s20 *(baseline)* | 179 | 17 | 13,810 | 94.9% | 528 | 14,559 |
| A2 | SD unet mixed_pr s20 | 185 | 17 | 10,782 | 93.8% | 476 | 11,490 |
| A3 | SD vae w8a8 s20 | 170 | 17 | 13,866 | 97.4% | 146 | 14,229 |
| **A4** | **SD mixed_pr + vae w8a8 s20** | **166** | **16** | **10,950** | **97.0%** | **130** | **11,286** |
| A5 | LCM fp16 s4 *(LCM baseline)* | 163 | 9 | 1,364 | 67.4% | 468 | 2,023 |
| **A6** | **LCM vae w8a8 s4** | **173** | **9** | **1,366** | **80.7%** | **126** | **1,693** |

*E2E = Tokenize + TextEnc + UNet + VAEDec + Postproc(~15–27ms).*

SD의 경우 UNet이 E2E의 **95% 이상**을 점유하며, step 수 감소가 가장 직접적인 latency 절감 수단이다. LCM은 step 수(4)가 적어 UNet 비중이 67~81%로 낮아지고 VAEDec 비중이 상대적으로 증가한다.

### 2.2 Cold Start

**표 2.2.** Cold Start Breakdown (ms)

| ID | Config | Session Init | 1st Infer | Cold E2E |
|---|---|---|---|---|
| A1 | SD fp16 s20 | 6,879 | 14,617 | 21,496 |
| A2 | SD unet mixed_pr s20 | 4,864 | 11,632 | 16,496 |
| A3 | SD vae w8a8 s20 | 6,378 | 14,290 | 20,668 |
| **A4** | **SD mixed_pr + vae w8a8 s20** | **4,775** | **11,403** | **16,178** |
| A5 | LCM fp16 s4 | 6,573 | 2,094 | 8,667 |
| **A6** | **LCM vae w8a8 s4** | **6,245** | **1,754** | **7,999** |

*Session Init = QNN HTP graph compile 포함. 앱 최초 실행 또는 모델 교체 시 발생, 이후 재사용 시 skip. Cold E2E = Session Init + 1st Inference (앱 최초 실행 → 첫 이미지).*

### 2.3 System Resource

**표 2.3.** System Resource (burst mode)

| ID | Config | Thermal Idle (°C) | Thermal End (°C) | Power Idle (mW) | Power AvgInfer (mW) | Mem Total (MB) |
|---|---|---|---|---|---|---|
| A1 | SD fp16 s20 | 30.9 | 52.0 | 627 | 5,580 | 2,399 |
| A2 | SD unet mixed_pr s20 | 27.3 | 43.8 | 525 | 4,638 | 1,765 |
| A3 | SD vae w8a8 s20 | 28.9 | 51.6 | 545 | 5,297 | 2,354 |
| **A4** | **SD mixed_pr + vae w8a8 s20** | **31.3** | **46.5** | **494** | **4,525** | **1,709** |
| A5 | LCM fp16 s4 | 31.6 | 39.5 | 510 | 4,367 | 2,372 |
| **A6** | **LCM vae w8a8 s4** | **30.9** | **37.5** | **610** | **3,999** | **2,320** |

*Thermal: SoC 온도. Power: BatteryManager 기반 시스템 전체 소비 전력. Mem = NPU(HTP) + App RSS 추정치.*

### 2.4 Precision 효과 분석

**표 2.4.** SD precision 비교 (A1 대비)

| | A2 (unet mixed_pr) | A3 (vae w8a8) | A4 (mixed_pr + w8a8) |
|---|---|---|---|
| E2E | −21% | −2% | −22% |
| Power AvgInfer | −17% | −5% | −19% |
| Mem Total | −26% | −2% | −29% |

UNet MIXED_PR이 E2E/전력/메모리의 주된 절감원이다. VAE W8A8은 VAEDec만 단축(528→130ms, −75%)하여 전체 E2E 기여는 제한적이나, 두 기법을 결합한 A4가 모든 지표에서 최대 절감을 달성한다.

**표 2.5.** LCM precision 비교 (A5 대비)

| | A6 (vae w8a8) |
|---|---|
| E2E | −16% |
| Power AvgInfer | −8% |
| Mem Total | −2% |

LCM에서 VAEDec가 E2E의 23%(A5)를 차지하며, vae w8a8이 이를 −73%(468→126ms) 단축한다. 메모리 절감은 미미한데, LCM UNet이 fp16으로 NPU 메모리의 대부분을 차지하기 때문이다.

### 2.5 품질 평가

평가 방법: 동일 seed(=42) 고정 → z_T 동일 전제 하에 LPIPS로 이미지 차이 측정.

**표 2.6.** 품질 평가 (LPIPS + CLIP Score)

| ID | Config | CLIP Score | LPIPS (vs base) | 판정 |
|---|---|---|---|---|
| A1 | SD fp16 *(base)* | 35.76 | — | base |
| A2 | SD unet mixed_pr | 36.30 | 0.6692 | CLIP 동등(±2 이내) → 품질 유지 |
| A3 | SD vae w8a8 | 35.65 | 0.0845 | 미세한 차이 (육안 구분 어려움) |
| A4 | SD mixed_pr + w8a8 | 35.97 | 0.6746 | CLIP 동등 → 품질 유지 |
| A5 | LCM fp16 *(LCM base)* | 30.14 | — | base |
| A6 | LCM vae w8a8 | 30.42 | 0.0289 | ✅ 사실상 동일 |

- **VAE w8a8 LPIPS ≪ UNet mixed_pr LPIPS**: seed 고정으로 VAE only 교체 시 z_T·trajectory 동일 → LPIPS가 VAE decode 품질만 반영. UNet 교체 시 step마다 오차 누적 → trajectory 분기 → LPIPS 높아짐. 이는 설계 의도와 일치하는 내적 일관성.
- **UNet MIXED_PR LPIPS 0.67(SD)**: 다른 이미지이지만 CLIP 차이 ±2 이내 → "다른 이미지, 동등 품질"로 판정.

### 2.6 Best Precision 선정

**표 2.7.** Best Precision 선정

| Model | Best Precision | 근거 |
|---|---|---|
| **SD v1.5** | **A4: mixed_pr + vae w8a8** | E2E −22%, Mem −29%, 전력 −19%, CLIP 동등 |
| **LCM-LoRA** | **A6: vae w8a8** | E2E −16%, VAEDec −73%, LPIPS 0.03 (사실상 동일) |

---

## 3. Step Sweep 결과

> 목적: step-quality tradeoff 확인. 고정: best precision, burst mode, 5 trials 2 warmup.

### 3.1 SD v1.5 Step Sweep (precision: mixed_pr + vae w8a8)

**표 3.1.** SD v1.5 Step Sweep

| ID | Steps | E2E (ms) | UNet/step (ms) | CLIP Score | 전력 Δ(mW) |
|---|---|---|---|---|---|
| A4 | 20 | 11,286 | 546.7 | 35.97 | +4,031 |
| B1 | 30 | 16,535 | 538.6 | 34.06 | +4,000 |
| B2 | 50 | 27,441 | 540.5 | 35.69 | +4,509 |

- **UNet per-step이 step 수와 무관하게 ~540ms로 일정**: NPU 처리량이 포화 상태. E2E는 step 수에 선형 비례.
- **CLIP Score 세 config 모두 33~36 (양호)**: step 수 증가가 prompt 반영도를 개선하지 않음.
- → **step 늘릴 이유 없음. A4(s20)가 best.**

### 3.2 LCM-LoRA Step Sweep (precision: vae w8a8)

**표 3.2.** LCM-LoRA Step Sweep

| ID | Steps | E2E (ms) | UNet/step (ms) | VAEDec (ms) | CLIP Score |
|---|---|---|---|---|---|
| A6 | 4 | 1,693 | 341.0 | 126 | 30.42 |
| B3 | 8 | 3,040 | 339.2 | 127 | 30.91 |

- CLIP 차이 ±0.5 이내(동등): s8이 s4 대비 품질 이득 없음.
- E2E +80% 증가.
- → **A6(s4)가 best.**

### 3.3 SD vs LCM 비교

**표 3.3.** SD vs LCM Best Config 비교

| | SD A4 (s20) | LCM A6 (s4) | LCM/SD 비 |
|---|---|---|---|
| E2E warm (ms) | 11,286 | 1,693 | **×0.15** (−85%) |
| Cold E2E (ms) | 16,178 | 7,999 | **×0.49** (−51%) |
| UNet (ms) | 10,950 | 1,366 | ×0.12 |
| VAEDec (ms) | 130 | 126 | ≈동일 |
| 전력 Δ (mW) | +4,031 | +3,389 | −16% |
| CLIP Score | 35.97 | 30.42 | −5.6 (보통 수준) |
| Mem Total (MB) | 1,709 | 2,320 | +36% |

LCM이 warm E2E를 **85% 단축**. Cold E2E(세션 초기화 포함) 기준으로도 16.2s→8.0s(−51%)로 유의미한 차이. CLIP Score는 SD보다 ~5 낮으나 27~33 구간("보통")으로 실용 가능 수준. 메모리는 LCM UNet(fp16)이 크기 때문에 오히려 더 높음.

---

## 4. Balanced Perf Mode 결과

섹션 2의 burst mode 측정에서 각 실험 config의 system resource 사용량은 다음과 같다:

**표 4.1.** Burst mode Thermal + Power 측정

| ID | Config | Idle (°C) | Cold 1회 후 (°C) | Warm 종료 (°C) | Idle (mW) | AvgInfer (mW) |
|---|---|---|---|---|---|---|
| A1 | SD fp16 | 30.9 | **51.2** | 52.0 | 627 | 5,580 |
| A3 | SD vae w8a8 | 28.9 | **49.6** | 51.6 | 545 | 5,297 |
| A2 | SD unet mixed_pr | 27.3 | 42.2 | 43.8 | 525 | 4,638 |
| A4 | SD mixed_pr + vae w8a8 | 31.3 | **47.3** | 46.5 | 494 | 4,525 |
| A6 | LCM vae w8a8 | 30.9 | 43.0 | 37.5 | 610 | 3,999 |

**표 4.2.** 모바일 SoC 전력-thermal 가이드라인

| 시스템 총 전력 | Thermal 영향 | 허용 시나리오 |
|---|---|---|
| **~3W 이하** | 열 축적 미미, 안정적 지속 가능 | 상시 background 추론 |
| **3~5W** | 수 초~수십 초 허용, 장시간 시 SoC 40°C+ | 단발성 on-demand 생성 |
| **5W 이상** | 수 분 내 throttling 구간 진입, 50°C+ 도달 | burst 단발만 권장 |

Android 기기의 thermal 제어는 다음 시스템 레이어에서 작동하며, burst 수준의 발열을 지속적으로 허용하지 않을 수 있다:
- **커널 thermal framework**: thermal zone별 trip point 도달 시 자동 clock throttle. Snapdragon 8 Gen 2의 경우 일반적으로 SoC 50~60°C 구간에서 단계적 throttling 시작, 85~95°C에서 critical shutdown
- **OEM thermal daemon**: 삼성 GOS(Game Optimizing Service) 등이 앱별 성능을 제한. 정확한 임계값은 OEM 비공개이나, 지속 42~45°C 이상에서 개입하는 것으로 알려져 있음

대부분의 서비스 환경은 cold inference지만, 지속 수행 시 thermal 제한에 걸릴 수 있고, 이를 방지하기 위해 앱 내부에서 현재 thermal 상태를 감지하고 burst→balanced 전환 정책이 필요할 수 있다.
이에 따른 latency 변화에 대해 탐구하고자 한다.

QNN HTP는 NPU clock 수준을 제어하는 `htp_performance_mode`를 제공한다:

**표 4.3.** QNN HTP Performance Mode

| Mode | 설명 |
|---|---|
| `burst` | 최대 clock · 최고 throughput · 전력/발열 최대 |
| `balanced` | 성능과 전력의 균형 — OS 기본값에 가까움 |
| `power_saver` | 최저 clock · 최소 전력 |

`balanced`를 비교 대상으로 선택한 이유는 (1) 연속 추론 시 thermal 대응으로 전환할 가능성이 높고, (2) background 추론 시 기기가 기본적으로 이 수준에서 동작하기 때문이다.

> 고정: A4/A6 best config, balanced HTP perf mode, 5 trials 2 warmup.

**표 4.4.** Burst vs Balanced 비교

| | E2E warm (ms) | Cold E2E (ms) | UNet/step (ms) | Power Idle (mW) | Power AvgInfer (mW) | 온도 end (°C) |
|---|---|---|---|---|---|---|
| SD burst (A4) | 11,286 | 16,178 | 546.7 | 494 | 4,525 | 46.5 |
| SD balanced (C1) | 12,789 | 17,624 | 618.5 | 504 | 3,433 | 45.0 |
| **Δ** | **+13%** | **+9%** | **+13%** | — | **−24%** | −1.5 |
| LCM burst (A6) | 1,693 | 7,999 | 341.0 | 610 | 3,999 | 37.5 |
| LCM balanced (C2) | 1,931 | 8,360 | 395.1 | 645 | 3,127 | 37.9 |
| **Δ** | **+14%** | **+5%** | **+16%** | — | **−22%** | ≈동일 |

*Cold E2E의 Δ가 warm E2E보다 작은 이유: Session Init(QNN HTP graph compile)은 perf mode 영향을 받지 않아 burst/balanced 거의 동일(SD ~4,780ms, LCM ~6,300ms). balanced의 latency 증가는 추론(UNet/VAE) 구간에만 적용된다.*

**결론**: balanced mode는 warm latency +13~14% 손해, 추론 중 전력 −22~24% 절감. Cold E2E 기준으로는 증가 폭이 더 작아(+5~9%) 실서비스 영향이 제한적이다. background 추론 또는 배터리 절약 시나리오에서 유효한 선택지.

---

## 5. 연속 추론 안정성 결과

> 목적: best config(LCM/SD)을 burst 모드로 쿨다운 없이 10회 연속 실행해 latency 안정성과 thermal drift 측정.

### 5.1 Per-Trial Latency 추세

**표 5.1.** 연속 추론 Per-Trial Latency 추세

| ID | Config | Trial 1 (ms) | Trial 10 (ms) | Drift | 온도 상승 |
|---|---|---|---|---|---|
| D1 | LCM s4 vae_w8a8 burst | 1,638 | 1,673 | **+2.1%** (+35ms) | 44.6→54.0°C (+9.4°C) |
| D2 | SD s20 mixed_pr+w8a8 burst | 11,108 | 11,116 | **+0.07%** (+8ms) | 45.0→53.2°C (+8.2°C) |

### 5.2 해석

**D1 (LCM):** 온도 +9.4°C 상승에 따른 경미한 drift(+2.1%) 확인. UNet per-step이 339ms→343ms(+1.2%)로 HTP burst clock이 미세 감소. Full thermal throttling(보통 온도 70°C↑에서 발생)은 아니며, trial 10 latency(1,673ms)는 cloud 대비 ×0.17을 유지.

**D2 (SD):** 11초 단위 장시간 추론으로 NPU가 이미 sustained clock 상태를 유지 → drift 사실상 없음(+0.07%).

**결론:** 두 config 모두 10회 연속 burst 추론에서 full thermal throttling 없이 실용적 latency 유지. LCM은 경미한 drift(+2.1%)가 측정됐으나 feasibility 판정에 영향 없음.

---

## 6. 결론: On-Device Feasibility 판정

### 6.1 KPI 종합: Burst Mode

기준: Galaxy S26 Ultra cloud 실행 latency ~10s (표 0.1 참조). 본 실험 해상도는 512×512.
> 참고: cloud의 1024×1024 생성과 직접 비교 시, on-device 512 생성 + SR upscale 파이프라인이 현실적 (Appendix B.1 참조).

**표 6.1.** Burst Mode KPI 종합 (best precision 기준)

| KPI | SD v1.5 A4 (s20) | LCM A6 (s4) | Cloud 기준 |
|---|---|---|---|
| E2E warm | 11,286ms (11.3s) | 1,693ms (1.7s) | ~10s |
| Cold E2E | 16,178ms (16.2s) | 7,999ms (8.0s) | ~10s |
| UNet/step | 546.7ms | 341.0ms | — |
| Memory (NPU+App) | 1,709MB | 2,320MB | — |
| Power AvgInfer | 4,525mW | 3,999mW | — |
| Thermal end | 46.5°C | 37.5°C | — |
| CLIP Score | 35.97 | 30.42 | — |

### 6.2 KPI 종합: Balanced Mode

**표 6.2.** Balanced Mode KPI 종합 (best precision 기준)

| KPI | SD v1.5 C1 (s20) | LCM C2 (s4) | Cloud 기준 |
|---|---|---|---|
| E2E warm | 12,789ms (12.8s) | 1,931ms (1.9s) | ~10s |
| Cold E2E | 17,624ms (17.6s) | 8,360ms (8.4s) | ~10s |
| UNet/step | 618.5ms | 395.1ms | — |
| Memory (NPU+App) | 1,706MB | 2,324MB | — |
| Power AvgInfer | 3,433mW | 3,127mW | — |
| Thermal end | 45.0°C | 37.9°C | — |
| CLIP Score | 35.97 | 30.42 | — |

*CLIP Score는 perf mode에 무관 (동일 모델·precision). balanced는 burst 대비 latency +13~14%, 전력 −22~24%.*

### 6.3 Feasibility 판정

**SD v1.5 (품질 충분, latency 미달)**
- CLIP Score 35.97(양호)로 다양한 task의 backbone UNet으로 활용 가능한 품질 수준
- 그러나 warm 11.3s로 cloud 기준(~10s)을 초과, cold 16.2s는 1.6배. balanced에서는 12.8s/17.6s로 더 악화
- Thermal 46.5°C로 OEM 개입 구간(42~45°C)에 근접하여 지속 운용에도 주의 필요
- → **on-device로 cloud를 대체하기 어렵다**

**LCM-LoRA (latency 충분, 품질 제한)**
- warm 1.7s / cold 8.0s로 burst/balanced 모두 cloud 기준 충족. Thermal 37.5°C로 안정적
- 그러나 CLIP Score 30.42(보통)로 SD v1.5 대비 프롬프트 충실도 −5.6. few-step distillation으로 속도를 얻는 대신 prior의 표현력 일부를 희생
- 단발성 생성(스티커, 빠른 프리뷰)에는 실용 가능하나, 스타일 변환·inpainting 등 다양한 task로 확장할 backbone UNet으로 쓰기에는 **quality가 부족**

**핵심 딜레마**: 다양한 task 확장을 고려하면 UNet의 quality가 중요하고, 이를 위해서는 SD v1.5 수준의 backbone이 필요하다. 그러나 SD v1.5는 현재 모바일 SoC에서 ~10s KPI를 충족하지 못한다. **이것이 현재 핸드폰이 diffusion 기반 이미지 생성에 cloud 전략을 유지하고 있는 근본적인 이유로 판단된다.**

### 6.4 아키텍처 효율 향상의 가능성

다만 본 실험은 **아키텍처 효율이 향상되면 현재 하드웨어에서도 on-device가 가능하다**는 것을 동시에 보여준다.

- LCM-LoRA는 SD v1.5와 동일한 UNet 구조에서 step 수를 20→4로 줄여 E2E를 **85% 단축**(11.3s→1.7s)
- UNet이 E2E의 95%를 지배하므로, UNet per-step 효율 또는 필요 step 수를 줄이는 아키텍처 개선이 가장 직접적인 레버
- 만약 SD v1.5 수준의 quality를 유지하면서 LCM 수준의 step 효율을 달성하는 모델이 나온다면, Snapdragon 8 Gen 2에서도 ~10s KPI 충족이 가능

이는 on-device diffusion의 병목이 하드웨어 성능이 아니라 **모델 아키텍처의 효율성**에 있음을 시사한다.

### 6.5 Key Findings

1. **UNet이 E2E의 95%를 지배한다** — SD v1.5의 UNet이 E2E의 94~97%를 점유. Step 수 감소가 가장 직접적인 latency 절감 수단이다.

2. **SD v1.5는 quality 충분, latency 미달** — CLIP 35.97로 backbone 품질은 충분하나 warm 11.3s로 cloud 기준 초과. 다양한 task 확장의 backbone으로 필요하지만 on-device KPI를 맞추지 못한다.

3. **LCM-LoRA는 latency 충분, quality 제한** — warm 1.7s로 cloud 대비 ×0.17이지만 CLIP 30.42로 backbone UNet quality가 부족. 단발성 생성에는 가능하나 multi-task 확장에는 아쉽다.

4. **현재 cloud 전략은 합리적** — quality와 latency를 동시에 만족하는 on-device 모델이 없으므로, cloud 유지가 현실적 선택이다. 아키텍처 효율 향상(quality 유지 + few-step)이 on-device 전환의 전제 조건.

5. **Mixed Precision이 품질 유지하며 KPI 개선** — UNet MIXED_PR + VAE W8A8(A4)은 E2E −22%, Mem −29%, 전력 −19% 절감, CLIP 동등. LCM vae w8a8(A6)은 VAEDec −73%, LPIPS 0.03(사실상 동일).

6. **Step 증가는 품질 이득 없음** — SD s20→s50, LCM s4→s8 모두 CLIP ±2 이내. UNet per-step latency는 step 수와 무관하게 일정하므로 E2E는 step에 선형 비례.

7. **Balanced mode: latency +13~14%, 전력 −22~24%** — NPU clock scaling이 모델 무관하게 일관 적용. Background 추론 시 유효한 선택지.

8. **연속 추론에서도 thermal throttling 없음** — 10회 연속 burst 시 LCM +2.1%, SD +0.07% drift.

### 6.6 권장 운영점

현재 조건에서 on-device를 적용한다면:

**표 6.3.** 권장 운영점

| Use Case | 권장 Config | E2E (warm) | Cold E2E | 비고 |
|---|---|---|---|---|
| 단발성 빠른 생성 (스티커 등) | LCM s4 vae w8a8 burst (A6) | 1.7s | 8.0s | quality 제한 감수 |
| Background / 배터리 절약 | LCM s4 vae w8a8 balanced (C2) | 1.9s | 8.4s | quality 제한 감수 |
| 고품질 필요 시 | Cloud 유지 권장 | — | ~10s | SD 수준 quality 보장 |

---

## Appendix A: 실험 ID 매핑

**표 A.1.** 실험 ID 매핑

| ID | Model | Steps | Precision | Perf Mode |
|---|---|---|---|---|
| A1 | SD v1.5 | 20 | FP16 | burst |
| A2 | SD v1.5 | 20 | unet mixed_pr | burst |
| A3 | SD v1.5 | 20 | vae w8a8 | burst |
| A4 | SD v1.5 | 20 | mixed_pr + vae w8a8 | burst |
| A5 | LCM | 4 | FP16 | burst |
| A6 | LCM | 4 | vae w8a8 | burst |
| B1 | SD v1.5 | 30 | mixed_pr + vae w8a8 | burst |
| B2 | SD v1.5 | 50 | mixed_pr + vae w8a8 | burst |
| B3 | LCM | 8 | vae w8a8 | burst |
| C1 | SD v1.5 | 20 | mixed_pr + vae w8a8 | balanced |
| C2 | LCM | 4 | vae w8a8 | balanced |
| D1 | LCM | 4 | vae w8a8 | burst (sustained 10회) |
| D2 | SD v1.5 | 20 | mixed_pr + vae w8a8 | burst (sustained 10회) |

## Appendix B: 확장 가능성 검토

### B.1 고해상도(720/1024) 출력 Feasibility

SD v1.5 / LCM-LoRA를 512, 720, 1024 해상도로 GPU(RTX 4070) txt2img 생성하여 품질을 비교했다.

**실험 조건**: 4개 프롬프트(cat watercolor, cyberpunk, golden retriever, temple) × 2 모델(SD s30, LCM s8) × 3 해상도, seed 고정.

**표 B.1.** 해상도별 GPU 품질 비교

| 해상도 | SD s30 | LCM s8 | GPU 추론 시간 (SD/LCM) |
|---|---|---|---|
| 512 | 기준 품질 | 기준 품질 | 2.6s / 0.5s |
| 720 | 구도·디테일 양호 | 약간의 디테일 손실, 사용 가능 | 5.3s / 1.0s (~2×) |
| 1024 | 구도 변화, 프롬프트별 편차 | 흐릿, 과포화 경향 | 14s / 2.3s (~5×) |

**모바일 배포 시 고해상도 native 생성의 문제점:**
1. UNet + VAE Decoder 재컴파일 필수 (latent size 64→90/128, calibration 데이터 재생성)
2. SD v1.5는 512 학습 모델 — 고해상도에서 품질 보장 안 됨 (실험에서 확인)
3. 연산량 quadratic 증가 — 720: ~2×, 1024: ~4× (NPU 추론 시간·메모리·발열 모두 악화)
4. 해상도마다 별도 모델 바이너리 필요 → 앱 용량 증가

**결론**: 고해상도 출력은 512 생성 후 **경량 SR(Super Resolution) 모델로 upscale**하는 파이프라인이 합리적이다.
- 기존 512 파이프라인(검증 완료된 KPI) 유지
- SR 모델은 upscale 전용 학습이라 품질이 native 고해상도보다 우수
- Single-pass CNN으로 가볍고 빠름 (수십 ms 수준)
- QAI Hub에 Real-ESRGAN, XLSR 등 모바일 최적화 SR 모델이 이미 존재

### B.2 img2img 파이프라인 확장 가능성

기존 txt2img 파이프라인(Text Encoder + UNet + VAE Decoder)에 **VAE Encoder를 추가**하면 img2img(스타일 변환) 지원이 가능하다.

```
img2img: VAE Encoder(input→latent) + noise → UNet(denoise, text-guided) → VAE Decoder → output
```

- VAE Encoder ONNX(`vae_encoder_fp32.onnx`)는 이미 export 완료 상태
- Text embedding은 UNet cross-attention으로 합류 — 이미지 latent와 별도 경로
- `strength` 파라미터로 원본 유지 정도 조절 (0=원본 유지, 1=완전 재생성)

**GPU 실험 결과** (COCO val2017 이미지 3장, SD s30 / LCM s8):
- strength 0.5: 원본 구도 유지, 프롬프트 영향 미미
- strength 0.75: 프롬프트 주도 생성, 원본 구도 거의 소실
- 스타일 변환 목적이면 strength 0.3~0.5가 적절

**모바일 배포 시**: VAE Encoder를 QNN NPU용으로 컴파일·추가하면 앱 파이프라인 확장 가능. 단, VAE Encoder 추론 시간(~수십 ms)이 E2E에 추가된다.

### B.3 모바일 Multi-Service 확장: Shared UNet Backbone 전략

모바일에서 diffusion 기반 기능(txt2img, img2img, style transfer, inpainting 등)을 여러 개 제공하려면, 기능마다 별도 UNet을 탑재하는 것은 비현실적이다.

**UNet이 핵심인 이유:**
- SD v1.5의 생성 능력과 prior는 대부분 UNet에 집중 (859M params, 3.3GB FP32)
- 기능별 UNet 교체 시: 앱 용량 수 GB 단위 증가, 모델 로딩 시간, NPU 세션 초기화·컴파일 비용 반복 발생
- 본 실험의 cold start 측정(섹션 2.2)에서도 세션 초기화에 수 초 소요 확인

**Shared Backbone + Lightweight Adapter 구조:**

```
[Task-specific Adapter]     [Shared UNet Backbone]     [Shared VAE Decoder]
  ├ Text Encoder (txt2img)         │                          │
  ├ VAE Encoder (img2img)    ──→   UNet (cross-attention)  ──→ VAE Decoder ──→ Output
  ├ ControlNet (구조 조건)          │                          │
  └ Reference Adapter (스타일)      │                          │
```

- **UNet + VAE Decoder는 공유**: NPU 세션 1회 로드, 기능 전환 시 재로딩 불필요
- **조건 입력만 교체**: text embedding, image latent, mask, structure map 등은 UNet의 cross-attention 또는 concat 입력으로 합류
- **기능 전환 비용 최소화**: adapter 모듈은 수~수십 MB 수준으로, UNet(~1.7GB QNN binary) 대비 경량

**Multi-service 확장과 backbone quality 문제:**

다양한 task(txt2img, img2img, style transfer, inpainting 등)로 확장하려면 backbone UNet의 **quality가 핵심**이다. 각 task는 UNet의 prior(구도·디테일·프롬프트 충실도)에 의존하므로, backbone quality가 낮으면 모든 task의 출력 품질이 함께 하락한다.

본 실험 결과, 이 관점에서 두 모델 모두 한계가 있다:
- **SD v1.5**: CLIP 35.97로 backbone quality 충분. 그러나 warm 11.3s로 ~10s KPI 미달 → on-device 배포 불가
- **LCM-LoRA**: warm 1.7s로 latency 충분. 그러나 CLIP 30.42로 few-step distillation이 prior 표현력을 희생 → 다양한 task의 backbone으로 쓰기에는 quality 부족

이것이 현재 핸드폰이 diffusion 이미지 생성에 cloud 전략을 유지하는 이유와 일치한다. Quality와 latency를 동시에 만족하는 on-device backbone이 아직 존재하지 않기 때문이다.

**향후 가능성:**
- SD v1.5 수준의 quality를 유지하면서 few-step 추론이 가능한 아키텍처(예: consistency distillation 개선, 효율적 UNet 설계)가 등장하면, 이 shared backbone 구조로 on-device multi-service 전환이 가능
- 본 실험의 LCM 결과(1.7s)는 **아키텍처 효율만 확보되면 현재 SoC에서도 충분히 동작함**을 보여주는 proof-of-concept

**한계:**
- 특정 스타일이나 도메인(예: anime 특화)처럼 prior 자체를 바꿔야 하는 경우 LoRA나 UNet fine-tuning 필요
- ControlNet 등 구조 조건 모듈은 별도 QNN 컴파일 필요 — 단, UNet 크기의 일부(~30%)

**결론**: multi-service 확장은 **"Shared UNet backbone + lightweight adapters"** 구조가 합리적이나, 현재는 on-device에서 backbone quality와 latency를 동시에 만족하는 모델이 없어 **cloud 유지가 현실적**이다. 아키텍처 효율이 향상되면 이 구조로의 on-device 전환이 가능하다.
