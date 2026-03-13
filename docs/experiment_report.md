# On-Device Text-to-Image Generation: Mobile Diffusion Feasibility Report

> 측정 기기: Samsung Galaxy S23 Ultra (Snapdragon 8 Gen 2, 12GB RAM)
> 실험 완료일: 2026-03-13

---

## 0. 개요

### 0.1 문제 설정

Samsung Galaxy S26 Ultra에는 diffusion 기반 이미지 생성 기능이 탑재되어 있으며, 현재 대부분의 생성 기능은 **서버(cloud)에서 실행**된다.

| 기능 | 실행 위치 | 측정 latency |
|---|---|---|
| AI Eraser, Move | On-device | ~10s |
| 스타일 (화풍 변환) | Cloud | ~12–20s |
| 이미지 생성 (배경화면 등) | Cloud | ~8s (1024×1024) |
| 스티커 생성 | Cloud | ~9s (720×720) |
| 스티커 세트 확장 | Cloud | ~25s (9개) |

Creative Studio의 이미지/스티커 생성 기능은 Text Encoder + UNet + VAE Decoder 구성의 Stable Diffusion 계열 모델을 기반으로 분석된다. 만약 on-device로 전환하면 서버 비용과 네트워크 의존성을 제거할 수 있으나, device의 메모리·전력·발열 자원을 직접 소비하게 된다.

**핵심 질문:**
1. SD v1.5 파이프라인을 on-device로 실행하면 현재 cloud latency(~8–25s)와 비교해 어느 수준인가?
2. 어떤 조건(step 수, precision, backend)에서 latency·resource 측면의 on-device feasibility가 성립하는가?
3. LCM-LoRA(few-step)는 SD v1.5 대비 mobile KPI를 얼마나 개선하는가?

### 0.2 실험 축

- **Model Variant**: SD v1.5 (EulerDiscrete scheduler, CFG 7.5) vs LCM-LoRA (DDIM-style, CFG 1.0)
- **Steps**: SD v1.5 — 20/30/50, LCM-LoRA — 4/8
- **Precision**: FP16 / W8A8 / MIXED_PR (Conv·MatMul·Gemm INT8, LayerNorm FP32)
- **Backend**: QNN HTP(NPU)
- **Perf Mode**: burst (Phase 1/2/4) / balanced (Phase 3)

### 0.3 측정 KPI

| KPI | 설명 |
|---|---|
| E2E Latency | Tokenize ~ VAE Decode 전체 (ms, warm mean) |
| Stage Breakdown | Text Enc / UNet total / VAE Dec 비중 |
| UNet per-step | step당 추론 시간 |
| Cold Start | Session load + first inference (ms) |
| Peak Memory | NPU(HTP) + App RSS 추정 (MB) |
| Avg Power | 추론 구간 소비전력 delta (mW) |
| Thermal | 시작/종료 온도 (°C) |

---

## 1. 실험 완료 현황

| 항목 | 상태 |
|---|---|
| SD v1.5 txt2img 파이프라인 구현 (Android Kotlin) | ✅ |
| Text Encoder / VAE / UNet base / UNet LCM ONNX export | ✅ |
| QAI Hub compile (FP16, W8A8, MIXED_PR variants) | ✅ |
| Component-level 양자화 품질 평가 (CosSim/PSNR) | ✅ |
| Phase 1 — Precision Burst (A1~A6, QNN NPU) | ✅ |
| Phase 1 품질 평가 (LPIPS + CLIP Score) | ✅ |
| Phase 2 — Step Sweep Burst (B1~B3) | ✅ |
| Phase 2 품질 평가 | ✅ |
| Phase 3 — Balanced Perf Mode (C1, C2) | ✅ |
| Phase 4 — Sustained Stability (D1, D2) | ✅ |

---

## 2. 양자화 품질 스크리닝

> 양자화 전략 수립·실행·디버깅 기술 내러티브: [`docs/model_optimization.md`](model_optimization.md)

평가 방법: FP32 ORT CPU 출력 vs 양자화 모델 출력 CosSim (`scripts/sd/eval_sd_quant_quality.py`)

| Component | Variant | CosSim | RMSE | Grade | 배포 여부 |
|---|---|---|---|---|---|
| vae_encoder | W8A8 (QAI Hub) | 0.9810 | 0.163 | Marginal | 조건부 |
| text_encoder | W8A16 (AIMET) | 0.9849 | 0.178 | Marginal | 조건부 |
| vae_decoder | W8A8 (QAI Hub) | 0.9827 | 0.136 | Marginal | 조건부 |
| vae_decoder | W8A16 (AIMET) | **0.9999** | 0.005 | Excellent | ✅ |
| unet_base | MIXED_PR | **0.9968** | 0.075 | Good | ✅ |
| unet_lcm | MIXED_PR | 0.9875 | 0.143 | Marginal | 조건부 |
| unet_lcm | W8A16 (AIMET) | 0.7292 | 0.729 | Poor | ❌ |
| unet_base/lcm | INT8 QDQ full | — | — | — | ❌ compile 실패 |

**주요 발견:**
- **UNet INT8 QDQ full compile 실패**: HTP `LayerNormalization`이 full INT8 input/output 미지원 → MIXED_PR로 우회 (Conv·MatMul·Gemm만 INT8, LayerNorm FP32 유지)
- **unet_lcm W8A16**: AIMET calibration 결과 on-device 실행 시 품질 급락 (CosSim 0.7292) → 실험 전 제외
- **vae_decoder W8A16**: qai-hub-models 컴파일 모델에 `/Div+/Clip` postproc 내장 → [0,1] 출력. 앱 `normalized=true` 분기 처리됨

CosSim Marginal 등급(0.980~0.995)은 컴포넌트 단위 스크리닝 통과 후 Phase 1 end-to-end 품질 평가(LPIPS/CLIP Score)로 최종 판정.

---

## 3. Phase 1 — Precision 비교 결과

> 목적: best precision config 선정. 고정: SD s20 / LCM s4, burst mode, 5 trials 2 warmup.

### 3.1 E2E Latency & Stage Breakdown (warm mean, ms)

| ID | Config | E2E | UNet | UNet% | VAEDec | Power Δ(mW) | Mem Total(MB) |
|---|---|---|---|---|---|---|---|
| A1 | SD fp16 s20 *(baseline)* | 14,559 | 13,810 | 94.9% | 528 | +4,953 | 2,399 |
| A2 | SD unet mixed_pr s20 | 11,490 | 10,782 | 93.8% | 476 | +4,113 | 1,765 |
| A3 | SD vae w8a8 s20 | 14,229 | 13,866 | 97.4% | 146 | +4,752 | 2,354 |
| **A4** | **SD mixed_pr + vae w8a8 s20** | **11,286** | **10,950** | **97.0%** | **130** | **+4,031** | **1,709** |
| A5 | LCM fp16 s4 *(LCM baseline)* | 2,023 | 1,364 | 67.4% | 468 | +3,856 | 2,372 |
| **A6** | **LCM vae w8a8 s4** | **1,693** | **1,366** | **80.7%** | **126** | **+3,389** | **2,320** |

*Mem Total = NPU(HTP) 상주 메모리(QAI Hub profiling) + App RSS 추정치*

**SD precision 비교 (A1 대비):**

| | A2 (unet mixed_pr) | A3 (vae w8a8) | A4 (mixed_pr + w8a8) |
|---|---|---|---|
| E2E | −21% | −2% | −22% |
| 전력 Δ | −17% | −4% | −19% |
| Mem Total | −26% | −2% | −29% |
| UNet/step (ms) | 538 (−22%) | 692 (≈동일) | 547 (−21%) |
| VAEDec (ms) | 476 (≈동일) | 146 (−72%) | 130 (−75%) |

→ UNet MIXED_PR이 latency/전력의 주된 절감원. VAE W8A8은 VAEDec만 단축(−75%)하여 전체 E2E 기여는 제한적.

**LCM precision 비교 (A5 대비):**

| | A6 (vae w8a8) |
|---|---|
| E2E | −16% (2,023 → 1,693ms) |
| 전력 Δ | −12% |
| VAEDec | −73% (468 → 126ms) |

→ LCM에서 VAEDec가 E2E의 23%(A5)를 차지하며, vae w8a8으로 큰 폭 단축.

### 3.2 UNet 지배성

SD의 경우 UNet이 E2E의 **95% 이상**을 점유. Step 수 감소가 가장 큰 latency 절감 수단임을 시사.

LCM은 step 수(4)가 적어 UNet 비중이 67~81%로 낮아지며, VAEDec 비중이 상대적으로 증가.

### 3.3 Cold Start

| ID | Session Init | 1st Infer | Cold E2E |
|---|---|---|---|
| A1 | 6,879ms | 14,617ms | 21,496ms |
| A4 | 4,775ms | 11,403ms | 16,178ms |
| A5 | 6,573ms | 2,094ms | 8,667ms |
| A6 | 6,245ms | 1,754ms | 7,999ms |

Session Init에 QNN HTP graph compile 포함 — 앱 최초 실행 또는 모델 교체 시 발생. 이후 재사용 시 skip.

### 3.4 Phase 1 품질 평가 (LPIPS + CLIP Score)

평가 방법: 동일 seed(=42) 고정 → z_T 동일 전제 하에 LPIPS로 이미지 차이 측정.

| ID | Config | CLIP Score | LPIPS (vs base) | 판정 |
|---|---|---|---|---|
| A1 | SD fp16 *(base)* | 35.76 | — | base |
| A2 | SD unet mixed_pr | 34.70 | 0.6690 | CLIP 동등(±2 이내) → 품질 유지 |
| A3 | SD vae w8a8 | 35.55 | 0.0305 | ✅ 사실상 동일 |
| A4 | SD mixed_pr + w8a8 | 35.97 | 0.6700 | CLIP 동등 → 품질 유지 |
| A5 | LCM fp16 *(LCM base)* | 30.14 | — | base |
| A6 | LCM vae w8a8 | 30.91 | 0.0300 | ✅ 사실상 동일 |

- **VAE w8a8 LPIPS ≪ UNet mixed_pr LPIPS**: seed 고정으로 VAE only 교체 시 z_T·trajectory 동일 → LPIPS가 VAE decode 품질만 반영. UNet 교체 시 step마다 오차 누적 → trajectory 분기 → LPIPS 높아짐. 이는 설계 의도와 일치하는 내적 일관성.
- **UNet MIXED_PR LPIPS 0.67(SD)**: 다른 이미지이지만 CLIP 차이 ±2 이내 → "다른 이미지, 동등 품질"로 판정.

### 3.5 Best Precision 선정

| Model | Best Precision | 근거 |
|---|---|---|
| **SD v1.5** | **A4: mixed_pr + vae w8a8** | E2E −22%, Mem −29%, 전력 −19%, CLIP 동등 |
| **LCM-LoRA** | **A6: vae w8a8** | E2E −16%, VAEDec −73%, LPIPS 0.03 (사실상 동일) |

---

## 4. Phase 2 — Step Sweep 결과

> 목적: step-quality tradeoff 확인. 고정: best precision, burst mode, 5 trials 2 warmup.

### 4.1 SD v1.5 Step Sweep (precision: mixed_pr + vae w8a8)

| ID | Steps | E2E (ms) | UNet/step (ms) | CLIP Score | 전력 Δ(mW) |
|---|---|---|---|---|---|
| A4 | 20 | 11,286 | 546.7 | 35.97 | +4,031 |
| B1 | 30 | 16,535 | 538.6 | 34.06 | +4,000 |
| B2 | 50 | 27,441 | 540.5 | 35.69 | +4,509 |

- **UNet per-step이 step 수와 무관하게 ~540ms로 일정**: NPU 처리량이 포화 상태. E2E는 step 수에 선형 비례.
- **CLIP Score 세 config 모두 33~36 (양호)**: step 수 증가가 prompt 반영도를 개선하지 않음.
- → **step 늘릴 이유 없음. A4(s20)가 best.**

### 4.2 LCM-LoRA Step Sweep (precision: vae w8a8)

| ID | Steps | E2E (ms) | UNet/step (ms) | VAEDec (ms) | CLIP Score |
|---|---|---|---|---|---|
| A6 | 4 | 1,693 | 341.0 | 126 | 30.91 |
| B3 | 8 | 3,040 | 339.2 | 127 | 30.91 |

- CLIP 차이 0.00: s8이 s4 대비 품질 이득 없음.
- E2E +80% 증가.
- → **A6(s4)가 best.**

### 4.3 SD vs LCM 비교

| | SD A4 (s20) | LCM A6 (s4) | LCM/SD 비 |
|---|---|---|---|
| E2E (ms) | 11,286 | 1,693 | **×0.15** (−85%) |
| UNet (ms) | 10,950 | 1,366 | ×0.12 |
| VAEDec (ms) | 130 | 126 | ≈동일 |
| 전력 Δ (mW) | +4,031 | +3,389 | −16% |
| CLIP Score | 35.97 | 30.91 | −5.1 (보통 수준) |
| Mem Total (MB) | 1,709 | 2,320 | +36% |

LCM이 E2E latency를 **85% 단축**. CLIP Score는 SD보다 ~5 낮으나 27~33 구간("보통")으로 실용 가능 수준. 메모리는 LCM UNet(fp16)이 크기 때문에 오히려 더 높음.

---

## 5. Phase 3 — Balanced Perf Mode 결과

> 목적: burst 대비 balanced mode에서의 latency/power tradeoff 측정. background 추론 시나리오.
> 고정: A4/A6 best config, balanced HTP perf mode, 5 trials 2 warmup.

| | SD burst (A4) | SD balanced (C1) | Δ | LCM burst (A6) | LCM balanced (C2) | Δ |
|---|---|---|---|---|---|---|
| E2E (ms) | 11,286 | 12,789 | **+13%** | 1,693 | 1,931 | **+14%** |
| UNet/step (ms) | 546.7 | 618.5 | +13% | 341.0 | 395.1 | +16% |
| 전력 Δ (mW) | +4,031 | +2,929 | **−27%** | +3,389 | +2,482 | **−27%** |
| 온도 end (°C) | 46.5 | 45.0 | −1.5 | 37.5 | 37.9 | ≈동일 |

**결론**: balanced mode는 latency +13~14% 손해, 전력 −27% 절감. 비율이 일관됨(두 모델 모두 동일 trade-off 패턴). background 추론 또는 배터리 절약 시나리오에서 유효한 선택지.

---

## 6. Phase 4 — 연속 추론 안정성 결과

> 목적: best config(LCM/SD)을 burst 모드로 쿨다운 없이 10회 연속 실행해 latency 안정성과 thermal drift 측정.

### 6.1 Per-Trial Latency 추세

| ID | Config | Trial 1 (ms) | Trial 10 (ms) | Drift | 온도 상승 |
|---|---|---|---|---|---|
| D1 | LCM s4 vae_w8a8 burst | 1,638 | 1,673 | **+2.1%** (+35ms) | 44.6→54.0°C (+9.4°C) |
| D2 | SD s20 mixed_pr+w8a8 burst | 11,108 | 11,116 | **+0.07%** (+8ms) | 45.0→53.2°C (+8.2°C) |

### 6.2 해석

**D1 (LCM):** 온도 +9.4°C 상승에 따른 경미한 drift(+2.1%) 확인. UNet per-step이 339ms→343ms(+1.2%)로 HTP burst clock이 미세 감소. Full thermal throttling(보통 온도 70°C↑에서 발생)은 아니며, trial 10 latency(1,673ms)는 cloud 대비 ×0.21을 유지.

**D2 (SD):** 11초 단위 장시간 추론으로 NPU가 이미 sustained clock 상태를 유지 → drift 사실상 없음(+0.07%).

**결론:** 두 config 모두 10회 연속 burst 추론에서 full thermal throttling 없이 실용적 latency 유지. LCM은 경미한 drift(+2.1%)가 측정됐으나 feasibility 판정에 영향 없음.

---

## 7. Feasibility 판정

### 7.1 Cloud latency 대비

기준: Galaxy S26 Ultra cloud 실행 latency. 본 실험 해상도는 512×512.
> 1024×1024 추정: UNet latency 동일(latent 64×64 고정), VAEDec ×4. A6 기준 ≈ 2,071ms.

| Config | E2E (ms) | Cloud 기준 (~8s) | 판정 |
|---|---|---|---|
| SD A4 burst (s20) | 11,286 | 8,000 | ❌ cloud 초과 (×1.4) |
| SD C1 balanced (s20) | 12,789 | 8,000 | ❌ cloud 초과 (×1.6) |
| LCM A6 burst (s4) | 1,693 | 8,000 | ✅ **cloud 대비 ×0.21** |
| LCM C2 balanced (s4) | 1,931 | 8,000 | ✅ **cloud 대비 ×0.24** |

### 7.2 KPI별 판정

| KPI | SD v1.5 A4 | LCM A6 | 기준 |
|---|---|---|---|
| Latency | ❌ 11.3s (cloud 초과) | ✅ 1.7s (cloud 대비 ×0.21) | cloud ~8s 이하 |
| Memory | ✅ 1,709MB (NPU+App) | ⚠️ 2,320MB (NPU+App) | crash-free 운용 |
| Thermal | ⚠️ end 46.5°C | ✅ end 37.5°C | throttling 없음 |
| Quality | ✅ CLIP 35.97 (양호) | ⚠️ CLIP 30.91 (보통) | >33 양호 / 27~33 조건부 |

### 7.3 On-device Feasibility 판정

| Scenario | Config | 판정 | 비고 |
|---|---|---|---|
| Foreground, 즉시 응답 | LCM A6 burst | ✅ **Feasible** | 1.7s, cloud 대비 5× 빠름 |
| Background, 배터리 절약 | LCM C2 balanced | ✅ **Feasible** | 1.9s, 전력 −27% |
| Foreground, SD 품질 우선 | SD A4 burst | ❌ **Not Feasible** | 11.3s > cloud 8s |
| Background, SD | SD C1 balanced | ❌ **Not Feasible** | 12.8s, 더 느림 |

---

## 8. Key Findings

### 8.1 UNet이 E2E의 95%를 지배한다

SD v1.5의 UNet이 E2E latency의 94~97%를 점유. 이 때문에 step 수 감소가 가장 직접적인 latency 절감 수단이다. VAE/Text Encoder 최적화는 SD에서 상대적 기여가 제한적이다.

### 8.2 LCM-LoRA가 on-device feasibility의 열쇠

SD v1.5(s20) 11.3s는 cloud 기준(~8s)을 초과해 on-device 이점이 없다. LCM-LoRA(s4)는 1.7s로 cloud 대비 5× 빠르며, Snapdragon 8 Gen 2에서 **실사용 가능한 on-device 이미지 생성이 가능하다**는 것을 보인다.

### 8.3 Mixed Precision이 품질 유지하며 KPI 개선

UNet MIXED_PR + VAE W8A8 조합(A4)은 SD fp16(A1) 대비 E2E −22%, Mem −29%, 전력 −19% 절감하면서 CLIP Score 동등(35.97 vs 35.76)을 유지. 단, UNet precision 변경으로 trajectory가 바뀌므로 LPIPS는 높게 나오며 이는 품질 저하가 아닌 경로 분기를 반영한다.

LCM vae w8a8(A6)은 VAEDec를 −73% 단축하면서 LPIPS 0.03(사실상 동일)을 달성.

### 8.4 Step 증가는 SD/LCM 모두 품질 이득 없음

SD s20→s50, LCM s4→s8 모두 CLIP Score 차이가 ±2 이내로 유의미하지 않다. UNet per-step latency는 step 수와 무관하게 일정(~540ms SD, ~340ms LCM)하므로 E2E는 step에 선형 비례. Step을 늘릴 근거가 없다.

### 8.5 Balanced mode: latency +14%, 전력 −27%로 일관된 trade-off

SD/LCM 모두 동일한 비율(latency +13~14%, 전력 −27%)을 보이며, NPU의 clock scaling이 두 모델에 동일하게 적용됨을 시사. Background 추론 등 배터리가 중요한 시나리오에서 유효한 선택지.

### 8.6 LCM의 메모리 역설

LCM UNet이 LCM-LoRA weight 포함 FP16 모델이라 SD UNet MIXED_PR보다 NPU 메모리가 더 크다(1,793MB vs 1,151MB). LCM이 latency/전력에서 유리하지만 메모리에서는 SD best config보다 불리하다.

### 8.7 연속 추론에서도 thermal throttling 없음

10회 연속 burst 추론(쿨다운 없음) 시 LCM +2.1%, SD +0.07% drift. Full throttling은 발생하지 않았으며, LCM feasibility 판정(×0.21 vs cloud)은 연속 사용 환경에서도 유지된다.

### 8.8 권장 운영점

| Use Case | 권장 Config | E2E | 전력 Δ |
|---|---|---|---|
| Foreground 빠른 생성 | LCM s4 vae w8a8 burst (A6) | 1.7s | +3,389mW |
| Background / 배터리 절약 | LCM s4 vae w8a8 balanced (C2) | 1.9s | +2,482mW |
| SD 품질 필요 시 (허용 latency > 11s) | SD s20 mixed_pr+w8a8 burst (A4) | 11.3s | +4,031mW |

---

## Appendix A: 측정 방법 요약

- **Latency**: `System.nanoTime()` wall-clock, warmup 2회 제외한 warm trial mean
- **Memory**: VmRSS(`/proc/self/status`) + QAI Hub profiling NPU 메모리 추정 합산
- **Power**: `BatteryManager.BATTERY_PROPERTY_CURRENT_NOW × voltage`, 시스템 전체 소비전력
- **Thermal**: `/sys/class/thermal/thermal_zone*/temp` (SoC 온도)
- **품질 평가**: LPIPS(AlexNet, Zhang et al. CVPR 2018), CLIP Score(`openai/clip-vit-base-patch16` logit scale)
- **실험 통제**: Airplane mode, 고정 brightness, trial 간 cooldown 60s (burst mode)

## Appendix B: 실험 ID 매핑

| ID | Model | Steps | Precision | Perf Mode | Phase |
|---|---|---|---|---|---|
| A1 | SD v1.5 | 20 | FP16 | burst | 1 |
| A2 | SD v1.5 | 20 | unet mixed_pr | burst | 1 |
| A3 | SD v1.5 | 20 | vae w8a8 | burst | 1 |
| A4 | SD v1.5 | 20 | mixed_pr + vae w8a8 | burst | 1 |
| A5 | LCM | 4 | FP16 | burst | 1 |
| A6 | LCM | 4 | vae w8a8 | burst | 1 |
| B1 | SD v1.5 | 30 | mixed_pr + vae w8a8 | burst | 2 |
| B2 | SD v1.5 | 50 | mixed_pr + vae w8a8 | burst | 2 |
| B3 | LCM | 8 | vae w8a8 | burst | 2 |
| C1 | SD v1.5 | 20 | mixed_pr + vae w8a8 | balanced | 3 |
| C2 | LCM | 4 | vae w8a8 | balanced | 3 |
| D1 | LCM | 4 | vae w8a8 | burst | 4 (sustained 10회) |
| D2 | SD v1.5 | 20 | mixed_pr + vae w8a8 | burst | 4 (sustained 10회) |
