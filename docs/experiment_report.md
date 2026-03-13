# On-Device Text-to-Image Generation: Mobile Diffusion Feasibility Report (Draft)

---

## 0. 개요

### 0.1 문제 설정

Samsung Galaxy S26 Ultra에는 diffusion 기반 이미지 생성 기능이 탑재되어 있으며, 현재 대부분의 생성 기능은 **서버(cloud)에서 실행**된다.

| 기능 | 실행 위치 | 측정 latency |
|---|---|---|
| AI Eraser, Move | **On-device** | ~10s |
| 스타일 (화풍 변환) | Cloud | ~12–20s |
| 이미지 생성 (배경화면 등) | Cloud | ~8s (1024×1024) |
| 스티커 생성 | Cloud | ~9s (720×720) |
| 스티커 세트 확장 | Cloud | ~25s (9개) |

Creative Studio의 이미지/스티커 생성 기능은 Text Encoder + UNet + VAE Decoder 구성의 Stable Diffusion 계열 모델을 기반으로 분석된다. 이 컴포넌트들은 구조적으로 mobile NPU에서 실행 가능한 범주에 있다. 만약 on-device로 전환하면 서버 비용과 네트워크 의존성을 제거할 수 있으나, device의 메모리·전력·발열 자원을 직접 소비하게 된다.

본 프로젝트의 핵심 질문:
- SD v1.5 파이프라인을 on-device로 실행하면 현재 cloud latency(~8–25s)와 비교했을 때 어느 수준인가?
- 어떤 조건(step 수, precision, backend)에서 latency·resource 측면의 on-device feasibility가 성립하는가?
- LCM-LoRA(few-step)는 SD v1.5 대비 mobile KPI를 얼마나 개선하는가?

### 0.2 실험 축

- **Model Variant**: SD v1.5 (EulerDiscrete scheduler, CFG 7.5) vs LCM-LoRA (DDIM-style, CFG 1.0)
- **Steps**: SD v1.5 — 20/30/50, LCM-LoRA — 2/4/8
- **Precision**: FP16 / W8A8 / MIXED_PR / W8A16
- **Backend**: CPU(FP32) / QNN GPU / QNN HTP(NPU)

### 0.3 측정 KPI

| KPI | 설명 |
|---|---|
| E2E Latency | Tokenize ~ VAE Decode 전체 (ms) |
| Stage Breakdown | Text Enc / UNet total / VAE Dec 비중 |
| UNet per-step | Step당 추론 시간, step 수 감소 실효성 |
| Cold Start | Session load + first inference (ms) |
| Peak Memory | 생성 중 최대 메모리 (MB) |
| Avg Power | 생성 구간 평균 소비전력 (mW) |
| Thermal Drift | 연속 실행 시 온도 상승 (°C/trial) |

---

## 1. 현재 상태 (2026-03-13 기준)

### 완료

| 항목 | 상태 |
|---|---|
| SD v1.5 txt2img 파이프라인 설계 | ✅ |
| Text Encoder / VAE / UNet base / UNet LCM ONNX export | ✅ |
| QAI Hub compile (FP16, W8A8, MIXED_PR, W8A16 variants) | ✅ |
| Component-level 양자화 품질 평가 (CosSim) | ✅ |
| deploy_config.json — 배포 가능 모델 목록 확정 | ✅ |
| Android 앱 구현 (Txt2ImgPipeline, Scheduler, 배치 실험) | ✅ |
| Scheduler LCM/EulerDiscrete 분기 수정 | ✅ |
| VAE Decoder W8A16 postproc 분기 수정 | ✅ |
| 실험 설계 정의 (Phase 1/2) | ✅ |

### 진행 예정

- Phase 1/2 on-device 실측 (Galaxy S23)
- 결과 CSV 수집 및 분석 (`analysis/parse_txt2img_csv.py`)

---

## 2. 양자화 품질 스크리닝 결과

> 양자화 전략 수립, 실행, 디버깅 과정의 기술 내러티브: [`docs/model_optimization.md`](model_optimization.md)

`scripts/sd/eval_sd_quant_quality.py` 기준. 평가 방법: FP32 ORT CPU 출력 vs 양자화 모델 출력 CosSim.

| Component | Variant | CosSim | RMSE | Grade | 배포 여부 |
|---|---|---|---|---|---|
| vae_encoder | W8A8 (QAI Hub) | 0.9810 | 0.163 | Marginal | 조건부 |
| text_encoder | W8A16 (AIMET) | 0.9849 | 0.178 | Marginal | 조건부 |
| vae_decoder | W8A8 (QAI Hub) | 0.9827 | 0.136 | Marginal | 조건부 |
| vae_decoder | W8A16 (AIMET) | **0.9999** | 0.005 | **Excellent** | ✅ |
| unet_base | MIXED_PR | **0.9968** | 0.075 | **Good** | ✅ |
| unet_lcm | MIXED_PR | 0.9875 | 0.143 | Marginal | 조건부 |
| unet_lcm | W8A16 (AIMET) | 0.7292 | 0.729 | **Poor** | ❌ |
| unet_base | W8A16 (AIMET) | — | — | — | ⚠️ OOM (S23 inference) |
| unet_base/lcm | INT8 QDQ full | — | — | — | ❌ compile 실패 |

**주요 발견**:
- UNet full INT8 QDQ compile 실패: HTP `LayerNormalization` op이 full INT8 input/output 미지원 → MIXED_PR로 우회 (Conv·MatMul·Gemm만 INT8)
- unet_lcm W8A16: AIMET calibration 결과 on-device 실행 시 품질 급락(0.7292). 실제 이미지 품질 추가 확인 예정
- vae_decoder W8A16: qai-hub-models 컴파일 모델에 `/Div+/Clip` postproc 내장 → [0,1] 출력. 앱 코드에서 `normalized=true` 분기 처리됨

---

## 3. 예정 평가 결과 (Phase 1/2)

> 아래는 실측 후 채워질 항목이다.

### 3.1 Phase 1 — Single-Run Feasibility

#### P1-1: Model Variant 비교 (SD v1.5 20step vs LCM-LoRA 4step)

| Metric | SD v1.5 FP16 | LCM-LoRA FP16 | 개선 |
|---|---|---|---|
| E2E Latency (ms) | — | — | — |
| UNet total (ms) | — | — | — |
| Peak Memory (MB) | — | — | — |
| Avg Power (mW) | — | — | — |

#### P1-2: Step Sweep — SD v1.5 (20/30/50)

| Steps | E2E (ms) | UNet (ms) | UNet % |
|---|---|---|---|
| 20 | — | — | — |
| 30 | — | — | — |
| 50 | — | — | — |

#### P1-3: Step Sweep — LCM-LoRA (2/4/8)

| Steps | E2E (ms) | UNet (ms) | UNet % |
|---|---|---|---|
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |

#### P1-4: Backend × Precision

| Backend | Precision | E2E (ms) | Power (mW) |
|---|---|---|---|
| NPU | FP16 | — | — |
| NPU | W8A8 | — | — |
| GPU | FP16 | — | — |
| CPU | FP32 | — | — |

#### P1-5: Mixed Precision

| Text Enc | UNet | VAE Dec | E2E (ms) | Memory (MB) |
|---|---|---|---|---|
| FP16 | FP16 | FP16 | — | — |
| FP16 | W8A8 | FP16 | — | — |
| W8A16 | MIXED_PR | W8A16 | — | — |

#### P1-6: Parallel Init

| Init Mode | Wall-clock (ms) | TextEnc+UNet+VAE Sum (ms) |
|---|---|---|
| Sequential | — | — |
| Parallel | — | — |

### 3.2 Phase 2 — Sustained Feasibility

#### P2-1: SD v1.5 Sustained (FP16 vs W8A8, 10 trials)

| Metric | FP16 | W8A8 |
|---|---|---|
| Trial 1 E2E (ms) | — | — |
| Trial 10 E2E (ms) | — | — |
| Thermal slope (°C/trial) | — | — |

#### P2-2: LCM-LoRA Sustained (FP16 vs W8A8, 4 steps, 10 trials)

| Metric | FP16 | W8A8 |
|---|---|---|
| Trial 1 E2E (ms) | — | — |
| Trial 10 E2E (ms) | — | — |
| Thermal slope (°C/trial) | — | — |

---

## 4. Feasibility 판정 기준

> Phase 1/2 실측 후 threshold 확정 예정.

**Latency 기준점 (Galaxy S26 Ultra cloud 실행 기준)**:

| 기능 | Cloud Latency | On-device 목표 |
|---|---|---|
| 이미지 생성 (1024×1024) | ~8s | 이하이면 on-device 우위 |
| 스티커 생성 (720×720) | ~9s | 이하이면 on-device 우위 |
| 스타일 변환 | ~12–20s | 이하이면 조건부 전환 가능 |

본 실험의 대상 해상도는 512×512 (실험 범위). 1024×1024 대비 UNet은 동일, VAE만 4배 차이.

**판정 기준**:
- **Latency**: cloud 대비 이하 → on-device 우위 / 근사 → 조건부 / 초과 → cloud 유지
- **Memory**: S23 12GB에서 crash-free foreground 운용
- **Thermal**: sustained 10회에서 throttling 급증 없음
- **Quality**: CosSim Marginal 이상 (시각 평가 병행)

---

## 5. Key Findings (TBD)

- UNet loop 지배성: Phase 1 실측 후 기재
- LCM-LoRA 실효 개선폭: Phase 1 실측 후 기재
- On-device feasible region: Phase 1/2 결과로 경계 도출
- 권장 운영점: latency-quality-power 트리플 tradeoff 기준
