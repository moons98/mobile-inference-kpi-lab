# On-Device Text-to-Image Generation: Mobile Diffusion Feasibility Study — 실험 설계

---

## 0. 프로젝트 개요

### 0.1 문제정의

Cloud 기반 text-to-image 생성이 보편화되었으나, mobile on-device 환경에서의 실행 가능성은 아직 충분히 검증되지 않았다. 본 프로젝트의 핵심 질문은 다음이다.

**Diffusion 기반 text-to-image 생성을 mobile on-device 환경에서 실사용 가능한 수준으로 수행할 수 있는가?**

이를 단일 기능 구현 문제가 아니라, 시스템 관점의 **feasibility 분석**으로 정의한다.

- Baseline: SD v1.5의 standard denoising (20–50 steps)
- Optimized: LCM-LoRA를 통한 few-step generation (2–8 steps)

핵심 비교는 **동일 backbone에서 step 수 감소가 mobile KPI를 얼마나 개선하는가**이다.

### 0.2 프로젝트 목표

Android 단말(Snapdragon NPU 타겟)에서 diffusion text-to-image 파이프라인의 실행 가능성을 아래 KPI로 정량화한다.

- Latency
- Memory
- Power
- Thermal
- Quality

그리고 다음 경계(boundary)를 도출한다.

- 어떤 조건(step 수, precision, backend)에서 on-device가 실사용 가능한가
- 어떤 조건에서는 cloud offload가 필요한가

### 0.3 연구 질문

1. SD v1.5 기반 text-to-image 생성을 mobile on-device에서 수행할 수 있는가?
2. LCM-LoRA(few-step)는 SD v1.5 대비 mobile KPI를 얼마나 개선하는가?
3. UNet 반복 구간이 전체 latency에서 차지하는 비중은 얼마인가?
4. 어느 조건(step/precision/backend)부터 on-device latency가 실사용 가능 구간에 들어오는가?

### 0.4 가설

1. SD v1.5의 standard step(20+)은 mobile에서 실시간 생성이 어렵다.
2. LCM-LoRA의 few-step(4–8) 생성은 on-device feasibility를 유의미하게 개선한다.
3. UNet 반복 구간이 지배적 병목이며, step 수 감소가 가장 큰 latency/energy 개선을 만든다.

### 0.5 분석 대상 시나리오

본 실험은 동일 SD v1.5 backbone에서 두 가지 생성 모드를 비교한다.

- **SD v1.5 (Baseline)**: standard scheduler, 20–50 steps
- **LCM-LoRA (Optimized)**: LCM scheduler + LoRA adapter, 2–8 steps

공통 backbone은 동일하며, LCM-LoRA는 UNet에 LoRA weight를 적용하고 scheduler를 교체하는 방식이다.

---

## 1. 시스템 관점 정의

### 1.1 비교 축 (Primary Axes)

- **Model Variant**: SD v1.5 (baseline) vs LCM-LoRA (optimized)
- **Steps**: SD v1.5 20/30/50, LCM-LoRA 2/4/6/8
- **Precision**: FP16 vs INT8(W8A8 또는 mixed)
- **Backend**: CPU / GPU / NPU(QNN)

### 1.2 Runtime Stack

- Android App (Kotlin)
- ONNX Runtime + QNN EP
- CPU EP / QNN GPU / QNN HTP(NPU)
- Stage-level profiling + system telemetry 수집

### 1.3 파이프라인 모델

Text Prompt → Text Encoder → Initial Noise(1,4,64,64) → Denoising Loop(UNet × N steps) → VAE Decode → Output Image(512×512)

- SD v1.5: standard scheduler (PNDM/DDIM 등), 20–50 steps
- LCM-LoRA: LCM scheduler, 2–8 steps, UNet에 LoRA weight 적용

### 1.4 측정 단위

- **Inference Trial**: 이미지 생성 1회
- **Cold Start Trial**: session load + first inference (HTP JIT 포함)
- **Sustained Trial**: 연속 실행(thermal/power drift 관찰, cooldown 없음)

---

## 2. KPI 정의

### 2.1 Product KPI

- Full E2E Latency (ms)
- Cold Start Time (ms)
- Peak Memory (MB)
- Avg Power (mW)
- Thermal Drift (°C/trial)
- Quality Score (정량 + 시각평가)

### 2.2 Stage KPI

- Text Encoding (tokenize + text encoder inference)
- UNet loop total / per-step
- VAE Decode
- Runtime overhead (tensor create/copy/fence)

### 2.3 품질 KPI

- Perceptual: FID, CLIP Score (가능한 범위에서)
- Visual: artifact 빈도, prompt 부합도
- Human panel(optional): A/B blind preference

> SD v1.5와 LCM-LoRA 간 품질 비교는 동일 prompt set에서 수행하며, step 수 변화에 따른 품질-속도 tradeoff를 중심으로 해석한다.

---

## 3. 실험 환경

### 3.1 Hardware

- Samsung Galaxy S23 Ultra (SM-S918N)
- Snapdragon 8 Gen 2
- RAM 12GB

### 3.2 Software

- Android API 34
- ONNX Runtime 1.23.x
- QNN SDK 2.42.x

### 3.3 통제 조건

- Airplane mode
- 고정 brightness
- 고정 governor/perf profile
- Trial 간 cooldown: 최소 60s → 온도 35°C 도달 시 완료, 최대 180s
- Phase 2(Sustained)는 cooldown 없이 연속 실행
- 고정 seed / 고정 입력셋

---

## 3.4 실험 전 체크리스트

실험 시작 전 아래 항목을 순서대로 확인한다.

### 필수 (측정값 신뢰성 직결)

| 항목 | 이유 |
|------|------|
| **충전기 분리** | 충전 전류가 전력 측정값을 오염. 앱이 `is_charging` 플래그로 감지하나, 근본적으로 분리 필요 |
| **Airplane mode ON** | 셀룰러/Wi-Fi 트래픽이 CPU 및 전력에 영향 |
| **화면 밝기 고정** | 디스플레이 소비전력을 일정하게 유지 (화면은 꺼지지 않음 — 앱이 `FLAG_KEEP_SCREEN_ON` 적용) |
| **백그라운드 앱 종료** | 메모리/CPU 경합 제거. 설정 → 앱 → 최근 앱 모두 닫기 |
| **배터리 잔량 40% 이상** | 저배터리 시 기기가 자체 throttling |
| **디바이스 온도 ≤ 35°C 확인 후 시작** | 이미 thermal throttle 상태에서 시작하면 Trial 1부터 왜곡 |

### 권장

| 항목 | 이유 |
|------|------|
| Do Not Disturb ON | 알림 인터럽트 차단 |
| Developer options → "Stay awake (while charging)" OFF | 충전 없이도 화면 유지는 앱이 처리 |
| ADB 연결은 USB가 아닌 Wi-Fi ADB 사용 (or 분리) | USB 연결 시 충전 전류 유입 가능성 |
| 실험 직전 재부팅 (선택) | 메모리 파편화 초기화, 재현성 향상 |

### 확인 명령 (adb)

```bash
# 온도 확인 (head -5 대신 Select-Object)
adb shell cat /sys/class/thermal/thermal_zone*/temp | Select-Object -First 5

# 충전 상태 확인 (grep 대신 Select-String)
adb shell dumpsys battery | Select-String "status"

# 배터리 잔량
adb shell dumpsys battery | Select-String "level"
```

---

## 4. 실험 설계

### 4.1 Phase 1 — Single-Run Feasibility

목적: 조건별 latency/memory/quality baseline 확보
공통 설정: 5 trials, 2 warmup, burst mode, trial 간 cooldown 적용

#### P1-1: Model Variant 비교

- SD v1.5 (20 steps) vs LCM-LoRA (4 steps, guidance=1.0)
- 동일 backend/precision(QNN NPU FP16)에서 비교

#### P1-2: Step Sweep — SD v1.5

- 20 / 30 / 50 steps
- step 증가에 따른 latency-quality tradeoff 곡선

#### P1-3: Step Sweep — LCM-LoRA

- 2 / 4 / 8 steps
- few-step feasibility 하한 탐색

#### P1-4: Backend × Precision Sweep

- Backend: CPU(FP32) / GPU / NPU
- Precision: FP16 vs W8A8
- CPU는 시간상 1 trial

#### P1-5: Mixed Precision (컴포넌트별 조합)

- FP16 full / W8A8 full
- FP16 + UNet W8A8 / W8A8 + VAE FP16
- 컴포넌트별 precision이 KPI에 미치는 영향

#### P1-6: Parallel Init (Cold Start 최적화)

- Sequential init (baseline) vs Parallel init
- 3개 ORT 세션(Text Enc, UNet, VAE Dec)의 순차/병렬 초기화 비교
- 각 세션 로드 시간 + 전체 cold start 비교

### 4.2 Phase 2 — Sustained Feasibility

목적: 실제 사용 패턴에서 열/전력/성능 안정성 검증
공통 설정: 10 trials, cooldown 없음, sustained_high mode

#### P2-1: Sustained — SD v1.5

- QNN NPU FP16 vs W8A8
- 연속 10회 생성, Trial 1 vs Trial N drift 비교

#### P2-2: Sustained — LCM-LoRA

- QNN NPU FP16 vs W8A8, 4 steps
- 연속 10회 생성

수집 항목:

- Inference drift (trial별 latency 변화)
- UNet per-step drift
- Thermal slope (°C/trial)
- Energy per generation (idle baseline delta)
- Memory stability

---

## 5. 입력 설계

### 5.1 프롬프트 셋

- SD v1.5 / LCM-LoRA 비교를 위해 동일 prompt set 사용
- 카테고리: 인물, 풍경, 오브젝트, 추상 등 다양성 확보
- 난이도: 단순 구도 / 복잡 구도 구분

### 5.2 재현성

- 고정 seed
- 고정 prompt set
- 동일 noise init 반복 사용

---

## 6. 분석 프레임워크

### 6.1 핵심 분석 관점

1. **UNet 지배성 검증**: 전체 latency 중 UNet loop 비중
2. **LCM-LoRA 실효성**: few-step 생성이 KPI를 얼마나 개선하는가
3. **Step-Quality Tradeoff**: step 감소에 따른 품질 열화 곡선
4. **Precision trade-off**: W8A8 도입 시 품질 손실 대비 시스템 이득
5. **Backend 비교**: CPU/GPU/NPU 간 latency-power 효율 비교
6. **Decision Boundary**: on-device vs cloud offload 경계

### 6.2 Feasibility 판정 기준(초안)

- Latency: 사용자가 수용 가능한 생성 대기 시간 이내
- Thermal: sustained 시 throttling 급증 없음
- Memory: foreground app 안정 운용 가능 범위
- Quality: 사용자 인지 가능한 열화가 허용 한도 이내

> 최종 threshold 값은 Phase 1/2 실측 결과로 확정한다.

---

## 7. 실행 순서

1. 모델 준비: SD v1.5 UNet/VAE/Text Encoder export, LCM-LoRA weight 병합 및 export
2. 단일 실행 측정: Phase 1 매트릭스 수행
3. 상위 config 선별
4. 연속 실행 측정: Phase 2 수행
5. boundary 도출: on-device feasible region 정량화
6. 보고서 정리: cloud/offload decision guideline 제시

---

## Appendix A: 결과 기록 스키마 (요약)

### Record 1 — Generation Summary

trial_id, model_variant, steps, precision, backend,
latency_e2e_ms, text_enc_ms, unet_total_ms, unet_step_mean_ms, vae_dec_ms,
peak_memory_mb, avg_power_mw, thermal_start_c, thermal_end_c,
pipeline_wall_clock_ms, trial_wall_clock_ms,
native_heap_mb, pss_mb,
quality_notes

### Record 2 — UNet Step Detail

trial_id, step_index, input_create_ms, session_run_ms,
output_copy_ms, scheduler_ms, step_total_ms,
thermal_c, power_mw,
memory_mb, native_heap_mb

### Record 3 — Cold Start

trial_id, model_variant, backend, precision,
session_load_ms (text_enc / unet / vae_dec 개별 포함),
first_inference_ms, cold_start_total_ms, warmup_total_ms,
peak_memory_after_load_mb,
thermal_zone_type, is_charging, idle_baseline_power_mw

---

## Appendix B: 용어 정리

- **SD v1.5**: Stable Diffusion v1.5, standard text-to-image diffusion model
- **LCM-LoRA**: Latent Consistency Model LoRA, few-step 생성을 가능하게 하는 LoRA adapter
- **Feasibility**: latency/memory/power/thermal/quality를 동시에 만족하는 운용 가능성
- **Decision Boundary**: on-device 처리와 cloud offload를 가르는 조건 경계
