# On-Device Text-to-Image Generation: Mobile Diffusion Feasibility Study — 실험 설계

---

## 0. 프로젝트 개요

### 0.1 문제정의

Samsung Galaxy S26 Ultra 기준으로 photo/creative 기능은 실행 위치가 나뉜다.

**On-device 기능** (로컬 추론):
- AI Eraser (ai 지우개): ~10s
- Move (피사체 이동 + 배경 생성)

**Cloud 기능** (서버 추론):
- 만들기 (텍스트 기반 이미지 편집): 이미지 크기 12M → 9M
- 스타일 (화풍 변환): 원본 해상도 유지(512 배수), ~12s (통신 상태에 따라 최대 20s)
- Creative Studio:
  - 텍스트 기반 이미지 생성 (배경화면 등): 1024×1024, ~8s
  - 스티커 생성: 720×720, ~9s
  - 스티커 세트 확장: 9개 이미지, ~25s

Creative Studio의 이미지/스티커 생성 기능은 Text Encoder + UNet + VAE Decoder로 구성되는 Stable Diffusion 계열 모델을 기반으로 할 것으로 분석된다. 이 컴포넌트들은 크기가 크지만 구조적으로 mobile NPU에서 실행 가능한 범주에 있다.

현재는 서버에서 실행되어 네트워크 latency가 포함된다. 만약 같은 모델을 on-device에서 실행한다면:
- 서버 비용 없이 오프라인에서도 동작 가능
- 하지만 device 자원(메모리, 전력, 발열) 사용이 불가피

**본 프로젝트의 핵심 질문**:

> SD v1.5 기반 text-to-image 파이프라인을 Samsung Galaxy S23(Snapdragon 8 Gen 2)에서 on-device로 실행할 경우, 어느 조건에서 실사용 가능한 latency와 시스템 자원 비용을 달성할 수 있는가?

이를 단일 기능 구현이 아닌 **실행 위치 결정을 위한 feasibility 분석**으로 정의한다.

### 0.2 프로젝트 목표

Android 단말(Snapdragon NPU 타겟)에서 diffusion text-to-image 파이프라인의 실행 가능성을 아래 KPI로 정량화한다.

- Latency — 현재 cloud 실행 latency(~8–25s)와 비교 가능한 수치 확보
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
4. 어느 조건(step/precision/backend)부터 on-device latency가 현재 cloud 수준(~8–25s)과 경쟁 가능한가?

### 0.4 가설

1. SD v1.5의 standard step(20+)은 mobile에서 실시간 생성이 어렵다.
2. LCM-LoRA의 few-step(4–8) 생성은 on-device feasibility를 유의미하게 개선한다.
3. UNet 반복 구간이 지배적 병목이며, step 수 감소가 가장 큰 latency/energy 개선을 만든다.

### 0.5 분석 대상 시나리오

본 실험은 동일 SD v1.5 backbone에서 두 가지 생성 모드를 비교한다.

- **SD v1.5 (Baseline)**: EulerDiscrete scheduler, 20–50 steps, CFG guidance_scale 7.5
- **LCM-LoRA (Optimized)**: LCM-LoRA adapter fused UNet, DDIM-style scheduler, 2–8 steps, guidance_scale 1.0 (CFG 비활성화)

공통 backbone(SD v1.5)은 동일하며, LCM-LoRA는 UNet에 LoRA weight를 fuse한 별도 모델이다. 두 variant 모두 동일한 Text Encoder / VAE Decoder를 공유한다.

---

## 1. 시스템 관점 정의

### 1.1 비교 축 (Primary Axes)

- **Model Variant**: SD v1.5 (baseline) vs LCM-LoRA (optimized)
- **Steps**: SD v1.5 20/30/50, LCM-LoRA 2/4/8
- **Precision**: FP16 / W8A8 (`_int8_qdq.onnx`, QAI Hub W8A8) / MIXED_PR (Conv·MatMul·Gemm INT8, LayerNorm FP32) / W8A16 (AIMET weight-only)
- **Backend**: CPU(FP32) / QNN GPU / QNN HTP(NPU)

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

## 2.5 모델 준비 (Pre-Experiment)

실험 전에 완료된 작업이다. 상세 내용은 `docs/weights_inventory.md` 참조.

### 2.5.1 Export

- `scripts/sd/export_sd_to_onnx.py` — Text Encoder / VAE Encoder / VAE Decoder (opset 18, FP32)
- `scripts/sd/export_sd_lcm_unet.py` — UNet base / LCM-LoRA fused UNet (external data 형식)

### 2.5.2 Quantization

QAI Hub 및 AIMET 기반으로 아래 precision variant를 생성했다.

| Precision | 방법 | 대상 |
|---|---|---|
| W8A8 | QAI Hub quantize (MinMax, 64 samples) | VAE Encoder, VAE Decoder |
| MIXED_PR | `quant_runpod.py` (Conv/MatMul/Gemm INT8, LayerNorm FP32) | UNet base, UNet LCM |
| W8A16 | AIMET weight-only (`qai-hub-models` export) | Text Encoder, VAE Decoder, UNet base, UNet LCM |

### 2.5.3 QAI Hub Compile

- Target: Samsung Galaxy S23 (Snapdragon 8 Gen 2), `--target_runtime precompiled_qnn_onnx --qairt_version 2.42`
- 산출물: `{name}.onnx` (stub) + `{name}.bin` (QNN context binary)

### 2.5.4 품질 스크리닝

`scripts/sd/eval_sd_quant_quality.py` — component-level CosSim (FP32 vs quantized) 측정.
결과: `exp_outputs/quantization/sd_quant_quality.txt`

| Component | Variant | CosSim | Grade | 배포 |
|---|---|---|---|---|
| vae_encoder | W8A8 | 0.9810 | Marginal | 조건부 |
| text_encoder | W8A16 | 0.9849 | Marginal | 조건부 |
| vae_decoder | W8A8 | 0.9827 | Marginal | 조건부 |
| vae_decoder | W8A16 | **0.9999** | Excellent | ✅ |
| unet_base | MIXED_PR | **0.9968** | Good | ✅ |
| unet_lcm | MIXED_PR | 0.9875 | Marginal | 조건부 |
| unet_lcm | W8A16 | 0.7292 | Poor | ❌ |
| unet_base/lcm | INT8 QDQ full | — | — | ❌ compile 실패 (HTP LayerNorm INT8 미지원) |

배포 가능 모델 목록: `scripts/deploy/deploy_config.json`

---

## 3. 실험 환경

### 3.1 Hardware

- Samsung Galaxy S23 Ultra (SM-S918N)
- Snapdragon 8 Gen 2
- RAM 12GB

### 3.2 Software

- Android API 34
- ONNX Runtime 1.24.3
- QNN SDK (QAIRT) 2.42.0 — QAI Hub compile job 버전과 일치

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

실행 순서: Phase 1 → (best precision 선정) → Phase 2 → Phase 3
상세 실험 목록 및 진행 상태: `docs/experiment_runs.md` 참조

### 4.1 Phase 1 — Precision Burst

목적: best precision config 선정
공통 설정: steps 고정 (SD 20 / LCM 4), 5 trials, 2 warmup, burst mode, trial 간 cooldown 적용
사용 가능한 precision 옵션: fp16 (baseline) / unet mixed_pr / vae w8a8 / 조합

| ID | Model | Steps | Precision | 목적 |
|---|---|---|---|---|
| A1 | SD v1.5 | 20 | fp16 | baseline |
| A2 | SD v1.5 | 20 | unet mixed_pr | UNet 양자화 효과 |
| A3 | SD v1.5 | 20 | vae w8a8 | VAE 양자화 효과 |
| A4 | SD v1.5 | 20 | unet mixed_pr + vae w8a8 | 조합 효과 |
| A5 | LCM | 4 | fp16 | LCM baseline |
| A6 | LCM | 4 | vae w8a8 | LCM VAE 양자화 효과 |

Phase 1 완료 후: CLIP Score + LPIPS로 precision 열화 확인 → best precision 선정

> **W8A16 전면 제외**: component-level CosSim은 합격이나 실제 파이프라인 on-device 추론 시 가시적 품질 저하 확인.
> **unet_lcm mixed_pr 제외**: 실제 추론 결과 품질 저하 확인.

### 4.2 Phase 2 — Step Sweep Burst

목적: step-quality tradeoff 곡선 도출 (SD 20/30/50, LCM 4/8)
공통 설정: best precision 고정 (Phase 1 결과 후 결정), 5 trials, 2 warmup, burst mode, trial 간 cooldown 적용

| ID | Model | Steps | 목적 |
|---|---|---|---|
| A1 | SD v1.5 | 20 | 기준점 (Phase 1 공유) |
| B1 | SD v1.5 | 30 | step sweep |
| B2 | SD v1.5 | 50 | step sweep (품질 상한 기준) |
| A5 | LCM | 4 | 기준점 (Phase 1 공유) |
| B3 | LCM | 8 | step sweep |

Phase 2 완료 후: B2(SD 50step)을 base image로 CLIP Score + LPIPS 계산 → step-quality 곡선 도출

### 4.3 Phase 3 — Balanced Perf Mode

목적: burst 대비 balanced perf mode에서의 latency/power 차이 측정
공통 설정: best config 고정 (Phase 1/2 결과 후 결정), 5 trials, 2 warmup, **balanced mode**, trial 간 cooldown 적용
배경: background 추론 시나리오 — 사용자가 명시적으로 요청하지 않는 경우 기기가 balanced 상태일 수 있음

| ID | Model | Steps | Precision | Perf Mode |
|---|---|---|---|---|
| C1 | SD v1.5 | best step | best precision | balanced |
| C2 | LCM | best step | best precision | balanced |

> 앱에 balanced perf mode 추가 필요 (현재 burst / sustained_high만 구현됨)

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

### 6.2 Reference Point: 현재 Cloud 실행 Latency

On-device feasibility 판단의 기준점으로 사용. Galaxy S26 Ultra 실측 기준.

| 기능 | 해상도 | Cloud Latency | 비고 |
|---|---|---|---|
| Creative Studio — 이미지 생성 | 1024×1024 | ~8s | 배경화면, 텍스트 기반 |
| Creative Studio — 스티커 생성 | 720×720 | ~9s | 1개 |
| Creative Studio — 스티커 세트 | 720×720 × 9 | ~25s | 9개 동시 |
| 스타일 (화풍 변환) | 원본 해상도 | ~12–20s | 네트워크 포함 |

On-device가 이 수치 이하(또는 근접)를 달성한다면 on-device 전환의 기술적 근거가 된다.

### 6.3 Feasibility 판정 기준(초안)

- **Latency**: cloud 실행 latency(~8–25s)와 비교. on-device가 이하이면 우위, 근사하면 조건부, 초과이면 cloud 유지 권장
- **Thermal**: sustained 10회 생성 시 throttling 급증 없음
- **Memory**: foreground app 안정 운용 (crash-free, S23 12GB RAM 기준)
- **Quality**: 사용자 인지 가능한 artifact 없음 (CosSim Marginal 이상)

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
