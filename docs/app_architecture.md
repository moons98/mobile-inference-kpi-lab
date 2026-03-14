# Mobile Inference KPI Lab — 앱 아키텍처 및 실험 구현

> **목적**: 이 프로젝트의 앱이 어떻게 구성되어 있고, 실험이 어떻게 구현·실행되는지를 복기할 수 있도록 정리한 문서.
> **관련 문서**: 실험 설계 → [experiment_design.md](experiment_design.md) · 양자화 전략 → [model_optimization.md](model_optimization.md) · 실험 결과 → [experiment_report.md](experiment_report.md)

---

## Table of Contents

- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 전체 시스템 구조](#2-전체-시스템-구조)
- [3. Android 앱 아키텍처](#3-android-앱-아키텍처)
  - [3.1 Layer 구조](#31-layer-구조)
  - [3.2 핵심 클래스 관계](#32-핵심-클래스-관계)
  - [3.3 Text-to-Image 파이프라인](#33-text-to-image-파이프라인)
  - [3.4 OrtRunner — ONNX Runtime 세션 래퍼](#34-ortrunner--onnx-runtime-세션-래퍼)
  - [3.5 Scheduler — 노이즈 스케줄러](#35-scheduler--노이즈-스케줄러)
  - [3.6 KpiCollector — 시스템 메트릭 수집](#36-kpicollector--시스템-메트릭-수집)
- [4. 실험 프레임워크](#4-실험-프레임워크)
  - [4.1 배치 실험 구조](#41-배치-실험-구조)
  - [4.2 BenchmarkRunner — 실험 오케스트레이터](#42-benchmarkrunner--실험-오케스트레이터)
  - [4.3 Cooldown 및 Thermal 관리](#43-cooldown-및-thermal-관리)
  - [4.4 CSV 데이터 수집](#44-csv-데이터-수집)
- [5. 모델 배포 파이프라인](#5-모델-배포-파이프라인)
  - [5.1 Export → Quantize → Compile → Deploy](#51-export--quantize--compile--deploy)
  - [5.2 Per-Component Precision Map](#52-per-component-precision-map)
  - [5.3 QNN HTP 실행 흐름](#53-qnn-htp-실행-흐름)
- [6. 분석 파이프라인](#6-분석-파이프라인)
- [7. 프로젝트 디렉토리 구조](#7-프로젝트-디렉토리-구조)

---

## 1. 프로젝트 개요

이 프로젝트는 **SD v1.5 text-to-image 파이프라인을 Samsung Galaxy S23(Snapdragon 8 Gen 2) NPU에서 on-device로 실행**할 때의 feasibility를 정량 분석하는 연구다.

핵심 질문: 현재 cloud에서 실행되는 이미지 생성(~8–25s)을 on-device로 전환하면, 어떤 조건(step 수, precision, perf mode)에서 실사용 가능한 latency와 자원 비용을 달성할 수 있는가?

이를 위해 다음을 구축했다:
1. **Android 벤치마크 앱** — ONNX Runtime + QNN EP 기반 on-device 추론 및 KPI 측정
2. **Python 스크립트 셋** — 모델 export, 양자화, 품질 평가, 결과 분석
3. **실험 프레임워크** — JSON 기반 배치 실험 정의, 자동 실행, CSV 데이터 수집

---

## 2. 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PC (개발 환경)                                                              │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────────┐  │
│  │ scripts/sd/           │  │ scripts/deploy/       │  │ analysis/         │  │
│  │ ① Export (ONNX)      │  │ ③ ADB Push           │  │ ⑥ CSV 파싱        │  │
│  │ ② Quantize (AIMET)   │──│   모델→디바이스         │  │   CLIP/LPIPS 평가  │  │
│  │   QAI Hub Compile    │  │                       │  │   리포트 생성       │  │
│  └──────────┬───────────┘  └──────────┬────────────┘  └────────┬──────────┘  │
│             │                         │                        ↑             │
│       QAI Hub Cloud              adb push                 adb pull           │
│     (compile job)             /sdcard/sd_models/       CSV + 생성 이미지      │
└─────────────┼─────────────────────────┼────────────────────────┼─────────────┘
              │                         │                        │
              ▼                         ▼                        │
┌─────────────────────────────────────────────────────────────────┼─────────────┐
│  Samsung Galaxy S23 Ultra (Snapdragon 8 Gen 2)                  │             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  Android App (com.example.kpilab)                                      │  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌────────────────────┐  ┌────────────────────────┐  │  │
│  │  │ MainActivity  │→│ BenchmarkRunner     │→│ Txt2ImgPipeline         │  │  │
│  │  │ (UI, 실험 선택) │  │ (실험 루프, CSV)    │  │ (추론 파이프라인)        │  │  │
│  │  └──────────────┘  └─────────┬──────────┘  └──────────┬─────────────┘  │  │
│  │                              │                        │                │  │
│  │                   ┌──────────▼──────────┐    ┌────────▼──────────┐     │  │
│  │                   │ KpiCollector         │    │ OrtRunner ×3      │     │  │
│  │                   │ (thermal/power/mem)  │    │ (TextEnc/UNet/VAE)│     │  │
│  │                   └─────────────────────┘    └────────┬──────────┘     │  │
│  └───────────────────────────────────────────────────────┼────────────────┘  │
│                                                          │                   │
│  ┌───────────────────────────────────────────────────────▼────────────────┐  │
│  │  ONNX Runtime 1.24.3 + QNN Execution Provider                         │  │
│  │  ④ Session Load (ONNX stub + QNN binary) → ⑤ NPU Inference           │  │
│  └───────────────────────────────────────────────────────┬────────────────┘  │
│                                                          │                   │
│  ┌───────────────────────────────────────────────────────▼────────────────┐  │
│  │  Qualcomm Hexagon HTP (NPU)                                           │  │
│  │  QAIRT 2.42.0 — precompiled QNN context binary 실행                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**번호별 흐름:**
1. PC에서 PyTorch → ONNX export (FP32, opset 18)
2. 양자화 (QAI Hub W8A8 / AIMET MIXED_PR) + QAI Hub compile → QNN binary (.bin)
3. ADB로 모델 파일을 디바이스에 push
4. 앱이 ONNX stub + QNN binary를 ONNX Runtime 세션으로 로드
5. NPU에서 추론 실행, 앱이 latency/thermal/power/memory 수집
6. CSV를 PC로 pull하여 분석 스크립트로 처리

---

## 3. Android 앱 아키텍처

### 3.1 Layer 구조

```
┌────────────────────────────────────────────────────────────┐
│  UI Layer                                                   │
│  MainActivity — 실험 세트 선택, 진행 표시, 결과 이미지 출력     │
│  BatchProgress — 배치 실험 진행률 관리                         │
└───────────────────┬────────────────────────────────────────┘
                    │ 제어
┌───────────────────▼────────────────────────────────────────┐
│  Experiment Layer                                           │
│  BenchmarkRunner — 실험 루프 제어 (trial, warmup, cooldown)   │
│  BenchmarkConfig — 실험 파라미터 (steps, precision, perf)     │
│  ExperimentSet/Loader — JSON 기반 배치 실험 정의 로드          │
└───────────────────┬────────────────────────────────────────┘
                    │ 호출
┌───────────────────▼────────────────────────────────────────┐
│  Pipeline Layer                                              │
│  Txt2ImgPipeline — SD/LCM 추론 파이프라인 오케스트레이션        │
│  Scheduler — EulerDiscrete (SD) / DDIM-style (LCM)           │
│  Tokenizer — CLIP BPE 토크나이저                              │
│  ImagePreprocessor — VAE float → Bitmap 변환                 │
└───────────────────┬────────────────────────────────────────┘
                    │ 사용
┌───────────────────▼────────────────────────────────────────┐
│  Runtime Layer                                               │
│  OrtRunner — ONNX Runtime 세션 래퍼 (EP 선택, I/O 관리)       │
│  KpiCollector — /sys/class/thermal, BatteryManager, /proc    │
└───────────────────┬────────────────────────────────────────┘
                    │
            ONNX Runtime + QNN EP → Hexagon HTP (NPU)
```

### 3.2 핵심 클래스 관계

```
MainActivity
 ├─ ExperimentSetLoader.load("experiment_sets_txt2img.json")
 │   └→ List<ExperimentSet> (Phase별 실험 그룹)
 ├─ BenchmarkRunner
 │   ├─ 상태: IDLE → INITIALIZING → WARMING_UP → RUNNING → STOPPING
 │   ├─ Txt2ImgPipeline (생성/파괴)
 │   ├─ KpiCollector (시스템 메트릭 폴링)
 │   └─ CSV Writer (3가지 record type)
 └─ UI 업데이트 (coroutine, LiveData)
```

**`MainActivity`**: 사용자가 실험 세트(Phase 1~4)를 선택하면 BenchmarkRunner에 위임. 실시간으로 온도/전력/진행률을 UI에 표시하고 생성 이미지를 화면에 출력.

**`BenchmarkRunner`**: 실험의 핵심 오케스트레이터. 상태 머신으로 관리되며, 각 trial마다 파이프라인 호출 → KPI 수집 → CSV 기록 → cooldown 순으로 진행.

**`Txt2ImgPipeline`**: 3개의 OrtRunner(TextEncoder, UNet, VAEDecoder)를 소유하며, 하나의 `generate()` 호출로 텍스트 프롬프트에서 이미지까지의 전체 추론을 수행.

### 3.3 Text-to-Image 파이프라인

하나의 이미지를 생성하는 `Txt2ImgPipeline.generate()` 호출의 전체 흐름:

```
generate(prompt: String, seed: Long = 42)
 │
 ├─ ① Tokenize (CPU)
 │    BPE 토크나이저로 프롬프트 인코딩 → int32[1, 77]
 │    CLIP vocab.json 기반, padding/truncation 포함
 │
 ├─ ② Text Encode (NPU)
 │    OrtRunner(textEncoder).run(tokenIds)
 │    → float[1, 77, 768] (text embeddings)
 │    SD: uncond embedding도 생성 (CFG 7.5)
 │    LCM: uncond 불필요 (CFG 1.0)
 │
 ├─ ③ Initial Noise (CPU)
 │    Gaussian noise 생성, seed 고정
 │    → float[1, 4, 64, 64] (latent space)
 │
 ├─ ④ UNet Denoising Loop (NPU, N steps)
 │    for step in 0..<steps:
 │    │  ┌─ inputCreate: latent + timestep + embedding 텐서 구성
 │    │  ├─ sessionRun: OrtRunner(unet).run(...)
 │    │  │    → noise prediction float[1, 4, 64, 64]
 │    │  ├─ outputCopy: 결과 텐서 → latent 업데이트
 │    │  └─ schedulerStep: Scheduler.step(sample, modelOutput)
 │    │       SD: x_next = x + (σ_next − σ) × noise_pred
 │    │       LCM: predicted x0 계산 → next alpha로 blend
 │    │
 │    각 step마다 StepDetail 기록 (inputCreate/sessionRun/outputCopy/schedulerStep)
 │
 ├─ ⑤ VAE Decode (NPU)
 │    OrtRunner(vaeDecoder).run(latent)
 │    → float[1, 3, 512, 512]
 │
 └─ ⑥ Postprocess (CPU)
      ImagePreprocessor: [-1,1] → [0,255] 변환
      precision 분기: normalized=true (W8A16) vs false (FP16)
      → Bitmap (512×512 RGB)
```

**SD vs LCM의 차이점:**

| 항목 | SD v1.5 | LCM-LoRA |
|------|---------|----------|
| UNet 모델 | unet_base (.bin) | unet_lcm (LoRA fused, .bin) |
| Steps | 20 / 30 / 50 | 2 / 4 / 8 |
| Scheduler | EulerDiscrete | DDIM-style (alpha blending) |
| CFG guidance_scale | 7.5 (uncond + cond 두 번 호출) | 1.0 (cond만 한 번) |
| UNet 호출 횟수/step | 2 (uncond + cond) | 1 |

SD는 CFG(Classifier-Free Guidance) 때문에 step당 UNet을 **두 번** 호출한다. LCM은 guidance_scale=1.0으로 CFG를 비활성화하여 한 번만 호출. 이것이 step당 latency 차이(SD ~547ms vs LCM ~341ms)의 주요 원인 중 하나다.

### 3.4 OrtRunner — ONNX Runtime 세션 래퍼

각 컴포넌트(TextEncoder, UNet, VAEDecoder)마다 하나의 OrtRunner 인스턴스가 생성된다.

```
OrtRunner(modelPath, executionProvider, options)
 │
 ├─ Session 생성
 │    ONNX stub (.onnx) + QNN context binary (.bin) 로드
 │    Execution Provider 선택:
 │      CPU_EP    — FP32 CPU 추론 (디버그용)
 │      QNN_GPU   — QNN GPU backend
 │      QNN_NPU   — QNN HTP backend (주 실험 대상)
 │
 │    QNN NPU 옵션:
 │      backend_path: "libQnnHtp.so"
 │      htp_performance_mode: "burst" / "balanced" / "power_saver"
 │      enable_htp_fp16_precision: "1" (FP16 활성화)
 │
 ├─ run(inputs) → outputs
 │    FloatBuffer/IntBuffer 기반 I/O 텐서 관리
 │    System.nanoTime()로 session run 시간 측정
 │
 └─ close()
      세션 해제, 메모리 반환
```

**QNN context binary란?**
QAI Hub에서 ONNX 모델을 target device(Galaxy S23)에 맞게 미리 컴파일한 결과물이다. `.onnx` 파일은 graph 구조만 담은 stub이고, `.bin` 파일이 실제 NPU에서 실행되는 precompiled graph다. 이렇게 하면 디바이스에서 JIT 컴파일 없이 바로 실행 가능하지만, 첫 세션 로드 시 HTP graph initialization이 발생한다(cold start의 주요 원인).

### 3.5 Scheduler — 노이즈 스케줄러

Scheduler는 UNet의 noise prediction을 받아 latent를 점진적으로 denoising하는 역할이다. 모델 variant에 따라 다른 알고리즘을 사용한다.

**EulerDiscrete (SD v1.5):**
```
σ 스케줄 계산 (1000 timesteps에서 N개 선택)
for step in 0..<N:
    σ_curr, σ_next = sigmas[step], sigmas[step+1]
    noise_pred = unet(latent / sqrt(σ²+1), timestep, embedding)
    latent = latent + (σ_next - σ_curr) × noise_pred
```

**DDIM-style (LCM-LoRA):**
```
alpha 스케줄 계산
for step in 0..<N:
    noise_pred = unet(latent, timestep, embedding)
    x0_pred = (latent - sqrt(1-α) × noise_pred) / sqrt(α)
    latent = sqrt(α_next) × x0_pred + sqrt(1-α_next) × noise
```

두 스케줄러 모두 Kotlin으로 CPU에서 실행된다. 전체 E2E에서 스케줄러 자체의 overhead는 미미하다(수 ms).

### 3.6 KpiCollector — 시스템 메트릭 수집

```
KpiCollector
 │
 ├─ Thermal
 │    /sys/class/thermal/thermal_zone*/temp 읽기
 │    여러 thermal zone 중 SoC 온도에 해당하는 것 선택
 │    ÷ 1000 변환 (millidegree → °C)
 │
 ├─ Power
 │    BatteryManager.BATTERY_PROPERTY_CURRENT_NOW (μA)
 │    × BatteryManager.getIntProperty(VOLTAGE_NOW) (μV)
 │    → mW 변환
 │    시스템 전체 소비전력 (앱 단독이 아닌 기기 전체)
 │
 ├─ Memory
 │    /proc/self/status → VmRSS (앱 RSS)
 │    Debug.getNativeHeapSize() (native heap)
 │    QAI Hub profiling 추정치 (NPU 메모리) — 별도 조회
 │
 └─ Device Info
      Build.SOC_MODEL (SM8550 = Snapdragon 8 Gen 2)
      Build.VERSION.SDK_INT (API level)
      충전 상태 감지 (is_charging flag)
```

**측정 주기:** BenchmarkRunner가 1초 간격(SYSTEM_METRICS_INTERVAL_MS = 1000)으로 폴링. Trial 시작/종료 시점의 스냅샷도 별도 기록.

---

## 4. 실험 프레임워크

### 4.1 배치 실험 구조

실험은 JSON으로 선언적으로 정의된다 (`assets/experiment_sets_txt2img.json`):

```json
{
  "version": 2,
  "defaults": {
    "sdBackend": "QNN_NPU",
    "sdPrecision": "FP16",
    "guidanceScale": 7.5,
    "seed": 42,
    "totalTrials": 5,
    "warmupTrials": 2,
    "cooldownSec": 60
  },
  "experimentSets": [
    {
      "id": "phase1_precision_sd15",
      "name": "[Phase1] Precision — SD v1.5 (A1~A4)",
      "experiments": [
        { "steps": 20 },                                          // A1: FP16 baseline
        { "steps": 20, "precUnet": "MIXED_PR" },                  // A2: UNet 양자화
        { "steps": 20, "precVaeDec": "W8A8" },                    // A3: VAE 양자화
        { "steps": 20, "precUnet": "MIXED_PR", "precVaeDec": "W8A8" }  // A4: 둘 다
      ]
    },
    {
      "id": "phase4_sustained",
      "phase": "SUSTAINED",
      "totalTrials": 10,
      "cooldownSec": 0,
      "experiments": [ ... ]
    }
  ]
}
```

**핵심 설계 결정:**
- **선언적 실험 정의**: 실험 파라미터를 코드가 아닌 JSON에 분리. 새 실험 추가 시 코드 변경 불필요.
- **defaults 상속**: 공통 설정은 defaults에 두고, 개별 실험에서 override만 선언.
- **Phase 분리**: SINGLE_GENERATE(burst, cooldown 있음) vs SUSTAINED(연속, cooldown 없음)를 phase 필드로 구분.

### 4.2 BenchmarkRunner — 실험 오케스트레이터

```
BenchmarkRunner.runExperiment(config)
 │
 ├─ 상태: IDLE → INITIALIZING
 │    Txt2ImgPipeline 생성 (OrtRunner ×3 세션 로드)
 │    Cold start timing 기록
 │
 ├─ 상태: → WARMING_UP
 │    warmupTrials(2)회 추론 — 결과 버림
 │    목적: HTP JIT 최적화, 캐시 웜업
 │
 ├─ 상태: → RUNNING
 │    for trial in 0..<totalTrials:
 │    │
 │    │  ┌─ 시스템 메트릭 스냅샷 (trial 시작)
 │    │  │   thermal, power, memory
 │    │  │
 │    │  ├─ pipeline.generate(prompt, seed)
 │    │  │   → StageTiming + List<StepDetail>
 │    │  │
 │    │  ├─ 시스템 메트릭 스냅샷 (trial 종료)
 │    │  │
 │    │  ├─ CSV Record 생성 & 기록
 │    │  │   ├─ GenerateSummaryRecord (E2E, stage별 latency, resource)
 │    │  │   ├─ UnetStepRecord × N (step별 상세)
 │    │  │   └─ ColdStartRecord (trial=0일 때만)
 │    │  │
 │    │  └─ Cooldown (SINGLE_GENERATE phase만)
 │    │       waitForThermalTarget(35°C, min=60s, max=180s)
 │    │       60초 대기 → 35°C 미달 시 추가 대기 → 최대 180초
 │    │
 │    생성 이미지 저장 (PNG)
 │
 └─ 상태: → IDLE
      Pipeline 해제, 리소스 정리
```

**Warmup이 필요한 이유:**
- 첫 1~2회 추론은 HTP 내부 JIT 최적화, 버퍼 할당, 캐시 적재 등으로 인해 latency가 높다.
- Warmup 2회 후의 "warm" trial이 steady-state 성능을 대표한다.
- 실험 리포트의 수치는 모두 warm trial mean이다.

### 4.3 Cooldown 및 Thermal 관리

모바일 기기에서의 정확한 벤치마크를 위해 trial 간 thermal 상태를 통제한다:

```
waitForThermalTarget(targetTemp=35°C)
 │
 ├─ 최소 60초 대기 (무조건)
 │
 ├─ 현재 온도 확인
 │    if temp ≤ 35°C → 완료
 │    else → 추가 대기 (1초 간격 폴링)
 │
 └─ 최대 180초 도달 시 → 강제 진행
      (로그에 경고 출력)
```

**통제 조건 체크리스트** (실험 시작 전):
- 충전기 분리 (전력 측정 오염 방지)
- Airplane mode ON (네트워크 트래픽 차단)
- 화면 밝기 고정 (디스플레이 전력 일정 유지)
- 백그라운드 앱 종료 (CPU/메모리 경합 제거)
- 배터리 40% 이상 (저배터리 throttling 방지)
- 기기 온도 ≤ 35°C (초기 상태 통제)

### 4.4 CSV 데이터 수집

앱은 3종의 CSV 파일을 생성한다:

**① GenerateSummaryRecord** (`{session}_generate.csv`)

| 필드 | 설명 |
|------|------|
| trial_id | 0부터 순번 |
| model_variant | SD_V15 / LCM_LORA |
| steps, precision, backend | 실험 config |
| tokenize_ms, text_enc_ms, unet_total_ms, vae_dec_ms | 단계별 latency |
| e2e_ms | 전체 latency |
| peak_memory_mb | 최대 메모리 |
| avg_power_mw | 추론 구간 평균 전력 |
| thermal_start_c, thermal_end_c | 온도 변화 |

**② UnetStepRecord** (`{session}_unet_steps.csv`)

| 필드 | 설명 |
|------|------|
| trial_id, step_index | trial 내 step 번호 |
| input_create_ms | 입력 텐서 구성 시간 |
| session_run_ms | ORT 세션 실행 시간 |
| output_copy_ms | 출력 복사 시간 |
| scheduler_ms | 스케줄러 연산 시간 |

**③ ColdStartRecord** (`{session}_cold_start.csv`)

| 필드 | 설명 |
|------|------|
| session_init_ms | ORT 세션 로드 시간 (3개 모델 합산) |
| first_inference_ms | 첫 추론 시간 |
| cold_e2e_ms | session_init + first_inference |

---

## 5. 모델 배포 파이프라인

### 5.1 Export → Quantize → Compile → Deploy

```
                    PC (Python)                              QAI Hub Cloud
                    ───────────                              ─────────────
① Export            PyTorch → ONNX (FP32, opset 18)
   scripts/sd/      ├─ text_encoder_fp32.onnx
   export_sd_*.py   ├─ unet_base_fp32.onnx + .data (external data)
                    ├─ unet_lcm_fp32.onnx + .data
                    └─ vae_decoder_fp32.onnx

② Quantize         ┌─ QAI Hub quantize API (W8A8)     ────→  QDQ ONNX
   (선택적)          │   64 samples calibration
                    └─ AIMET op-selective (MIXED_PR)   ────→  mixed QDQ ONNX
                        Conv/MatMul/Gemm INT8, LayerNorm FP32
                        RunPod Linux 환경 (x86 AIMET 의존성)

③ Compile                                              ────→  QAI Hub compile job
   target: Galaxy S23                                          --target_runtime
   QAIRT 2.42.0                                                precompiled_qnn_onnx
                                                        ────→  {name}.onnx (stub)
                                                               {name}.bin  (QNN binary)

④ Deploy            adb push → /sdcard/sd_models/
   scripts/deploy/   ├─ text_encoder_fp16.onnx + .bin
   push_models.sh    ├─ unet_base_mixed_pr.onnx + .bin
                     ├─ unet_lcm_fp16.onnx + .bin
                     ├─ vae_decoder_w8a8.onnx + .bin
                     └─ ... (precision variant별)
```

### 5.2 Per-Component Precision Map

이 프로젝트의 핵심 설계 중 하나는 **컴포넌트별 독립 precision 설정**이다. 하나의 파이프라인 내에서 각 컴포넌트가 서로 다른 precision으로 동작한다:

```
실험 A4 (best config for SD):
  Text Encoder  → FP16      (237MB .bin)
  UNet          → MIXED_PR  (1,151MB .bin, Conv/MatMul INT8)
  VAE Decoder   → W8A8      (57MB .bin)

실험 A6 (best config for LCM):
  Text Encoder  → FP16      (237MB .bin)
  UNet LCM      → FP16      (1,651MB .bin)
  VAE Decoder   → W8A8      (57MB .bin)
```

**왜 per-component인가?**
- 컴포넌트마다 연산 구조가 다르다: VAE(Conv 중심), UNet(Conv + Attention), TextEncoder(Transformer)
- 품질 민감도가 다르다: UNet의 누적 오차(N steps)가 가장 크고, VAE는 1회 변환이라 상대적으로 robust
- HTP 제약이 다르다: LayerNorm full INT8 미지원으로 UNet은 MIXED_PR 필수, VAE는 W8A8 가능
- 이 설계로 "어떤 컴포넌트에 어떤 precision을 적용하면 KPI가 얼마나 변하는가"를 개별 측정 가능

### 5.3 QNN HTP 실행 흐름

```
ONNX Runtime Session
 │
 ├─ Session 생성 시:
 │    .onnx stub 파싱 (graph 구조)
 │    .bin QNN context binary 로드
 │    HTP graph initialization (cold start의 주요 비용)
 │    htp_performance_mode 설정 (burst/balanced)
 │
 ├─ 추론 시 (session.run):
 │    Input: CPU FloatBuffer → QNN internal format 변환
 │    Execution: Hexagon HTP 코어에서 연산
 │    Output: QNN → CPU FloatBuffer 복사
 │    모든 연산은 NPU에서 실행 (CPU fallback 최소화)
 │
 └─ 성능 제어:
      burst     — 최대 NPU clock, 최고 throughput, 발열 최대
      balanced  — 중간 clock, 전력-성능 균형
      power_saver — 최저 clock, 전력 최소
```

---

## 6. 분석 파이프라인

디바이스에서 수집된 CSV를 PC에서 분석한다:

```
analysis/
 │
 ├─ parse_txt2img_csv.py
 │    CSV 파싱 (GENERATE_SUMMARY / UNET_STEP_DETAIL / COLD_START)
 │    warmup trial 제외, warm trial mean/std 계산
 │    단계별 latency 분석 (UNet 비중, per-step 일관성)
 │    메모리 합산 (App RSS + NPU 추정치)
 │    텍스트 + 차트 리포트 생성
 │
 ├─ clip_score_phase1.py
 │    Phase 1 생성 이미지의 CLIP Score 계산
 │    openai/clip-vit-base-patch16 모델 사용
 │    이미지-텍스트 정렬도 (logit scale)
 │
 ├─ clip_score_phase2.py
 │    Phase 2 step sweep 품질 평가
 │    B2(SD 50step)를 base로 LPIPS + CLIP Score
 │    step-quality tradeoff 곡선 도출
 │
 └─ parse_phase4_sustained.py
      Phase 4 sustained 10회 연속 추론 분석
      trial별 latency drift, 온도 상승 추세
      thermal throttling 구간 탐지
```

**품질 평가 2단계 설계:**

1. **양자화 스크리닝 (CosSim)** — 양자화된 컴포넌트의 출력 텐서를 FP32와 비교. 파이프라인 전체를 돌리지 않고 빠르게 품질 열화를 감지. Poor(< 0.980) 등급은 실험 투입 전 제외.

2. **E2E 품질 평가 (LPIPS + CLIP)** — 파이프라인 전체를 실행한 최종 이미지로 평가. LPIPS는 지각적 유사도(낮을수록 유사), CLIP Score는 프롬프트 반영도(높을수록 양호). 동일 seed로 z_T를 고정하여, 이미지 차이가 순전히 모델(양자화)의 영향인지 통제.

---

## 7. 프로젝트 디렉토리 구조

```
mobile-inference-kpi-lab/
│
├── android/                             # Android 벤치마크 앱
│   └── app/src/main/
│       ├── java/com/example/kpilab/
│       │   ├── MainActivity.kt          # UI, 실험 세트 선택
│       │   ├── BenchmarkRunner.kt       # 실험 루프 오케스트레이터
│       │   ├── Txt2ImgPipeline.kt       # SD/LCM 추론 파이프라인
│       │   ├── OrtRunner.kt             # ONNX Runtime 세션 래퍼
│       │   ├── Scheduler.kt             # EulerDiscrete / DDIM-style
│       │   ├── Tokenizer.kt             # CLIP BPE 토크나이저
│       │   ├── KpiCollector.kt          # Thermal/Power/Memory 수집
│       │   ├── ImagePreprocessor.kt     # VAE 출력 → Bitmap
│       │   ├── BenchmarkConfig.kt       # 실험 파라미터 데이터 클래스
│       │   ├── InputMode.kt             # Enum: Precision, Variant, EP
│       │   └── batch/                   # 배치 실험 관리
│       │       ├── ExperimentSet.kt
│       │       └── ExperimentSetLoader.kt
│       └── assets/
│           ├── experiment_sets_txt2img.json  # 배치 실험 정의
│           └── vocab.json                   # CLIP 토크나이저 어휘
│
├── scripts/
│   ├── sd/                              # 모델 export & 양자화
│   │   ├── export_sd_to_onnx.py         # TextEnc/VAE ONNX export
│   │   ├── export_sd_lcm_unet.py        # UNet export (base + LCM)
│   │   ├── eval_sd_quant_quality.py     # CosSim 품질 스크리닝
│   │   └── quant_runpod.py              # AIMET MIXED_PR 양자화
│   ├── deploy/                          # 디바이스 배포
│   │   ├── deploy_config.json           # 모델 파일 매핑
│   │   └── push_models_to_device.sh     # ADB push 스크립트
│   └── inference/                       # PC 기반 추론 비교
│       ├── resolution_compare.py
│       └── text2img_sd_lcm_compare.py
│
├── analysis/                            # 결과 분석 스크립트
│   ├── parse_txt2img_csv.py             # CSV 파싱 & 리포트
│   ├── clip_score_phase1.py             # CLIP Score 계산
│   ├── clip_score_phase2.py             # LPIPS + CLIP (step sweep)
│   └── parse_phase4_sustained.py        # Sustained drift 분석
│
├── docs/                                # 문서
│   ├── experiment_design.md             # 실험 설계 & 방법론
│   ├── experiment_report.md             # 실험 결과 종합
│   ├── model_optimization.md            # 양자화 전략 내러티브
│   ├── app_architecture.md              # ← 이 문서
│   ├── weights_inventory.md             # 모델 파일 목록
│   └── reference_runtime.md             # vAI 런타임 참고
│
├── weights/sd_v1.5/                     # 모델 파일 (git 제외)
│   ├── onnx/                            # ONNX + QNN binary
│   └── calib_data/                      # 양자화 calibration 데이터
│
├── outputs/                             # 실험 결과
│   ├── exp/                             # CSV, 분석 리포트
│   └── quantization/                    # CosSim 평가 결과
│
└── logs/                                # Phase별 실험 로그
    ├── phase1/ ~ phase4/
```
