# Mobile NPU Inference KPI 실험 설계

---

## 0. 실험 목적

Snapdragon 8 Gen 2 기반 Android 단말에서 YOLOv8 객체 탐지 모델을 대상으로, **execution backend**, **precision 정책**, **model scale**이 inference KPI에 미치는 영향을 정량 분석한다.

### 핵심 질문

1. **Backend 선택**: CPU / GPU / NPU 중 어떤 backend가 latency, power, thermal 측면에서 최적인가?
2. **Precision 효과**: 동일 NPU backend에서 FP16 vs INT8 QDQ vs INT8 QIO가 실측 성능에 미치는 영향은?
3. **Model Scale**: 모델 크기(n/s/m)가 커질 때 각 backend에서의 latency 증가율은 어떻게 달라지는가?
4. **Sustained 안정성**: Peak 성능이 sustained 운영에서도 유지되는가? Thermal throttling은 발생하는가?

---

## 1. System Architecture

### 1.1 Runtime Stack

```
Android App (Kotlin)
  ├─ CameraX (ImageAnalysis) → Real-time frame input
  └─ ONNX Runtime 1.23.2
       ├─ CPU EP (ARM Cortex-X3)
       ├─ QNN EP → QNN SDK v2.34 → Adreno GPU
       └─ QNN EP → QNN SDK v2.34 → Hexagon HTP (NPU)
```

### 1.2 왜 ORT + QNN EP인가

- **Graph Partitioning 가시성**: QNN이 지원하지 않는 op은 CPU EP로 자동 fallback되며, partition log로 식별 가능
- **Hybrid 실행**: NPU + CPU 혼합 실행을 자연스럽게 지원하여 실제 모바일 환경 재현
- **ORT Profiling**: session.run() 내부를 NPU compute / CPU fallback / fence / ORT overhead로 분해 가능

### 1.3 Inference Pipeline

CameraX real-time pipeline을 기본으로 한다.

```
Camera(YUV420) → Acquire → PreProc → session.run() → PostProc → [KPI Record]
```

| 단계 | 설명 | 실행 위치 |
|------|------|-----------|
| **Acquire** | CameraX ImageAnalysis에서 ImageProxy 수신 → ARGB Bitmap 변환 | Camera HAL → CPU |
| **PreProc** | Letterbox resize + normalize + CHW transpose | CPU (Kotlin) |
| **Inference** | OnnxTensor 생성 + session.run() + output 추출 | JNI + EP (CPU/GPU/NPU) |
| **PostProc** | Confidence filter + NMS + coordinate transform | CPU (Kotlin) |

#### Timing 측정 범위

```
├── Acquire ──┤├── PreProc ──┤├───── Inference ─────┤├── PostProc ──┤
                              ├─InCreate─┤├─Run─┤├─OutCopy─┤
E2E = ─────────────────────────────────────────────────────────────────
```

- **E2E latency**: Acquire ~ PostProc 전체
- **Inference breakdown**: InCreate (tensor 생성) + Run (session.run) + OutCopy (output 추출)
- **Acquire**는 별도 측정하여 camera overhead를 분리

#### Input Source Modes

벤치마크 목적에 따라 입력 소스를 선택할 수 있다.

| Mode | 설명 | Acquire 포함 | 용도 |
|------|------|:---:|------|
| **Camera Live** | CameraX에서 매 frame 수신 | O | Production pipeline 시뮬레이션 (기본값) |
| **Camera Single** | Camera에서 1장 캡처 후 반복 사용 | 초회만 | 변인 통제 + camera frame 품질 반영 |
| **Static Image** | Assets의 sample_image.jpg 반복 사용 | X | 순수 latency 비교 (fallback) |

Demo mode (detection overlay on preview)를 UI에서 on/off 할 수 있으나, overlay 렌더링은 E2E 측정 범위 밖에서 수행하므로 벤치마크 결과에 영향 없음.

---

## 2. 실험 환경

### 2.1 Hardware

| 항목 | 값 |
|------|-----|
| Device | Samsung Galaxy S24 Ultra (SM-S918N) |
| SoC | Snapdragon 8 Gen 2 (SM8550) |
| CPU | Cortex-X3 (1) + Cortex-A715 (4) + Cortex-A510 (3) |
| GPU | Adreno 740 |
| NPU | Hexagon Tensor Processor (HTP) |

### 2.2 Software

| 항목 | 값 |
|------|-----|
| Runtime | ONNX Runtime 1.23.2 |
| QNN SDK | v2.34 |
| Android | API 34 |
| QNN Options | backend=HTP, perf_mode=burst, context_cache=0 |

### 2.3 실험 조건 통제

| 항목 | 설정 | 근거 |
|------|------|------|
| CPU Governor | `performance` 고정 | Frequency scaling에 의한 latency variance 제거 |
| Camera Executor | Single thread executor | Scheduler interference 방지, analysis thread 격리 |
| 화면 | Always-on, 최소 밝기 | Display 부하 최소화 |
| 네트워크 | Airplane mode | Background traffic 제거 |
| 충전 | 벤치마크 중 미충전 | 충전 시 thermal / power 측정 왜곡 방지 |

### 2.3 Target Models

| Model | Params | FP32 Size | INT8 Size | Input Shape |
|-------|--------|-----------|-----------|-------------|
| YOLOv8n | 3.2M | 12.5 MB | 3.4 MB | [1, 3, 640, 640] |
| YOLOv8s | 11.2M | 42.8 MB | 11.0 MB | [1, 3, 640, 640] |
| YOLOv8m | 25.9M | 101.4 MB | 25.7 MB | [1, 3, 640, 640] |

### 2.4 Backend별 Precision 선택 근거

각 backend는 하드웨어 아키텍처에 따라 최적 precision이 다르다. 각 backend의 **production-optimal precision**을 사용한다.

| Backend | Model Precision | 근거 |
|---------|----------------|------|
| CPU | FP32 | ORT CPU EP 기본 동작 |
| GPU | FP32 | QNN GPU EP 기본 동작. 내부 실행 precision은 Adreno 구현에 의존 |
| NPU | FP16 (runtime) | FP32 모델 + `enable_htp_fp16_precision=1`로 HTP에서 FP16 변환 실행 |
| NPU | INT8 QDQ | 사전 양자화(Static). HTP 최적. Boundary Q/DQ 2개 CPU fallback |
| NPU | INT8 QIO | QDQ에서 boundary Q/DQ 제거. 100% NPU offload |

---

## 3. Phase 1 — Burst Latency Test

### 3.1 목적

- Thermal 영향 없는 상태에서 **순수 inference latency** 비교
- Backend × Precision × Model Scale의 peak performance 평가
- Cold start 비용 측정

### 3.2 실행 방식

```
Warmup: 10 iterations (연속, 결과 제외)
Profiling: 50 iterations (ORT profiling 수집)
Main loop:
  for i in 1..100:
      run full E2E inference
      sleep 500ms
Input: Camera Single (1장 캡처 후 반복)
```

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| Iterations | 100 | 통계적 신뢰도 확보 (P50/P95/min/max) |
| Sleep | 500ms (2 Hz) | Thermal 축적 방지, scheduling interference 최소화 |
| Warmup | 10 iterations | JIT, cache warming, HTP initialization 안정화 |
| Input | Camera Single | 변인 통제 (동일 frame 반복) |

### 3.3 실험 구성

#### 실험 1: Backend × Precision (YOLOv8n 고정)

| # | Backend | Precision | Model |
|---|---------|-----------|-------|
| 1 | CPU | FP32 | YOLOv8n |
| 2 | GPU | FP32 | YOLOv8n |
| 3 | NPU | FP16 | YOLOv8n |
| 4 | NPU | INT8 QDQ | YOLOv8n |
| 5 | NPU | INT8 QIO | YOLOv8n |

#### 실험 2: Model Scale (최적 config 고정)

| # | Model | Backend | Precision |
|---|-------|---------|-----------|
| 1 | YOLOv8n | NPU | INT8 QDQ |
| 2 | YOLOv8s | NPU | INT8 QDQ |
| 3 | YOLOv8m | NPU | INT8 QDQ |

> 실험 1에서 최적 backend/precision 확인 후 해당 설정으로 model scale 비교

### 3.4 측정 Metrics

| 카테고리 | Metric | 설명 |
|---------|--------|------|
| **Latency** | E2E P50 / P95 / Min | 전체 pipeline latency 분포 |
| | Inference P50 / Min | session.run() only |
| **Breakdown** | Acquire / PreProc / InCreate / Run / OutCopy / PostProc | E2E 구간별 소요 시간 |
| **ORT Profiling** | NPU(ms) / CPU(ms) / Fence(ms) / ORT(ms) / NPU% | session.run 내부 분해 |
| | CPU Ops | CPU fallback된 op 목록 |
| **Cold Start** | Load / Session / 1st Inference | 모델 로드 ~ 첫 추론 완료 |
| **Graph** | Total / QNN / CPU nodes / Coverage% | Graph partitioning 결과 |

---

## 4. Phase 2 — Sustained Throughput Test

### 4.1 목적

- Production 환경 시뮬레이션: **실측 throughput** 및 **frame drop rate**
- **Thermal throttling** 유무 검증
- **Power consumption** 비교

### 4.2 실행 방식

```
Warmup: 10 iterations (연속)
Profiling: 50 iterations (ORT profiling)
Main loop (5 minutes):
  target = 30 Hz (33ms interval)
  for each frame:
      acquire camera frame
      run full E2E inference
      if elapsed < 33ms: sleep remaining
      else: record frame_drop
Input: Camera Live (매 frame 새로 수신)
```

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| Target FPS | 30 Hz | 모바일 real-time inference 표준 |
| Duration | 5 분 | Thermal saturation 관찰에 충분 |
| Input | Camera Live | Production pipeline 재현 |

### 4.3 실험 구성

Phase 1에서 유망한 **3~4개 configuration** 선별.

선별 기준:
- E2E P50 < 33ms (30 FPS 가능) 또는 경계선 config
- Backend 다양성 (GPU, NPU 각 1개 이상)

### 4.4 측정 Metrics

| 카테고리 | Metric | 설명 |
|---------|--------|------|
| **Throughput** | Avg FPS / Frame Drop Rate | 실제 처리율 및 target 미달 비율 |
| **Latency Drift** | First 30s P50 / Last 30s P50 / Drift% | Stable(±3%) / Slight(3~10%) / Throttling(>10%) |
| **Jitter** | P95 - P50 gap | GC spike 등 outlier 간접 관찰 |
| **Thermal** | Temp Slope (C/min) / Peak Temp | 온도 상승 추세 |
| **Power** | Avg Power (mW) / Energy per Inference (mJ) | 전력 효율 |
| **Memory** | Peak (MB) / Delta (MB) | 메모리 사용 및 leak 감지 |

---

## 5. 분석 프레임워크

### 5.1 Phase 1 분석 관점

- **Backend 비교**: E2E P50 bar chart. Inference vs Pre+Post 비율로 CPU floor 가시화
- **Precision 비교**: 동일 NPU에서 FP16 / INT8 QDQ / INT8 QIO의 inference, coverage, cold start 비교
- **Model Scale**: n/s/m의 E2E 및 NPU compute 증가율. CPU 대비 NPU에서 sublinear한지 확인

### 5.2 Phase 2 분석 관점

- **Latency over time**: 시계열 그래프로 thermal throttling 유무 확인
- **Frame drop**: Target 30 FPS 대비 실제 달성률. Config별 비교

### 5.3 Discussion 포인트

1. **Static vs Live Frame**: Camera live frame에서의 YUV→RGB 변환, scene complexity, GC pressure가 latency에 미치는 추가 영향
2. **CPU Floor**: E2E 중 inference 외 overhead 비중. Zero-copy 경로 또는 GPU 전처리 사용 시 예상 개선폭
3. **INT8 QIO Paradox**: session.run 최소이나 boundary quantize/dequantize가 Kotlin에서 수행되어 E2E 역전. Native 구현 시 예상 개선
4. **NPU Scale 효율성**: CPU에서 n→m 6x 느려지는 반면 NPU에서 ~1.1x인 이유. HTP 병렬 아키텍처 분석
5. **Cold Start Trade-off**: NPU session 생성 1~4초. Context cache 활성화 시 개선 가능

---

## 6. 실험 실행 순서

```
1. Phase 1 — Burst Latency
   ├── 실험 1: Backend × Precision (YOLOv8n, 5 configs)
   ├── 결과 분석 → 최적 backend/precision 확인
   ├── 실험 2: Model Scale (n/s/m, 최적 config)
   └── 결과 분석 → Phase 2 config 선별

2. Phase 2 — Sustained Throughput
   ├── 선별된 3~4 configs, 각 5분, Camera Live
   └── 결과 분석 → thermal/power/frame drop

3. Report
   ├── Phase 1/2 결과 종합
   ├── Discussion
   └── Production 배포 권장 configuration
```
