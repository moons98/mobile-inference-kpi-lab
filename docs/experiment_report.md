# Snapdragon NPU Offload의 실질적 효과: Coverage, Precision, E2E Pipeline이 Sustained KPI에 미치는 영향 분석

---

# 0. Context — 모바일 NPU 시대의 실질적 병목은 어디에 있는가

이 글은 Snapdragon 기반 Android 단말에서 YOLOv8n 객체 탐지 모델을 대상으로, NPU offload coverage, precision 정책, 그리고 E2E pipeline 구성이 실제 sustained KPI에 미치는 영향을 정량 분석한 내용이다. "NPU에 올리면 빨라진다"는 단순한 명제를 넘어, 왜 coverage가 100%에 가까워도 E2E latency가 비례하여 감소하지 않는지, 그리고 그 병목이 어디에서 발생하는지를 실측 데이터를 통해 밝히는 데 초점을 두었다.

본 실험은 ONNX Runtime + QNN Execution Provider 구조 위에서 진행되었다. ORT의 graph partitioning 메커니즘을 활용하여 NPU가 처리 가능한 연산과 CPU로 fallback되는 연산을 명확히 분리하고, 각 영역이 E2E latency에 기여하는 비중을 개별적으로 측정하였다.

실험 결과, NPU offload는 inference 단계에서 CPU 대비 약 12배의 latency 감소를 달성하였으나, E2E 기준으로는 약 4.5~4.8배의 개선에 그쳤다. 이는 CPU에서 수행되는 전후처리(preprocessing/postprocessing)가 E2E latency의 하한선으로 작용하기 때문이다. 이러한 관찰을 바탕으로, ONNX 모델에 전처리/후처리/NMS를 순차적으로 통합(bake-in)하여 NPU offload 범위를 확장하는 추가 실험을 진행하였으며, pipeline 구성 변화가 E2E KPI에 미치는 영향을 정량적으로 확인하였다.

---

# 1. Background & Goal

## 1.1 문제 인식

모바일 SoC에 탑재된 NPU(Neural Processing Unit)는 inference 연산을 전용 하드웨어로 가속할 수 있다. 그러나 실제 모바일 inference pipeline은 모델 추론만으로 완결되지 않는다.

```
[Image Input] → CPU Pre → NPU Inference → CPU Post → [Result]
```

전처리(letterbox resize, normalize, CHW transpose)와 후처리(confidence filter, NMS, coordinate transform)는 여전히 CPU에서 수행되며, 이 구간의 latency는 NPU 성능과 무관하게 일정 수준 이하로 줄일 수 없다. 따라서 NPU가 아무리 빨라도 E2E latency에는 구조적 하한이 존재한다.

## 1.2 핵심 질문

본 실험은 다음 세 가지 질문에 답하고자 설계되었다.

1. NPU offload coverage가 높아도 왜 E2E latency는 비례 감소하지 않는가?
2. Precision(INT8 QDQ vs FP16 runtime)이 coverage 및 inference latency에 실제로 어떤 영향을 주는가?
3. 전후처리를 ONNX 모델에 통합(bake-in)하여 NPU offload 범위를 확장하면 E2E latency를 얼마나 개선할 수 있는가?

## 1.3 Target Model

**YOLOv8n** (Object Detection, ONNX)

| 항목 | 값 |
|------|-----|
| Parameters | ~3.2M |
| Model size (FP32) | 12.5 MB |
| Model size (INT8 QDQ) | 3.4 MB |
| Input shape | [1, 3, 640, 640] |
| Output | 84 × 8400 (80 classes + 4 box coords) |

YOLOv8n을 선택한 이유:

- 일부 shape/activation op이 존재하여 partition 관찰 가능
- 가벼워 반복 실험 및 sustained 측정 용이
- 모델 1개에 집중하여 변인 통제 가능

---

# 2. System Architecture

## 2.1 QNN (Qualcomm Neural Network)

QNN은 Snapdragon SoC의 AI 가속기(HTP/NPU, Adreno GPU, CPU)를 제어하는 저수준 inference runtime SDK이다.

- ONNX 모델을 QNN IR로 변환
- Backend(HTP/GPU/CPU) 선택 및 graph compile
- Op mapping, fusion, 실제 NPU dispatch 수행

## 2.2 ORT + QNN EP 구조

ONNX Runtime(ORT)은 framework-independent inference engine이며, Execution Provider(EP) 구조를 통해 다양한 backend를 연결할 수 있다.

```
ONNX Runtime
   ├─ CPU EP
   └─ QNN EP → QNN → HTP (NPU)
```

QNN EP는 ORT와 QNN runtime을 연결하는 어댑터 계층이다. ORT는 graph orchestration 및 partition을 담당하며, 실제 NPU dispatch는 QNN이 수행한다.

## 2.3 왜 ORT + QNN EP를 사용했는가

본 프로젝트의 목적은 NPU 실행 시 발생하는 fallback을 분석하고, 각 실행 영역이 KPI에 미치는 영향을 정량화하는 것이다. 이를 위해 Pure QNN이 아닌 ORT + QNN EP 구조를 채택하였다.

### Fallback 가시성 확보

ORT는 graph partition을 수행하며, QNN이 지원하지 않는 op를 CPU EP로 자동 분리한다. Partition log를 통해 fallback node를 명확히 식별할 수 있다. Pure QNN 사용 시에는 unsupported op에서 compile 에러가 발생하며, 자동 fallback 구조가 없어 분석용 hybrid 실행이 어렵다.

### Hybrid 실행 환경 구성

ORT는 일부 op는 QNN(HTP), 일부 op는 CPU 형태의 혼합 실행을 자연스럽게 지원한다. 이는 실제 모바일 환경에서 발생하는 fallback 시나리오를 현실적으로 재현하는 데 필수적이다.

### 실험 반복성과 개발 효율성

Graph 수정 후 즉시 재실행이 가능하고, CPU baseline 비교가 용이하며, fallback 전후 KPI 차이를 정량화할 수 있다. QNN 단독 사용 시에는 offline compile 및 graph 재생성이 필요하여 분석 파이프라인 구축에 비효율적이다.

---

# 3. Experiment Design

## 3.1 실험 환경

| 항목 | 값 |
|------|-----|
| Device | Samsung Galaxy S24 Ultra (SM-S918N) |
| SoC | Snapdragon 8 Gen 2 (SM8550) |
| NPU | Hexagon HTP (Tensor Processor) |
| Runtime | ONNX Runtime 1.23.2 |
| QNN SDK | v2.34 |
| QNN Options | backend=HTP, perf=burst, cache=0 |
| Measurement | 300 inferences per run, 5 Hz (200ms interval) |
| Warmup | 10 iterations (auto-excluded) |
| Duration | ~5 minutes per run |

### Timing Definition

E2E latency는 CPU wall-time 기준으로 측정하였다.

- **PreProc**: Image preprocessing (letterbox resize + normalize + CHW transpose)
- **Inference**: ONNX Runtime `session.run()` 호출 (EP에 의해 가속)
- **PostProc**: Confidence filter + NMS + coordinate transform
- **E2E Latency**: PreProc + Inference + PostProc

## 3.2 Backend별 Precision 선택 근거

### 3.2.1 Precision은 독립 변수가 아니다

각 backend는 하드웨어 아키텍처에 따라 최적 precision이 다르다:

| Backend | FP32 | FP16 | INT8 |
|---------|------|------|------|
| CPU (ARM Cortex) | **기본 동작** | SW 에뮬레이션 | NEON dot product로 가능하나 ORT CPU EP는 FP32 동작 |
| GPU (Adreno) | **기본 동작** | HW 네이티브 지원 (QNN EP에 런타임 옵션 없음, FP16 모델 필요) | dequant→FP32 변환 후 연산 |
| NPU (HTP) | 가능하나 비효율적 | **런타임 변환** (`enable_htp_fp16_precision`) | **HW 최적화** (QDQ 정적 양자화) |

- **CPU**는 범용 프로세서로 FP32 연산이 기본이며, INT8 전용 가속 유닛이 없다.
- **GPU**는 FP32/FP16 부동소수점 연산에 특화되어 있으며, INT8을 넣어도 내부적으로 dequantize 후 부동소수점으로 처리한다.
- **NPU(HTP)**는 저정밀도 고병렬 연산에 특화된 프로세서로, INT8/FP16에서 최대 throughput을 달성한다. FP32로 구동하면 HW 특성을 활용하지 못하는 비현실적 시나리오가 된다.

### 3.2.2 Op Coverage도 backend마다 다르다

동일 ONNX 그래프라도 backend별 지원 op set이 달라 실행 구성이 달라진다:

| Backend | Coverage | Fallback Ops |
|---------|----------|-------------|
| CPU EP | 100% (전부 CPU) | — |
| QNN GPU | 99.6% | Softmax |
| QNN NPU (FP16) | 100.0% | — |
| QNN NPU (INT8 QDQ) | 99.8% | QuantizeLinear, DequantizeLinear |

같은 모델이라도 backend에 따라 일부 op이 CPU로 fallback되므로, precision을 통일하더라도 **실행되는 그래프 구조 자체가 동일하지 않다**.

### 3.2.3 채택한 비교 방식

위 제약 조건에 따라 본 프로젝트는 두 가지 비교 축을 분리하였다.

**축 1: Backend 비교 — 각 backend의 production-optimal precision 사용**

| Backend | Precision | 근거 |
|---------|-----------|------|
| CPU | FP32 | ORT CPU EP 기본 동작 모드 |
| GPU | FP32 | QNN GPU EP에 FP16 런타임 옵션 부재 |
| NPU | FP16 | `enable_htp_fp16_precision=1` 런타임 변환 |
| NPU | INT8 | QDQ 정적 양자화 모델, HTP 최적 precision |

이는 "실전 배포 시 어떤 backend가 최적인가"에 답하는 비교이며, MLPerf Mobile 등 업계 벤치마크에서도 각 HW의 best config로 비교하는 것이 표준이다.

**축 2: Precision 비교 — 동일 backend(NPU) 내에서 precision만 변경**

NPU FP16 vs NPU INT8 비교를 통해 같은 하드웨어에서 precision이 latency, power, coverage에 미치는 영향을 분리 분석한다. 이 경우 hardware가 통제되므로 정당한 대조군이 성립한다.

## 3.3 실험 1: Execution Path & Precision Impact

NPU offload coverage와 precision 정책이 E2E KPI에 미치는 영향을 분석한다.

| Case | EP | Precision | 전처리 | 추론 | 후처리 |
|------|-----|-----------|--------|------|--------|
| **A** | CPU | FP32 | CPU | CPU | CPU |
| **B** | QNN_NPU | FP16 (runtime) | CPU | NPU | CPU |
| **C** | QNN_NPU | INT8 (QDQ) | CPU | NPU | CPU |

- Case A는 CPU-only baseline으로, NPU offload 효과의 참조 기준이 된다.
- Case B는 FP32 모델을 runtime에서 FP16으로 변환하여 NPU에 offload한다.
- Case C는 사전 양자화(Static INT8 QDQ)된 모델을 NPU에 offload한다.

## 3.4 실험 2: E2E Pipeline Offload Impact

실험 1에서 관찰된 CPU 전후처리 병목을 해소하기 위해, ONNX 모델 자체에 전처리/후처리/NMS를 순차적으로 통합(bake-in)하여 NPU offload 범위를 확장한다.

| Variant | 모델 구성 | NPU Offload 범위 |
|---------|-----------|------------------|
| **Baseline** | YOLOv8n | Inference only |
| **+Pre** | YOLOv8n(Pre) | Pre + Inference |
| **+E2E** | YOLOv8n(E2E) | Pre + Inference + Post |
| **+E2E+NMS** | YOLOv8n(E2E+NMS) | Pre + Inference + Post + NMS |

모든 variant는 동일한 QNN_NPU EP, FP16 precision으로 실행하여 변인을 pipeline 구성 차이로 한정한다.

---

# 4. Experiment 1 Results — Execution Path & Precision Impact

## 4.1 Graph Partitioning & Coverage

**Table 1 — Graph Partitioning**

| Case | Total Nodes | QNN Nodes | CPU Nodes | Coverage | Fallback Ops |
|------|-------------|-----------|-----------|----------|--------------|
| A: CPU FP32 | — | — | — | N/A | — |
| B: NPU FP16 | 241 | 241 | 0 | 100.0% | — |
| C: NPU INT8 | 958 | 956 | 2 | 99.8% | QuantizeLinear, DequantizeLinear |

FP16 runtime 변환 시 모든 241개 노드가 QNN EP에서 실행되어 100% coverage를 달성하였다.

INT8 QDQ 모델은 양자화/역양자화 노드가 삽입되어 전체 노드 수가 958개로 증가하였다. 이 중 2개의 QuantizeLinear/DequantizeLinear 노드만 CPU fallback되었으며, 99.8%의 coverage를 보였다. 그래프 입출력 경계에 위치한 QDQ 노드가 fallback된 것으로, 실질적으로 모든 연산이 NPU에서 수행된다.

---

## 4.2 Latency Performance

**Table 2 — E2E Latency**

| Case | E2E P50 (ms) | E2E P95 (ms) | Mean (ms) | MaxFPS |
|------|-------------|-------------|-----------|--------|
| A: CPU FP32 | 105.63 | 122.49 | 107.16 | 9.5 |
| B: NPU FP16 | 23.50 | 27.25 | 23.61 | 42.6 |
| C: NPU INT8 | 21.97 | 28.47 | 22.05 | 45.5 |

**Table 3 — E2E Breakdown (P50)**

| Case | PreProc (ms) | Inference (ms) | PostProc (ms) | E2E (ms) |
|------|-------------|----------------|--------------|----------|
| A: CPU FP32 | 11.26 | 92.47 | 0.89 | 105.63 |
| B: NPU FP16 | 13.99 | 7.81 | 1.72 | 23.50 |
| C: NPU INT8 | 11.31 | 9.36 | 1.35 | 21.97 |

---

### Observation

NPU offload는 inference 단계에서 극적인 효과를 보였다. CPU FP32 대비 FP16 NPU는 inference P50이 92.47ms → 7.81ms로 약 **11.8배 감소**하였다. 그러나 E2E P50은 105.63ms → 23.50ms로 **4.5배 감소**에 그쳤다.

이 차이는 전후처리 구간에서 발생한다. NPU FP16의 경우 PreProc(13.99ms) + PostProc(1.72ms) = 15.71ms가 CPU에서 소요되며, 이는 E2E P50(23.50ms)의 **66.8%**를 차지한다. 즉 NPU inference가 아무리 빨라도, CPU 전후처리가 E2E latency의 하한선으로 작용한다.

```
E2E latency ≥ PreProc + PostProc ≈ 15.7ms  (CPU floor)
```

Precision 관점에서 INT8 QDQ는 FP16 대비 E2E P50이 23.50ms → 21.97ms로 약 6.5% 감소하였다. 그러나 inference 단계만 보면 INT8(9.36ms)이 FP16(7.81ms)보다 오히려 **약 20% 느렸다**. E2E 개선은 inference 가속이 아닌 PreProc 시간 감소(13.99ms → 11.31ms)에서 발생하였다. 이는 INT8 모델의 크기(3.4MB vs 12.5MB)가 메모리 접근 패턴에 영향을 줄 수 있음을 시사하나, inference 자체의 latency 증가는 INT8 QDQ 그래프의 추가 노드 오버헤드(958 vs 241 nodes)에 기인하는 것으로 보인다.

P95 기준에서도 INT8(28.47ms)이 FP16(27.25ms)보다 높아, INT8이 꼬리 지연(tail latency)에서도 더 불안정한 경향을 보였다.

---

## 4.3 Cold Start

**Table 4 — Cold Start Breakdown**

| Case | Total (ms) | Load (ms) | Session (ms) | 1st Inference (ms) |
|------|-----------|-----------|-------------|-------------------|
| A: CPU FP32 | 132 | 15 | 22 | 95.70 |
| B: NPU FP16 | 2,878 | 32 | 2,831 | 15.44 |
| C: NPU INT8 | 1,275 | 6 | 1,257 | 12.16 |

### Observation

NPU를 사용하는 Case B, C에서 cold start가 크게 증가하였다. 이는 Session 생성 단계에서 QNN EP가 HTP 그래프 컴파일을 수행하기 때문이다.

FP16(2,831ms)이 INT8(1,257ms)보다 session 생성이 약 2.3배 느렸다. 이는 FP16 모델의 더 큰 그래프 크기(12.5MB vs 3.4MB)가 컴파일 시간에 영향을 주기 때문으로 보인다.

반면 1st inference는 NPU가 CPU보다 크게 빨랐으며(15.44ms vs 95.70ms), steady-state inference와 유사한 수준이었다. 즉 cold start 병목은 inference가 아닌 **graph compilation**에 집중되어 있다.

> QNN Context Cache를 활성화하면 컴파일된 그래프를 재사용하여 session 생성 시간을 크게 줄일 수 있으나, 본 실험에서는 cache=0으로 설정하여 매 실행마다 컴파일을 수행하였다.

---

## 4.4 Sustained Performance & Thermal

**Table 5 — Thermal Drift (Latency P50 변화)**

| Case | First 30s P50 (ms) | Last 30s P50 (ms) | Drift % | Verdict |
|------|-------------------|-------------------|---------|---------|
| A: CPU FP32 | 105.73 | 105.65 | -0.1% | Stable |
| B: NPU FP16 | 23.25 | 23.61 | +1.5% | Stable |
| C: NPU INT8 | 22.33 | 21.46 | -3.9% | Warmup effect |

**Table 6 — System Resources**

| Case | Power (mW) | Thermal Slope (°C/min) | Memory Peak (MB) |
|------|-----------|----------------------|-----------------|
| A: CPU FP32 | 2,887.9 | +1.14 | 309 |
| B: NPU FP16 | 433.4 | -0.49 | 422 |
| C: NPU INT8 | 413.3 | -0.15 | 366 |

### Observation

5분간의 sustained 실행에서 세 case 모두 drift가 ±4% 이내로 안정적이었다.

전력 소비에서 NPU의 효율이 두드러졌다. CPU FP32(2,888mW) 대비 NPU FP16(433mW)은 **약 6.7배 낮은 전력**을 소비하였다. INT8(413mW)은 FP16 대비 소폭 추가 절감을 보였다.

온도 상승률(thermal slope) 역시 NPU가 우수하였다. CPU는 +1.14°C/min으로 지속적인 발열이 관찰된 반면, NPU는 -0.49~-0.15°C/min으로 오히려 냉각되는 추세를 보였다. 이는 NPU가 동일 workload를 CPU 대비 훨씬 적은 에너지로 처리하기 때문이다.

메모리 사용에서는 NPU가 CPU보다 높은 peak memory를 보였다(422MB vs 309MB). QNN EP의 HTP 그래프 컴파일 및 NPU buffer 할당에 추가 메모리가 필요하기 때문이다. INT8 QDQ(366MB)는 FP16(422MB) 대비 56MB 낮은 peak를 기록하였다.

---

## 4.5 Insight

실험 1의 결과를 종합하면 다음과 같다.

1. **NPU offload는 inference 단계에서 압도적 효과(11.8×)를 보이나, E2E 기준으로는 CPU 전후처리에 의해 개선 폭이 제한된다(4.5×).** Coverage 100%임에도 E2E latency의 약 2/3가 CPU 구간에서 발생한다.

2. **INT8 QDQ는 inference latency를 개선하지 않았다.** QDQ 노드 삽입으로 인한 그래프 복잡도 증가가 양자화 이점을 상쇄하였다. E2E 개선은 inference가 아닌 전처리 구간에서 발생하였다.

3. **NPU는 전력 효율에서 강점이 크다.** CPU 대비 약 6.7배 낮은 전력으로 4.5배 빠른 처리를 달성하여, 성능 per watt 기준으로는 약 30배의 효율 차이를 보였다.

4. **Cold start는 graph compilation에 의해 지배된다.** Steady-state inference 전환은 빠르지만, 세션 생성에 1.3~2.9초가 소요되어 앱 초기 로딩에 영향을 줄 수 있다.

이러한 결과는 "추론 가속" 자체보다 **E2E pipeline 전체를 NPU offload 범위로 확장하는 것**이 실질적 latency 개선의 핵심 경로임을 시사한다. 이를 검증하기 위해 실험 2를 설계하였다.

---

# 5. Experiment 2 Results — E2E Pipeline Offload Impact

## 5.1 모델 Variant 구성

실험 1에서 CPU 전후처리(PreProc + PostProc ≈ 15.7ms)가 E2E latency의 하한선으로 작용함을 확인하였다. 이를 해소하기 위해, ONNX 모델 자체에 전처리/후처리/NMS 연산을 순차적으로 통합(bake-in)하여 NPU offload 범위를 확장하는 접근을 시도하였다.

| Variant | 설명 | Input Shape |
|---------|------|-------------|
| Baseline (YOLOv8n) | 원본 모델 (inference only) | [1, 3, 640, 640] |
| +Pre (YOLOv8n(Pre)) | Letterbox + Normalize 포함 | [1, 640, 640, 3] |
| +E2E (YOLOv8n(E2E)) | Pre + Transpose + Decode 포함 | [1, 640, 640, 3] |
| +E2E+NMS (YOLOv8n(E2E+NMS)) | Pre + Decode + NMS 포함 | [1, 640, 640, 3] |

+Pre 이상의 variant는 입력이 NHWC(uint8)로 변경되어, CPU-side CHW transpose가 불필요해진다.

---

## 5.2 Graph Partitioning & Coverage

**Table 7 — Graph Partitioning (Experiment 2)**

| Variant | Total Nodes | QNN Nodes | CPU Nodes | Coverage | Fallback Ops |
|---------|-------------|-----------|-----------|----------|--------------|
| Baseline | 241 | 241 | 0 | 100.0% | — |
| +Pre | 242 | 242 | 0 | 100.0% | — |
| +E2E | 248 | 248 | 0 | 100.0% | — |
| +E2E+NMS | 250 | 249 | 1 | 99.6% | NonMaxSuppression |

전처리와 후처리 연산은 모두 QNN EP에서 지원되어 100% coverage를 유지하였다. 유일한 예외는 E2E+NMS variant의 NonMaxSuppression op으로, 이는 QNN에서 미지원하여 CPU fallback이 발생하였다.

---

## 5.3 Latency Performance

**Table 8 — E2E Latency (Experiment 2)**

| Variant | E2E P50 (ms) | E2E P95 (ms) | Mean (ms) | MaxFPS |
|---------|-------------|-------------|-----------|--------|
| Baseline | 23.13 | 26.70 | 23.22 | 43.2 |
| +Pre | 21.23 | 23.95 | 21.40 | 47.1 |
| +E2E | 20.74 | 23.23 | 20.78 | 48.2 |
| +E2E+NMS | 20.52 | 22.99 | 20.62 | 48.7 |

**Table 9 — E2E Breakdown (P50)**

| Variant | PreProc (ms) | Inference (ms) | PostProc (ms) | Pre+Post (ms) | E2E (ms) |
|---------|-------------|----------------|--------------|---------------|----------|
| Baseline | 13.16 | 7.92 | 2.15 | 15.30 | 23.13 |
| +Pre | 10.62 | 8.13 | 2.63 | 13.25 | 21.23 |
| +E2E | 11.20 | 7.91 | 1.74 | 12.94 | 20.74 |
| +E2E+NMS | 11.01 | 9.55 | 0.02 | 11.03 | 20.52 |

---

### Observation

전후처리를 모델에 통합한 결과, E2E P50은 23.13ms(Baseline) → 20.52ms(E2E+NMS)로 약 **11.3% 개선**되었다. CPU 전후처리 합계(Pre+Post P50)는 15.30ms → 11.03ms로 **27.9% 감소**하였다.

그러나 개선 효과는 기대보다 제한적이었다. 그 원인을 breakdown에서 분석할 수 있다.

**전처리 offload (+Pre)의 효과**: PreProc이 13.16ms → 10.62ms로 약 2.5ms 감소하였다. Letterbox resize와 normalize 연산이 NPU로 이전되면서 CPU 부담이 줄었다. 그러나 NHWC→NCHW 전처리 일부와 이미지 decode는 여전히 CPU에서 수행되어 10ms 이상의 전처리 시간이 남아있다.

**후처리 offload (+E2E)의 효과**: PostProc이 2.15ms → 1.74ms로 소폭 감소하였다. Decode(box coordinate transform)가 모델에 포함되어 PostProc 부담이 줄었으나, 원래 PostProc 자체가 작아 절대적 이득이 제한적이었다.

**NMS offload (+E2E+NMS)의 효과**: PostProc이 1.74ms → 0.02ms로 거의 제거되었다. 그러나 inference latency가 7.91ms → 9.55ms로 약 **1.6ms 증가**하였다. NMS 연산이 NPU에서 실행되면서 inference 시간이 늘어났고, PostProc 절감(1.72ms)과 inference 증가(1.64ms)가 거의 상쇄되었다. 결과적으로 E2E 개선은 +E2E 대비 **0.22ms(1.1%)**에 그쳤다.

각 variant의 baseline 대비 E2E 개선율을 정리하면 다음과 같다.

**Table 10 — E2E Improvement Summary**

| Variant | E2E P50 (ms) | vs Baseline | 누적 개선 |
|---------|-------------|-------------|-----------|
| Baseline | 23.13 | — | — |
| +Pre | 21.23 | -1.90ms (-8.2%) | -8.2% |
| +E2E | 20.74 | -2.39ms (-10.3%) | -10.3% |
| +E2E+NMS | 20.52 | -2.61ms (-11.3%) | -11.3% |

전처리 bake-in(+Pre)이 가장 큰 단일 개선(-1.90ms)을 제공하였으며, 이후 단계의 한계 효용은 급격히 감소하였다.

---

## 5.4 Cold Start

**Table 11 — Cold Start (Experiment 2)**

| Variant | Total (ms) | Load (ms) | Session (ms) | 1st Inference (ms) |
|---------|-----------|-----------|-------------|-------------------|
| Baseline | 2,773 | 23 | 2,735 | 15.91 |
| +Pre | 1,658 | 19 | 1,623 | 16.03 |
| +E2E | 1,880 | 25 | 1,843 | 12.35 |
| +E2E+NMS | 1,992 | 21 | 1,955 | 16.73 |

### Observation

모델에 연산을 bake-in하면 cold start가 감소할 것으로 예상할 수 있으나, 결과는 복잡하였다. +Pre(1,658ms)가 Baseline(2,773ms)보다 크게 빠른 것은 사실이나, +E2E(1,880ms), +E2E+NMS(1,992ms)로 갈수록 다시 증가하는 추세를 보였다. 이는 bake-in된 연산이 QNN 그래프 컴파일 복잡도를 높이기 때문이다.

그럼에도 모든 variant가 FP16 Baseline(2,773ms)보다 빠른 cold start를 기록하였다. 이는 E2E variant의 ONNX 모델이 원본 대비 더 효율적인 그래프 구조를 가질 가능성을 시사한다.

---

## 5.5 Sustained Performance & System Resources

**Table 12 — Thermal Drift (Experiment 2)**

| Variant | First 30s P50 (ms) | Last 30s P50 (ms) | Drift % | Verdict |
|---------|-------------------|-------------------|---------|---------|
| Baseline | 23.06 | 23.48 | +1.8% | Stable |
| +Pre | 20.87 | 21.58 | +3.4% | Slight throttle |
| +E2E | 20.65 | 20.76 | +0.5% | Stable |
| +E2E+NMS | 19.93 | 20.35 | +2.1% | Stable |

**Table 13 — System Resources (Experiment 2)**

| Variant | Power (mW) | Thermal Slope (°C/min) | Memory Peak (MB) |
|---------|-----------|----------------------|-----------------|
| Baseline | 416.5 | -0.00 | 411 |
| +Pre | 452.1 | -0.02 | 391 |
| +E2E | 422.7 | -0.03 | 399 |
| +E2E+NMS | 455.1 | -0.00 | 410 |

### Observation

Sustained 성능은 모든 variant에서 안정적이었다(drift ±3.4% 이내). +Pre에서 관찰된 +3.4%의 slight throttle은 통계적으로 유의미하지 않을 수 있는 수준이다.

전력 소비는 variant 간 차이가 크지 않았다(416~455mW). 이는 모든 variant가 동일한 QNN_NPU EP + FP16 조건에서 실행되기 때문이며, pipeline 구성 변화가 전력에 미치는 영향은 미미함을 보여준다.

메모리 사용도 유사한 범위(391~411MB)를 보였으며, bake-in에 의한 추가 메모리 부담은 관찰되지 않았다.

---

## 5.6 Insight

실험 2의 결과는 다음과 같이 요약된다.

1. **전처리 bake-in이 가장 효과적인 단일 개선 수단이다.** CPU-side letterbox + normalize를 NPU로 이전하여 E2E를 약 8% 개선하였다.

2. **NMS offload는 투자 대비 효과가 제한적이다.** PostProc을 거의 제거하였으나, NPU-side NMS의 inference 증가가 이를 상쇄하여 추가 개선은 약 1%에 그쳤다.

3. **E2E pipeline의 "CPU floor"는 완전히 제거되지 않는다.** 전후처리를 최대한 offload하여도 이미지 decode, 메모리 복사, Android framework 오버헤드 등으로 인해 약 11ms의 하한이 남아있다.

4. **Sustained 안정성과 전력 효율은 pipeline 구성에 영향받지 않는다.** 동일 EP/precision 조건에서는 bake-in 여부와 무관하게 안정적인 sustained 성능을 유지한다.

---

# 6. Discussion

## 6.1 Coverage와 E2E Latency의 비선형 관계

실험 1과 2를 종합하면, NPU offload coverage와 E2E latency의 관계는 비선형적이다.

**Table 14 — Coverage vs E2E 전체 비교**

| Configuration | Coverage | E2E P50 (ms) | Inference P50 (ms) | CPU Floor (ms) | CPU Floor 비율 |
|---------------|----------|-------------|-------------------|---------------|---------------|
| CPU FP32 | N/A | 105.63 | 92.47 | 12.15 | 11.5% |
| NPU FP16 (Baseline) | 100.0% | 23.50 | 7.81 | 15.71 | 66.8% |
| NPU INT8 QDQ | 99.8% | 21.97 | 9.36 | 12.66 | 57.6% |
| NPU FP16 (+E2E+NMS) | 99.6% | 20.52 | 9.55 | 11.03 | 53.7% |

CPU-only baseline에서는 CPU floor(전후처리)가 E2E의 11.5%에 불과하지만, NPU offload 후에는 inference가 급격히 줄면서 CPU floor의 상대적 비중이 53~67%로 역전된다.

이는 모바일 inference에서 NPU 가속의 효과를 극대화하기 위해서는 **inference 단계의 최적화만으로는 불충분**하며, E2E pipeline 전체의 설계가 중요함을 보여준다.

## 6.2 Precision 선택에 대한 재고

본 실험에서 INT8 QDQ는 기대와 달리 inference latency를 개선하지 못하였다. ORT + QNN EP 환경에서 INT8의 이점이 제한되는 원인으로 다음을 고려할 수 있다.

- **QDQ 노드 오버헤드**: Static quantization에 의한 QuantizeLinear/DequantizeLinear 노드 삽입(241 → 958 nodes)이 NPU graph의 복잡도를 높임
- **HTP 내부 최적화**: Snapdragon HTP는 FP16에 대해서도 내부적으로 최적화된 실행 경로를 제공하여, INT8의 이론적 이점이 실제로 실현되지 않을 수 있음
- **Cold start trade-off**: INT8의 작은 모델 크기(3.4MB)가 세션 생성(1,257ms vs 2,831ms)에서 이점을 제공하지만, steady-state inference에서는 이점 없음

본 실험 결과만으로는 INT8의 일반적 열위를 단정할 수 없으나, ORT + QNN EP 환경에서 "INT8이 항상 FP16보다 빠르다"는 가정은 성립하지 않았다.

## 6.3 한계 및 향후 과제

본 실험의 한계와 향후 확장 가능성은 다음과 같다.

**단일 모델 한계**: YOLOv8n은 비교적 작은 모델이다. 대형 모델(YOLOv8-l/x, LLM 등)에서는 fallback 비율이 높아지거나, sustained thermal throttling이 두드러질 수 있다.

**Sequential pipeline 한계**: 현재 실험은 모두 sequential pipeline(Pre → Infer → Post)에서 수행되었다. Overlapped pipeline(Frame N+1 Pre / Frame N Infer / Frame N-1 Post 동시 실행)을 구현하면 steady-state throughput을 개선할 수 있을 것으로 예상된다.

**GPU 전처리 미검증**: CPU 전처리를 OpenGL ES 기반 GPU 전처리로 대체하면 CPU floor를 추가 감소시킬 가능성이 있다.

**Context Cache 미사용**: QNN Context Cache를 활성화하면 cold start를 크게 줄일 수 있으나, 본 실험에서는 매 실행마다 컴파일을 수행하여 worst-case cold start를 측정하였다.

---

# 7. Conclusion

본 실험은 Snapdragon 8 Gen 2 기반 단말에서 YOLOv8n 모델을 대상으로, NPU offload coverage, precision 정책, E2E pipeline 구성이 sustained KPI에 미치는 영향을 정량 분석하였다.

**Table 15 — 주요 결과 요약**

| 구분 | 핵심 결과 |
|------|-----------|
| NPU Offload 효과 | Inference 11.8× 가속, E2E 4.5× 가속 (CPU floor가 하한 작용) |
| Coverage 특성 | FP16 100%, INT8 QDQ 99.8% (실질적 전체 offload) |
| Precision 비교 | INT8 QDQ ≠ 더 빠른 inference. QDQ 노드 오버헤드가 양자화 이점 상쇄 |
| E2E Pipeline Offload | 전처리 bake-in으로 E2E 8.2% 개선, NMS까지 포함 시 11.3% |
| 전력 효율 | NPU는 CPU 대비 6.7× 낮은 전력, 성능/watt 약 30× |
| Sustained 안정성 | 5분간 모든 configuration에서 drift ±4% 이내 |
| Cold Start | NPU session 생성(HTP 컴파일)이 1.3~2.9초 소요 |

실험을 통해 확인한 가장 중요한 사실은, 모바일 NPU inference의 실질적 병목은 **inference 연산 자체가 아닌 E2E pipeline 구조**에 있다는 것이다. NPU coverage가 100%에 도달하더라도 CPU에서 수행되는 전후처리가 E2E latency의 절반 이상을 차지하며, 이 구조적 제약을 해소하기 위해서는 pipeline 전체의 offload 전략이 필요하다.

그러나 전후처리를 최대한 NPU로 이전하더라도 약 11ms의 CPU floor가 남아있었으며, 이를 더 줄이기 위해서는 overlapped pipeline scheduling이나 GPU 전처리와 같은 heterogeneous scheduling 접근이 필요할 것으로 보인다.
