# Experiment Design

## Goal

> Snapdragon 기반 Android 단말에서
> NPU offload coverage, precision 정책, pipeline scheduling 구조가
> Sustained E2E KPI에 미치는 영향을 정량 분석한다.

**핵심 질문:**

1. NPU offload coverage가 높아도 왜 E2E latency는 비례 감소하지 않는가?
2. Precision(INT8)이 coverage 및 inference latency에 실제로 어떤 영향을 주는가?
3. Sequential vs Overlapped pipeline 설계가 sustained throughput을 얼마나 개선하는가?

## Target Model

**YOLOv8n** (Object Detection, ONNX)

- 일부 shape/activation op 존재 → partition 관찰 가능
- 가벼워 반복 실험 및 sustained 측정 용이
- 모델 1개에 집중하여 변인 통제

---

## 실험 구성

### 실험 1: Execution Path & Precision Impact

**목적:**
- Offload coverage와 E2E KPI의 관계 분석
- Precision 정책이 partition 및 latency에 미치는 영향 분석

| Case | EP | Precision | 전처리 | 추론 | 후처리 |
|------|-----|-----------|--------|------|--------|
| **A** | CPU | FP32 | CPU | CPU | CPU |
| **B** | QNN_NPU | FP16 (runtime) | CPU | NPU | CPU |
| **C** | QNN_NPU | INT8 (QDQ) | CPU | NPU | CPU |

**반드시 측정:**

📊 Coverage & Partition

| 항목 | 단위 | 설명 |
|------|------|------|
| Offload Coverage | % | NPU 노드 수 / 전체 노드 수 |
| Fallback op 목록 | - | CPU로 fallback된 연산 |
| Partition segment 수 | - | QNN EP가 생성한 subgraph 수 |
| FLOPs-weighted coverage | % | 보조 지표 (가능한 경우) |

⏱ Latency Metrics

| 항목 | 단위 | 설명 |
|------|------|------|
| Inference Latency | ms | P50 / P95 |
| E2E Breakdown | ms | Pre / Infer / Post 각각 |
| Cold Start | ms | 모델 로드 + 세션 생성 (Context Cache on/off) |
| Session Creation | ms | 세션 생성 시간 |

🔥 Sustained Metrics

| 항목 | 단위 | 설명 |
|------|------|------|
| Sustained Drift | % | First 30s vs Last 30s P50 변화 |
| Time-series Latency | ms | 시간에 따른 latency 추이 |
| Thermal Trend | °C | 보조 지표 |

**반드시 보여줄 것:**
- Coverage 90% 이상이어도 E2E 감소율이 제한되는 이유 → CPU pre/post가 하한으로 작용
- INT8이 inference latency는 줄여도 E2E가 비례 감소하지 않는 경우
- Precision 변경이 coverage에 영향을 주는지 여부
- Sustained 구간에서 FP16 vs INT8 drift 차이

### 실험 2: Pipeline Scheduling Impact (TODO)

**목적:**
- Sequential vs Overlapped pipeline 구조 차이 분석
- Latency hiding 효과 검증
- Sustained throughput 개선 확인

> 실험 1에서 가장 효율적인 precision 선택 (예: INT8)

| Case | Scheduling |
|------|------------|
| **A** | Sequential (pre → infer → post) |
| **B** | Overlapped (Frame N+1 pre / N infer / N-1 post 동시 실행) |

> GPU 전처리는 옵션. 핵심은 "병렬화" 구조.

**반드시 측정:**

| 항목 | 단위 | 설명 |
|------|------|------|
| Steady-state FPS | fps | 안정 상태 처리량 |
| E2E Latency | ms | 프레임당 총 지연시간 |
| CPU Utilization | % | CPU 사용률 |
| Sustained Drift | % | First 30s vs Last 30s P50 변화 |
| Latency Convergence | ms | E2E ≈ max(pre+post, infer) 수렴 여부 |

**반드시 보여줄 것:**
- Inference latency는 동일해도 throughput은 달라질 수 있음
- Overlap 시 steady-state FPS 증가
- Thermal 구간에서 drift 감소 여부
- Heterogeneous scheduling의 실질적 효과

> 실험 2는 overlapped pipeline 인프라 구현 후 진행. 현재 TODO.

### Optional: GPU Pre/Post Offload

시간이 허용될 경우 수행.

| Case | 전처리 | 추론 | 후처리 |
|------|--------|------|--------|
| **CPU-only** | CPU | NPU | CPU |
| **GPU-pre** | GPU | NPU | CPU |

**목적:**
- CPU 병목 감소 확인
- Pre latency 감소량 정량화

> GPU는 OpenGL ES 기반 최소 구현 권장.

---

## 최종 산출물

| 산출물 | 설명 |
|--------|------|
| **Coverage table** | Case별 coverage %, fallback ops, partition count |
| **Latency breakdown 표** | Case별 Pre / Infer / Post / E2E / P50 / P95 |
| **Sustained drift 그래프** | 시간에 따른 latency 변화 + precision별 비교 |
| **Precision 비교표** | Precision별 coverage / infer ms / E2E ms / drift % |
| **Pipeline 비교표** (TODO) | Scheduling별 FPS / E2E ms / drift % |
