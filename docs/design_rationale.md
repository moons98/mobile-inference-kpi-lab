# Design Rationale

## 1. QNN (Qualcomm Neural Network)

QNN은 Snapdragon SoC의 AI 가속기(HTP/NPU, Adreno GPU, CPU)를 제어하는 저수준 inference runtime SDK이다.

**QNN의 역할:**

- ONNX 모델을 QNN IR로 변환
- Backend(HTP/GPU/CPU) 선택
- Graph compile 및 lowering
- Op mapping 및 fusion
- 실제 NPU dispatch 수행

> QNN은 Snapdragon 가속기에 연산을 실제로 실행시키는 런타임 계층이다.

## 2. ORT + QNN EP 구조

### ONNX Runtime (ORT)

ORT는 framework-independent inference engine이며,
Execution Provider(EP) 구조를 통해 다양한 backend를 연결할 수 있다.

### QNN EP

QNN EP는 ORT와 QNN runtime을 연결하는 어댑터 계층이다.

```
ONNX Runtime
   ├─ CPU EP
   └─ QNN EP → QNN → HTP (NPU)
```

실제 NPU dispatch는 QNN이 수행하며,
ORT는 graph orchestration 및 partition을 담당한다.

## 3. 왜 Pure QNN이 아니라 ORT + QNN EP를 사용했는가

본 프로젝트의 목적은:

> Snapdragon NPU 실행 시 발생하는 fallback을 분석하고,
> KPI(지연, 처리량, sustained 성능)에 미치는 영향을 정량화하는 것

이를 위해 ORT + QNN EP 구조를 채택하였다.

### 3.1 Fallback 가시성 확보

ORT는 graph partition을 수행하며, QNN이 지원하지 않는 op를 CPU EP로 자동 분리한다.
Partition log를 통해 fallback node를 명확히 식별할 수 있다.

Pure QNN 사용 시:

- Unsupported op는 compile 단계에서 에러 발생
- 자동 fallback 구조가 없음
- 분석용 hybrid 실행이 어려움

### 3.2 Hybrid 실행 환경 구성

ORT는 일부 op는 QNN(HTP), 일부 op는 CPU 형태의 혼합 실행을 자연스럽게 지원한다.
이는 실제 모바일 환경에서 발생하는 fallback 시나리오를 현실적으로 재현하는 데 필수적이다.

### 3.3 실험 반복성과 개발 효율성

- Graph 수정 후 즉시 재실행 가능
- CPU baseline 비교 용이
- Fallback 전후 KPI 차이 정량화 가능

QNN 단독 사용 시에는 offline compile 및 graph 재생성이 필요하여
분석 파이프라인 구축에 비효율적이다.

### 요약

ONNX Runtime은 graph partitioning을 통해 NPU에서 실행 가능한 subgraph만 QNN EP에 위임한다.
이를 통해 QNN 단독 사용 시 발생하는 compile 실패 문제를 피하고,
미지원 연산은 CPU에서 fallback 실행하는 hybrid 구조를 구성할 수 있다.

이러한 구조는 Snapdragon NPU 활용 가능 영역과 fallback 영역을 명확히 구분하고,
각 영역이 KPI에 미치는 영향을 정량적으로 분석하는 데 적합하다.
