# Execution Profiles

이 문서는 Mobile Inference KPI Lab의 실행 흐름과 각 컴포넌트의 내부 동작을 상세히 설명합니다.

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                        MainActivity                              │
│                    (UI, 사용자 설정 입력)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ BenchmarkConfig
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BenchmarkRunner                             │
│              (벤치마크 오케스트레이션, 상태 관리)                    │
│                                                                  │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │  Inference Loop  │          │ System Metrics   │             │
│  │  (추론 실행)      │   ║║     │ (KPI 수집)        │             │
│  │  intervalMs 간격  │ 병렬    │  1초 간격          │             │
│  └────────┬─────────┘          └────────┬─────────┘             │
└───────────┼─────────────────────────────┼───────────────────────┘
            │                             │
            ▼                             ▼
┌───────────────────────┐    ┌───────────────────────┐
│       OrtRunner       │    │     KpiCollector      │
│  (ONNX Runtime 래퍼)   │    │  (Thermal/Power/Mem)  │
└───────────┬───────────┘    └───────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────┐
│              ONNX Runtime Session                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ QNN EP  │  │ QNN EP  │  │  CPU    │           │
│  │  (NPU)  │  │  (GPU)  │  │ Default │           │
│  └─────────┘  └─────────┘  └─────────┘           │
└───────────────────────────────────────────────────┘
```

---

## 실행 단계 (Execution Phases)

### 1. 초기화 단계 (INITIALIZING)

**파일**: [OrtRunner.kt](../android/app/src/main/java/com/example/kpilab/OrtRunner.kt)

```
BenchmarkRunner.start()
    └─> 이전 runner release (있으면)
    └─> OrtRunner.initialize(modelType, executionProvider, useNpuFp16, useContextCache)
            ├─> OrtEnvironment.getEnvironment()      // ONNX Runtime 환경 생성
            ├─> configureExecutionProvider()         // QNN EP 또는 CPU 설정
            │       ├─> FP16 옵션 적용 (useNpuFp16)
            │       └─> Context Cache 설정 (useContextCache)
            ├─> loadModelFromAssets()                // 모델 파일 로드
            ├─> ortEnv.createSession()               // 세션 생성 + HTP 컴파일
            ├─> extractIOInfo()                      // 입출력 정보 추출
            └─> allocateInputData()                  // 입력 버퍼 할당
```

**QNN Execution Provider 설정**:

```kotlin
// NPU (HTP) 설정
val qnnOptions = mutableMapOf<String, String>()
qnnOptions["backend_path"] = "libQnnHtp.so"
qnnOptions["htp_performance_mode"] = "burst"
qnnOptions["htp_graph_finalization_optimization_mode"] = "3"

// FP16 정밀도 (UI 옵션)
qnnOptions["enable_htp_fp16_precision"] = if (useNpuFp16) "1" else "0"

// Context Cache (UI 옵션)
if (useContextCache) {
    val cachePath = "${cacheDir}/qnn_${model}_${precision}.bin"
    qnnOptions["qnn_context_cache_enable"] = "1"
    qnnOptions["qnn_context_cache_path"] = cachePath
}

options.addQnn(qnnOptions)

// GPU 설정
qnnOptions["backend_path"] = "libQnnGpu.so"
```

### 2. Warm-up 단계 (WARMING_UP)

**조건**: `warmUpEnabled = true`

```kotlin
fun runWarmUp(iterations: Int = 10) {
    for (i in 0 until iterations) {
        runInferenceInternal()  // 10회 사전 실행
    }
}
```

**효과**:
- ONNX Runtime 그래프 최적화 완료
- QNN 백엔드 초기화 안정화
- 초기 지연시간 스파이크 제거
- 메모리 할당 패턴 안정화

### 3. 실행 단계 (RUNNING)

**병렬 실행 흐름**:

```
┌────────────────────────────────────────┐
│         Main Inference Loop            │
│  ┌──────────────────────────────────┐  │
│  │  while (elapsed < duration)      │  │
│  │       ├─> runInference()         │  │
│  │       │     └─> OnnxTensor 생성   │  │
│  │       │     └─> session.run()    │  │
│  │       │     └─> latency 측정     │  │
│  │       ├─> update progress        │  │
│  │       └─> delay(intervalMs)      │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
              ║ (병렬 코루틴)
┌────────────────────────────────────────┐
│      System Metrics Collection         │
│  ┌──────────────────────────────────┐  │
│  │  while (elapsed < duration)      │  │
│  │       ├─> kpiCollector.collectAll()  │
│  │       │     ├─ readThermal()     │  │
│  │       │     ├─ readPower()       │  │
│  │       │     └─ readMemory()      │  │
│  │       ├─> runner.logSystemMetrics()  │
│  │       └─> delay(1000ms)          │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

**타이밍 설정**:

| 항목 | 주기 | 설명 |
|------|------|------|
| Inference | `1000/frequencyHz` ms | 1Hz=1000ms, 5Hz=200ms, 10Hz=100ms |
| Thermal/Power | 1000ms | 시스템 메트릭 |
| Memory | 5000ms | VmRSS (더 느린 주기로 측정) |

### 4. 종료 단계 (STOPPING → IDLE)

```
BenchmarkRunner.stop()
    ├─> state = STOPPING
    ├─> benchmarkJob.cancel()
    ├─> systemMetricsJob.cancel()
    └─> cleanup()
            └─> state = IDLE
```

---

## 구성 옵션 (Configuration Options)

### BenchmarkConfig

**파일**: [BenchmarkConfig.kt](../android/app/src/main/java/com/example/kpilab/BenchmarkConfig.kt)

```kotlin
data class BenchmarkConfig(
    val modelType: OnnxModelType = OnnxModelType.MOBILENET_V2,
    val executionProvider: ExecutionProvider = ExecutionProvider.QNN_NPU,
    val frequencyHz: Int = 5,
    val warmUpEnabled: Boolean = false,
    val durationMinutes: Int = 5,
    // NPU precision: true = FP16, false = FP32 (FP32 모델에만 적용)
    val useNpuFp16: Boolean = true,
    // QNN context cache: HTP 컴파일 그래프 캐싱
    val useContextCache: Boolean = false
) {
    val intervalMs: Long = (1000.0 / frequencyHz).toLong()
    val durationMs: Long = durationMinutes * 60 * 1000L
}
```

**옵션 설명**:
- `useNpuFp16`: FP32 모델을 NPU에서 FP16으로 변환하여 실행 (더 빠름, 약간의 정밀도 손실)
- `useContextCache`: 첫 실행 시 HTP 컴파일 결과를 캐싱, 이후 빠른 로드

### OnnxModelType (지원 모델)

**파일**: [OrtRunner.kt](../android/app/src/main/java/com/example/kpilab/OrtRunner.kt)

| 모델 | 파일명 | 입력 크기 | 포맷 | 양자화 |
|------|--------|-----------|------|--------|
| MobileNetV2 | `mobilenetv2_torchvision.onnx` | 1x3x224x224 | NCHW | FP32 |
| MobileNetV2 (INT8 Dynamic) | `mobilenetv2_int8_dynamic.onnx` | 1x3x224x224 | NCHW | INT8 Dynamic |
| MobileNetV2 (INT8 QDQ) | `mobilenetv2_int8_qdq.onnx` | 1x3x224x224 | NCHW | INT8 QDQ |
| YOLOv8n | `yolov8n_ultralytics.onnx` | 1x3x640x640 | NCHW | FP32 |
| YOLOv8n (INT8 Dynamic) | `yolov8n_int8_dynamic.onnx` | 1x3x640x640 | NCHW | INT8 Dynamic |
| YOLOv8n (INT8 QDQ) | `yolov8n_int8_qdq.onnx` | 1x3x640x640 | NCHW | INT8 QDQ |

> **Note**: 모든 모델은 **FLOAT 입력**을 받습니다. INT8 모델의 양자화/역양자화는 모델 내부에서 처리됩니다.

### ExecutionProvider (실행 경로)

| Provider | Backend Library | 설명 |
|----------|-----------------|------|
| QNN_NPU | `libQnnHtp.so` | Hexagon Tensor Processor (NPU) |
| QNN_GPU | `libQnnGpu.so` | Qualcomm Adreno GPU |
| CPU | (기본값) | ONNX Runtime CPU EP |

**QNN NPU 옵션 상세**:

| 옵션 | 값 | 설명 |
|------|-----|------|
| `backend_path` | `libQnnHtp.so` | HTP(NPU) 백엔드 라이브러리 |
| `htp_performance_mode` | `burst` | 최고 성능 모드 |
| `htp_graph_finalization_optimization_mode` | `3` | 최대 그래프 최적화 |
| `enable_htp_fp16_precision` | `0` / `1` | FP16 정밀도 사용 (UI: NPU FP16) |
| `qnn_context_cache_enable` | `0` / `1` | HTP 그래프 캐싱 (UI: Context Cache) |
| `qnn_context_cache_path` | `path/to/cache.bin` | 캐시 파일 경로 |

**그래프 최적화 타이밍**:
1. **OrtSession 생성 시**: ONNX Runtime 그래프 최적화
2. **QNN EP 변환 시**: QNN 그래프로 변환
3. **첫 추론 시**: HTP 컴파일 (Context Cache 저장 시점)

Context Cache가 활성화되면 3단계에서 컴파일된 그래프가 파일로 저장되어, 다음 실행 시 1-3단계를 건너뛰고 캐시에서 바로 로드합니다.

---

## KPI 수집 (Metrics Collection)

### KpiCollector

**파일**: [KpiCollector.kt](../android/app/src/main/java/com/example/kpilab/KpiCollector.kt)

#### Thermal (온도)

**탐색 우선순위**:

```
1. /sys/class/thermal/thermal_zone{0-5}/temp
2. /sys/devices/virtual/thermal/thermal_zone{0-1}/temp
3. /sys/class/hwmon/hwmon{0-1}/temp1_input
4. /sys/class/sec/temperature/value (Samsung)
5. /sys/class/thermal/thermal_zone9/temp (MTK)
6. Battery Temperature (최종 fallback)
```

**단위 변환**:
```kotlin
// 대부분의 기기는 millidegrees (예: 38500 → 38.5°C)
val tempC = if (tempRaw > 1000) tempRaw / 1000f else tempRaw.toFloat()
```

#### Power (전력)

```kotlin
// BatteryManager API 사용
val currentMicroAmps = batteryManager.getIntProperty(
    BatteryManager.BATTERY_PROPERTY_CURRENT_NOW
)
val voltageMv = batteryStatus.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1)

// Power (mW) = |Current (μA)| × Voltage (mV) / 10^6
val powerMw = (abs(currentMicroAmps) * voltageMv) / 1_000_000f
```

**주의사항**:
- 기기별 정확도 차이 있음
- 일부 기기는 충전 중 음수 전류값 보고

#### Memory (메모리)

```kotlin
// /proc/self/status 파싱
// VmRSS: 245678 kB → 245 MB
```

- 단위: MB
- VmRSS = Resident Set Size (실제 물리 메모리 사용량)
- 5초마다 측정 (더 느린 주기)

---

## 데이터 기록 (Data Logging)

### KpiRecord 구조

**파일**: [OrtRunner.kt](../android/app/src/main/java/com/example/kpilab/OrtRunner.kt)

```kotlin
data class KpiRecord(
    val timestamp: Long,       // Unix timestamp (ms)
    val eventType: Int,        // 0: INFERENCE, 1: SYSTEM
    val latencyMs: Float,      // 추론 지연시간 (INFERENCE만)
    val thermalC: Float,       // 온도 (SYSTEM만)
    val powerMw: Float,        // 전력 (SYSTEM만)
    val memoryMb: Int,         // 메모리 (SYSTEM, 5초마다, 미측정=-1)
    val isForeground: Boolean  // 포그라운드 여부
)
```

### CSV Export 형식

```csv
# device_manufacturer,Samsung
# device_model,SM-S928N
# soc_manufacturer,QTI
# soc_model,SM8650
# android_version,14
# api_level,34
# runtime,ONNX Runtime
# execution_provider,QNN_NPU
# model,MobileNetV2 (ONNX)
# session_id,ort_mnv2_npu_5hz_w_1706789012345
#
timestamp,event_type,latency_ms,thermal_c,power_mw,memory_mb,is_foreground
1706789012345,INFERENCE,12.34,,,,true
1706789013000,SYSTEM,,38.2,2150,245,true
1706789014000,SYSTEM,,38.5,2200,,true
```

**필드 설명**:
- `event_type`: INFERENCE(추론) 또는 SYSTEM(시스템 메트릭)
- `latency_ms`: INFERENCE 이벤트에서만 값 존재
- `thermal_c`, `power_mw`: SYSTEM 이벤트에서만 값 존재
- `memory_mb`: SYSTEM 이벤트에서 5초마다 값 존재, 그 외 비어있음

---

## 상태 전이 (State Transitions)

```
IDLE ──[start()]──> INITIALIZING
                         │
                    [initialize()]
                         │
                         ▼
              ┌──[warmUp=false]──┐
              │                  │
              ▼                  │
         WARMING_UP              │
              │                  │
         [runWarmUp()]           │
              │                  │
              └────────┬─────────┘
                       ▼
                    RUNNING
                       │
          ┌──[stop()]──┴──[duration 완료]──┐
          │                                │
          ▼                                ▼
       STOPPING ─────[cleanup()]────> IDLE
```

### BenchmarkProgress

```kotlin
data class BenchmarkProgress(
    val state: BenchmarkState,      // 현재 상태
    val elapsedMs: Long,            // 경과 시간
    val totalMs: Long,              // 전체 시간
    val inferenceCount: Int,        // 추론 횟수
    val lastLatencyMs: Float,       // 마지막 지연시간
    val lastThermalC: Float,        // 마지막 온도
    val lastPowerMw: Float,         // 마지막 전력
    val lastMemoryMb: Int           // 마지막 메모리
) {
    val progressPercent: Int        // 진행률 (%)
    val throughput: Float           // 초당 추론 횟수 (inf/s)
}
```

---

## 실험 시나리오 예시

### 시나리오 1: NPU 성능 한계 테스트

```kotlin
BenchmarkConfig(
    modelType = OnnxModelType.MOBILENET_V2,
    executionProvider = ExecutionProvider.QNN_NPU,
    frequencyHz = 10,
    warmUpEnabled = false,
    durationMinutes = 5
)
```

**목적**: NPU의 10Hz 지속 실행 시 thermal throttling 확인

**예상 관찰점**:
- 초기 latency vs 5분 후 latency 비교
- thermal 상승 곡선
- throttling 시점 파악

### 시나리오 2: Warm-up 효과 측정

```kotlin
// A: Warm-up OFF
BenchmarkConfig(
    warmUpEnabled = false,
    executionProvider = ExecutionProvider.QNN_NPU,
    frequencyHz = 5,
    durationMinutes = 5
)

// B: Warm-up ON
BenchmarkConfig(
    warmUpEnabled = true,  // 10회 사전 실행
    executionProvider = ExecutionProvider.QNN_NPU,
    frequencyHz = 5,
    durationMinutes = 5
)
```

**목적**: 초기 latency spike 제거 효과 비교

**비교 지표**:
- 처음 10개 inference의 latency 평균
- latency 표준편차

### 시나리오 3: Execution Provider 비교

```kotlin
// NPU
BenchmarkConfig(
    executionProvider = ExecutionProvider.QNN_NPU,
    modelType = OnnxModelType.MOBILENET_V2,
    frequencyHz = 5,
    durationMinutes = 5
)

// GPU
BenchmarkConfig(
    executionProvider = ExecutionProvider.QNN_GPU,
    ...
)

// CPU
BenchmarkConfig(
    executionProvider = ExecutionProvider.CPU,
    ...
)
```

**목적**: 동일 조건에서 실행 경로별 KPI 차이 분석

**비교 지표**:
- Latency: mean, P50, P95
- Thermal: 최대값, 상승률 (°C/min)
- Power: 평균 소비량

### 시나리오 4: 양자화 효과 측정

```kotlin
// FP32
BenchmarkConfig(
    modelType = OnnxModelType.MOBILENET_V2,
    executionProvider = ExecutionProvider.QNN_NPU,
    useNpuFp16 = true,
    ...
)

// INT8 Dynamic
BenchmarkConfig(
    modelType = OnnxModelType.MOBILENET_V2_INT8_DYNAMIC,
    executionProvider = ExecutionProvider.QNN_NPU,
    ...
)

// INT8 QDQ
BenchmarkConfig(
    modelType = OnnxModelType.MOBILENET_V2_INT8_QDQ,
    executionProvider = ExecutionProvider.QNN_NPU,
    ...
)
```

**목적**: INT8 양자화의 성능/전력 개선 효과 측정

### 시나리오 5: FP16 vs FP32 비교

```kotlin
// FP16 (빠름, 약간의 정밀도 손실)
BenchmarkConfig(
    modelType = OnnxModelType.MOBILENET_V2,
    executionProvider = ExecutionProvider.QNN_NPU,
    useNpuFp16 = true,
    frequencyHz = 10,
    durationMinutes = 5
)

// FP32 (정확, 느림)
BenchmarkConfig(
    modelType = OnnxModelType.MOBILENET_V2,
    executionProvider = ExecutionProvider.QNN_NPU,
    useNpuFp16 = false,
    frequencyHz = 10,
    durationMinutes = 5
)
```

**목적**: NPU FP16 정밀도의 성능 향상 측정

### 시나리오 6: Context Cache 효과 측정

```kotlin
// Cache OFF (매번 컴파일)
BenchmarkConfig(
    modelType = OnnxModelType.MOBILENET_V2,
    executionProvider = ExecutionProvider.QNN_NPU,
    useContextCache = false,
    ...
)

// Cache ON (첫 실행 후 캐시 사용)
BenchmarkConfig(
    modelType = OnnxModelType.MOBILENET_V2,
    executionProvider = ExecutionProvider.QNN_NPU,
    useContextCache = true,
    ...
)
```

**목적**: QNN Context Cache의 초기화 시간 단축 효과 측정

**측정 방법**:
1. 앱 종료 후 재시작
2. Cache OFF로 벤치마크 시작 → 초기화 시간 기록
3. 앱 종료 후 재시작
4. Cache ON으로 벤치마크 시작 → 초기화 시간 비교

---

## 디버깅 가이드

### QNN EP 초기화 실패 시

Logcat 필터: `OrtRunner`

```
=== OrtRunner Initialization ===
Model: MobileNetV2 (ONNX)
Requested EP: QNN NPU (HTP)
=== Configuring QNN Execution Provider (NPU) ===
  backend_path = libQnnHtp.so
  htp_performance_mode = burst
  ...
QNN EP (NPU) configured
```

**실패 시 확인 사항**:
1. 모델 파일이 assets에 있는지
2. ONNX 모델이 QNN 지원 op만 사용하는지 (`analyze_ops.py`)
3. 기기가 Snapdragon인지 (QNN은 Qualcomm 전용)

### Thermal 읽기 실패 시

```
=== Searching for readable thermal path ===
  /sys/class/thermal/thermal_zone0/temp: exists=true, canRead=false
  ...
=== No readable thermal path found ===
Battery temperature fallback: 32.5 °C
```

**해결책**: Battery temperature fallback 사용 (자동)

### Memory 측정값이 비어있을 때

Memory는 5초마다만 측정됩니다. CSV에서 `memory_mb`가 비어있는 행은 정상입니다.

```csv
1706789013000,SYSTEM,,38.2,2150,245,true   # 측정됨
1706789014000,SYSTEM,,38.5,2200,,true      # 미측정 (정상)
1706789015000,SYSTEM,,38.7,2180,,true      # 미측정 (정상)
```
