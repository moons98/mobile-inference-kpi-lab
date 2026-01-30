# Mobile Inference KPI Lab

Android 단말(Snapdragon)에서 **ONNX Runtime + QNN Execution Provider** 기반 on-device inference를 실행하고, 실행 경로 및 정책 변화가 Latency, Power, Thermal KPI에 미치는 영향을 분석하는 프로젝트.

## 프로젝트 목적

1. **실행 경로별 KPI 비교**: NPU (HTP), GPU, CPU
2. **실행 정책 영향 분석**: Frequency (1/5/10Hz), Warm-up 유무
3. **NPU 디버깅**: QNN EP의 상세 로그를 통한 지원/미지원 Op 파악
4. **개선안 도출**: 정책/그래프 조정으로 KPI 개선

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         MainActivity                             │
│  (Model, Execution Provider, Frequency, Duration 설정)           │
└─────────────────────────────┬───────────────────────────────────┘
                              │ BenchmarkConfig
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BenchmarkRunner                            │
│  - 벤치마크 생명주기 관리 (IDLE→INIT→WARMUP→RUNNING→STOP)         │
│  - 2개의 병렬 코루틴: 추론 루프 + 시스템 메트릭 수집                │
└─────────┬───────────────────────────────────────┬───────────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│     OrtRunner       │                 │    KpiCollector     │
│ (ONNX Runtime)      │                 │   (시스템 메트릭)    │
│ - QNN EP 설정        │                 │ - Thermal (°C)      │
│ - 모델 로드/추론      │                 │ - Power (mW)        │
│ - Latency 측정       │                 │ - Memory (MB)       │
└─────────────────────┘                 └─────────────────────┘
          │                                       │
          └───────────────┬───────────────────────┘
                          ▼
                  ┌───────────────┐
                  │  CSV Export   │
                  └───────────────┘
```

## 프로젝트 구조

```
mobile-inference-kpi-lab/
├── android/                    # Android 앱
│   └── app/
│       └── src/main/
│           ├── java/.../       # Kotlin 소스
│           │   ├── MainActivity.kt
│           │   ├── BenchmarkRunner.kt
│           │   ├── BenchmarkConfig.kt
│           │   ├── OrtRunner.kt      # ONNX Runtime 래퍼
│           │   └── KpiCollector.kt   # 시스템 메트릭
│           ├── assets/         # ONNX 모델 파일
│           └── res/            # UI 리소스
├── scripts/                    # 모델 관련 스크립트
│   ├── export_to_onnx.py       # ONNX 모델 export (torchvision/ultralytics)
│   ├── analyze_ops.py          # Op 분석 (QNN EP 호환성)
│   ├── graph_transform.py      # 그래프 변환
│   └── setup_calibration_data.py  # Calibration 데이터 다운로드
├── analysis/                   # 분석 도구
│   ├── scripts/
│   │   ├── parse_logs.py
│   │   └── plot_kpi.py
│   └── notebooks/
│       ├── 01_baseline.ipynb
│       └── 02_policy_comparison.ipynb
└── docs/
```

## 필수 환경

- Android Studio Hedgehog (2023.1.1) 이상
- JDK 17
- Python 3.10+ (분석 스크립트용)

> **Note**: QNN SDK 별도 설치 불필요. ONNX Runtime Android QNN AAR에 포함됨.

## 설정 방법

### 1. 모델 준비

**방법 1: torchvision/ultralytics에서 export**
```bash
# FP32 모델
python scripts/export_to_onnx.py --export-mobilenetv2

# INT8 양자화 모델 (QNN EP 지원)
python scripts/setup_calibration_data.py --download-imagenet  # calibration data
python scripts/export_to_onnx.py --export-mobilenetv2-quantized --quant-method static
```

**방법 2: 직접 배치**

ONNX 모델을 `android/app/src/main/assets/`에 배치:
```
assets/
├── mobilenetv2.onnx              # FP32 MobileNetV2
├── mobilenetv2_int8_dynamic.onnx # INT8 Dynamic 양자화
├── mobilenetv2_int8_qdq.onnx     # INT8 QDQ 양자화
├── yolov8n.onnx                  # FP32 YOLOv8n
├── yolov8n_int8_dynamic.onnx     # INT8 Dynamic 양자화
└── yolov8n_int8_qdq.onnx         # INT8 QDQ 양자화
```

### 2. Android 앱 빌드

```bash
cd android
./gradlew assembleDebug
```

### 3. 앱 설치 및 실행

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 앱 사용법

### 실험 설정

| 옵션 | 선택지 | 설명 |
|------|--------|------|
| Model | 6종 (아래 표 참조) | 추론 모델 |
| Execution Provider | NPU / GPU / CPU | 추론 실행 경로 (아래 참조) |
| Frequency | 1Hz / 5Hz / 10Hz | 추론 빈도 |
| Warm-up | On / Off | 워밍업 실행 여부 (10회 추론) |
| NPU FP16 | On / Off | FP32 모델을 FP16으로 변환 (NPU에서 더 빠름) |
| Context Cache | On / Off | QNN 컴파일 그래프 캐싱 (재시작 시 빠른 로드) |
| Duration | 5min / 10min | 실험 시간 |

### 지원 모델

| 모델 | 파일명 | 입력 크기 | 양자화 |
|------|--------|-----------|--------|
| MobileNetV2 | `mobilenetv2.onnx` | 1x3x224x224 | FP32 |
| MobileNetV2 (INT8 Dynamic) | `mobilenetv2_int8_dynamic.onnx` | 1x3x224x224 | INT8 Dynamic |
| MobileNetV2 (INT8 QDQ) | `mobilenetv2_int8_qdq.onnx` | 1x3x224x224 | INT8 QDQ |
| YOLOv8n | `yolov8n.onnx` | 1x3x640x640 | FP32 |
| YOLOv8n (INT8 Dynamic) | `yolov8n_int8_dynamic.onnx` | 1x3x640x640 | INT8 Dynamic |
| YOLOv8n (INT8 QDQ) | `yolov8n_int8_qdq.onnx` | 1x3x640x640 | INT8 QDQ |

> **Note**: INT8 모델도 FLOAT 입력을 받습니다. 양자화/역양자화는 모델 내부에서 처리됩니다.

### Execution Provider 동작 방식

ONNX Runtime은 세션 생성 시 그래프를 분할(partitioning)하여 각 Op을 실행할 EP를 결정합니다:

| 선택 | 동작 |
|------|------|
| **NPU** | QNN HTP 백엔드 사용. 미지원 Op은 자동으로 CPU fallback |
| **GPU** | QNN GPU 백엔드 사용. 미지원 Op은 자동으로 CPU fallback |
| **CPU** | CPU만 사용 (fallback 없음) |

- Fallback은 **세션 생성 시 결정**되며, 런타임에 변경되지 않음
- CSV export의 `execution_provider` 필드에 실제 사용된 EP가 기록됨
- Logcat에서 `OrtRunner` 태그로 어떤 Op이 어디서 실행되는지 확인 가능

### 측정 KPI

#### Raw Metrics (앱에서 수집)

| KPI | 단위 | 수집 주기 | 설명 |
|-----|------|----------|------|
| **Latency** | ms | 매 추론 | 단일 추론 지연시간 |
| **Thermal** | °C | 1초 | SoC 온도 |
| **Power** | mW | 1초 | 전력 소비 |
| **Memory** | MB | 5초 | VmRSS 메모리 |

#### Cold Start Metrics (CSV 헤더에 기록)

| 메트릭 | 단위 | 설명 |
|--------|------|------|
| **model_load_ms** | ms | ONNX 모델 파일 로드 시간 |
| **session_create_ms** | ms | ORT 세션 생성 시간 (QNN 컴파일 포함) |
| **first_inference_ms** | ms | 첫 번째 추론 지연시간 |
| **total_cold_ms** | ms | 전체 Cold Start 시간 (위 3개 합계) |

#### Summary Statistics (분석 도구에서 계산)

| 메트릭 | 단위 | 설명 |
|--------|------|------|
| **p50, p95** | ms | Latency 백분위수 |
| **fps** | inf/s | 초당 추론 횟수 (throughput) |
| **first_30s_p50** | ms | 처음 30초 동안의 Latency p50 |
| **last_30s_p50** | ms | 마지막 30초 동안의 Latency p50 |
| **latency_drift_pct** | % | (last_30s - first_30s) / first_30s × 100 (thermal throttling 지표) |

### 데이터 Export

1. 벤치마크 완료 후 "EXPORT CSV" 클릭
2. 파일명 형식: `kpi_{Model}_{EP}_{timestamp}.csv`
3. 파일 위치: `/sdcard/Android/data/com.example.kpilab/files/Documents/`
4. ADB로 추출:
   ```bash
   adb pull /sdcard/Android/data/com.example.kpilab/files/Documents/ ./logs/
   ```

## Batch Mode (배치 모드)

여러 실험을 연속으로 자동 실행하는 기능입니다. 각 실험 완료 후 자동으로 CSV가 저장되며, 실험 간 30초 쿨다운이 적용됩니다.

### 사용 방법

1. **Batch Mode 체크박스** 활성화
2. **Experiment Set** 드롭다운에서 실행할 실험 세트 선택
3. **START** 버튼 클릭
4. 모든 실험 완료까지 자동 실행 (중간에 STOP 가능)

### 사전 정의된 실험 세트

| 파일 | 실험 세트 | 설명 |
|------|----------|------|
| `experiment_sets_mobilenet.json` | EP 비교 | NPU / GPU / CPU 비교 |
| | 양자화 비교 (NPU) | FP32 / INT8 Dynamic / INT8 QDQ |
| | INT8 EP 비교 | INT8 QDQ 모델의 NPU / GPU / CPU 비교 |
| | 주파수 비교 | 1Hz / 5Hz / 10Hz |
| | Warmup 효과 | Warmup On / Off |
| | 전체 비교 | 5개 실험 (모든 모델 × NPU) |
| `experiment_sets_yolo.json` | EP 비교 | NPU / GPU / CPU 비교 |
| | 양자화 비교 (NPU) | FP32 / INT8 Dynamic / INT8 QDQ |
| | INT8 EP 비교 | INT8 QDQ 모델의 NPU / GPU / CPU 비교 |
| | 주파수 비교 | 1Hz / 5Hz / 10Hz |
| | 전체 비교 | 5개 실험 (모든 모델 × NPU) |

### 자동 Export

- 각 실험 완료 시 자동으로 CSV 파일 저장
- 실험 간 30초 쿨다운 (열 발산 대기)
- 완료된 파일 목록이 UI에 표시됨

### 커스텀 실험 세트

`assets/` 또는 외부 저장소에 `experiment_sets_*.json` 파일을 추가하여 커스텀 실험 세트를 정의할 수 있습니다:

```json
{
  "version": 1,
  "defaults": {
    "frequencyHz": 10,
    "durationMinutes": 5,
    "warmUpEnabled": true,
    "useNpuFp16": true,
    "useContextCache": false
  },
  "experimentSets": [
    {
      "id": "my_custom_set",
      "name": "커스텀 실험",
      "experiments": [
        { "model": "MOBILENET_V2", "executionProvider": "QNN_NPU" },
        { "model": "MOBILENET_V2", "executionProvider": "CPU", "frequencyHz": 5 }
      ]
    }
  ]
}
```

**모델명**: `MOBILENET_V2`, `MOBILENET_V2_INT8_DYNAMIC`, `MOBILENET_V2_INT8_QDQ`, `YOLOV8N`, `YOLOV8N_INT8_DYNAMIC`, `YOLOV8N_INT8_QDQ`

**Execution Provider**: `QNN_NPU`, `QNN_GPU`, `CPU`

**우선순위**: 외부 저장소 (`Documents/`) > assets (동일 ID는 외부 우선)

## CSV 포맷

로그 파일명: `kpi_{Model}_{EP}_{timestamp}.csv` (예: `kpi_MobileNetV2_QNNNPU_20260130_150850.csv`)

```csv
# device_manufacturer,Samsung
# device_model,SM-S928N
# soc_manufacturer,QTI
# soc_model,SM8650
# android_version,14
# api_level,34
# app_version,1.0
# app_build,1
# runtime,ONNX Runtime
# ort_version,1.17.0
# execution_provider,QNN_NPU
# model,MobileNetV2
# model_file,mobilenetv2.onnx
# precision,FP32
# frequency_hz,5
# warmup_iters,10
# model_load_ms,45
# session_create_ms,1823
# first_inference_ms,12.34
# total_cold_ms,1880
# session_id,ort_mnv2_npu_5hz_w_1705123456789
#
timestamp,event_type,latency_ms,thermal_c,power_mw,memory_mb,is_foreground
1705123456789,INFERENCE,12.34,,,,true
1705123457000,SYSTEM,,38.2,2150,245,true
1705123458000,SYSTEM,,38.5,2200,,true
```

> **Note**:
> - `memory_mb`가 비어있으면 해당 interval에서 측정되지 않음 (5초마다 측정)
> - `session_create_ms`는 QNN EP 사용 시 HTP 그래프 컴파일 시간을 포함

## 분석 방법

### 환경 설정

```bash
pip install -r requirements.txt
```

### 분석 실행

```bash
# 단일 로그 분석
python analysis/scripts/parse_logs.py logs/kpi_MobileNetV2_QNNNPU_xxx.csv

# 여러 로그 비교 분석
python analysis/scripts/parse_logs.py logs/

# 시각화
python analysis/scripts/plot_kpi.py logs/kpi_MobileNetV2_QNNNPU_xxx.csv logs/

# Jupyter notebook
jupyter notebook analysis/notebooks/
```

## 실험 매트릭스

| # | EP | Model | FP16 | Cache | Freq | Warm-up | 목적 |
|---|-----|-------|------|-------|------|---------|------|
| 1 | NPU | MobileNetV2 | On | Off | 10Hz | Off | NPU FP16 baseline |
| 2 | NPU | MobileNetV2 | Off | Off | 10Hz | Off | NPU FP32 baseline |
| 3 | NPU | MobileNetV2 | On | Off | 10Hz | On | Warm-up 효과 |
| 4 | GPU | MobileNetV2 | - | - | 10Hz | Off | GPU baseline |
| 5 | CPU | MobileNetV2 | - | - | 10Hz | Off | CPU baseline |
| 6 | NPU | MobileNetV2 (INT8 QDQ) | - | Off | 10Hz | Off | INT8 양자화 효과 |
| 7 | NPU | YOLOv8n | On | Off | 5Hz | Off | 큰 모델 성능 |
| 8 | NPU | MobileNetV2 | On | On | 10Hz | Off | Context Cache 효과 |

> **Note**: FP16 옵션은 FP32 모델에만 적용됩니다. INT8 모델은 항상 INT8로 실행됩니다.

## ONNX Runtime QNN EP 옵션

`OrtRunner.kt`에서 설정 가능한 주요 옵션:

```kotlin
val qnnOptions = mutableMapOf<String, String>()
qnnOptions["backend_path"] = "libQnnHtp.so"          // NPU 백엔드
qnnOptions["htp_performance_mode"] = "burst"         // 성능 모드
qnnOptions["htp_graph_finalization_optimization_mode"] = "3"

// FP16 정밀도 (UI 옵션: NPU FP16)
qnnOptions["enable_htp_fp16_precision"] = "1"        // FP32 → FP16 변환 (NPU에서 더 빠름)

// Context Cache (UI 옵션: Context Cache)
qnnOptions["qnn_context_cache_enable"] = "1"         // HTP 컴파일 그래프 캐싱
qnnOptions["qnn_context_cache_path"] = "path/to/cache.bin"
```

### FP16 Precision
- FP32 모델을 NPU에서 FP16으로 실행하여 성능 향상
- INT8 양자화 모델에는 영향 없음 (이미 INT8로 실행)
- 약간의 정밀도 손실이 있을 수 있음

### Context Cache
- 첫 실행 시 HTP 컴파일 결과를 파일로 저장
- 이후 실행 시 캐시된 그래프를 로드하여 초기화 시간 단축
- 모델/설정별로 별도 캐시 파일 생성 (`qnn_{model}_{precision}.bin`)
- OFF: 항상 새로 컴파일 (캐시 무시)
- ON: 캐시 있으면 사용, 없으면 생성

## Op 분석

QNN EP 호환성 확인 (Snapdragon 8 Gen 2 기준):

```bash
python scripts/analyze_ops.py path/to/model.onnx
```

## 개선 액션

### (A) 실행 정책 개선
- Frequency 조정 (10Hz → 5Hz)
- Warm-up 활성화
- INT8 양자화 모델 사용

### (B) 그래프 변환
```bash
python scripts/graph_transform.py \
    path/to/model.onnx \
    --fix-batch 1 \
    --replace-hardswish \
    --validate
```

## 참고 자료

- [ONNX Runtime QNN EP Documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
- [ONNX Model Zoo](https://github.com/onnx/models)
