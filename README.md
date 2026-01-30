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
├── mobilenetv2_torchvision.onnx           # FP32 MobileNetV2
└── mobilenetv2_torchvision_quantized.onnx # INT8 MobileNetV2 (QDQ format)
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
| Model | MobileNetV2 / MobileNetV2 (INT8) | 추론 모델 |
| Execution Provider | NPU / GPU / CPU | 추론 실행 경로 |
| Frequency | 1Hz / 5Hz / 10Hz | 추론 빈도 |
| Warm-up | On / Off | 워밍업 실행 여부 (10회 추론) |
| Duration | 5min / 10min | 실험 시간 |

### 측정 KPI

| KPI | 단위 | 수집 주기 | 설명 |
|-----|------|----------|------|
| **Latency** | ms | 매 추론 | 단일 추론 지연시간 |
| **Throughput** | inf/s | 실시간 계산 | 초당 추론 횟수 |
| **Thermal** | °C | 1초 | SoC 온도 |
| **Power** | mW | 1초 | 전력 소비 |
| **Memory** | MB | 5초 | VmRSS 메모리 |

### 데이터 Export

1. 벤치마크 완료 후 "EXPORT CSV" 클릭
2. 파일 위치: `/sdcard/Android/data/com.example.kpilab/files/Documents/`
3. ADB로 추출:
   ```bash
   adb pull /sdcard/Android/data/com.example.kpilab/files/Documents/ ./data/
   ```

## CSV 포맷

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
# session_id,ort_mnv2_npu_5hz_w_1705123456789
#
timestamp,event_type,latency_ms,thermal_c,power_mw,memory_mb,is_foreground
1705123456789,INFERENCE,12.34,,,, true
1705123457000,SYSTEM,,38.2,2150,245,true
1705123458000,SYSTEM,,38.5,2200,,true
```

> **Note**: `memory_mb`가 비어있으면 해당 interval에서 측정되지 않음 (5초마다 측정)

## 분석 방법

### 환경 설정

```bash
pip install -r requirements.txt
```

### 분석 실행

```bash
# 단일 로그 분석
python analysis/scripts/parse_logs.py data/kpi_log_xxx.csv

# 시각화
python analysis/scripts/plot_kpi.py data/kpi_log_xxx.csv

# Jupyter notebook
jupyter notebook analysis/notebooks/
```

## 실험 매트릭스

| # | EP | Model | Freq | Warm-up | 목적 |
|---|-----|-------|------|---------|------|
| 1 | NPU | MobileNetV2 | 10Hz | Off | NPU 한계 확인 |
| 2 | NPU | MobileNetV2 | 10Hz | On | Warm-up 효과 |
| 3 | GPU | MobileNetV2 | 10Hz | Off | GPU baseline |
| 4 | CPU | MobileNetV2 | 10Hz | Off | CPU baseline |
| 5 | NPU | MobileNetV2 (INT8) | 10Hz | Off | 양자화 효과 |
| 6 | NPU | MobileNetV2 | 5Hz | Off | Frequency 영향 |

## ONNX Runtime QNN EP 옵션

`OrtRunner.kt`에서 설정 가능한 주요 옵션:

```kotlin
val qnnOptions = mutableMapOf<String, String>()
qnnOptions["backend_path"] = "libQnnHtp.so"          // NPU 백엔드
qnnOptions["htp_performance_mode"] = "burst"         // 성능 모드
qnnOptions["htp_graph_finalization_optimization_mode"] = "3"
qnnOptions["enable_htp_fp16_precision"] = "1"        // FP16 정밀도
```

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
