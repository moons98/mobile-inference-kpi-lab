# Mobile Inference KPI Lab

Snapdragon 기반 Android 단말에서 **NPU offload coverage**, **precision 정책**, **pipeline scheduling 구조**가 **Sustained E2E KPI**에 미치는 영향을 정량 분석하는 프로젝트.

- **Runtime**: ONNX Runtime + QNN Execution Provider
- **Model**: YOLOv8n (Object Detection)
- **Pipeline**: 전처리(CPU) → 추론(NPU/CPU) → 후처리(CPU)

## 핵심 실험

### 실험 1: Execution Path & Precision Impact

Offload coverage와 precision 정책이 E2E KPI에 미치는 영향을 분석한다.

| Case | EP | Precision | 구성 |
|------|-----|-----------|------|
| **A** | CPU | FP32 | pre/infer/post 모두 CPU (Baseline) |
| **B** | QNN_NPU | FP16 (runtime) | pre/post CPU, infer NPU |
| **C** | QNN_NPU | INT8 (QDQ) | pre/post CPU, infer NPU |

**측정 항목**: Coverage %, fallback ops, E2E breakdown (pre/infer/post), P50/P95, cold start, sustained drift, power

### 실험 2: Pipeline Scheduling Impact (TODO)

Sequential vs Overlapped pipeline이 sustained throughput에 미치는 영향을 분석한다.

| Case | Scheduling |
|------|------------|
| **A** | Sequential (pre → infer → post) |
| **B** | Overlapped (Frame N+1 pre / N infer / N-1 post 동시 실행) |

**측정 항목**: Steady-state FPS, E2E latency, CPU utilization, sustained drift

## E2E Pipeline

```
Bitmap (원본)
  │
  ├─ [전처리 — CPU] ─────────────────────────┐
  │  Resize → Letterbox Pad → Normalize      │
  │  → HWC→CHW 변환                          │
  │                                           ▼
  │                            Float[1, 3, 640, 640]
  │                                           │
  │                            [추론 — NPU/GPU/CPU]
  │                            YOLOv8n forward pass
  │                                           │
  │                                           ▼
  │                            Float[1, 84, 8400]
  │                                           │
  ├─ [후처리 — CPU] ─────────────────────────┘
  │  Argmax → Confidence Filter (0.25)
  │  → Coord 변환 → Per-class NMS (IoU 0.45)
  │
  ▼
List<Detection>
```

## 최종 산출물

- **Coverage table**: Case별 coverage %, fallback ops, partition count
- **Latency breakdown 표**: Case별 Pre / Infer / Post / E2E / P50 / P95
- **Sustained drift 그래프**: 시간에 따른 latency 변화 + precision별 비교
- **Precision 비교표**: Precision별 coverage / infer ms / E2E ms / drift %
- **Pipeline 비교표** (TODO): Scheduling별 FPS / E2E ms / drift %

## 빠른 시작

### 1. 모델 준비

```bash
# FP32 모델
python scripts/export_to_onnx.py --export-yolov8n

# INT8 양자화 모델
python scripts/export_to_onnx.py --export-yolov8n-quantized --quant-method static
```

또는 ONNX 모델을 `android/app/src/main/assets/`에 직접 배치.

### 2. 빌드 및 설치

```bash
cd android
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 3. 벤치마크 실행

1. 앱에서 Model, EP, Frequency 등 설정
2. START 버튼 클릭
3. 완료 후 EXPORT CSV 클릭

### 4. 데이터 분석

```bash
# 로그 추출
adb pull /sdcard/Android/data/com.example.kpilab/files/Documents/ ./logs/yolov8n/

# 비교 분석
python analysis/scripts/parse_logs.py logs/yolov8n/
python analysis/scripts/plot_kpi.py logs/yolov8n/
```

## 측정 KPI

| KPI | 단위 | 설명 |
|-----|------|------|
| **Latency (E2E)** | ms | 전처리 + 추론 + 후처리 총 지연시간 |
| **Preprocess** | ms | 이미지 전처리 (letterbox, normalize, CHW) |
| **Inference** | ms | 순수 모델 추론 시간 (ONNX Runtime) |
| **Postprocess** | ms | 후처리 (confidence filter, NMS, 좌표 변환) |
| **Detection Count** | 개 | 프레임당 검출 객체 수 |
| **Offload Coverage** | % | 전체 graph node 중 QNN EP(GPU/NPU)에서 실행되는 비율 |
| **Fallback Ops** | - | CPU로 fallback된 연산 목록 |
| **Thermal** | °C | SoC 온도 |
| **Power** | mW | 전력 소비 |
| **Memory** | MB | VmRSS 메모리 |
| **Cold Start** | ms | 모델 로드 + 세션 생성 시간 |

## NPU (HTP) 사용 요구사항

NPU 실행을 위해서는 단말에 **QNN Skel 라이브러리**가 필요합니다.

| 조건 | 설명 |
|------|------|
| **시스템 드라이버** | 단말 펌웨어에 QNN 드라이버 포함 필요 |
| **Custom Skel** | 없으면 QNN SDK에서 수동 설치 가능 (개발용) |

**진단 방법**:
```bash
# QNN 라이브러리 확인 (없으면 NPU 사용 불가)
adb shell ls /vendor/lib64/ | grep -i qnn

# SNPE만 있으면 QNN 미지원 (레거시)
adb shell ls /vendor/lib64/ | grep -i snpe
```

**Custom Skel 설정** (시스템에 QNN 없는 경우):
```bash
# QNN SDK에서 Skel push
adb shell mkdir -p /data/local/tmp/qnn
adb push libQnnHtpV73Skel.so /data/local/tmp/qnn/
adb push libQnnHtpV73.so /data/local/tmp/qnn/
adb shell chmod 755 /data/local/tmp/qnn/*.so
```

> 자세한 내용: [troubleshooting.md](docs/troubleshooting.md#custom-qnn-skel-라이브러리-설정-개발용)

## 앱 설정 옵션

| 옵션 | 설명 |
|------|------|
| **Model** | YOLOv8n (FP32/INT8) |
| **Execution Provider** | NPU / GPU / CPU |
| **Frequency** | 1Hz / 5Hz / 10Hz |
| **NPU FP16** | FP32 모델을 FP16으로 실행 (NPU 전용) |
| **Context Cache** | QNN 컴파일 그래프 캐싱 |
| **Batch Mode** | 여러 실험 자동 연속 실행 |

> **Note**: Warm-up (10회)은 항상 자동 실행되어 steady-state 성능을 측정합니다.

## 프로젝트 구조

```
mobile-inference-kpi-lab/
├── android/                    # Android 벤치마크 앱
│   └── app/src/main/
│       ├── java/.../kpilab/
│       │   ├── OrtRunner.kt          # E2E 파이프라인 (전처리→추론→후처리)
│       │   ├── YoloPostProcessor.kt   # YOLO 후처리 (NMS, 좌표 변환)
│       │   ├── BenchmarkRunner.kt     # 벤치마크 루프 + CSV export
│       │   ├── LogcatCapture.kt       # ORT 파티션 로그 캡처
│       │   └── KpiCollector.kt        # Thermal/Power/Memory 수집
│       └── assets/
│           ├── *.onnx                 # YOLO 모델 파일
│           ├── sample_image.jpg       # 벤치마크 입력 이미지
│           └── experiment_sets_*.json # 배치 실험 정의
├── analysis/scripts/           # Python 분석 도구
│   ├── parse_logs.py           # CSV 파싱 + 리포트 생성
│   └── plot_kpi.py             # KPI 시각화
├── scripts/                    # 모델 export 스크립트
├── logs/                       # 원본 데이터 (adb pull)
├── outputs/                    # 분석 결과물 (플롯, 리포트)
└── docs/                       # 상세 문서
```

## 문서

| 문서 | 설명 |
|------|------|
| [design_rationale.md](docs/design_rationale.md) | ORT + QNN EP 구조 선택 이유 및 설계 근거 |
| [experiment_design.md](docs/experiment_design.md) | 실험 설계 및 검증 체크리스트 |
| [execution_profiles.md](docs/execution_profiles.md) | EP별 동작 원리, Cold Start, 디버깅 가이드 |
| [qnn_options.md](docs/qnn_options.md) | QNN EP 옵션 상세 (FP16, Cache, 최적화 모드) |
| [troubleshooting.md](docs/troubleshooting.md) | 오류 해결 가이드 (QNN 버전 불일치 등) |

## 참고 자료

- [ONNX Runtime QNN EP Documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
