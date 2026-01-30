# Mobile Inference KPI Lab

Android 단말(Snapdragon)에서 **ONNX Runtime + QNN Execution Provider** 기반 on-device inference를 실행하고, 실행 경로 및 정책 변화가 Latency, Power, Thermal KPI에 미치는 영향을 분석하는 프로젝트.

## 프로젝트 목적

1. **실행 경로별 KPI 비교**: NPU (HTP), GPU, CPU
2. **실행 정책 영향 분석**: Frequency, Warm-up, FP16/INT8
3. **NPU 디버깅**: QNN EP 로그를 통한 지원/미지원 Op 파악

## 빠른 시작

### 1. 모델 준비

```bash
# FP32 모델
python scripts/export_to_onnx.py --export-mobilenetv2

# INT8 양자화 모델
python scripts/export_to_onnx.py --export-mobilenetv2-quantized --quant-method static
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
adb pull /sdcard/Android/data/com.example.kpilab/files/Documents/ ./logs/mobilenetv2/

# 비교 분석
python analysis/scripts/parse_logs.py logs/mobilenetv2/
python analysis/scripts/plot_kpi.py logs/mobilenetv2/
```

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
| **Model** | MobileNetV2, YOLOv8n (FP32/INT8) |
| **Execution Provider** | NPU / GPU / CPU |
| **Frequency** | 1Hz / 5Hz / 10Hz |
| **NPU FP16** | FP32 모델을 FP16으로 실행 (NPU 전용) |
| **Context Cache** | QNN 컴파일 그래프 캐싱 |
| **Batch Mode** | 여러 실험 자동 연속 실행 |

> **Note**: Warm-up (10회)은 항상 자동 실행되어 steady-state 성능을 측정합니다.

## 측정 KPI

| KPI | 단위 | 설명 |
|-----|------|------|
| **Latency** | ms | 단일 추론 지연시간 |
| **Thermal** | °C | SoC 온도 |
| **Power** | mW | 전력 소비 |
| **Memory** | MB | VmRSS 메모리 |
| **Cold Start** | ms | 모델 로드 + 세션 생성 시간 |

## 프로젝트 구조

```
mobile-inference-kpi-lab/
├── android/                    # Android 앱
├── scripts/                    # ONNX 모델 export/분석 스크립트
├── analysis/                   # 분석 도구 (parse_logs.py, plot_kpi.py)
├── logs/                       # 원본 데이터 (adb pull)
├── outputs/                    # 분석 결과물 (플롯, 리포트)
└── docs/                       # 상세 문서
```

## 문서

| 문서 | 설명 |
|------|------|
| [execution_profiles.md](docs/execution_profiles.md) | EP별 동작 원리, Cold Start, 디버깅 가이드 |
| [qnn_options.md](docs/qnn_options.md) | QNN EP 옵션 상세 (FP16, Cache, 최적화 모드) |
| [troubleshooting.md](docs/troubleshooting.md) | 오류 해결 가이드 (QNN 버전 불일치 등) |

## 참고 자료

- [ONNX Runtime QNN EP Documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
