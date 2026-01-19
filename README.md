# Mobile Inference KPI Lab

Android 단말(Snapdragon)에서 QNN 기반 on-device inference를 실행하고, 실행 경로 및 정책 변화가 Latency, Power, Thermal KPI에 미치는 영향을 분석하는 프로젝트.

## 프로젝트 목적

1. **실행 경로별 KPI 비교**: NPU-only, NPU+Fallback, GPU-only
2. **실행 정책 영향 분석**: Frequency (1/5/10Hz), Warm-up 유무
3. **개선안 도출**: 엔진 변경 없이 정책/그래프 조정으로 KPI 개선

## 프로젝트 구조

```
mobile-inference-kpi-lab/
├── android/                    # Android 앱
│   └── app/
│       └── src/main/
│           ├── java/.../       # Kotlin 소스
│           ├── cpp/            # Native C++ (QNN)
│           └── res/            # UI 리소스
├── models/                     # 모델 관련
│   └── scripts/
│       ├── convert_to_dlc.py   # QNN 변환
│       ├── analyze_ops.py      # Op 분석
│       └── graph_transform.py  # 그래프 변환
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
- NDK 25.0+
- QNN SDK 2.x (Qualcomm AI Engine Direct)
- Python 3.10+

## 설정 방법

### 1. QNN SDK 설정

```bash
export QNN_SDK_ROOT=/path/to/qnn-sdk
```

### 2. 모델 준비

```bash
cd models/scripts

# MobileNetV3 다운로드 및 변환
python convert_to_dlc.py --download-mobilenetv3

# Op 분석 (fallback 가능성 확인)
python analyze_ops.py ../original/mobilenetv3_small.onnx
```

### 3. Android 앱 빌드

```bash
cd android
./gradlew assembleDebug
```

### 4. 앱 설치 및 실행

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 앱 사용법

### 실험 설정

| 옵션 | 선택지 | 설명 |
|------|--------|------|
| Execution Path | NPU-only / NPU+FB / GPU-only | 추론 실행 경로 |
| Frequency | 1Hz / 5Hz / 10Hz | 추론 빈도 |
| Warm-up | On / Off | 워밍업 실행 여부 |
| Duration | 5min / 10min | 실험 시간 |

### 측정 KPI

- **Latency**: 추론 지연시간 (ms)
- **Thermal**: SoC 온도 (°C)
- **Power**: 전력 소비 (mW)
- **Memory**: 메모리 사용량 (MB)

### 데이터 Export

1. 벤치마크 완료 후 "EXPORT CSV" 클릭
2. 파일 위치: `/sdcard/Android/data/com.example.kpilab/files/Documents/`
3. ADB로 추출:
   ```bash
   adb pull /sdcard/Android/data/com.example.kpilab/files/Documents/ ./data/
   ```

## 분석 방법

### 환경 설정

```bash
cd analysis
pip install -r requirements.txt
```

### 분석 실행

```bash
# 단일 로그 분석
python scripts/parse_logs.py data/kpi_log_xxx.csv

# 시각화
python scripts/plot_kpi.py data/kpi_log_xxx.csv

# Jupyter notebook
jupyter notebook notebooks/
```

## 실험 매트릭스

| # | Path | Freq | Warm-up | 목적 |
|---|------|------|---------|------|
| 1 | NPU_ONLY | 10Hz | Off | NPU 한계 확인 |
| 2 | NPU_ONLY | 10Hz | On | Warm-up 효과 |
| 3 | NPU_FALLBACK | 10Hz | Off | 현실적 default |
| 4 | GPU_ONLY | 10Hz | Off | GPU baseline |
| 5 | NPU_FALLBACK | 5Hz | Off | Frequency 영향 |
| 6 | NPU_FALLBACK | 1Hz | Off | 저빈도 정책 |

## 개선 액션

### (A) 실행 정책 개선
- Frequency 조정 (10Hz → 5Hz)
- Warm-up 활성화
- Hot/Cold path 분리

### (B) 그래프 변환
```bash
python models/scripts/graph_transform.py \
    models/original/model.onnx \
    --fix-batch 1 \
    --replace-hardswish \
    --validate
```

## CSV 포맷

```csv
timestamp,event_type,latency_ms,thermal_c,power_mw,memory_mb,is_foreground
1705123456789,INFERENCE,12.3,,,, true
1705123457000,SYSTEM,, 38.2,2150,245,true
```

## 참고 자료

- [QNN SDK Documentation](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- [ONNX Model Zoo](https://github.com/onnx/models)
