# ONNX Runtime QNN EP 옵션 가이드

QNN Execution Provider의 상세 설정 옵션 가이드입니다.

## 기본 설정

`OrtRunner.kt`에서 설정되는 QNN EP 옵션:

```kotlin
val qnnOptions = mutableMapOf<String, String>()
qnnOptions["backend_path"] = "libQnnHtp.so"          // NPU 백엔드
qnnOptions["htp_performance_mode"] = "burst"         // 성능 모드
qnnOptions["htp_graph_finalization_optimization_mode"] = "3"

// FP16 정밀도 (UI 옵션: NPU FP16)
qnnOptions["enable_htp_fp16_precision"] = "1"        // FP32 → FP16 변환

// Context Cache (UI 옵션: Context Cache)
qnnOptions["qnn_context_cache_enable"] = "1"
qnnOptions["qnn_context_cache_path"] = "path/to/cache.bin"
```

---

## 주요 옵션 상세

### backend_path

QNN 백엔드 라이브러리 경로:

| 값 | 설명 |
|----|------|
| `libQnnHtp.so` | NPU (Hexagon Tensor Processor) - 시스템 경로 |
| `libQnnGpu.so` | GPU (Adreno) - 시스템 경로 |
| `{custom_path}/libQnnHtp.so` | NPU - 커스텀 경로 |

**커스텀 경로 사용** ([QnnLibraryManager.kt](../android/app/src/main/java/com/example/kpilab/QnnLibraryManager.kt)):
```kotlin
val qnnLibPath = QnnLibraryManager.getLibraryPath()  // /data/data/.../files/qnn_libs
qnnOptions["backend_path"] = "$qnnLibPath/libQnnHtp.so"
```

### skel_library_dir

DSP에서 실행할 Skel 라이브러리 디렉토리:

```kotlin
qnnOptions["skel_library_dir"] = "/data/data/.../files/qnn_libs"
```

> **중요**: 시스템에 QNN 드라이버가 없는 경우 필수. Skel 라이브러리(`libQnnHtpV73Skel.so`)가 이 경로에 있어야 함.

### htp_performance_mode

NPU 성능 모드 설정:

| 모드 | 설명 | 사용 사례 |
|------|------|----------|
| `burst` | 최대 성능, 높은 전력 | 단기간 고성능 필요 시 |
| `sustained_high` | 지속 가능한 고성능 | 장시간 추론 |
| `balanced` | 성능/전력 균형 | 일반 사용 |
| `power_saver` | 저전력 모드 | 배터리 절약 |

### htp_graph_finalization_optimization_mode

그래프 최적화 수준:

| 모드 | 최적화 수준 | 컴파일 시간 | 런타임 성능 |
|------|------------|------------|------------|
| 0 | 없음 | 가장 빠름 | 가장 느림 |
| 1 | 기본 | 빠름 | 보통 |
| 2 | 중간 | 보통 | 좋음 |
| **3** | 적극적 (기본값) | 느림 | 매우 좋음 |
| 4 | 최대 | 가장 느림 | 최고 |

> **Note**: 모드 3이 컴파일 시간과 런타임 성능의 합리적인 균형점입니다. Context Cache를 사용하면 컴파일 시간 오버헤드를 줄일 수 있습니다.

---

## FP16 Precision

### 개요
- FP32 모델을 NPU에서 FP16으로 실행하여 성능 향상
- INT8 양자화 모델에는 영향 없음 (이미 INT8로 실행)
- 약간의 정밀도 손실이 있을 수 있음

### 설정
```kotlin
qnnOptions["enable_htp_fp16_precision"] = "1"  // ON
qnnOptions["enable_htp_fp16_precision"] = "0"  // OFF (기본값)
```

### 성능 영향
- 일반적으로 2배 가량의 throughput 향상
- 메모리 대역폭 절감
- 정밀도 손실은 대부분의 추론 작업에서 무시 가능

---

## Context Cache

### 개요
- 첫 실행 시 HTP 컴파일 결과를 파일로 저장
- 이후 실행 시 캐시된 그래프를 로드하여 초기화 시간 단축
- 모델/설정별로 별도 캐시 파일 생성

### 설정
```kotlin
qnnOptions["qnn_context_cache_enable"] = "1"
qnnOptions["qnn_context_cache_path"] = "${cacheDir}/qnn_${model}_${precision}.bin"
```

### 캐시 파일 명명
```
qnn_yolov8n.onnx_fp16.bin   # YOLOv8n FP16
qnn_yolov8n.onnx_fp32.bin   # YOLOv8n FP32
qnn_yolov8n.onnx_fp16.bin       # YOLOv8n FP16
```

### Cold Start 시간 비교

| 모델 | Cache OFF | Cache ON | 개선율 |
|------|-----------|----------|--------|
| YOLOv8n | ~2초 | ~0.3초 | 85% |
| YOLOv8n | ~5초 | ~0.5초 | 90% |

---

## Op 분석

QNN EP 호환성 확인:

```bash
python scripts/analyze_ops.py path/to/model.onnx
```

### 출력 예시
```
=== ONNX Model Op Analysis ===
Model: yolov8n.onnx
Total ops: 154

Supported by QNN HTP:
  - Conv: 52
  - BatchNormalization: 52
  - Relu: 35
  - Add: 10
  ...

Potentially unsupported:
  - Softmax: 1 (may fallback to CPU)
```

---

## 그래프 변환

QNN EP 호환성을 높이기 위한 그래프 변환:

```bash
python scripts/graph_transform.py \
    path/to/model.onnx \
    --fix-batch 1 \
    --replace-hardswish \
    --validate
```

### 주요 변환 옵션

| 옵션 | 설명 |
|------|------|
| `--fix-batch N` | 동적 배치를 고정 배치로 변경 |
| `--replace-hardswish` | HardSwish를 QNN 호환 연산으로 대체 |
| `--fuse-bn` | Conv + BatchNorm 융합 |
| `--validate` | 변환 후 출력 검증 |

---

## Custom QNN 라이브러리 번들링

시스템에 QNN 드라이버가 없는 경우, 앱에 QNN 라이브러리를 번들링하여 사용할 수 있습니다.

### 구조

```
assets/qnn_libs/
├── libQnnHtp.so           # HTP 백엔드 (2.6MB)
├── libQnnHtpPrepare.so    # 그래프 준비 (84MB) ← 필수!
├── libQnnHtpV73Stub.so    # V73 Stub (0.6MB)
├── libQnnHtpV73Skel.so    # V73 Skel (10MB)
├── libQnnSystem.so        # 시스템 (2.7MB)
└── libQnnGpu.so           # GPU 백엔드 (6.4MB, 선택)
```

### 설정 코드

```kotlin
// MainActivity.kt - 앱 시작 시
OrtRunner.initializeQnnLibraries(this)

// OrtRunner.kt - 세션 생성 시
val qnnLibPath = QnnLibraryManager.getLibraryPath()
if (qnnLibPath != null) {
    qnnOptions["backend_path"] = "$qnnLibPath/libQnnHtp.so"
    qnnOptions["skel_library_dir"] = qnnLibPath
}
```

### 버전 호환성

| ORT 버전 | 빌드 시 QNN | 번들 QNN | 호환성 |
|---------|------------|---------|--------|
| 1.23.2  | 2.37.1     | 2.37.x  | ✅ 완벽 |
| 1.23.2  | 2.37.1     | 2.42.x  | ✅ 동작 (API 호환) |
| 1.23.2  | 2.37.1     | 2.25.x  | ❌ 실패 |

> **주의**: 모든 라이브러리는 **동일한 QNN SDK 버전**에서 가져와야 합니다.

---

## 참고 자료

- [ONNX Runtime QNN EP Documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
- [QNN SDK Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html)
