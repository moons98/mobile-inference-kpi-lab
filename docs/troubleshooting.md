# Troubleshooting Guide

ONNX Runtime + QNN EP 사용 시 발생할 수 있는 문제와 해결 방법을 정리합니다.

---

## QNN 관련 오류

### QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES

**증상**:
```
QNN SetupBackend failed Failed to create device.
Error: QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES: Loaded libraries are of incompatible versions
```

**원인**: ONNX Runtime AAR에 포함된 QNN SDK 버전과 단말의 QNN 드라이버 버전 불일치

**진단**:
ORT 로그에서 버전 확인:
```
libQnnHtp.so interface version: 2.25.0    ← AAR의 QNN SDK
Backend build version: v2.33.0.xxx        ← 단말의 QNN 드라이버
```

**해결**:

1. **ONNX Runtime 버전 업데이트** (권장)
   ```kotlin
   // build.gradle.kts
   // 기존
   implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.22.0")  // QNN 2.25.0

   // 변경
   implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.23.2")  // QNN 2.37.1
   ```

2. **버전 호환성 표**
   | ORT 버전 | QNN SDK | 비고 |
   |---------|---------|------|
   | 1.22.0 | 2.25.0 | |
   | 1.23.2 | 2.37.1 | 최신 (2026-01) |

3. **단말 드라이버 업데이트**: OTA 업데이트 또는 vendor 이미지 업데이트 필요

**참고**:
- [Maven Central - onnxruntime-android-qnn](https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-android-qnn)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)

---

### Unable to load Skel Library

**증상**:
```
DspTransport.openSession qnn_open failed, 0x80000406
Unable to load Skel Library. transportStatus: 8
Error in verify skel version
```

**원인**: HTP skel 라이브러리 로드 실패 (QNN 버전 불일치의 세부 오류)

**해결**: 위 `QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES` 해결 방법 참조

---

### QNN_DEVICE_ERROR_INVALID_CONFIG

**증상**:
```
QNN SetupBackend failed Failed to create device.
Error: QNN_DEVICE_ERROR_INVALID_CONFIG: Invalid config values
Failed to create transport for device, error: 1002
```

**원인**: HTP device 생성 시 설정 오류 (주로 skel 라이브러리 또는 DSP 접근 문제)

**확인 사항**:
1. 단말이 HTP (Hexagon Tensor Processor) 지원하는지 확인
2. QNN skel 라이브러리가 단말에 설치되어 있는지 확인
3. SELinux 정책으로 DSP 접근이 차단될 수 있음

**진단 방법**:
```bash
# 플랫폼 확인
adb shell getprop ro.board.platform

# QNN 라이브러리 확인
adb shell ls -la /vendor/lib64/ | grep -E "Qnn|qnn"

# SNPE 라이브러리 확인 (QNN과 다름)
adb shell ls -la /vendor/lib64/ | grep -E "Snpe|snpe"

# DSP 디바이스 확인
adb shell ls -la /dev/adsp*
```

**kalama (SM8550) 플랫폼 사례**:
```
# 문제 상황: DSP 디바이스는 존재하지만 QNN HTP 초기화 실패
/dev/adsp-devma                 ← DSP 디바이스 존재
/dev/adsp-smd

# 라이브러리 확인 결과
libSnpeHtpV73Stub.so           ← SNPE HTP stub만 존재
libSnpeHtpV73Skel.so (없음)    ← QNN skel 없음
libQnnHtp.so (AAR 내장)        ← ORT AAR의 QNN 라이브러리
```

**원인 분석**:
- 단말에 **SNPE** (Snapdragon Neural Processing Engine) 라이브러리만 설치됨
- **QNN** (Qualcomm AI Engine Direct) skel 라이브러리 없음
- ONNX Runtime은 QNN을 사용하므로 호환되지 않음
- 하드웨어(HTP)는 지원하지만 시스템 드라이버가 QNN 미지원

**해결 방법**:
1. **GPU EP 사용** (권장 우회책): Adreno GPU는 대부분 정상 동작
   ```kotlin
   executionProvider = ExecutionProvider.QNN_GPU  // NPU 대신 GPU 사용
   ```

2. **Custom Skel 라이브러리 로드** (개발/테스트용): 아래 섹션 참조

3. **시스템 업데이트**: OTA 또는 vendor 이미지에 QNN 드라이버 포함 필요

4. **Qualcomm AI Hub 확인**: 단말별 QNN 지원 여부 확인
   - https://aihub.qualcomm.com/

---

### Custom QNN Skel 라이브러리 설정 (개발용)

시스템에 QNN Skel이 없는 단말에서 HTP를 테스트하려면 QNN SDK에서 Skel 라이브러리를 직접 push할 수 있습니다.

> ⚠️ **주의**: 개발/테스트 목적으로만 사용. unsigned 라이브러리는 production 배포 불가.

**필요 조건**:
- Qualcomm AI Engine Direct (QNN) SDK 설치
- 단말 플랫폼에 맞는 Hexagon 버전 확인 (예: SM8550 = V73)

**설정 방법**:

1. **QNN SDK에서 Skel 라이브러리 위치 확인**:
   ```
   C:\Qualcomm\QNN\{version}\qairt\{version}\lib\hexagon-v{XX}\unsigned\
   ├── libQnnHtpV73Skel.so    ← 필수 (DSP에서 실행)
   └── libQnnHtpV73.so        ← 필수 (HTP 구현체)
   ```

2. **단말에 Push**:
   ```bash
   # 디렉토리 생성
   adb shell mkdir -p /data/local/tmp/qnn

   # Skel 라이브러리 push
   adb push libQnnHtpV73Skel.so /data/local/tmp/qnn/
   adb push libQnnHtpV73.so /data/local/tmp/qnn/

   # 권한 설정
   adb shell chmod 755 /data/local/tmp/qnn/*.so
   ```

3. **앱 코드에서 경로 설정** ([OrtRunner.kt](../android/app/src/main/java/com/example/kpilab/OrtRunner.kt#L147-L156)):
   ```kotlin
   // QNN EP 옵션에 skel_library_dir 추가
   val customSkelPath = "/data/local/tmp/qnn"
   val skelFile = java.io.File("$customSkelPath/libQnnHtpV73Skel.so")
   if (skelFile.exists()) {
       qnnOptions["skel_library_dir"] = customSkelPath
   }
   ```

**버전 호환성**:
| ORT AAR 버전 | 내장 QNN SDK | 권장 Skel 버전 |
|-------------|-------------|---------------|
| 1.23.2 | v2.37.1 | v2.37.x |
| 1.22.0 | v2.31.0 | v2.31.x |

> 버전이 다르면 `QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES` 오류 발생 가능

**확인 방법**:
```bash
# 앱 실행 후 로그 확인
adb logcat -s OrtRunner:V | grep -i skel
# 예상 출력: "Custom skel library dir: /data/local/tmp/qnn"
```

---

### NPU fallback to CPU

**증상**:
- NPU 선택했지만 CPU와 비슷한 latency
- ORT 로그에 "QNN nodes: 0"

**확인 방법**:
```bash
adb logcat -s onnxruntime:V | grep -E "SetupBackend|GetCapability|Error"
```

**원인**:
1. QNN 초기화 실패 → CPU fallback
2. 모델에 QNN 미지원 Op 포함

**해결**:
1. QNN 초기화 오류 확인 및 해결
2. `scripts/analyze_ops.py`로 모델 Op 분석
3. 미지원 Op가 많으면 GPU EP 사용 고려

---

## SNPE vs QNN 이해하기

Qualcomm의 AI 추론 프레임워크는 두 가지 세대가 있으며, 서로 **호환되지 않습니다**.

### 개요

| 항목 | SNPE | QNN |
|------|------|-----|
| 정식 명칭 | Snapdragon Neural Processing Engine | Qualcomm AI Engine Direct |
| 세대 | 구세대 (레거시) | 신세대 (현재 주력) |
| 출시 | 2017년경 | 2021년경 |
| 모델 포맷 | `.dlc` (Deep Learning Container) | `.bin` (QNN context binary) |
| 지원 프레임워크 | TensorFlow, Caffe, ONNX (변환 필요) | ONNX Runtime, TFLite 직접 지원 |

### 아키텍처 비교

```
SNPE 아키텍처:
  앱 → SNPE SDK → DLC 모델 → SNPE Runtime → libSnpeHtp*.so → HTP

QNN 아키텍처:
  앱 → ONNX Runtime → QNN EP → QNN Runtime → libQnnHtp*.so → HTP
```

### Stub/Skel 라이브러리 구조

Qualcomm DSP 통신은 **Stub-Skel** 패턴을 사용합니다:

```
┌──────────────┐          ┌──────────────┐
│   앱 (ARM)   │          │  DSP (HTP)   │
│  Stub.so ────┼── IPC ───┼── Skel.so    │
│  (호출자)    │          │  (실행자)    │
└──────────────┘          └──────────────┘
```

| 라이브러리 | 위치 | 역할 |
|-----------|------|------|
| `libSnpeHtpV73Stub.so` | `/vendor/lib64/` | SNPE에서 HTP 호출 |
| `libQnnHtpV73Stub.so` | AAR 또는 vendor | QNN에서 HTP 호출 |
| `*Skel.so` | DSP 펌웨어 | DSP에서 실제 실행 |

### 단말에 SNPE만 있고 QNN이 없는 경우

**진단**:
```bash
# SNPE 라이브러리 확인 (있으면 SNPE 지원)
adb shell ls /vendor/lib64/ | grep -i snpe

# QNN 라이브러리 확인 (없으면 QNN 미지원)
adb shell ls /vendor/lib64/ | grep -i qnn
```

**결과 해석**:
- `libSnpeHtp*.so` 있음 + `libQnnHtp*.so` 없음 → **SNPE만 지원**
- ONNX Runtime QNN EP는 QNN 드라이버 필요 → **HTP 사용 불가**

**해결 방안**:

| 방안 | 실현 가능성 | 설명 |
|------|------------|------|
| GPU EP 사용 | ✅ 즉시 가능 | Adreno GPU는 QNN 지원 확인됨 |
| SNPE SDK 사용 | ⚠️ 개발 필요 | ONNX → DLC 변환 후 SNPE API 직접 호출 |
| 시스템 업데이트 | ❓ OEM 의존 | QNN 드라이버 포함된 펌웨어 필요 |

---

## 모델 관련 오류

### Model file not found

**증상**:
```
Failed to load model: mobilenetv2.onnx
```

**해결**:
1. 모델 파일이 `android/app/src/main/assets/`에 있는지 확인
2. 파일명이 `OnnxModelType`의 `fileName`과 일치하는지 확인

### Invalid model format

**증상**:
```
Invalid model format or unsupported opset
```

**해결**:
1. ONNX opset 버전 확인 (권장: opset 17 이하)
2. `scripts/export_to_onnx.py`로 재변환

---

## 측정 관련 문제

### Thermal 읽기 실패

**증상**:
```
=== No readable thermal path found ===
Battery temperature fallback: 32.5 °C
```

**원인**: `/sys/class/thermal/` 경로에 읽기 권한 없음 (비root 단말)

**영향**: Battery 온도로 fallback (SoC 온도보다 부정확)

**해결**: Root 권한 필요 또는 Battery 온도 사용 감수

---

### Power 측정값이 0

**증상**: CSV에서 `power_mw`가 계속 0

**원인**:
- 충전 중일 때 `BATTERY_PROPERTY_CURRENT_NOW`가 음수 또는 0
- 일부 단말에서 미지원

**해결**:
1. 충전기 분리 후 테스트
2. 단말의 power 측정 지원 여부 확인

---

### Memory 값이 비어있음

**증상**: CSV에서 `memory_mb` 열이 대부분 비어있음

**원인**: 정상 동작 - Memory는 5초마다만 측정

**참고**:
```csv
timestamp,event_type,latency_ms,thermal_c,power_mw,memory_mb,is_valid
1706789013000,SYSTEM,,38.2,2150,245,true   # 측정됨
1706789014000,SYSTEM,,38.5,2200,,true      # 미측정 (정상)
```

---

## 빌드 관련 오류

### Gradle sync 실패

**증상**:
```
Could not resolve com.microsoft.onnxruntime:onnxruntime-android-qnn:x.x.x
```

**해결**:
1. 인터넷 연결 확인
2. `settings.gradle.kts`에 mavenCentral 확인:
   ```kotlin
   repositories {
       google()
       mavenCentral()
   }
   ```

### APK 설치 실패

**증상**:
```
INSTALL_FAILED_NO_MATCHING_ABIS
```

**원인**: 단말 아키텍처와 APK 불일치

**해결**:
`build.gradle.kts`에서 abiFilters 확인:
```kotlin
ndk {
    abiFilters += listOf("arm64-v8a")  // 64bit ARM만 지원
}
```

---

## 로그 수집

### ORT 로그 확인
```bash
# 실시간 로그
adb logcat -s onnxruntime:V OrtRunner:I

# QNN 관련만
adb logcat | grep -E "QNN|qnn|Qnn"
```

### 앱 로그 추출
```bash
adb pull /sdcard/Android/data/com.example.kpilab/files/Documents/ ./logs/
```

### 전체 logcat 저장
```bash
adb logcat -d > full_logcat.txt
```

---

## 개발 중 겪었던 주요 이슈

프로젝트 개발 과정에서 해결한 주요 문제들을 기록합니다.

### 1. TFLite → ONNX Runtime 마이그레이션

**배경**: 초기에는 TFLite 기반이었으나, QNN EP를 통한 NPU 지원을 위해 ONNX Runtime으로 전환

**주요 변경**:
- `TFLiteRunner.kt` → `OrtRunner.kt`
- DLC 포맷 → ONNX 포맷
- Interpreter API → OrtSession API

**교훈**: EP(Execution Provider) 기반 아키텍처가 다양한 하드웨어 지원에 유리

---

### 2. ONNX Runtime 버전 정보 API 미지원

**문제**: Android에서 `OrtEnvironment.getVersionString()` 같은 버전 조회 API가 없음

**증상**:
```
Unresolved reference: getVersionString
```

**해결**: `build.gradle.kts`에서 버전을 `BuildConfig`로 노출
```kotlin
// build.gradle.kts
buildFeatures {
    buildConfig = true
}

// OrtRunner.kt에서는 하드코딩
private val ORT_VERSION = "1.23.2"
```

---

### 3. Kotlin Pair 타입 접근 오류 (Batch Mode)

**문제**: `ExperimentSetLoader`에서 `Pair<List, String?>` 반환 시 `.first.isNotEmpty()` 호출 오류

**증상**:
```
Unresolved reference: isNotEmpty
Type mismatch: inferred type is Any but List was expected
```

**원인**: Kotlin에서 `Pair`의 제네릭 타입이 제대로 추론되지 않음

**해결**: 명시적 타입 캐스팅 또는 data class 사용
```kotlin
// Before
val (experiments, error) = loader.load()
if (experiments.isNotEmpty()) { ... }  // Error

// After
val result = loader.load()
if ((result.first as List<*>).isNotEmpty()) { ... }
```

---

### 4. WARMING_UP 상태가 UI에 표시 안됨

**문제**: Warm-up 실행 중에도 UI가 IDLE 상태로 표시

**원인**: `initialize()` 내부에서 warm-up을 실행하여 StateFlow 업데이트가 UI에 반영되기 전에 완료

**해결**: `runWarmUp()`을 별도 함수로 분리하고, 상태 업데이트 후 실행
```kotlin
// Before: initialize() 내부에서 warm-up 실행
// After: 별도 단계로 분리
_progress.value = progress.copy(state = BenchmarkState.WARMING_UP)
runWarmUp(iterations)  // 이제 UI에 표시됨
```

---

### 5. OrtRunner 리소스 누수

**문제**: 벤치마크 재시작 시 이전 OrtSession이 해제되지 않아 메모리 누수

**증상**: 여러 번 실행 후 메모리 증가, 가끔 크래시

**해결**: 새 벤치마크 시작 전 이전 runner 명시적 해제
```kotlin
// BenchmarkRunner.kt
ortRunner?.let {
    Log.i(TAG, "Releasing previous runner")
    it.release()
    ortRunner = null
}
```

---

### 6. 모델 파일명 불일치

**문제**: Export 스크립트와 앱의 모델 파일명이 다름

**증상**:
```
Failed to load model: mobilenetv2_torchvision.onnx
```

**원인**:
- Export 스크립트: `mobilenetv2.onnx`
- 앱 코드: `mobilenetv2_torchvision.onnx`

**해결**: `OnnxModelType` enum의 `fileName`을 export 스크립트와 일치시킴
```kotlin
enum class OnnxModelType(val displayName: String, val fileName: String) {
    MOBILENET_V2("MobileNetV2", "mobilenetv2.onnx"),
    // ...
}
```

---

### 7. Thermal 경로 단말별 차이

**문제**: thermal zone 경로가 단말마다 다름

**증상**: 일부 단말에서 thermal 값이 항상 0

**해결**: 여러 경로를 시도하고 읽기 가능한 첫 번째 경로 사용
```kotlin
private val THERMAL_PATHS = listOf(
    "/sys/class/thermal/thermal_zone0/temp",
    "/sys/class/thermal/thermal_zone1/temp",
    // ... 추가 경로들
)
```

---

### 8. CSV Memory 컬럼 파싱 오류

**문제**: Memory가 5초마다만 측정되어 빈 값이 많아 파싱 오류

**증상**: pandas에서 `ValueError: cannot convert float NaN to integer`

**해결**: 빈 값을 명시적으로 처리
```python
# parse_logs.py
df['memory_mb'] = pd.to_numeric(df['memory_mb'], errors='coerce')
```
