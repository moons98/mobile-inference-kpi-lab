# QNN SDK 설정 가이드

## 1. QNN SDK 다운로드

Qualcomm Developer Network에서 다운로드:
https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk

## 2. 환경 변수 설정

```bash
export QNN_SDK_ROOT=/path/to/qnn-sdk
```

## 3. 라이브러리 복사

### 앱에 번들할 라이브러리

```bash
# 목적지
DEST=android/app/src/main/jniLibs/arm64-v8a/

# 필수 라이브러리 복사
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so $DEST
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnGpu.so $DEST
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so $DEST
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so $DEST

# HTP Stub (타겟 디바이스에 맞는 버전)
# Snapdragon 8 Gen 2 (v73) 예시:
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV73Stub.so $DEST

# 여러 디바이스 지원 시 모든 버전 복사:
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV66Stub.so $DEST
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV68Stub.so $DEST
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV69Stub.so $DEST
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV73Stub.so $DEST
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so $DEST
```

### 디바이스에 push할 Skeleton 라이브러리

```bash
# Signed skeleton (production)
adb push $QNN_SDK_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so /data/local/tmp/

# 또는 앱 디렉토리에 복사
adb push $QNN_SDK_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so \
    /data/local/tmp/com.example.kpilab/
```

## 4. 디바이스별 HTP 버전

| Chipset | SoC Model | HTP Version | Stub Library |
|---------|-----------|-------------|--------------|
| Snapdragon 8 Gen 3 | SM8650 | V75 | libQnnHtpV75Stub.so |
| Snapdragon 8 Gen 2 | SM8550 | V73 | libQnnHtpV73Stub.so |
| Snapdragon 8 Gen 1 | SM8450 | V69 | libQnnHtpV69Stub.so |
| Snapdragon 888 | SM8350 | V68 | libQnnHtpV68Stub.so |
| Snapdragon 865 | SM8250 | V66 | libQnnHtpV66Stub.so |

## 5. 런타임 탐지

앱은 자동으로 디바이스를 탐지하고 적절한 HTP 버전을 선택합니다:

```kotlin
val runner = NativeRunner()
Log.i("Device", runner.getDeviceInfo())  // "Snapdragon 8 Gen 2 (SM8550) - HTP v73"
Log.i("HTP", "Supported: ${runner.isHtpSupported()}")
Log.i("HTP", "Version: v${runner.getHtpVersion()}")
```

## 6. 모델 변환

```bash
# ONNX → DLC
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter \
    -i models/original/mobilenetv3_small.onnx \
    -o models/converted/mobilenetv3_small.dlc

# 특정 HTP 버전 타겟
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter \
    -i models/original/mobilenetv3_small.onnx \
    -o models/converted/mobilenetv3_small.dlc \
    --htp_socs sm8550
```

## 7. 문제 해결

### "Failed to load backend" 오류

1. jniLibs에 라이브러리가 있는지 확인
2. ABI 필터 확인 (`arm64-v8a`)
3. 라이브러리 권한 확인

### "HTP not supported" 경고

1. 디바이스가 Qualcomm Snapdragon인지 확인
2. 해당 디바이스의 HTP Stub 라이브러리가 있는지 확인
3. GPU-only 모드로 fallback 사용

### Skeleton 로드 실패

```bash
# Skeleton 파일 위치 확인
adb shell ls -la /data/local/tmp/libQnnHtpV*Skel.so

# SELinux 문제 시 (개발 중에만)
adb shell setenforce 0
```

## 8. 파일 구조 (최종)

```
android/app/src/main/
├── jniLibs/
│   └── arm64-v8a/
│       ├── libQnnHtp.so
│       ├── libQnnGpu.so
│       ├── libQnnSystem.so
│       ├── libQnnHtpPrepare.so
│       ├── libQnnHtpV69Stub.so
│       ├── libQnnHtpV73Stub.so
│       └── libQnnHtpV75Stub.so
├── assets/
│   └── mobilenetv3_small.dlc
└── cpp/
    └── (source files)
```
