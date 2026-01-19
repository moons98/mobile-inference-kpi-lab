# Models

벤치마킹용 TFLite 모델 다운로드

## 지원 모델

| 모델 | 입력 크기 | 출력 | 데이터 타입 | 용도 |
|------|----------|------|------------|------|
| MobileNetV2 FP32 | 224x224x3 | 1001 classes | FP32 | 이미지 분류 |
| MobileNetV2 INT8 | 224x224x3 | 1001 classes | INT8 | 이미지 분류 (NPU 최적화) |
| YOLOv8n FP32 | 640x640x3 | Detection boxes | FP32 | 객체 탐지 |
| YOLOv8n INT8 | 640x640x3 | Detection boxes | INT8 | 객체 탐지 (NPU 최적화) |

## 모델 다운로드

### 전체 다운로드

```bash
cd models
python scripts/convert_to_tflite.py --download-all
```

### 개별 다운로드

```bash
# MobileNetV2 (TensorFlow Hub에서 직접 다운로드)
python scripts/convert_to_tflite.py --download-mobilenetv2-fp32
python scripts/convert_to_tflite.py --download-mobilenetv2-int8

# YOLOv8n (Ultralytics에서 다운로드 후 TFLite 변환)
pip install ultralytics
python scripts/convert_to_tflite.py --download-yolov8n-fp32
python scripts/convert_to_tflite.py --download-yolov8n-int8
```

### 다운로드 상태 확인

```bash
python scripts/convert_to_tflite.py --status
python scripts/convert_to_tflite.py --list
```

## 모델 파일 위치

다운로드된 모델은 자동으로 Android assets 폴더에 저장됩니다:

```
android/app/src/main/assets/
├── mobilenetv2_fp32.tflite   # MobileNetV2 FP32 (약 14MB)
├── mobilenetv2_int8.tflite   # MobileNetV2 INT8 (약 3.5MB)
├── yolov8n_fp32.tflite       # YOLOv8n FP32 (약 12MB)
└── yolov8n_int8.tflite       # YOLOv8n INT8 (약 3MB)
```

## FP32 vs INT8

| 항목 | FP32 | INT8 (Quantized) |
|------|------|------------------|
| NPU 성능 | 보통 | 최고 (2-4x 빠름) |
| 모델 크기 | 큼 | 작음 (약 1/4) |
| 정확도 | 높음 | 약간 낮음 |
| 메모리 사용 | 높음 | 낮음 |

NPU에서 최대 성능을 측정하려면 **INT8** 모델을 사용하세요.

## 모델 소스

- **MobileNetV2**: [TensorFlow Hub](https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1)
- **YOLOv8n**: [Ultralytics](https://docs.ultralytics.com/models/yolov8/) - PyTorch에서 TFLite로 변환
