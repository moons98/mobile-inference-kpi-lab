# Models

모델 변환 및 분석 스크립트

## 디렉토리 구조

```
models/
├── original/       # 원본 ONNX 모델
├── converted/      # QNN DLC 변환 모델
├── optimized/      # 그래프 최적화 모델
└── scripts/        # 변환/분석 스크립트
```

## 스크립트 사용법

### 1. 모델 다운로드 및 변환

```bash
# MobileNetV3-Small 다운로드 및 DLC 변환
python scripts/convert_to_dlc.py --download-mobilenetv3

# 수동 변환
python scripts/convert_to_dlc.py -i original/model.onnx -o converted/model.dlc
```

### 2. Op 분석

```bash
# NPU 호환성 분석
python scripts/analyze_ops.py original/mobilenetv3_small.onnx

# JSON 출력
python scripts/analyze_ops.py original/model.onnx --json analysis.json
```

### 3. 그래프 변환

```bash
# 모든 최적화 적용
python scripts/graph_transform.py original/model.onnx -o optimized/model.onnx --all

# 개별 최적화
python scripts/graph_transform.py original/model.onnx \
    --fix-batch 1 \
    --replace-hardswish \
    --remove-identity \
    --validate
```

## 필수 패키지

```bash
pip install onnx numpy onnxruntime
```
