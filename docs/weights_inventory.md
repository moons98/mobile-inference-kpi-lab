# Weights Inventory

weights/ 디렉토리의 모델 파일 목록 및 용도 정리.
gitignore 대상이므로 git에 포함되지 않음. 이 문서로 재현 가능하도록 기록.

## SD v1.5 Inpainting (`weights/sd_v1.5_inpaint/`)

### VAE Encoder (`onnx/vae_encoder_*`)

| 파일 | 크기 | 생성 방법 | 상태 | .bin |
|---|---|---|---|---|
| `vae_encoder_fp32.onnx` | 130MB | `scripts/sd/export_sd_to_onnx.py` | ✅ | `j57dwqdl5` (63MB) |
| `vae_encoder_int8_qdq.onnx` | 33MB | `scripts/sd/quant_runpod.py` | ✅ | — |
| `vae_encoder_qai_int8.onnx` + `.data` | 0.4MB + 131MB | QAI Hub quantize (`jgjlm961p`) | ✅ | `j5q2kj245` (30MB) |

### Text Encoder (`onnx/text_encoder_*`)

| 파일 | 크기 | 생성 방법 | 상태 | .bin |
|---|---|---|---|---|
| `text_encoder_fp32.onnx` | 470MB | `scripts/sd/export_sd_to_onnx.py` | ✅ | `jpydw9d7p` (218MB) |
| `text_encoder_qai_int8.onnx` + `.data` | 0.6MB + 470MB | QAI Hub quantize (`jpev1q085`) | ✅ | `jglkzjk8p` (102MB) |

> INT8 QDQ 로컬 양자화는 미생성 (품질 파괴 위험)

### UNet (`onnx/unet_*`)

859M params, FP32=3.2GB, FP16=1.6GB. 9ch input (4 latent + 4 masked image + 1 mask).

| 파일 | 크기 | 생성 방법 | 상태 | .bin |
|---|---|---|---|---|
| `unet_fp32.onnx` + `.data` | 0.8MB + 3.2GB | `scripts/sd/export_sd_to_onnx.py` | ✅ opset 18 | `jgov0yex5` (1.5GB) |
| — | — | QAI Hub quantize (`jgdv9w3rg`) | ❌ OOM | — |

### VAE Decoder (`onnx/vae_decoder_*`)

| 파일 | 크기 | 생성 방법 | 상태 | .bin |
|---|---|---|---|---|
| `vae_decoder_fp32.onnx` | 189MB | `scripts/sd/export_sd_to_onnx.py` | ✅ | `jpx1jw11g` (91MB) |
| `vae_decoder_int8_qdq.onnx` | 48MB | `scripts/sd/quant_runpod.py` | ✅ | — |
| `vae_decoder_qai_int8.onnx` + `.data` | 0.5MB + 189MB | QAI Hub quantize (`j57dwm1n5`) | ✅ | `j561jk10p` (46MB) |

### Calibration Data (`calib_data/`)

| 파일 | 크기 | 용도 |
|---|---|---|
| `calib_vae_encoder.npz` | 192MB | VAE Encoder 양자화 calibration (64 samples, 3×512×512 float32) |
| `calib_text_encoder.npz` | 39KB | Text Encoder 양자화 calibration (64 samples, 77 int64) |
| `calib_unet.npz` | 23MB | UNet 양자화 calibration (64 samples, multi-input) |
| `calib_vae_decoder.npz` | 4MB | VAE Decoder 양자화 calibration (64 samples, 4×64×64 float32) |

## YOLO-seg (`weights/yolov8n_seg/onnx/`)

| 파일 | 크기 | 생성 방법 | 상태 | .bin |
|---|---|---|---|---|
| `yolov8n-seg_fp32.onnx` | 13MB | `scripts/yolo/export_yolo_seg.py` | ✅ | `j5mz2jqqp` FP16 (6.3MB) |
| `yolov8n-seg_int8_qdq_noh.onnx` | 6.9MB | `scripts/yolo/export_yolo_seg.py` | ✅ mAP@50 -1.8% | `j561jkn0p` (3.9MB) |
| `yolov8n-seg_int8_qdq.onnx` | 3.7MB | `scripts/yolo/export_yolo_seg.py` | ❌ mAP 0.0 | — |
| `yolov8n-seg_qai_int8.onnx` + `.data` | 0.3MB + 13MB | QAI Hub quantize (`j57dwmwq5`) | ❌ mAP 0.0 | — |

### 삭제 가능

- `yolov8n-seg_int8_qdq.onnx` — 전체 INT8, detection 파괴
- `yolov8n-seg_qai_int8.onnx` + `.data` — QAI Hub W8A8, detection 파괴

## 디바이스 배포 구성

### Precompiled (권장)
stub .onnx + .bin만 배포. cold start 212ms.
```
/sdcard/sd_models/yolov8n-seg.onnx  (434B stub)
/sdcard/sd_models/yolov8n-seg.bin   (8MB QNN binary)
```

### 기존 방식 (FP32 ONNX + on-device 컴파일)
```
/sdcard/sd_models/yolov8n-seg.onnx       (13MB)
/sdcard/sd_models/vae_encoder_fp32.onnx   (130MB)
/sdcard/sd_models/text_encoder_fp32.onnx  (470MB)
/sdcard/sd_models/unet_fp32.onnx          (0.8MB)
/sdcard/sd_models/unet_fp32.onnx.data     (3.2GB)
/sdcard/sd_models/vae_decoder_fp32.onnx   (189MB)
```

## QAI Hub Compile 참고

- Target: Samsung Galaxy S23 (Family), Snapdragon 8 Gen 2
- QAIRT 버전: 앱 QNN SDK와 매칭 필요 (현재 ORT 1.24.3 = QNN build v2.42.0)
- 컴파일 옵션: `--target_runtime precompiled_qnn_onnx --qairt_version 2.42`
- stub .onnx 내부 `ep_cache_context` 경로를 배포명에 맞게 수정 필요
- 대용량 ONNX (external data): 디렉토리 형식으로 업로드 (.onnx + .data를 같은 폴더에)
