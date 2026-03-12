# Weights Inventory

weights/ 디렉토리의 모델 파일 목록 및 용도 정리.
gitignore 대상이므로 git에 포함되지 않음. 이 문서로 재현 가능하도록 기록.

> 모든 ONNX 모델은 **opset 18** 로 export.

---

## 공유 컴포넌트 (VAE / Text Encoder)

SD v1.5 base와 LCM 모두 동일한 VAE Encoder/Decoder 및 Text Encoder를 사용한다.
현재 `weights/sd_v1.5/onnx/`에 배치.

### VAE Encoder

| 파일 | 크기 | 생성 방법 | 상태 | compile | profile |
|---|---|---|---|---|---|
| `vae_encoder_fp32.onnx` | 130MB | `scripts/sd/export_sd_to_onnx.py` | ✅ | `jgkymvrop` (76MB) | `jpev3o815` **201ms** / 82MB |
| `vae_encoder_qai_int8.onnx` | 131MB | QAI Hub W8A8 quantize (`j5w9nwxjp`) | ✅ | `jglkr418p` (39MB) | `jpvwxq9jg` **54ms** / 41MB |

### Text Encoder (CLIP ViT-L/14, 768-dim)

| 파일 | 크기 | 생성 방법 | 상태 | compile | profile |
|---|---|---|---|---|---|
| `text_encoder_fp32.onnx` | 470MB | `scripts/sd/export_sd_to_onnx.py` | ✅ | `jp2mnxjm5` (237MB) | `jgz7k28kp` **5.9ms** / 2MB |
| `text_encoder_w8a16.onnx` | 470MB | qai-hub-models SD v1.5 export (AIMET W8A16) | ✅ | `j561l217p` (156MB) | `jp18x2v2g` **3.3ms** / 2MB |

### VAE Decoder

| 파일 | 크기 | 생성 방법 | 상태 | compile | profile |
|---|---|---|---|---|---|
| `vae_decoder_fp32.onnx` | 189MB | `scripts/sd/export_sd_to_onnx.py` | ✅ | `j5q2o0wm5` (109MB) | `j5w9nw16p` **440ms** / 119MB |
| `vae_decoder_qai_int8.onnx` | 189MB | QAI Hub W8A8 quantize (`jg94e08v5`) | ✅ | `j561l2d0p` (57MB) | `jgjl4dwxp` **111ms** / 69MB |
| `vae_decoder_w8a16.onnx` | 189MB | qai-hub-models SD v1.5 export (AIMET W8A16) | ✅ | `jpev3ov05` (62MB) | `j57d327l5` **270ms** / 3MB |

### Calibration Data (`weights/sd_v1.5/calib_data/`)

| 파일 | 크기 | 용도 | 생성 방법 |
|---|---|---|---|
| `calib_vae_encoder.npz` | 192MB | VAE Encoder (64 samples, 3×512×512 float32) | `export_sd_to_onnx.py --generate-calib-data` |
| `calib_text_encoder.npz` | 39KB | Text Encoder (64 samples, 77 int64) | 동일 |
| `calib_vae_decoder.npz` | 4MB | VAE Decoder (64 samples, 4×64×64 float32) | 동일 |
| `calib_unet.npz` | 19MB | UNet (64 samples, 4×64×64 noisy latents + text embeddings) | `export_sd_lcm_unet.py --generate-calib-data` |

---

## SD v1.5 UNet (`weights/sd_v1.5/onnx/`)

4ch input (standard txt2img/img2img). VAE/Text Encoder는 상단 공유 컴포넌트 참조.

### UNet — Base (SD v1.5)

859M params. 4ch input (latent only). Baseline 비교용.

| 파일 | 크기 | 생성 방법 | 상태 | compile | profile |
|---|---|---|---|---|---|
| `unet_base_fp32.onnx` + `.data` | 0.8MB + 3.3GB | `export_sd_lcm_unet.py --export base` | ✅ | `jpryj609g` (1,651MB) |  |
| `unet_base_w8a16.onnx` + `.data` | 1.7MB + 3.3GB | qai-hub-models SD v1.5 export (AIMET W8A16) | ✅ | `jgjl4dl8p` (842MB) | `jgdvlnzeg` **113ms** / 2MB |

### UNet — LCM (SD v1.5 + LCM-LoRA fused)

LCM-LoRA (`latent-consistency/lcm-lora-sdv1-5`, 67.5M params) 를 base UNet에 fuse한 모델.
2-8 step 추론, guidance_scale 1.0-2.0, LCMScheduler 사용.

| 파일 | 크기 | 생성 방법 | 상태 | compile | profile |
|---|---|---|---|---|---|
| `unet_lcm_fp32.onnx` + `.data` | 0.8MB + 3.3GB | `export_sd_lcm_unet.py --export lcm` | ✅ | `jp87vq9q5` (1,651MB) | `jp2mnx0x5` **333ms** / 1,793MB |
| `unet_lcm_w8a16.onnx` + `.data` | 0.8MB + 3.3GB | `export_sd_lcm_unet_w8a16.py` (AIMET W8A16) | ✅ | `jgz7k21xp` (834MB) | `jp3m2nkng` **214ms** / 902MB |

---

## INT8 품질 평가 요약

`scripts/sd/quant_report_sd.py` (8 samples, component-level FP32 vs INT8)

이전 export (inpainting 시절) 기준 결과. opset 18 재export 후 재평가 필요.

| Component | Variant | CosSim | RMSE | PSNR | Grade |
|---|---|---|---|---|---|
| vae_encoder | QAI Hub W8A8 | 0.984 | 0.154 | 28.6dB | Marginal |
| text_encoder | QAI Hub W8A8 | **-0.048** | 1.295 | 28.1dB | **Poor** |
| vae_decoder | QAI Hub W8A8 | 0.968 | 0.153 | 19.4dB | Marginal |

> UNet base/LCM: QAI Hub quantize OOM (exit 137). RunPod에서 로컬 양자화 필요.
> 상세 리포트: `exp_outputs/quantization/sd_quantization.txt`

---

## QAI Hub Compile 참고

- Target: Samsung Galaxy S23 (Family), Snapdragon 8 Gen 2
- QAIRT 버전: 앱 QNN SDK와 매칭 필요 (현재 ORT 1.24.3 = QNN build v2.42.0)
- 컴파일 옵션: `--target_runtime precompiled_qnn_onnx --qairt_version 2.42`
- stub .onnx 내부 `ep_cache_context` 경로를 배포명에 맞게 수정 필요
- 대용량 ONNX (external data): 디렉토리 형식으로 업로드 (.onnx + .data를 같은 폴더에)

### W8A16 QNN Context Binary (qai-hub-models export)

`qai-outputs/sd_v1.5/` 에 저장. `qai_hub_models.models.stable_diffusion_v1_5.export` 사용.

- 소스: `qai-hub-models==0.47.0`, AIMET W8A16 양자화 (weight 8bit, activation FP16)
- UNet base / Text Encoder / VAE Decoder: `qai_hub_models.models.stable_diffusion_v1_5.export` 사용
- UNet LCM: `export_sd_lcm_unet_w8a16.py` — PyTorch에서 LCM-LoRA fuse 후 ONNX export → AIMET W8A16 → QAI Hub compile
- 컴파일: `--target_runtime precompiled_qnn_onnx --qairt_version 2.42`, QAIRT 2.42.0.251225135753_193295
- calibration data 불필요 (weight-only quantization)
- Linux(RunPod)에서 실행 필요 (`aimet_onnx`가 Linux 전용)

| 폴더 | 컴포넌트 | bin 크기 | job ID |
|---|---|---|---|
| `compile_w8a16_text_encoder_j561l217p/` | Text Encoder | 156MB | `j561l217p` |
| `compile_w8a16_unet_jgjl4dl8p/` | UNet (base) | 842MB | `jgjl4dl8p` |
| `compile_w8a16_unet_lcm_jgz7k21xp/` | UNet (LCM) | 834MB | `jgz7k21xp` |
| `compile_w8a16_vae_decoder_jpev3ov05/` | VAE Decoder | 62MB | `jpev3ov05` |
