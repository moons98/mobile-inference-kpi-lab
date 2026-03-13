# Mobile Inference KPI Lab — On-Device Text-to-Image

Snapdragon 기반 Android 단말에서 **SD v1.5 text-to-image 파이프라인**의 on-device 실행 가능성을 정량 분석하는 프로젝트.

- **Runtime**: ONNX Runtime 1.24.3 + QNN Execution Provider (QAIRT 2.42)
- **Model**: SD v1.5 (Text Encoder + UNet + VAE Decoder), LCM-LoRA variant 포함
- **Pipeline**: Prompt → Text Encode → Denoising Loop(UNet × N) → VAE Decode → 512×512 Image
- **Target Device**: Samsung Galaxy S23 (Snapdragon 8 Gen 2, Hexagon HTP NPU)

---

## 핵심 실험

### Phase 1 — Single-Run Feasibility

조건별 latency/memory/power/quality baseline 확보 (5 trials, burst mode).

| ID | 실험 | 목적 |
|---|---|---|
| P1-1 | SD v1.5 (20 step) vs LCM-LoRA (4 step) | variant 비교 |
| P1-2 | SD v1.5 step sweep (20/30/50) | latency-step 관계 |
| P1-3 | LCM-LoRA step sweep (2/4/8) | few-step 하한 탐색 |
| P1-4 | Backend × Precision (NPU/GPU/CPU, FP16/W8A8) | execution path 비교 |
| P1-5 | Mixed Precision (component-level FP16/W8A8/MIXED_PR) | UNet/VAE 개별 영향 |
| P1-6 | Parallel Init vs Sequential Init | cold start 최적화 |

### Phase 2 — Sustained Feasibility

연속 10회 생성으로 thermal/power/latency drift 검증 (cooldown 없음, sustained_high mode).

| ID | 실험 |
|---|---|
| P2-1 | SD v1.5 sustained (FP16 vs W8A8) |
| P2-2 | LCM-LoRA sustained (FP16 vs W8A8, 4 steps) |

---

## SD v1.5 파이프라인

```
Prompt (text)
  │
  ▼ [Tokenize — CPU]
int32[1, 77]
  │
  ▼ [Text Encoder — NPU]   (CLIP ViT-L/14, 470MB FP32 / 156MB W8A16)
float32[1, 77, 768]  (text embeddings)
  │
  ▼ [Initial Noise — CPU]  (Gaussian, 1×4×64×64)
  │   + Scheduler.setTimesteps(N)
  │
  ▼ [UNet Denoising Loop × N steps — NPU]
  │   SD v1.5: EulerDiscrete, CFG 7.5, N=20–50
  │   LCM-LoRA: DDIM-style, CFG 1.0, N=2–8
  │   UNet: 859M params (1,651MB FP16 / 1,151MB MIXED_PR)
  │
  ▼ [VAE Decoder — NPU]    (189MB FP32 / 62MB W8A16)
float32[1, 3, 512, 512]
  │
  ▼ [Postprocess — CPU]
Bitmap 512×512
```

---

## 배포 모델 구성

컴파일 완료 모델 목록: `scripts/deploy/deploy_config.json`
상세 스펙·품질: `docs/weights_inventory.md`

| 컴포넌트 | Variant | Bin 크기 | CosSim |
|---|---|---|---|
| text_encoder | FP16 | 237MB | — |
| text_encoder | W8A16 | 156MB | 0.9849 |
| vae_encoder | FP16 | 76MB | — |
| vae_encoder | W8A8 | 39MB | 0.9810 |
| vae_decoder | FP16 | 109MB | — |
| vae_decoder | W8A8 | 57MB | 0.9827 |
| vae_decoder | W8A16 | 62MB | **0.9999** |
| unet_base | FP16 | 1,651MB | — |
| unet_base | MIXED_PR | 1,151MB | **0.9968** |
| unet_lcm | FP16 | 1,651MB | — |
| unet_lcm | MIXED_PR | 1,150MB | 0.9875 |

---

## 빠른 시작

### 1. 모델 배포

```bash
# QAI Hub에서 컴파일된 모델을 device로 push
bash scripts/deploy/push_models_to_device.sh

# 필요 시 tokenizer asset 생성
python scripts/deploy/extract_tokenizer_assets.py
```

### 2. 빌드 및 설치

```bash
cd android
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 3. 벤치마크 실행

1. 실험 전 체크리스트 확인 (`docs/experiment_design.md` §3.4)
2. 앱에서 실험 세트 선택 (`experiment_sets_txt2img.json`)
3. RUN 버튼 → 완료 후 EXPORT CSV

### 4. 데이터 분석

```bash
# device에서 CSV 수집
adb pull /sdcard/Android/data/com.example.kpilab/files/Documents/ ./exp_outputs/runs/

# 분석
python analysis/parse_txt2img_csv.py exp_outputs/runs/
```

---

## 프로젝트 구조

```
mobile-inference-kpi-lab/
├── android/                          # Android 벤치마크 앱
│   └── app/src/main/
│       ├── java/.../kpilab/
│       │   ├── Txt2ImgPipeline.kt    # SD v1.5 / LCM-LoRA 파이프라인
│       │   ├── Scheduler.kt          # EulerDiscrete + LCM DDIM-style
│       │   ├── Tokenizer.kt          # CLIP BPE tokenizer
│       │   ├── ImagePreprocessor.kt  # VAE 출력 → Bitmap (precision 분기)
│       │   ├── OrtRunner.kt          # ONNX Runtime 세션 래퍼
│       │   ├── BenchmarkRunner.kt    # 실험 루프 + CSV export
│       │   ├── BenchmarkConfig.kt    # 실험 파라미터 (precision, variant, steps 등)
│       │   ├── InputMode.kt          # SdPrecision / SdComponent / ModelVariant enum
│       │   └── KpiCollector.kt       # Thermal / Power / Memory 수집
│       └── assets/
│           └── experiment_sets_txt2img.json  # 배치 실험 정의
├── scripts/
│   ├── sd/
│   │   ├── export_sd_to_onnx.py          # Text Encoder / VAE export
│   │   ├── export_sd_lcm_unet.py         # UNet base / LCM export
│   │   └── eval_sd_quant_quality.py      # 양자화 품질 평가 (CosSim)
│   └── deploy/
│       ├── deploy_config.json            # 배포 모델 목록
│       └── push_models_to_device.sh      # 모델 push 스크립트
├── analysis/
│   └── parse_txt2img_csv.py              # 실험 결과 파싱·리포트
├── exp_outputs/
│   ├── quantization/
│   │   ├── sd_quant_quality.txt          # 양자화 품질 리포트
│   │   └── sd_quant_eval_jobs.json       # QAI Hub inference job 기록
│   └── runs/                             # on-device 측정 결과 CSV
├── weights/                              # 모델 파일 (gitignore)
└── docs/
    ├── experiment_design.md              # 실험 설계 (이 프로젝트의 핵심 문서)
    ├── experiment_report.md              # 결과 리포트 (실측 후 채워짐)
    └── weights_inventory.md              # 모델 파일 목록 및 품질 현황
```

---

## 측정 KPI

| KPI | 단위 | 설명 |
|---|---|---|
| **E2E Latency** | ms | Tokenize ~ VAE Decode 전 구간 |
| **Text Enc** | ms | CLIP 텍스트 인코딩 |
| **UNet total** | ms | N step 전체 denoising |
| **UNet per-step** | ms | step당 평균 |
| **VAE Decode** | ms | 잠재 변수 → 이미지 |
| **Cold Start** | ms | Session load + first inference |
| **Peak Memory** | MB | 생성 중 최대 메모리 |
| **Avg Power** | mW | 생성 구간 평균 전력 |
| **Thermal Drift** | °C/trial | 연속 실행 시 온도 상승률 |

---

## 문서

| 문서 | 설명 |
|---|---|
| [experiment_design.md](docs/experiment_design.md) | 실험 설계, 모델 준비, 체크리스트 |
| [experiment_report.md](docs/experiment_report.md) | 결과 리포트 (실측 후 업데이트) |
| [model_optimization.md](docs/model_optimization.md) | 양자화 전략 설계 및 품질 검증 과정 (기술 내러티브) |
| [weights_inventory.md](docs/weights_inventory.md) | 모델 파일 목록, compile/profile/품질 현황 |

## 참고 자료

- [ONNX Runtime QNN EP Documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
- [Latent Consistency Models (LCM-LoRA)](https://latent-consistency-models.github.io/)
