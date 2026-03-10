# On-Device AI Eraser: Generative Object Removal on Android — 실험 설계

---

## 0. 프로젝트 개요

### 0.1 문제정의

모바일 사진 편집에서 "불필요한 객체 제거"는 가장 빈번한 사용자 요구 중 하나이다. 이를 on-device diffusion inpainting으로 구현하려면 두 가지 모델을 연속 실행해야 한다: (1) 객체 인식·마스크 추출을 위한 **YOLO-seg**, (2) 마스크 영역을 배경으로 자연스럽게 채우는 **SD v1.5 Inpainting pipeline**. 전체 파이프라인은 YOLO-seg → ROI crop → Inpainting(VAE Encoder, Text Encoder, Inpainting UNet, VAE Decoder) → blend의 2단 구조로, Inpainting UNet의 iterative denoising loop가 전체 latency의 대부분을 차지한다.

### 0.2 프로젝트 목표

Android 단말에서 **YOLO-seg + SD v1.5 Inpainting** 기반 **AI 객체 지우개** 기능을 on-device로 구현하고, **2단 파이프라인의 실행 경로 및 시스템 병목을 분석하여 모바일 제품 적용 가능성을 평가**한다.

### 0.3 핵심 질문

1. **실행 가능성**: YOLO-seg + Inpainting 2단 파이프라인이 모바일 단말에서 사용자가 수용 가능한 시간(< 10초) 내에 완료되는가?
2. **YOLO-seg 비용**: 객체 인식 + 마스크 추출이 전체 E2E에서 차지하는 비중은 얼마인가? backend별 차이는?
3. **Backend 효과**: Inpainting pipeline의 각 stage(VAE Enc / Text Enc / UNet / VAE Dec)에서 CPU / GPU / NPU의 latency 차이는 얼마인가?
4. **Precision 효과**: FP16 vs W8A8(INT8)에서 latency·memory 트레이드오프는 어떠한가? inpainting 품질은 유지되는가?
5. **병목 구간**: E2E latency에서 Inpainting UNet denoising loop의 비중은 얼마이며, step 수·strength에 따라 어떻게 변화하는가?
6. **ROI 크기 영향**: 작은 객체(~128²) vs 큰 객체(~400²)에서 crop→512² resize→inpaint→uncrop 결과의 blend 품질 차이는?
7. **Sustained 안정성**: 연속 객체 제거 시 thermal throttling으로 인한 성능 열화가 발생하는가?

### 0.4 제품 시나리오

**AI 객체 지우개(AI Object Eraser)** — 갤러리에서 선택한 사진에서 사용자가 tap으로 지정한 객체를 자동 인식·제거하고, 배경을 자연스럽게 채운다.

**사용자 흐름:** 사진 선택 → 객체 tap → YOLO-seg 마스크 추출 → 마스크 확인 → 지우기 실행 → Before/After 비교

#### 핵심 처리 흐름

1. **사용자가 지울 객체를 tap으로 선택** — YOLO-seg가 tap point를 포함하는 segmentation mask를 추출
2. **Mask bounding box 추출 + padding** — 객체만 딱 자르면 주변 문맥이 부족하므로, bbox를 확장하여 배경 포함
3. **Padded ROI crop → 512×512 resize** — NPU 고정 shape 요구사항. Inpainting UNet 입력 크기 통일
4. **Inpainting 수행** — ROI image + mask + prompt를 SD v1.5 Inpainting pipeline에 입력
5. **결과를 원본 ROI 크기로 역변환** — 512×512 → 원래 crop 크기로 resize
6. **Feathering/blending으로 원본에 합성** — mask 경계 alpha feathering으로 자연스러운 합성

> **ROI padding이 중요한 이유**: 객체 bbox만 딱 잘라서 inpainting하면 diffusion이 "무엇으로 메꿔야 하는지" 판단할 주변 문맥이 부족하다. padding으로 배경을 포함시켜야 경계가 자연스럽다.

#### Prompt 전략

자유 텍스트가 아닌 **고정 prompt**를 사용한다. 변인 통제(동일 prompt 반복)와 UX 단순화를 위해 내부적으로 고정한다.

| 용도 | Prompt (내부) |
|------|-------------|
| 객체 제거 | "remove the object and fill the background naturally" |

> 지우개 기능에서 prompt는 사용자에게 노출하지 않는다. 향후 "다른 것으로 대체" 기능 확장 시 prompt 입력을 추가할 수 있다.

#### Strength 파라미터

Inpainting에서 `strength`는 mask 영역을 얼마나 새로 생성할지를 제어한다. actual UNet steps = total_steps × strength.

| strength | 의미 | 실제 UNet steps (총 20 기준) |
|----------|------|---------------------------|
| 0.5 | 원본 흔적 일부 보존, 가벼운 제거 | 10 steps |
| **0.7** | **기본값** — 자연스러운 제거, 주변 배경과 조화 | 14 steps |
| 0.8 | 강한 제거, 거의 새로 생성 | 16 steps |
| 1.0 | 완전 재생성, 경계 어색할 수 있음 | 20 steps |

> strength=1.0은 mask 영역을 순수 noise에서 재생성하므로 주변과의 연속성이 끊길 수 있다. **0.7~0.8이 sweet spot**으로, padding 영역의 배경 정보가 latent에 남아 diffusion이 "주변과 어울리게" 채울 수 있다. 벤치마크에서는 strength를 변인으로 포함하여 latency-quality 트레이드오프를 측정한다.

#### Inpainting 해상도 전략

| 항목 | 결정 | 근거 |
|------|------|------|
| Inpaint 해상도 | **512×512 고정** | NPU는 고정 shape 필수 (QNN compile이 shape별). 다중 해상도 모델 로드는 메모리 초과 |
| ROI 크기 범위 | ~128² ~ ~500² | 512 resize 시 downscale 최소화. 원본 면적의 ~30% 이내 |
| ROI 상한 | 원본 면적의 ~30% | 초과 시 "객체가 너무 큽니다" 안내 |
| 큰 객체 대응 | **TODO: Tiled inpainting** | ROI > 512²일 때 겹치는 타일로 분할 처리. PoC 범위 밖, 제품화 시 구현 |

> 512² 고정 이유: 256/512/768 모델을 각각 QNN compile하여 로드하면 UNet만 ~5.1GB(FP16)로 메모리 초과. session swap은 QNN re-compile에 수십 초 소요. 512 단일 모델이 메모리·latency 모두 현실적이다.

---

## 1. System Architecture

### 1.1 Runtime Stack

```
Android App (Kotlin)
  ├─ Gallery Image Loader → 사진 입력
  ├─ Touch Interaction → tap point 입력
  ├─ CLIP Tokenizer (CPU) → 고정 prompt → token IDs
  └─ ONNX Runtime 1.23.2
       ├─ YOLO-seg session → 객체 인식 + mask 추출
       ├─ CPU EP (ARM Cortex-X3)
       ├─ QNN EP → QNN SDK v2.42.0 → Adreno GPU
       └─ QNN EP → QNN SDK v2.42.0 → Hexagon HTP (NPU)
```

### 1.2 왜 ORT + QNN EP인가

Qualcomm 단말에서 Pure QNN(직접 compile/dispatch)과 ORT + QNN EP(ONNX Runtime이 graph orchestration + QNN으로 dispatch) 두 경로가 있다. 본 프로젝트에서 ORT + QNN EP를 선택한 이유:

- **Graph Partitioning 가시성**: QNN 미지원 op을 CPU EP로 자동 분리하며, partition log로 fallback node를 식별할 수 있다. SD 모델의 CrossAttention, GroupNorm 등 NPU 지원이 불확실한 op 분석에 필수적이다.
- **Hybrid 실행**: NPU + CPU 혼합 실행을 자연스럽게 지원. Pure QNN에서는 불가능하다.
- **Multi-session 관리**: 5개 모델(YOLO-seg + SD 4개)을 독립 ORT session으로 생성하여 stage별 KPI 개별 측정 및 서로 다른 EP 지정이 가능하다.
- **ORT Profiling**: session.run() 내부를 NPU compute / CPU fallback / fence wait / ORT overhead로 분해하여 병목 원인을 구분할 수 있다.

### 1.3 AI Eraser Pipeline

#### Session 관리 전략

AI Eraser pipeline은 **5개 독립 모델**을 사용한다: YOLO-seg 1개 + Inpainting pipeline 4개(VAE Encoder / Text Encoder / Inpainting UNet / VAE Decoder).

| | 전략 A: 5 Session 동시 상주 | 전략 B: Session Swap |
|---|---|---|
| **Peak Memory** | ~1.8~2.1 GB | ~1.0~1.2 GB |
| **Stage 전환 비용** | ~0ms | 수 초~수십 초 (QNN compile) |
| **E2E 영향** | baseline | +10~30초 (실용 불가) |

**선택: 전략 A (5 Session 동시 상주)** — YOLO-seg가 ~30MB로 가벼워서 기존 4 session 대비 메모리 증가 미미. 12GB 단말에서 ~2.1GB는 전체의 18%로 충분하다.

> ORT Session 메모리 = Model Weights + QNN Context (weights의 20~40%) + ORT Buffers. SD 4 session ~1.7~2.0GB에 YOLO-seg ~30MB가 추가되어 **~1.8~2.1GB**가 현실적 예상치이다.

#### Pipeline 흐름 (AI Eraser)

```
[Gallery Image] + [User Tap]
      │               │
      ▼               │
 ┌─ Stage A: Object Detection ─────────────────────────────┐
 │  YOLO-seg ─── session 1 ─── session.run() 1회           │
 │  tap point → 해당 객체의 segmentation mask 추출           │
 └──────────────────────────────────────────────────────────┘
      │
      ▼
 ┌─ Stage B: ROI Preparation (CPU) ────────────────────────┐
 │  mask → bbox 추출 → padding 확장 → 정사각형 보정           │
 │  원본 이미지에서 padded ROI crop (정사각형)                 │
 │  mask도 동일 영역 crop                                    │
 │  ROI image + ROI mask → 512×512 resize                   │
 │  mask → 64×64 resize (latent space 크기)                  │
 └──────────────────────────────────────────────────────────┘
      │
      ▼
 ┌─ Stage C: Inpainting Pipeline ──────────────────────────┐
 │                                                          │
 │  Tokenize (CPU) ─── 고정 prompt → token IDs              │
 │       │                                                  │
 │       ▼                                                  │
 │  Text Encoder ─── session 2 ─── session.run() 1회        │
 │       │                                                  │
 │       ▼                                                  │
 │  ROI image 512² ──→ VAE Encoder ─── session 3 ───        │
 │       │                session.run() 1회                  │
 │       ▼                                                  │
 │  image_latent [1,4,64,64]                                │
 │       │                                                  │
 │  ┌─ masked_image_latent 생성 (CPU + VAE) ──────────┐     │
 │  │  masked_image = ROI_image × (1 − mask)          │     │
 │  │  → VAE Encoder ─── session 3 재사용              │     │
 │  │  → masked_image_latent [1,4,64,64]               │     │
 │  └──────────────────────────────────────────────────┘     │
 │       │                                                  │
 │  Add Noise (CPU) ←── strength                            │
 │       │  (image_latent에 noise 추가 → noisy latent)       │
 │       ▼                                                  │
 │  Inpainting UNet ─── session 4 ───                       │
 │  입력: noisy_latent(4ch) + masked_image_latent(4ch)      │
 │       + mask_64(1ch) = 9ch concat + timestep + text_cond │
 │  session.run() × actual_steps회 반복                      │
 │       │                                                  │
 │       ▼                                                  │
 │  VAE Decoder ─── session 5 ─── session.run() 1회         │
 │       │                                                  │
 │       ▼                                                  │
 │  inpainted ROI image 512²                                │
 └──────────────────────────────────────────────────────────┘
      │
      ▼
 ┌─ Stage D: Composite (CPU) ──────────────────────────────┐
 │  512² → 원본 ROI 크기로 resize                            │
 │  mask 경계 alpha feathering                               │
 │  원본 이미지에 blending 합성                               │
 └──────────────────────────────────────────────────────────┘
      │
      ▼
  [Result Image] → Before/After 표시 + KPI Log
```

> **Inpainting UNet vs base UNet 핵심 차이**: base UNet은 4ch 입력(latent)이지만, Inpainting UNet은 **9ch 입력**(4ch latent + 4ch masked_image_latent + 1ch mask)을 받는다. 별도 모델(`stable-diffusion-v1-5/stable-diffusion-inpainting`)이 필요하다.

### 1.4 Stage별 역할

| Stage | 모델 | 입력 | 출력 | 실행 위치 |
|-------|------|------|------|-----------|
| **YOLO-seg** | YOLOv8-seg | 원본 이미지 [1,3,640,640] | raw detections → NMS → mask decode → tap point 매칭 → segmentation mask | ORT Session + CPU 후처리 |
| **ROI Crop** | — | 원본 이미지 + mask | cropped ROI 512² + cropped mask 512² + mask 64×64 (latent space) | CPU (Kotlin) |
| **Tokenize** | — | 고정 prompt | token IDs [1, 77] | CPU (Kotlin) |
| **Text Encoder** | CLIP ViT-L/14 | token IDs [1, 77] | text embeddings [1, 77, 768] | ORT Session (CPU/GPU/NPU) |
| **VAE Encode** | AutoencoderKL (Encoder) | ROI image [1,3,512,512] | image_latent [1,4,64,64] | ORT Session (CPU/GPU/NPU) |
| **Masked Image Prep** | — (CPU) + VAE Encoder 재사용 | ROI image × (1−mask) → VAE Encode | masked_image_latent [1,4,64,64] | CPU + ORT Session (session 3 재사용) |
| **Add Noise** | — | image_latent + strength | noisy_latent [1,4,64,64] | CPU (scheduler) |
| **Inpaint UNet × M** | UNet2D (9ch) | noisy_latent(4) + masked_image_latent(4) + mask_64(1) + timestep + cond | predicted noise [1,4,64,64] | ORT Session (CPU/GPU/NPU) |
| **Scheduler** | — | predicted noise + current latent | next latent | CPU (math) |
| **VAE Decode** | AutoencoderKL (Decoder) | final latent [1,4,64,64] | inpainted ROI [1,3,512,512] | ORT Session (CPU/GPU/NPU) |
| **Composite** | — | inpainted ROI + 원본 + mask | 최종 합성 이미지 | CPU (Kotlin) |

> M = total_steps × strength. 예: total_steps=20, strength=0.7 → M=14 steps
> Inpainting은 512×512 고정이므로 latent는 항상 [1,4,64,64].
> VAE Encode는 2회 호출: (1) image_latent = VAE_Enc(ROI_image), (2) masked_image_latent = VAE_Enc(ROI_image × (1−mask)). Session 3을 재사용하므로 추가 메모리 없음.

### 1.5 KPI 측정 구조

AI Eraser pipeline에서는 **지우기 1회가 하나의 실험**이며, YOLO-seg + inpainting + blend를 포함한다. 이를 4개 계층으로 측정한다.

#### Level 0 — Product KPI

| 지표 | 정의 | 제품 의미 |
|------|------|----------|
| **Full E2E Latency** | Seg + Inpaint E2E 합산 (사용자 확인 대기 시간 제외) | "전체 연산 소요 시간" |
| **Seg Latency** | YOLO-seg forward + NMS + mask decode 포함 | "객체 인식은 즉각적인가?" |
| **Inpaint E2E** | ROI crop ~ composite 완료 (확인 버튼 후 시작) | "지우기 버튼 후 몇 초?" |
| **Cold Start Time** | 앱 시작 ~ 지우기 가능 상태 (5개 세션 로드) | "앱 열고 얼마나 기다리나?" |

```
시간축:

앱 시작                        Tap        마스크 확인          지우기 완료
  │                             │             │                  │
  ├──── Cold Start ────┤        ├─ Seg ─┤     ├── Inpaint E2E ──┤
  │  session load ×5   │        │       │     │  (roi_crop ~     │
  ▼                    ▼        ▼       ▼     ▼   composite)     ▼
  t=0               Ready     Tap    Mask   Confirm            Done
                                      표시   (사용자 확인)
                              ├ Seg ──┤     ├── Inpaint E2E ───┤
                              │Latency│     │                   │
                              ├───────── Full E2E ─────────────┤
                               (사용자 확인 대기 시간은 제외)
```

**Full E2E** = seg_latency + inpaint_e2e (사용자 확인 대기 시간 제외)

**Seg Latency** = yolo_seg_forward + nms + mask_decode + tap_point_selection

**Inpaint E2E** = roi_crop + masked_image_prep + tokenize + text_enc + vae_enc + add_noise + unet_loop + vae_dec + composite

> Seg latency가 중요한 이유: 사용자가 tap 후 마스크가 즉각 표시되어야 "반응하고 있다"고 느낀다. Seg가 100ms 이내면 이상적.

**Cold Start 시나리오:**

| 구분 | 정의 | 예상 시간 |
|------|------|----------|
| **Cold Start** | 5개 세션 생성 (QNN graph compilation 포함) | ~10~35초 |
| **Warm Start** | 세션 로드 완료 상태 | ~0ms overhead |
| **Context Cache Hit** | QNN 컴파일 결과가 캐시된 상태에서 세션 생성 | Cold의 ~50~70% |

#### Level 1 — Stage Breakdown

각 stage의 소요 시간으로 병목 구간을 식별한다.

| Stage | 예상 비중 |
|-------|----------|
| yolo_seg | ~1~3% |
| roi_crop + resize | < 0.5% |
| tokenize | < 0.1% |
| text_enc | ~1~2% |
| vae_enc (×2: image + masked_image) | ~6~10% |
| **unet_loop** | **~75~85%** |
| vae_dec | ~5~8% |
| composite (resize + blend) | < 1% |

#### Level 2 — UNet Per-Step

Inpainting UNet denoising loop 내부의 매 step을 개별 측정한다.

```
Inpainting UNet Step k:
  ├─ input_create   (9ch latent concat + timestep + conditioning → OnnxTensor)
  ├─ session_run    (Inpainting UNet forward pass)
  ├─ output_copy    (predicted noise → FloatArray)
  └─ scheduler_step (noise prediction → next latent, CPU math)
```

Step별 측정의 의미:
- **Step 간 latency 일정성**: NPU cache warming 효과 확인
- **Thermal 영향**: Step 후반 latency 증가 시 generation 내 thermal throttling 발생
- **Scheduler overhead**: session.run 외 CPU 연산 비중
- **9ch 입력 overhead**: base UNet(4ch) 대비 input_create 시간 차이 확인

#### Level 3 — System Metrics

지우기 실행 중 1초 간격으로 thermal / power / memory를 수집한다.

---

## 2. 실험 환경

### 2.1 Hardware

| 항목 | 값 |
|------|-----|
| Device | Samsung Galaxy S23 Ultra (SM-S918N) |
| SoC | Snapdragon 8 Gen 2 (SM8550) |
| CPU | Cortex-X3 (1) + Cortex-A715 (2) + Cortex-A710 (2) + Cortex-A510 (4) |
| GPU | Adreno 740 |
| NPU | Hexagon NPU (HTP v73) |
| RAM | 12 GB LPDDR5X |

### 2.2 Software

| 항목 | 값 |
|------|-----|
| Runtime | ONNX Runtime 1.23.2 |
| QNN SDK | v2.42.0 (업데이트 필요 시 상위 버전) |
| Android | API 34 |
| Models | SD v1.5 Inpainting (FP16 / W8A8), YOLO-seg (FP32 / INT8) |

### 2.3 실험 조건 통제

| 항목 | 설정 | 근거 |
|------|------|------|
| CPU Governor | `performance` 고정 | Frequency scaling에 의한 latency variance 제거 |
| 화면 | Always-on, 최소 밝기 | Display 부하 최소화 |
| 네트워크 | Airplane mode | Background traffic 제거 |
| 충전 | 벤치마크 중 미충전 | 충전 시 thermal / power 측정 왜곡 방지 |
| Cooldown | 최소 60초 + 온도 ≤35°C 대기 (최대 180초) | 실험 간 thermal state 정규화 |
| Input Image | **고정 테스트 이미지 3장** (ROI 크기별) | 변인 통제 |
| Input Mask | **사전 정의 binary mask** (테스트 이미지별) | ROI 재현성 보장 |
| Prompt | 고정 ("remove the object and fill the background naturally") | 변인 통제 |
| Seed | 고정 random seed | 동일 denoising trajectory 보장 |

> **사전 정의 마스크를 사용하는 이유**: 벤치마크 재현성을 위해 YOLO-seg의 실시간 마스크 대신 미리 생성한 마스크를 inpainting 입력으로 사용한다. YOLO-seg latency는 별도로 측정한다.

### 2.4 Target Models

#### YOLO-seg (객체 인식 + 마스크 추출)

| 항목 | 값 |
|------|-----|
| Model | YOLOv8n-seg (또는 YOLOv8s-seg) |
| Parameters | ~3.4M (nano) / ~11.8M (small) |
| 역할 | tap point 기반 instance segmentation mask 추출 |
| 입력 | 이미지 [1,3,640,640] |
| 출력 | bboxes + class + segmentation masks |
| 주요 Op | Conv2D, C2f, SPPF, Proto(mask head) |

> YOLO-seg는 detection + segmentation을 단일 forward pass로 수행. tap point를 포함하는 mask를 후보에서 선택하여 사용한다.

#### SD v1.5 Inpainting Pipeline

| Component | Parameters | 역할 | 주요 Op |
|-----------|-----------|------|---------|
| **VAE Encoder** | ~34M | ROI image → latent | Conv2D, GroupNorm, Downsample |
| **CLIP Text Encoder** | ~123M | Prompt → text embedding | Attention, LayerNorm, Linear |
| **Inpainting UNet** | ~860M | Iterative denoising (9ch 입력) | CrossAttention, GroupNorm, Conv2D, SiLU |
| **VAE Decoder** | ~83M | Latent → inpainted ROI image | Conv2D, GroupNorm, Upsample |

> Inpainting UNet은 base UNet과 동일 아키텍처이나 **입력 채널이 9ch** (4ch latent + 4ch masked_image_latent + 1ch mask). `stable-diffusion-v1-5/stable-diffusion-inpainting`에서 제공하며 별도 ONNX export 필요.

#### 모델 크기 (Precision별)

| Component | FP32 | FP16 | W8A8 (INT8) |
|-----------|------|------|-------------|
| YOLO-seg (nano) | ~13 MB | — | ~4 MB |
| VAE Encoder | ~130 MB | ~65 MB | ~35 MB |
| Text Encoder | ~490 MB | ~245 MB | ~125 MB |
| Inpainting UNet | ~3.4 GB | ~1.7 GB | ~860 MB |
| VAE Decoder | ~330 MB | ~165 MB | ~85 MB |
| **Total** | **~4.4 GB** | **~2.2 GB** | **~1.1 GB** |

> YOLO-seg는 FP32/INT8만 사용 (FP16은 별도 검증 필요). SD 모델 총합 대비 무시 가능한 크기.

### 2.5 Precision 전략

**SD Inpainting Pipeline:**

| Precision | Weight | Activation | 용도 | 근거 |
|-----------|--------|------------|------|------|
| FP32 | FP32 | FP32 | CPU baseline | CPU 전용 참고값, 타 precision 정확도 기준 |
| **FP16** | FP16 | FP16 | **Production 기본** | NPU/GPU에서 기본 실행 precision. 품질 기준선 + 실용적 메모리 (~2.2GB) |
| W8A8 | INT8 | INT8 | Aggressive 옵션 | 최소 메모리, 최대 throughput. inpainting 품질 손실 가능 |

**YOLO-seg:**

| Precision | 용도 | 근거 |
|-----------|------|------|
| FP32 | 정확도 기준 | mask 품질 기준선 |
| INT8 | Production | 경량 모델에서 INT8 효과 검증 |

> **W8A16 제외 사유**: ORT의 static_quant에는 W8A16이 지원되지 않음

### 2.6 테스트 데이터 설계

벤치마크 재현성을 위해 **고정 테스트 이미지 + 사전 정의 마스크**를 사용한다.

| 테스트 케이스 | 이미지 설명 | 대상 객체 | ROI 크기 (padding 포함) | 실험 의미 |
|-------------|-----------|----------|----------------------|----------|
| **Small** | 풍경 + 원거리 사람 | 사람 (~5% 면적) | ~128×128 → 512² resize | 작은 객체, upscale 비율 높음 |
| **Medium** | 실내 + 물건 | 테이블 위 컵 (~15% 면적) | ~256×256 → 512² resize | 일반적 사용 케이스 |
| **Large** | 거리 + 차량 | 차량 (~30% 면적) | ~400×400 → 512² resize | 큰 객체, padding 영역 적음 |

```
assets/
  test_images/
    scene_small.jpg          # 작은 객체 포함
    scene_medium.jpg         # 중간 객체 포함
    scene_large.jpg          # 큰 객체 포함
  test_masks/
    mask_small.png           # 사전 생성 binary mask
    mask_medium.png
    mask_large.png
  test_tap_points.json       # YOLO-seg E2E 측정용 tap 좌표
```

> ROI 크기별로 inpaint latency 자체는 동일(항상 512²)하지만, **crop/resize/blend CPU 비용**과 **최종 blend 품질**(upscale/downscale ratio)이 달라진다.

---

## 3. Phase 1 — Single Erase Profiling

### 3.1 목적

- AI Eraser 2단 파이프라인의 **stage별 latency breakdown** 측정
- **YOLO-seg** 단독 latency 및 backend별 비교
- **Inpainting pipeline** stage별 profiling (VAE Enc / Text Enc / UNet loop / VAE Dec)
- Backend × Precision 조합별 **peak performance** 비교
- Inpainting UNet per-step latency의 **선형성** 확인
- **Cold / Warm / Cache Hit** 세 가지 시나리오에서의 시작 비용 측정
- ROI 크기별 **blend 품질** 시각 평가

### 3.2 실행 방식

#### 3.2.1 Cold Start 측정 (config당 1회)

```
앱 프로세스 kill → 재시작
  → 5개 세션 순차 생성 (각각 timing)
  → YOLO-seg + SD 4개 세션 cold start 시간 기록

QNN context cache 활성화 후:
  앱 프로세스 kill → 재시작
  → 5개 세션 순차 생성 (cache hit timing)
  → 총 cache hit 시간 기록
```

#### 3.2.2 YOLO-seg Profiling (별도 측정)

YOLO-seg는 inpainting과 독립적으로 프로파일링한다.

```
세션 로드 완료 (warm state)
Warmup: 5회 inference (결과 제외)

for trial in 1..20:
    run YOLO-seg (고정 테스트 이미지)
    record: inference_ms, input_create_ms, nms_ms, mask_decode_ms, output_process_ms
    // cooldown 불필요 (단일 forward pass, 발열 미미)

Input: 고정 테스트 이미지 3장 × 고정 tap point
```

> YOLO-seg는 1회 forward pass (~10~50ms 예상)로 매우 빠르므로 trial 수를 20회로 늘려 variance를 줄인다.

#### 3.2.3 Inpaint Profiling (config당 5회)

```
세션 로드 완료 (warm state)
Warmup: 2회 full inpainting (결과 제외)

for trial in 1..5:
    run inpainting pipeline (사전 정의 mask 사용)
    모든 stage/step 개별 timing
    cooldown (≤35°C 대기)

Input: 고정 테스트 이미지 + 사전 정의 mask + 고정 prompt + 고정 seed
```

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| Trials | 5 | inpainting 1회가 충분히 긴 연산이므로 5회로 통계 확보 |
| Warmup | 2회 | Session initialization, cache warming |
| Input | 사전 정의 mask | 동일 ROI로 변인 통제 |
| Prompt | 고정 | "remove the object and fill the background naturally" |
| ORT Profiling | ON (trial별) | Stage 내부 NPU/CPU/fence 분해 |
| Cooldown | 매 trial 사이 | Thermal 누적 방지 |

### 3.3 실험 구성

#### 실험 1: Backend × Precision (Inpainting)

Inpainting UNet을 중심으로 backend와 precision 효과를 측정한다. SD 4개 모델 모두 동일 backend/precision으로 실행.

| # | Backend | Precision | Steps | Strength | ROI Size |
|---|---------|-----------|-------|----------|----------|
| 1 | CPU | FP32 | 20 | 0.7 | Medium |
| 2 | GPU | FP16 | 20 | 0.7 | Medium |
| 3 | NPU | FP16 | 20 | 0.7 | Medium |
| 4 | NPU | W8A8 | 20 | 0.7 | Medium |

> Config 1 (CPU FP32)은 baseline. 생성 시간이 매우 길 수 있으므로 1회만 실행하여 참고값으로 사용할 수 있다.
> ROI Size는 Medium(~256²)으로 고정. Strength 0.7 → actual UNet steps = 14.

#### 실험 2: YOLO-seg Backend × Precision

| # | Backend | Precision | 입력 해상도 |
|---|---------|-----------|-----------|
| 1 | CPU | FP32 | 640×640 |
| 2 | GPU | FP32 | 640×640 |
| 3 | NPU | FP32 | 640×640 |
| 4 | NPU | INT8 | 640×640 |

> YOLO-seg 단독 측정. mask 품질(mAP)도 함께 비교하여 INT8 양자화에 의한 mask 정확도 손실 확인.

#### 실험 3: Step 수 × Strength 영향 (최적 backend 고정)

| # | Steps | Strength | Actual Steps | Backend | Precision | ROI Size |
|---|-------|----------|-------------|---------|-----------|----------|
| 1 | 20 | 0.5 | 10 | NPU | W8A8 | Medium |
| 2 | 20 | 0.7 | 14 | NPU | W8A8 | Medium |
| 3 | 20 | 0.8 | 16 | NPU | W8A8 | Medium |
| 4 | 20 | 1.0 | 20 | NPU | W8A8 | Medium |
| 5 | 50 | 0.7 | 35 | NPU | W8A8 | Medium |

> Inpainting에서 strength 의미: 0.7~0.8이 sweet spot (주변 배경과 자연스러운 조화), 1.0은 경계 어색할 수 있음. latency-quality 트레이드오프 측정.

#### 실험 4: ROI 크기 영향 (최적 backend 고정)

| # | ROI Size | Crop → Resize | Backend | Precision | Steps | 측정 초점 |
|---|----------|---------------|---------|-----------|-------|----------|
| 1 | Small (~128²) | 128→512 (4×) | NPU | W8A8 | 20 | upscale 비율 높음, blend 품질 |
| 2 | Medium (~256²) | 256→512 (2×) | NPU | W8A8 | 20 | 일반 케이스 |
| 3 | Large (~400²) | 400→512 (1.3×) | NPU | W8A8 | 20 | padding 부족, 문맥 제한 |

> Inpaint 자체 latency는 동일(항상 512²)하므로, 이 실험의 주 목적은 **crop/resize/blend CPU 비용 차이**와 **최종 blend 품질의 시각적 평가**이다.

#### 실험 5: Stage별 Backend 혼합 (Optional)

| # | YOLO-seg | Text Encoder | UNet | VAE Decoder | 근거 |
|---|----------|-------------|------|-------------|------|
| 1 | NPU | NPU | NPU | NPU | 전체 NPU |
| 2 | NPU | CPU | NPU | GPU | UNet만 NPU, VAE는 GPU가 유리할 경우 |
| 3 | CPU | NPU | NPU | CPU | YOLO+VAE CPU, UNet NPU |

> 실험 1, 2 결과에서 stage별 최적 backend가 다를 경우에만 실행.

### 3.4 측정 Metrics

1.5절에서 정의한 4계층(Product KPI / Stage Breakdown / UNet Per-Step / System Metrics)으로 측정한다. 데이터는 Appendix C의 CSV 스키마로 기록한다. 추가로:

- **ORT Profiling** 활성화: session.run 내부를 NPU / CPU / Fence / ORT overhead로 분해
- **Graph partitioning** 결과: YOLO-seg 및 Inpainting UNet 각각의 NPU coverage(%) 수집
- **Inpainting 품질 평가**: ROI 크기별·strength별 blend 결과 시각 평가 (경계 자연스러움, 색상 일관성)

---

## 4. Phase 2 — Sustained Erase Test

### 4.1 목적

- **연속 객체 제거 시 성능 열화** 측정: 사용자가 한 사진에서 여러 객체를 연속으로 지우는 시나리오
- **Thermal throttling** 유무 검증: Inpainting UNet loop는 수 초간 지속 연산이므로 발열 가능성 높음
- **Power consumption** 비교: 지우기 1회당 에너지 비용
- **메모리 안정성**: 반복 inpainting 시 memory leak 여부 (특히 ROI crop/blend 과정의 Bitmap 관리)

### 4.2 실행 방식

```
Model Load: 5개 세션 생성 (1회)
Warmup: 2회 full inpainting
Main:
  for trial in 1..10:
      run full inpainting pipeline (사전 정의 mask 사용)
      record Erase Summary + UNet Step Detail
      record system metrics (thermal/power/memory) at 지우기 전후 + 1초 간격
      NO cooldown (연속 실행)
HTP Mode: sustained_high
```

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| Trials | 10 | 연속 10회 지우기로 thermal saturation 관찰 |
| Cooldown | 없음 | 연속 부하에서의 실제 사용 패턴 재현 |
| HTP Mode | sustained_high | 연속 실행에 적합한 성능/전력 균형 |
| Input | 고정 이미지 × 고정 mask × 고정 seed | Trial 간 동일 workload 보장 |
| YOLO-seg | 포함하지 않음 (사전 정의 mask 사용) | Inpainting 열화에 집중 |
| ORT Profiling | OFF | 연속 실행에서 profiling overhead 제거 |

Phase 2의 분석 포인트는 **trial 간 비교**이다. Trial 1의 UNet per-step mean과 Trial 10의 UNet per-step mean을 비교하여, 연속 사용 시 사용자가 체감할 성능 열화를 정량화한다.

### 4.3 실험 구성

Phase 1에서 확인된 최적 configuration으로 실행:

| # | Backend | Precision | Steps | Strength | ROI Size | 근거 |
|---|---------|-----------|-------|----------|----------|------|
| 1 | NPU | W8A8 | 20 | 0.7 | Medium | 예상 최적 config |
| 2 | NPU | FP16 | 20 | 0.7 | Medium | Precision 비교 |
| 3 | GPU | FP16 | 20 | 0.7 | Medium | Backend 비교 |

### 4.4 측정 Metrics

| 카테고리 | Metric | 설명 |
|---------|--------|------|
| **Latency Drift** | Trial 1 Inpaint E2E / Trial 10 Inpaint E2E / Drift% | Stable(±5%) / Slight(5~15%) / Throttling(>15%) |
| **UNet Drift** | Trial 1 per-step mean / Trial 10 per-step mean | UNet 구간 집중 열화 분석 |
| **Thermal** | Start Temp / Peak Temp / Slope (°C/trial) | 온도 상승 추세 |
| **Power** | Avg Power (mW) / Energy per Erase (mJ) | 지우기 1회당 에너지 |
| **Memory** | RSS after Trial 1 / Trial 10 / Delta | Memory leak 감지 (Bitmap 관리 포함) |

---

## 5. 분석 프레임워크

### 5.1 Phase 1 분석 관점

- **2단 파이프라인 분해**: Full E2E = YOLO-seg + Inpaint E2E. 각각의 비중과 병목 구간 식별
- **YOLO-seg 분석**: backend별 inference latency, INT8 양자화에 의한 mask 정확도 영향
- **Inpaint Stage Breakdown**: Stacked bar chart로 stage별 비중 표시. UNet이 75~85%+로 지배적일 것으로 예상
- **Backend 비교**: 동일 precision에서 CPU / GPU / NPU의 Inpainting UNet per-step latency 비교
- **Precision 비교**: FP16 vs W8A8의 latency·메모리·coverage 차이, inpainting 품질 (blend 경계 시각 평가)
- **Strength 영향**: strength별 actual steps 변화와 E2E latency·blend 품질 관계. 0.7~0.8 vs 1.0에서 경계 자연스러움 비교
- **ROI 크기 영향**: inpaint latency는 동일하므로 crop/blend CPU 비용 + blend 품질이 주요 분석 대상

### 5.2 Phase 2 분석 관점

- **Latency over trials**: 시계열 그래프로 thermal throttling onset 확인
- **UNet step-level 열화**: Trial 1의 step latency vs Trial 10의 step latency
- **Thermal ceiling**: 몇 번째 trial에서 온도가 포화되는가
- **Bitmap 메모리 안정성**: ROI crop/blend 반복 시 Bitmap 누수 여부

### 5.3 Discussion 포인트

1. **Inpainting UNet 병목과 최적화 경로**: UNet이 Inpaint E2E의 75~85%일 때, step 수 감소(Turbo/LCM)나 model distillation의 예상 효과. 9ch 입력이 base UNet 대비 추가하는 overhead 정량화
2. **Strength-Quality 트레이드오프**: strength=0.7 vs 0.8 vs 1.0에서 blend 경계 자연스러움 차이. strength 감소 시 actual steps 감소로 E2E가 선형으로 줄어드는가?
3. **ROI 크기-품질 트레이드오프**: 작은 ROI(128²→512²)는 과도한 upscale로 연산 낭비, 큰 ROI(400²→512²)는 padding 부족으로 문맥 제한. 최적 ROI 크기 범위 제안
4. **YOLO-seg의 E2E 영향**: seg latency가 ~10~50ms라면 전체 E2E에서 무시 가능. 단, mask 정확도가 inpainting 품질에 미치는 영향(잘못된 mask → 잘못된 제거)
5. **Blend 품질 분석**: alpha feathering 범위(px)에 따른 경계 자연스러움, 색상 일관성. ROI 크기별 최적 feathering 범위
6. **Cold Start 최적화**: QNN context cache 효과, 5개 세션 순차/병렬 로드, 앱 백그라운드 유지 전략
7. **FP16 vs W8A8**: W8A8의 메모리 절감 대비 latency 변화. 특히 inpainting에서 W8A8 양자화가 blend 경계 품질에 미치는 영향
8. **Multi-session Memory**: 5 session 동시 상주 시 실측 peak memory vs 예상 ~1.8~2.1GB. YOLO-seg 추가의 실질적 메모리 영향
9. **NPU Op Coverage**: Inpainting UNet의 CrossAttention, GroupNorm NPU 지원 여부. 9ch 입력 Conv의 NPU 호환성 확인
10. **Tiled Inpainting (TODO)**: ROI > 512²인 큰 객체 대응. 겹치는 512² 타일로 분할 + seam blending 전략. 제품화 시 구현 우선순위
11. **제품 적용 가능성 판단 기준**:

| 기준 | Target | 근거 |
|------|--------|------|
| Seg Latency | < 100ms | tap 후 마스크가 즉각 표시되어야 함 |
| Inpaint E2E | < 8초 | "지우기" 기능으로 수용 가능한 대기 시간 |
| Full E2E | < 10초 | seg + inpaint + blend 전체 |
| Cold Start (cache hit) | < 10초 | 앱 시작 후 대기 허용 범위 |
| Peak Memory | < 4 GB | 12GB 단말에서 OS + 앱 + 모델 공존 |
| Thermal | 연속 5회 지우기 후 throttling < 15% | 실사용 패턴에서 안정성 |
| Blend 품질 | 경계 미인지 | 일반 사용자가 제거 흔적을 인지하지 못하는 수준 |

---

## 6. 실험 실행 순서

```
0. 사전 검증 — QAI Hub Profiling
   ├── SD v1.5 Inpainting 4개 모델 각각 compile & profile (cloud)
   │   (Inpainting UNet, VAE Encoder는 별도 ONNX export)
   ├── YOLO-seg 모델 compile & profile
   ├── NPU op coverage 확인 (특히 Inpainting UNet 9ch 입력, CrossAttention)
   ├── FP16 / W8A8 latency·memory 비교
   └── 결과로 on-device 실행 가능성 판단 → Go/No-Go

1. PC 파이프라인 PoC
   ├── Python ONNX Runtime으로 SD v1.5 inpainting pipeline 동작 확인
   ├── Inpainting UNet + VAE Encoder ONNX export
   ├── 수동 mask → ROI crop → inpaint → blend 파이프라인 검증
   ├── YOLO-seg → tap-guided mask 선택 로직 검증
   ├── Strength별 inpainting 품질 확인 (0.5 / 0.7 / 0.8 / 1.0)
   ├── ROI 크기별 blend 품질 확인
   └── W8A8 모델 변환 및 inpainting 정확도 검증

2. Android 앱 구현
   ├── feature/ai-eraser 브랜치 생성
   ├── YOLO-seg session + mask 추출 로직 구현
   ├── MaskProcessor (tap→mask선택, bbox+padding, ROI crop) 구현
   ├── InpaintPipeline (9ch UNet orchestration) 구현
   ├── BlendProcessor (resize + alpha feathering + composite) 구현
   ├── 터치 기반 객체 선택 UI 구현
   ├── OrtRunner 5-session 관리 확장
   ├── Stage breakdown KPI 측정 포인트 추가
   └── Before/After UI + 벤치마크 대시보드 구현

3. Phase 1 — Single Erase Profiling
   ├── 실험 1: Backend × Precision — Inpainting (4 configs)
   ├── 실험 2: YOLO-seg Backend × Precision (4 configs)
   ├── 실험 3: Step 수 × Strength 영향 (5 configs)
   ├── 실험 4: ROI 크기 영향 (3 configs)
   └── 결과 분석 → Phase 2 config 선별

4. Phase 2 — Sustained Erase Test
   ├── 선별된 2~3 configs, 각 10회 연속 지우기
   └── 결과 분석 → thermal drift, power, memory stability

5. Report
   ├── 2단 파이프라인 stage breakdown 분석 (YOLO-seg + inpaint 병목 가시화)
   ├── Backend × Precision × Steps × Strength × ROI Size 종합 비교
   ├── Blend 품질 평가 (strength별, ROI 크기별)
   ├── Sustained 안정성 평가
   ├── 제품 적용 가능성 판단
   └── 최적화 로드맵 제안 (Tiled Inpainting, LCM/Turbo 등)
```

---

## Appendix A: QAI Hub 사전 검증 항목

QAI Hub에서 아래 항목을 cloud profiling으로 사전 확인한다.

| 항목 | 확인 내용 | 판단 기준 |
|------|----------|----------|
| Inpainting UNet Coverage | 9ch 입력 Conv, CrossAttention, GroupNorm HTP 지원 여부 | Coverage ≥ 85% |
| Inpainting UNet per-step | HTP에서 1 step latency | < 500ms (20 step 기준 Inpaint E2E < 12s) |
| YOLO-seg latency | HTP에서 single inference | < 50ms |
| Memory Peak | 5개 session 동시 로드 시 peak RSS | < 4 GB (12GB 단말 기준) |
| W8A8 정확도 | FP16 대비 inpainting 품질 | blend 경계 시각 평가 |
| INT8 YOLO-seg | FP32 대비 mask 품질 | mAP 비교 |
| Cold Start | 5개 모델 순차 compile/load | < 35s (1회성, 허용 범위 넓음) |

## Appendix B: 앱 아키텍처 변경 사항

### 기존 img2img 앱에서 변경되는 부분

| Component | 기존 (SD img2img) | 변경 (AI Eraser) |
|-----------|-------------------|-----------------|
| **Input** | Gallery image + preset prompt + strength | Gallery image + **tap point** + 고정 prompt |
| **Model** | 4개 SD session | **5개 session** (YOLO-seg + SD Inpainting 4개) |
| **Inference** | img2img orchestration | **2단**: YOLO-seg → ROI crop → inpainting → blend |
| **UNet** | Base UNet (4ch 입력) | **Inpainting UNet (9ch 입력)** |
| **Output** | 전체 이미지 스타일 변환 | **객체 제거 + 배경 채움** (원본 일부만 수정) |
| **UI** | Preset prompt 선택 + strength 조절 | **터치로 객체 선택** + 마스크 확인 + 지우기 |
| **Preview** | Progressive preview (UNet 중간) | **제거** (지우개에서 중간 preview 불필요) |

### 신규/교체 파일

| 파일 | 상태 | 역할 |
|------|------|------|
| `YoloSegRunner.kt` | **신규** | YOLO-seg session 관리, mask 추출, tap-guided 선택 |
| `MaskProcessor.kt` | **신규** | bbox 추출, padding, ROI crop, mask resize |
| `InpaintPipeline.kt` | **신규** | 9ch Inpainting UNet orchestration |
| `BlendProcessor.kt` | **신규** | ROI uncrop, alpha feathering, composite |
| `TouchImageView.kt` | **신규** | 터치 이벤트 처리, 마스크 오버레이 표시 |
| `SdPipeline.kt` | **교체** | img2img → inpainting으로 변경 |
| `Tokenizer.kt` | **유지** | CLIP tokenizer (동일) |
| `Scheduler.kt` | **유지** | EulerDiscrete scheduler (동일) |
| `ImagePreprocessor.kt` | **확장** | ROI crop/uncrop, mask resize, blend 로직 추가 |
| `OrtRunner.kt` | **유지** | Multi-session 관리 (5개로 확장) |
| `BenchmarkRunner.kt` | **리팩토링** | Erase trial 루프, YOLO-seg 분리 측정, Phase 1/2 |
| `MainActivity.kt` | **리팩토링** | 터치 UI, 마스크 확인, 지우기 흐름 |

### 앱 설정 옵션

| 옵션 | 값 | 설명 |
|------|-----|------|
| Backend (SD) | CPU / GPU / NPU | SD 4개 session 공통 |
| Backend (YOLO) | CPU / GPU / NPU | YOLO-seg session 독립 설정 가능 |
| Precision (SD) | FP32 / FP16 / W8A8 | Inpainting 모델 파일 선택 |
| Precision (YOLO) | FP32 / INT8 | YOLO-seg 모델 파일 선택 |
| Steps | 10 / 20 / 30 / 50 | Scheduler total step 수 |
| Strength | 0.5 / 0.7 / 0.8 / 1.0 | inpainting 강도 |
| ROI Padding | 1.2× / 1.5× / 2.0× | bbox 대비 padding 비율 |
| QNN Context Cache | ON / OFF | Cold start 최적화 |

### 모델 파일 관리

SD 모델은 수백 MB ~ 1GB+이므로 external storage에서 로드한다. YOLO-seg는 경량이므로 assets 또는 external storage.

```bash
# SD Inpainting 모델 배포 (adb push)
adb push models/vae_encoder_fp16.onnx /sdcard/sd_models/
adb push models/text_encoder_fp16.onnx /sdcard/sd_models/
adb push models/inpaint_unet_fp16.onnx /sdcard/sd_models/
adb push models/vae_decoder_fp16.onnx /sdcard/sd_models/

# YOLO-seg 모델
adb push models/yolov8n_seg.onnx /sdcard/sd_models/
adb push models/yolov8n_seg_int8.onnx /sdcard/sd_models/

# Tokenizer 리소스 (assets에 포함, ~2MB)
assets/
  vocab.json        # CLIP vocabulary
  merges.txt        # BPE merge rules
  special_tokens.txt

# 테스트 데이터 (assets에 포함)
assets/
  test_images/      # 고정 테스트 이미지 3장
  test_masks/       # 사전 정의 binary mask 3장
  test_tap_points.json  # YOLO-seg 측정용 tap 좌표
```

## Appendix C: CSV 출력 스키마

지우기 1회에 대해 네 종류의 record를 생성한다.

**Record Type 1: Erase Summary** (지우기 1회당 1행)

```
trial_id, prompt, steps, strength, actual_steps, roi_size, backend_sd, precision_sd, backend_yolo, precision_yolo,
full_e2e_ms, inpaint_e2e_ms,
yolo_seg_ms, roi_crop_ms, tokenize_ms, text_enc_ms, vae_enc_ms, masked_img_prep_ms, unet_total_ms, vae_dec_ms, composite_ms,
unet_per_step_mean_ms, unet_per_step_p95_ms, scheduler_overhead_ms,
peak_memory_mb, start_temp_c, end_temp_c, avg_power_mw
```

> Inpaint profiling(3.2.3)에서 사전 정의 mask를 사용할 경우, `yolo_seg_ms = 0`으로 기록하고 `full_e2e_ms = inpaint_e2e_ms`가 된다. YOLO-seg latency는 Record Type 4에서 별도 수집한다.

**Record Type 2: UNet Step Detail** (지우기 1회당 N행, step마다 1행)

```
trial_id, step_index, input_create_ms, session_run_ms, output_copy_ms,
scheduler_step_ms, step_total_ms,
thermal_c, power_mw
```

**Record Type 3: Cold Start** (앱 시작당 1행)

```
start_type (cold/warm/cache_hit),
yolo_seg_load_ms, vae_enc_load_ms, text_enc_load_ms, unet_load_ms, vae_dec_load_ms, total_load_ms,
peak_memory_after_load_mb
```

**Record Type 4: YOLO-seg Detail** (seg 측정당 1행)

```
trial_id, test_image, backend, precision,
inference_ms, input_create_ms, nms_ms, mask_decode_ms, output_process_ms,
mask_count, selected_mask_area_pct
```

## Appendix D: 기존 실험과의 연속성

본 프로젝트는 기존 Mobile NPU Inference KPI Lab의 프레임워크를 확장한다.

| 요소 | 기존 (YOLOv8 → SD img2img) | 확장 (AI Eraser) |
|------|---------------------------|-----------------|
| KPI 수집 | KpiCollector (thermal/power/memory) | **그대로 재활용** |
| Batch 실험 | ExperimentConfig + JSON | 확장 (ROI size, YOLO config 추가) |
| CSV Export | BenchmarkRunner | 확장 (seg/crop/blend stage 컬럼 추가) |
| 분석 스크립트 | parse_logs.py, plot_kpi.py | 확장 (2단 pipeline 시각화 추가) |
| QAI Hub 검증 | profile_qai_hub.py | 확장 (YOLO-seg + Inpainting UNet 지원) |
| Backend 비교 | CPU / GPU / NPU | 동일 (YOLO + SD 독립 설정 가능) |
| Phase 1/2 구조 | Single / Sustained | 적용 (단건 지우기 / 연속 지우기) |
| YOLO-seg | (원래 YOLOv8 detection으로 시작) | **segmentation으로 복귀** — 원점 회귀하여 mask 추출에 활용 |
