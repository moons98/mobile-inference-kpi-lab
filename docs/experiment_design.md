# On-Device Photo Assist: Generative Image Editing on Android — 실험 설계

---

## 0. 프로젝트 개요

### 0.1 문제정의

모바일 생성형 이미지 편집 기능은 사용자 경험 관점에서 낮은 지연과 단말 내 실행이 중요하지만, diffusion 계열 모델은 높은 연산량과 메모리 요구량 때문에 실제 제품 적용 시 병목이 발생한다. Stable Diffusion의 img2img pipeline은 VAE Encoder, Text Encoder, UNet (반복 실행), VAE Decoder의 4-stage 구조로 구성되며, UNet의 iterative denoising loop가 전체 latency의 대부분을 차지한다.

### 0.2 프로젝트 목표

Android 단말에서 Stable Diffusion v2.1 기반 **이미지 스타일 변환(img2img)** 기능을 on-device로 구현하고, **실행 경로 및 시스템 병목을 분석하여 모바일 제품 적용 가능성을 평가**한다.

### 0.3 핵심 질문

1. **실행 가능성**: SD v2.1의 img2img pipeline(4-stage)이 모바일 단말에서 사용자가 수용 가능한 시간 내에 완료되는가?
2. **Backend 효과**: VAE Encoder / Text Encoder / UNet / VAE Decoder 각각에서 CPU / GPU / NPU의 latency 차이는 얼마인가?
3. **Precision 효과**: FP16 vs W8A16에서 latency·memory 트레이드오프는 어떠한가? 생성 품질은 유지되는가?
4. **병목 구간**: E2E latency에서 UNet denoising loop의 비중은 얼마이며, step 수·strength·해상도에 따라 어떻게 변화하는가?
5. **Sustained 안정성**: 연속 생성 시 thermal throttling으로 인한 성능 열화가 발생하는가?

### 0.4 제품 시나리오

**이미지 스타일 변환(Image Style Transfer)** — 갤러리에서 선택한 사진을 preset prompt가 지정하는 스타일로 변환한다. 원본 구조(구도, 형태)를 유지하면서 화풍·분위기를 변경하는 img2img 방식이다.

**사용자 흐름:** 사진 선택 → preset prompt 탭 → strength 조절 (기본 0.7) → 생성 → Before/After 비교

#### Preset Prompt

자유 텍스트가 아닌 **사전 정의된 5개 preset 버튼**으로 구성한다. 변인 통제(동일 prompt 반복), tokenizer 검증 범위 축소, UX 단순화, 제품 현실성(Samsung Photo Editor 스타일 필터와 유사) 등의 이점이 있다.

| Preset | Prompt (내부) | 사용자 레이블 |
|--------|-------------|-------------|
| 1 | "sunset beach with golden light" | 노을 해변 |
| 2 | "snowy winter cityscape at night" | 겨울 도시 야경 |
| 3 | "lush green forest with sunlight" | 초록 숲 |
| 4 | "futuristic neon city" | 미래 도시 |
| 5 | "soft watercolor painting style" | 수채화 |

#### Strength 파라미터

img2img에서 `strength`는 원본 이미지를 얼마나 변형할지를 제어한다. actual UNet steps = total_steps × strength.

| strength | 의미 | 실제 UNet steps (총 20 기준) |
|----------|------|---------------------------|
| 0.3 | 원본 구조 강하게 유지, 미세 스타일 변환 | 6 steps |
| 0.5 | 균형 — 구조 유지 + 스타일 적용 | 10 steps |
| **0.7** | **기본값** — 스타일 강하게 적용, 구조 대략 유지 | 14 steps |
| 1.0 | 원본 무시, 순수 text-to-image와 동일 | 20 steps |

> strength가 낮을수록 UNet step 수가 줄어 E2E가 빨라지지만, 변환 효과도 약해진다. 벤치마크에서는 strength를 변인으로 포함하여 latency-quality 트레이드오프를 측정한다.

---

## 1. System Architecture

### 1.1 Runtime Stack

```
Android App (Kotlin)
  ├─ Gallery Image Loader → Static image input
  ├─ CLIP Tokenizer (CPU) → Text prompt → token IDs
  └─ ONNX Runtime 1.23.2
       ├─ CPU EP (ARM Cortex-X3)
       ├─ QNN EP → QNN SDK v2.42.0 → Adreno GPU
       └─ QNN EP → QNN SDK v2.42.0 → Hexagon HTP (NPU)
```

### 1.2 왜 ORT + QNN EP인가

Qualcomm 단말에서 Pure QNN(직접 compile/dispatch)과 ORT + QNN EP(ONNX Runtime이 graph orchestration + QNN으로 dispatch) 두 경로가 있다. 본 프로젝트에서 ORT + QNN EP를 선택한 이유:

- **Graph Partitioning 가시성**: QNN 미지원 op을 CPU EP로 자동 분리하며, partition log로 fallback node를 식별할 수 있다. SD 모델의 CrossAttention, GroupNorm 등 NPU 지원이 불확실한 op 분석에 필수적이다.
- **Hybrid 실행**: NPU + CPU 혼합 실행을 자연스럽게 지원. Pure QNN에서는 불가능하다.
- **Multi-session 관리**: 4개 모델을 독립 ORT session으로 생성하여 stage별 KPI 개별 측정 및 서로 다른 EP 지정이 가능하다.
- **ORT Profiling**: session.run() 내부를 NPU compute / CPU fallback / fence wait / ORT overhead로 분해하여 병목 원인을 구분할 수 있다.

### 1.3 Generation Pipeline

#### Session 관리 전략

img2img pipeline은 4개 독립 모델(VAE Encoder / TextEnc / UNet / VAE Decoder)을 순차 실행한다.

| | 전략 A: 4 Session 동시 상주 | 전략 B: Session Swap | 전략 C: Swap + Context Cache |
|---|---|---|---|
| **Peak Memory** | ~1.7~2.0 GB | ~1.0~1.2 GB | ~1.0~1.2 GB |
| **Stage 전환 비용** | ~0ms | 수 초~수십 초 (QNN compile) | 수백 ms~수 초 |
| **E2E 영향** | baseline | +10~30초 (실용 불가) | +1~5초 |

**선택: 전략 A (4 Session 동시 상주)** — 5~10초 E2E 목표에서 swap 비용은 허용 불가. 12GB 단말에서 ~2.0GB는 전체의 17%로 충분하다. Progressive preview에서 UNet loop 중간에 VAE Decoder를 즉시 호출해야 하므로 swap은 불가능하다.

> ORT Session 메모리 = Model Weights + QNN Context (weights의 20~40%) + ORT Buffers. 4 session 동시 상주 시 weights 합산 ~1.2GB에 QNN context 등이 추가되어 **~1.7~2.0GB**가 현실적 예상치이다. Peak Memory는 반드시 실측하여 확인한다.

#### Pipeline 흐름 (img2img)

```
[Gallery Image] ────────────────→ [Prompt Select] + [Strength Slider]
      │                                    │                │
      ▼                                    ▼                │
  Resize/Preprocess (CPU)            Tokenize (CPU)         │
      │                                    │                │
      ▼                                    ▼                │
  VAE Encoder ─── session 1 ───     Text Encoder ─── session 2 ───
  session.run() 1회                 session.run() 1회       │
      │                                    │                │
      ▼                                    │                │
  image_latent                             │                │
      │                                    │                │
      ▼                                    │                │
  Add Noise (CPU)  ←── strength ───────────┘────────────────┘
  (scheduler.add_noise with t_start)
      │
      ▼
  noisy_latent (초기값 = image_latent + noise)
      │
      ▼
 ┌─ Scheduler Loop ────────────────────────────────────────────┐
 │  actual_steps = total_steps × strength                       │
 │  for step in 1..actual_steps:                                │
 │    UNet ─── session 3 ─── session.run() actual_steps회 반복  │
 │    scheduler.step() (CPU)                                    │
 │    if preview_enabled && step ∈ preview_steps:               │
 │      VAE Decode → Progressive Preview                        │
 └──────────────────────────────────────────────────────────────┘
      │
      ▼
  VAE Decoder ─── session 4 ─── session.run() 1회 (final)
      │
      ▼
  [Result Image] → KPI Log
```

> text-to-image와의 핵심 차이: 초기 latent가 순수 noise가 아닌 **원본 이미지를 VAE Encode한 latent에 noise를 추가한 것**이며, UNet의 실제 step 수가 `total_steps × strength`로 줄어든다.

#### Progressive Preview

UNet denoising loop 중 특정 step에서 중간 latent를 VAE Decoder에 통과시켜 preview 이미지를 생성한다. 앱에서 **toggle로 ON/OFF 가능**하며, 벤치마크에서는 양쪽을 측정하여 overhead를 정량화한다.

| 항목 | Preview OFF | Preview ON (every 5 steps, 3회) |
|------|------------|-----------------|
| VAE Decode 횟수 | 1회 (final) | 4회 (preview 3 + final 1) |
| 추가 비용 | 0 ms | ~900 ms (VAE 3회 × ~300ms) |
| E2E 영향 | baseline | +10~15% |
| **First feedback** | E2E 완료 시 | **~1.8초** |

### 1.4 Stage별 역할

| Stage | 모델 | 입력 | 출력 | 실행 위치 |
|-------|------|------|------|-----------|
| **Preprocess** | — | gallery image | resized image [1,3,H,W] | CPU (Kotlin) |
| **VAE Encode** | AutoencoderKL (Encoder) | image [1,3,H,W] | image latent [1,4,H/8,W/8] | ORT Session (CPU/GPU/NPU) |
| **Add Noise** | — | image latent + strength | noisy latent [1,4,H/8,W/8] | CPU (scheduler) |
| **Tokenize** | — | prompt string | token IDs [1, 77] | CPU (Kotlin) |
| **Text Encoder** | CLIP ViT-L/14 | token IDs [1, 77] | text embeddings [1, 77, 1024] | ORT Session (CPU/GPU/NPU) |
| **UNet × M** | UNet2D | latent [1,4,H/8,W/8] + timestep + cond [1,77,1024] | predicted noise [1,4,H/8,W/8] | ORT Session (CPU/GPU/NPU) |
| **Scheduler** | — | predicted noise + current latent | next latent | CPU (math) |
| **VAE Decode** | AutoencoderKL (Decoder) | final latent [1,4,H/8,W/8] | image [1,3,H,W] | ORT Session (CPU/GPU/NPU) |

> M = total_steps × strength. 예: total_steps=20, strength=0.7 → M=14 steps
> SD v2.1 latent space: 1/8 해상도. latent [1,4,64,64] → output 512×512, latent [1,4,48,48] → output 384×384

### 1.5 KPI 측정 구조

SD pipeline에서는 **generation 1회가 그 자체로 하나의 실험**이며, 내부에 20~50개 step의 시계열 데이터를 포함한다. 이를 4개 계층으로 측정한다.

#### Level 0 — Product KPI

| 지표 | 정의 | 제품 의미 |
|------|------|----------|
| **E2E Latency** | Generate 버튼 ~ 최종 결과 표시 | "몇 초 걸리나?" |
| **First Preview Latency** | Generate 버튼 ~ 첫 preview 표시 | "언제 뭔가 보이나?" |
| **Cold Start Time** | 앱 시작 ~ Generate 가능 상태 (4개 세션 로드 + QNN compilation) | "앱 열고 얼마나 기다리나?" |

```
시간축:

앱 시작                                          Generate 버튼
  │                                                  │
  ├────── Cold Start ──────┤                          ├── First Preview ──┤── 나머지 steps ──┤
  │  session load ×4       │                          │                   │                  │
  │  (QNN compile)         │                          │                   │   Final Image    │
  ▼                        ▼                          ▼                   ▼       ▼          ▼
  t=0                   Ready                      Generate           Preview  Result      Done
```

**E2E** = preprocess + vae_enc + tokenize + text_enc + unet_loop + vae_dec(final)

**First Preview** = preprocess + vae_enc + tokenize + text_enc + unet_steps(1~K) + vae_dec(preview)

> First Preview가 중요한 이유: E2E가 10초여도 First Preview가 2초이면 사용자는 "빠르다"고 느낀다.

**Cold Start 시나리오:**

| 구분 | 정의 | 예상 시간 |
|------|------|----------|
| **Cold Start** | 첫 세션 생성 (QNN graph compilation 포함) | ~10~30초 |
| **Warm Start** | 세션 로드 완료 상태 | ~0ms overhead |
| **Context Cache Hit** | QNN 컴파일 결과가 캐시된 상태에서 세션 생성 | Cold의 ~50~70% |

#### Level 1 — Stage Breakdown

각 stage의 소요 시간으로 병목 구간을 식별한다.

| Stage | 예상 비중 |
|-------|----------|
| preprocess | < 0.1% |
| vae_enc | ~3~5% |
| tokenize | < 0.1% |
| text_enc | ~1~2% |
| unet_loop | **~80~90%** |
| vae_dec | ~5~8% |

#### Level 2 — UNet Per-Step

UNet denoising loop 내부의 매 step을 개별 측정한다.

```
UNet Step k:
  ├─ input_create   (latent + timestep + conditioning → OnnxTensor)
  ├─ session_run    (UNet forward pass)
  ├─ output_copy    (predicted noise → FloatArray)
  └─ scheduler_step (noise prediction → next latent, CPU math)
```

Step별 측정의 의미:
- **Step 간 latency 일정성**: NPU cache warming 효과 확인
- **Thermal 영향**: Step 후반 latency 증가 시 generation 내 thermal throttling 발생
- **Scheduler overhead**: session.run 외 CPU 연산 비중

#### Level 3 — System Metrics

generation 실행 중 1초 간격으로 thermal / power / memory를 수집한다.

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
| QAI Hub | SD v2.1 pre-compiled models (W8A16) |

### 2.3 실험 조건 통제

| 항목 | 설정 | 근거 |
|------|------|------|
| CPU Governor | `performance` 고정 | Frequency scaling에 의한 latency variance 제거 |
| 화면 | Always-on, 최소 밝기 | Display 부하 최소화 |
| 네트워크 | Airplane mode | Background traffic 제거 |
| 충전 | 벤치마크 중 미충전 | 충전 시 thermal / power 측정 왜곡 방지 |
| Cooldown | 최소 60초 + 온도 ≤35°C 대기 (최대 180초) | 실험 간 thermal state 정규화 |
| Input | 고정 이미지 + 고정 prompt | 변인 통제 |
| Seed | 고정 random seed | 동일 denoising trajectory 보장 |

### 2.4 Target Models

#### Stable Diffusion v2.1 Pipeline

| Component | Parameters | 역할 | 주요 Op |
|-----------|-----------|------|---------|
| **VAE Encoder** | ~34M | Pixel image → latent | Conv2D, GroupNorm, Downsample |
| **CLIP Text Encoder** | ~340M | Prompt → text embedding | Attention, LayerNorm, Linear |
| **UNet** | ~860M | Iterative denoising | CrossAttention, GroupNorm, Conv2D, SiLU |
| **VAE Decoder** | ~83M | Latent → pixel image | Conv2D, GroupNorm, Upsample |

> VAE Encoder는 AutoencoderKL의 encoder 부분으로, img2img에서 입력 이미지를 latent space로 변환하는 데 사용한다. QAI Hub에는 Decoder만 제공되므로 **Encoder는 별도 ONNX export**가 필요하다.

#### 모델 크기 (Precision별)

| Component | FP32 | FP16 | W8A16 |
|-----------|------|------|-------|
| VAE Encoder | ~130 MB | ~65 MB | ~35 MB |
| Text Encoder | ~490 MB | ~245 MB | ~130 MB |
| UNet | ~3.4 GB | ~1.7 GB | ~900 MB |
| VAE Decoder | ~330 MB | ~165 MB | ~90 MB |
| **Total** | **~4.4 GB** | **~2.2 GB** | **~1.15 GB** |

### 2.5 Precision 전략

| Precision | Weight | Activation | 용도 | 근거 |
|-----------|--------|------------|------|------|
| **FP16** | FP16 | FP16 | 품질 기준선 | 최고 생성 품질, 메모리 ~2.1GB |
| **W8A16** | INT8 | FP16 | **Production 기본** | QAI Hub 기본 설정. 품질 유지 + 메모리 ~1.1GB |
| W8A8 | INT8 | INT8 | Aggressive 옵션 | 최소 메모리, 품질 손실 가능 |

SD 모델에서 W8A16이 표준인 이유:
- UNet의 CrossAttention, GroupNorm은 activation quantization에 민감
- W8A8로는 생성 이미지에 artifact가 발생하기 쉬움
- W8A16은 weight 압축으로 메모리/대역폭 이점을 얻으면서 연산 정밀도 유지

---

## 3. Phase 1 — Single Generation Profiling

### 3.1 목적

- SD v2.1 pipeline의 **stage별 latency breakdown** 측정
- **E2E latency**와 **First Preview latency** 동시 측정
- Backend × Precision 조합별 **peak performance** 비교
- UNet per-step latency의 **선형성** 확인
- **Cold / Warm / Cache Hit** 세 가지 시나리오에서의 시작 비용 측정

### 3.2 실행 방식

#### 3.2.1 Cold Start 측정 (config당 1회)

```
앱 프로세스 kill → 재시작
  → 4개 세션 순차 생성 (각각 timing)
  → 총 cold start 시간 기록

QNN context cache 활성화 후:
  앱 프로세스 kill → 재시작
  → 4개 세션 순차 생성 (cache hit timing)
  → 총 cache hit 시간 기록
```

#### 3.2.2 Generation Profiling (config당 5회 × preview ON/OFF)

각 config에 대해 **preview OFF 5회 + preview ON 5회**를 실행한다. Preview OFF가 기본 baseline이며, ON과의 차이로 preview overhead를 정량화한다.

```
세션 로드 완료 (warm state)
Warmup: 2회 full generation (결과 제외)

[Preview OFF] — baseline E2E 측정
  for trial in 1..5:
      run generation (preview_enabled = false)
      모든 stage/step 개별 timing
      cooldown (≤35°C 대기)

[Preview ON] — First Preview latency + overhead 측정
  for trial in 1..5:
      run generation (preview_enabled = true, preview_interval = 5)
      모든 stage/step 개별 timing + preview VAE timing
      cooldown (≤35°C 대기)

Input: 고정 이미지 + 고정 prompt + 고정 seed
```

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| Trials | 5 × 2 (OFF/ON) | Preview 효과를 분리 측정 |
| Warmup | 2회 | Session initialization, cache warming |
| Input | 고정 | 동일 denoising trajectory로 변인 통제 |
| Preview interval | 5 steps (of 20) | 25% 간격 (step 5, 10, 15에서 preview) |
| ORT Profiling | ON (trial별) | Stage 내부 NPU/CPU/fence 분해 |
| Cooldown | 매 trial 사이 | Thermal 누적 방지 |

### 3.3 실험 구성

#### 실험 1: Backend × Precision

UNet을 중심으로 backend와 precision 효과를 측정한다. 4개 모델 모두 동일 backend/precision으로 실행한다.

| # | Backend | Precision | Steps | Strength | Resolution |
|---|---------|-----------|-------|----------|------------|
| 1 | CPU | FP32 | 20 | 0.7 | 512×512 |
| 2 | GPU | FP16 | 20 | 0.7 | 512×512 |
| 3 | NPU | FP16 | 20 | 0.7 | 512×512 |
| 4 | NPU | W8A16 | 20 | 0.7 | 512×512 |

> Config 1 (CPU FP32)은 baseline. 생성 시간이 매우 길 수 있으므로 1회만 실행하여 참고값으로 사용할 수 있다.
> Strength 0.7 → actual UNet steps = 14.

#### 실험 2: Step 수 × Strength 영향 (최적 backend 고정)

| # | Steps | Strength | Actual Steps | Backend | Precision | Resolution |
|---|-------|----------|-------------|---------|-----------|------------|
| 1 | 20 | 0.3 | 6 | NPU | W8A16 | 512×512 |
| 2 | 20 | 0.5 | 10 | NPU | W8A16 | 512×512 |
| 3 | 20 | 0.7 | 14 | NPU | W8A16 | 512×512 |
| 4 | 20 | 1.0 | 20 | NPU | W8A16 | 512×512 |
| 5 | 50 | 0.7 | 35 | NPU | W8A16 | 512×512 |

> Strength에 따른 actual step 수 변화와 E2E latency의 관계를 측정. strength=1.0은 text-to-image와 동일하므로 img2img overhead(VAE Encode)를 분리 측정 가능.

#### 실험 3: 해상도 영향 (최적 backend 고정)

| # | Resolution | Latent Shape | Backend | Precision | Steps |
|---|------------|-------------|---------|-----------|-------|
| 1 | 384×384 | [1,4,48,48] | NPU | W8A16 | 20 |
| 2 | 512×512 | [1,4,64,64] | NPU | W8A16 | 20 |
| 3 | 768×768 | [1,4,96,96] | NPU | W8A16 | 20 |

> SD v2.1 기본 해상도는 768×768이나, on-device에서는 메모리 제약으로 512 이하가 현실적일 수 있다.

#### 실험 4: Stage별 Backend 혼합 (Optional)

| # | Text Encoder | UNet | VAE Decoder | 근거 |
|---|-------------|------|-------------|------|
| 1 | NPU | NPU | NPU | 전체 NPU |
| 2 | CPU | NPU | GPU | UNet만 NPU, VAE는 GPU가 유리할 경우 |
| 3 | NPU | NPU | CPU | VAE CPU fallback 시 영향 측정 |

> 실험 1 결과에서 stage별 최적 backend가 다를 경우에만 실행.

### 3.4 측정 Metrics

1.5절에서 정의한 4계층(Product KPI / Stage Breakdown / UNet Per-Step / System Metrics)으로 측정한다. 데이터는 Appendix C의 CSV 스키마로 기록한다. 추가로 ORT Profiling을 활성화하여 session.run 내부를 NPU / CPU / Fence / ORT overhead로 분해하고, graph partitioning 결과(coverage%)를 수집한다.

---

## 4. Phase 2 — Sustained Generation Test

### 4.1 목적

- **연속 생성 시 성능 열화** 측정: 2회차, 3회차 생성에서 latency 증가 여부
- **Thermal throttling** 유무 검증: UNet loop는 수 초간 지속 연산이므로 발열 가능성 높음
- **Power consumption** 비교: 생성 1회당 에너지 비용
- **메모리 안정성**: 반복 생성 시 memory leak 여부

### 4.2 실행 방식

```
Model Load: 4개 세션 생성 (1회)
Warmup: 2회 full generation
Main:
  for trial in 1..10:
      run full generation pipeline
      record Generation Summary + UNet Step Detail
      record system metrics (thermal/power/memory) at generation 전후 + 1초 간격
      NO cooldown (연속 실행)
HTP Mode: sustained_high
```

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| Trials | 10 | 연속 10회 생성으로 thermal saturation 관찰 |
| Cooldown | 없음 | 연속 부하에서의 실제 사용 패턴 재현 |
| HTP Mode | sustained_high | 연속 실행에 적합한 성능/전력 균형 |
| Input | 고정 prompt × 고정 seed | Trial 간 동일 workload 보장 |
| ORT Profiling | OFF | 연속 실행에서 profiling overhead 제거 |

Phase 2의 분석 포인트는 **trial 간 비교**이다. Trial 1의 UNet per-step mean과 Trial 10의 UNet per-step mean을 비교하여, 연속 사용 시 사용자가 체감할 성능 열화를 정량화한다.

### 4.3 실험 구성

Phase 1에서 확인된 최적 configuration으로 실행:

| # | Backend | Precision | Steps | Resolution | 근거 |
|---|---------|-----------|-------|------------|------|
| 1 | NPU | W8A16 | 20 | 512×512 | 예상 최적 config |
| 2 | NPU | FP16 | 20 | 512×512 | Precision 비교 |
| 3 | GPU | FP16 | 20 | 512×512 | Backend 비교 |

### 4.4 측정 Metrics

| 카테고리 | Metric | 설명 |
|---------|--------|------|
| **Latency Drift** | Trial 1 E2E / Trial 10 E2E / Drift% | Stable(±5%) / Slight(5~15%) / Throttling(>15%) |
| **UNet Drift** | Trial 1 per-step mean / Trial 10 per-step mean | UNet 구간 집중 열화 분석 |
| **Thermal** | Start Temp / Peak Temp / Slope (°C/trial) | 온도 상승 추세 |
| **Power** | Avg Power (mW) / Energy per Generation (mJ) | 생성 1회당 에너지 |
| **Memory** | RSS after Trial 1 / Trial 10 / Delta | Memory leak 감지 |

---

## 5. 분석 프레임워크

### 5.1 Phase 1 분석 관점

- **Product KPI**: E2E만이 아니라 First Preview가 얼마나 빠른가가 핵심. Cold Start는 Cold / Cache Hit를 비교하여 cache 효과를 정량화
- **Stage Breakdown**: Stacked bar chart로 stage별 비중 표시. UNet이 75~90%+로 지배적일 것으로 예상
- **Preview 효과**: Preview ON/OFF 간 E2E 차이와 First Preview latency로 비용-효과 판단
- **Backend 비교**: 동일 precision에서 CPU / GPU / NPU의 UNet per-step latency 비교
- **Precision 비교**: FP16 vs W8A16의 latency·메모리·coverage 차이, 생성 이미지 품질 (visual + optional PSNR/SSIM)
- **Step 수 / 해상도 영향**: UNet total latency vs step 수의 선형성 확인, resolution 증가 시 per-step latency 증가율

### 5.2 Phase 2 분석 관점

- **Latency over trials**: 시계열 그래프로 thermal throttling onset 확인
- **UNet step-level 열화**: Trial 1의 step 20 latency vs Trial 10의 step 20 latency
- **Thermal ceiling**: 몇 번째 trial에서 온도가 포화되는가

### 5.3 Discussion 포인트

1. **UNet 병목과 최적화 경로**: UNet이 E2E의 70~90%일 때, step 수 감소(Turbo/LCM)나 model distillation의 예상 효과
2. **Strength-Latency 트레이드오프**: strength 감소 시 actual steps 감소로 E2E가 선형으로 줄어드는가? strength=0.3 vs 0.7에서 생성 품질 차이는 수용 가능한가?
3. **VAE Encoder 추가 비용**: img2img에서 추가된 VAE Encode stage의 E2E 영향 (~3~5%)
4. **Progressive Preview의 가치**: Preview overhead (~10~15%) 대비 First Preview Latency 단축의 UX 효과
5. **Cold Start 최적화**: QNN context cache 효과, 세션 순차/병렬 로드, 앱 백그라운드 유지 전략
6. **W8A16 vs FP16**: 메모리 절감 대비 latency 변화, NPU weight decompression overhead 여부
7. **Multi-session Memory**: 4 session 동시 상주 시 실측 peak memory vs 예상 ~1.7~2.0GB. 8GB 단말 확장 시 session swap 전략의 E2E 비용
8. **NPU Op Coverage**: CrossAttention, GroupNorm의 NPU 지원 여부. Fallback 발생 시 E2E 영향
9. **제품 적용 가능성 판단 기준**:

| 기준 | Target | 근거 |
|------|--------|------|
| First Preview | < 3초 | 사용자가 "반응하고 있다"고 느끼는 한계 |
| E2E (Final) | < 10초 | 편집 기능으로 수용 가능한 대기 시간 |
| Cold Start (cache hit) | < 10초 | 앱 시작 후 대기 허용 범위 |
| Peak Memory | < 4 GB | 12GB 단말에서 OS + 앱 + 모델 공존 |
| Thermal | 연속 5회 생성 후 throttling < 15% | 실사용 패턴에서 안정성 |

---

## 6. 실험 실행 순서

```
0. 사전 검증 — QAI Hub Profiling
   ├── SD v2.1 4개 모델 각각 compile & profile (cloud)
   │   (VAE Encoder는 별도 ONNX export 후 제출)
   ├── NPU op coverage 확인 (특히 UNet CrossAttention)
   ├── FP16 / W8A16 latency·memory 비교
   └── 결과로 on-device 실행 가능성 판단 → Go/No-Go

1. PC 파이프라인 PoC
   ├── Python ONNX Runtime으로 SD v2.1 img2img pipeline 동작 확인
   ├── VAE Encoder ONNX export (QAI Hub 미제공)
   ├── Tokenizer + Scheduler + strength 파라미터 검증
   ├── 해상도/step/strength별 출력 품질 확인
   └── W8A16 모델 변환 및 정확도 검증

2. Android 앱 구현
   ├── HEAD에서 feature/photo-assist 브랜치 생성
   ├── Camera 관련 제거, Gallery picker 추가
   ├── SdPipeline (img2img) + Tokenizer + Scheduler 구현
   ├── OrtRunner 4-session 관리 확장
   ├── Strength slider UI 추가
   ├── Stage breakdown KPI 측정 포인트 추가
   └── Before/After UI 구현

3. Phase 1 — Single Generation Profiling
   ├── 실험 1: Backend × Precision (4 configs)
   ├── 실험 2: Step 수 영향 (4 configs)
   ├── 실험 3: 해상도 영향 (3 configs)
   └── 결과 분석 → Phase 2 config 선별

4. Phase 2 — Sustained Generation Test
   ├── 선별된 2~3 configs, 각 10회 연속 생성
   └── 결과 분석 → thermal drift, power, memory stability

5. Report
   ├── Stage breakdown 분석 (병목 가시화)
   ├── Backend × Precision × Steps × Resolution 종합 비교
   ├── Sustained 안정성 평가
   ├── 제품 적용 가능성 판단
   └── 최적화 로드맵 제안
```

---

## Appendix A: QAI Hub 사전 검증 항목

QAI Hub에서 아래 항목을 cloud profiling으로 사전 확인한다.

| 항목 | 확인 내용 | 판단 기준 |
|------|----------|----------|
| NPU Coverage | UNet CrossAttention/GroupNorm HTP 지원 여부 | Coverage ≥ 85% |
| UNet per-step | HTP에서 1 step latency | < 500ms (20 step 기준 E2E < 12s) |
| Memory Peak | UNet session 로드 시 peak RSS | < 4 GB (12GB 단말 기준) |
| W8A16 정확도 | FP16 대비 생성 이미지 품질 | Visual inspection + optional PSNR |
| Cold Start | 4개 모델 순차 compile/load | < 30s (1회성, 허용 범위 넓음) |

## Appendix B: 앱 아키텍처 변경 사항

### 기존 YOLOv8 앱에서 변경되는 부분

| Component | 기존 (YOLOv8) | 변경 (SD v2.1) |
|-----------|--------------|----------------|
| **Input** | CameraX live/single frame | Gallery image picker + preset prompt buttons + strength slider |
| **Model** | 단일 YOLO session | 4개 독립 session, 메모리 동시 상주, 실행은 sequential |
| **Inference** | 단일 session.run() | img2img orchestration: VAE Enc → TextEnc → UNet loop → VAE Dec |
| **Output** | Detection bounding boxes | Style-transferred image (Bitmap) |
| **UI** | Camera preview + bbox overlay | Before/After slider + KPI dashboard + progress bar |
| **KPI** | FPS, frame drop, P50/P95 | E2E, First Preview, Cold Start, stage breakdown, per-step latency |

### 신규/교체 파일

| 파일 | 상태 | 역할 |
|------|------|------|
| `SdPipeline.kt` | **신규** | 4개 session orchestration, img2img strength 관리, preview flag |
| `Tokenizer.kt` | **신규** | CLIP tokenizer (vocab.json + merges.txt 기반, CPU) |
| `Scheduler.kt` | **신규** | EulerDiscrete scheduler. add_noise(strength) + step loop |
| `ImagePreprocessor.kt` | **신규** | Gallery image → resize → normalize → float tensor |
| `OrtRunner.kt` | **리팩토링** | Multi-session 관리, stage별 timing |
| `BenchmarkRunner.kt` | **리팩토링** | Generation trial 루프, Phase 1/2 |
| `MainActivity.kt` | **리팩토링** | Gallery picker, preset buttons, strength slider, before/after |
| `CameraManager.kt` | **제거** | Camera 불필요 |
| `YoloPostProcessor.kt` | **제거** | Detection 후처리 불필요 |

### 앱 설정 옵션

| 옵션 | 값 | 설명 |
|------|-----|------|
| Backend | CPU / GPU / NPU | 4개 session 공통 또는 개별 지정 |
| Precision | FP16 / W8A16 | 모델 파일 선택에 연동 |
| Steps | 10 / 20 / 30 / 50 | Scheduler total step 수 |
| Strength | 0.3 / 0.5 / 0.7 / 1.0 | actual steps = steps × strength |
| Resolution | 384 / 512 / 768 | 출력 해상도 (latent 크기 결정) |
| Progressive Preview | ON / OFF | Preview 활성화 여부 + interval 설정 |
| QNN Context Cache | ON / OFF | Cold start 최적화 |

### 모델 파일 관리

SD 모델은 수백 MB ~ 1GB+이므로 external storage에서 로드한다.

```bash
# 모델 배포 (adb push)
adb push models/vae_encoder_w8a16.onnx /sdcard/sd_models/
adb push models/text_encoder_w8a16.onnx /sdcard/sd_models/
adb push models/unet_w8a16.onnx /sdcard/sd_models/
adb push models/vae_decoder_w8a16.onnx /sdcard/sd_models/

# Tokenizer 리소스 (assets에 포함 가능, ~2MB)
assets/
  vocab.json        # CLIP vocabulary
  merges.txt        # BPE merge rules
  special_tokens.txt
```

## Appendix C: CSV 출력 스키마

generation 1회에 대해 세 종류의 record를 생성한다.

**Record Type 1: Generation Summary** (generation당 1행)

```
trial_id, prompt, steps, strength, actual_steps, resolution, backend, precision,
e2e_ms, first_preview_ms, first_preview_step,
preprocess_ms, vae_enc_ms, tokenize_ms, text_enc_ms, unet_total_ms, vae_dec_final_ms,
preview_count, preview_overhead_ms,
unet_per_step_mean_ms, unet_per_step_p95_ms, scheduler_overhead_ms,
peak_memory_mb, start_temp_c, end_temp_c, avg_power_mw
```

**Record Type 2: UNet Step Detail** (generation당 N행, step마다 1행)

```
trial_id, step_index, input_create_ms, session_run_ms, output_copy_ms,
scheduler_step_ms, step_total_ms, is_preview_step, preview_vae_ms,
thermal_c, power_mw
```

**Record Type 3: Cold Start** (앱 시작당 1행)

```
start_type (cold/warm/cache_hit),
vae_enc_load_ms, text_enc_load_ms, unet_load_ms, vae_dec_load_ms, total_load_ms,
peak_memory_after_load_mb
```

## Appendix D: 기존 YOLOv8 실험과의 연속성

본 프로젝트는 기존 Mobile NPU Inference KPI Lab의 프레임워크를 확장한다.

| 요소 | 기존 (YOLOv8) | 확장 (SD v2.1) |
|------|--------------|----------------|
| KPI 수집 | KpiCollector (thermal/power/memory) | **그대로 재활용** |
| Batch 실험 | ExperimentConfig + JSON | 확장 (steps, resolution, prompt 추가) |
| CSV Export | BenchmarkRunner | 확장 (stage breakdown 컬럼 추가) |
| 분석 스크립트 | parse_logs.py, plot_kpi.py | 확장 (stage breakdown 시각화 추가) |
| QAI Hub 검증 | profile_qai_hub.py (YOLOv8) | 확장 (SD 4개 모델 지원) |
| Backend 비교 | CPU / GPU / NPU | 동일 |
| Phase 1/2 구조 | Burst / Sustained | 적용 (단건 생성 / 연속 생성) |
