# SD v1.5 On-Device 배포를 위한 모델 최적화: 양자화 전략 설계와 품질 검증

---

## 0. 배경

이 문서는 SD v1.5 text-to-image 파이프라인을 Android NPU(Snapdragon 8 Gen 2 HTP)에서 실행하기 위해 진행한 모델 최적화 과정을 기록한다. 단순히 어떤 결과가 나왔는지가 아니라, 각 결정에 이른 과정과 그 과정에서 발견한 기술적 제약들을 중심으로 서술한다.

### 대상 파이프라인

SD v1.5 txt2img는 세 컴포넌트가 순차적으로 실행되는 구조다.

```
[Text Encoder] → [UNet × N steps] → [VAE Decoder]
  CLIP ViT-L/14     859M params         4ch → RGB
  470MB FP32        3.3GB FP32          189MB FP32
```

각 컴포넌트는 구조적 특성이 달라 단일 양자화 전략을 일괄 적용하기 어렵다. VAE는 Conv 중심, UNet은 Conv + Attention(LayerNorm, Gemm), Text Encoder는 Transformer(LayerNorm, Gemm)로 구성되며 이 차이가 이후 전략 분기의 핵심이 된다.

---

## 1. 전략 수립: 왜 컴포넌트별로 다르게 접근했는가

첫 번째 질문은 "어떤 precision을 어떤 컴포넌트에 적용할 것인가"였다.

단일 precision을 전체에 적용하는 방식(full W8A8, full W8A16)도 있지만, 컴포넌트별로 연산 구조가 다르고 품질 민감도도 달라서 개별 판단이 필요했다.

**UNet**: Conv, MatMul, Gemm이 지배적이나 LayerNorm이 곳곳에 포함된다. UNet의 denoising 품질은 노이즈 제거 정밀도에 직결되므로 activation 정밀도를 낮추면 품질 손실이 심할 수 있다. 따라서 weight quantization 위주로 접근하되, activation은 최대한 유지하는 방향을 우선 고려했다.

**VAE Decoder**: 잠재 공간(4ch 64×64)을 픽셀 공간(3ch 512×512)으로 변환하는 순수 Conv 구조. 구조 특성상 weight-only quantization이 상대적으로 안정적일 가능성이 높다.

**Text Encoder**: CLIP Transformer로 embedding 생성. inference 중 한 번만 실행되므로 latency 기여가 작다. 크기 감소 목적으로 W8A16 적용을 우선 검토.

이 분석으로부터 세 가지 precision 경로를 설정했다.

| Precision | 정의 | 주요 대상 |
|---|---|---|
| W8A8 | weight 8bit + activation 8bit (QDQ 노드 삽입) | VAE Encoder, VAE Decoder |
| MIXED_PR | Conv·MatMul·Gemm INT8, LayerNorm·나머지 FP32 | UNet base, UNet LCM |
| W8A16 | weight 8bit, activation FP16 (AIMET weight-only) | Text Encoder, VAE Decoder, UNet |

---

## 2. ONNX Export

양자화의 전제는 정확한 ONNX export다. opset 18을 기준으로 했으며, UNet(3.3GB)은 ONNX external data 형식으로 export했다 — `.onnx` 파일에는 graph 구조만 담고 weight는 별도 `.data` 파일로 분리된다. QAI Hub은 이 경우 두 파일을 같은 디렉터리에 묶어서 업로드해야 한다.

calibration data도 이 단계에서 함께 생성했다. UNet W8A8 시도를 위해 64개 샘플 기준의 `calib_unet.npz`를 만들었고, VAE Encoder/Decoder는 실제 이미지에서 추출한 latent 샘플을 사용했다.

---

## 3. 양자화 실행

### W8A8 — QAI Hub quantize

VAE Encoder, VAE Decoder에 적용. QAI Hub의 quantize job API를 사용했으며, MinMax calibration(64 samples) 방식이다. 결과물은 ONNX에 QDQ(Quantize-Dequantize) 노드가 삽입된 형태로 나온다. 이 상태에서 ORT CPU로 로컬 추론이 가능하기 때문에 품질 평가를 로컬에서 수행할 수 있다는 장점이 있다.

### MIXED_PR — AIMET 기반 부분 양자화

UNet에 full INT8 QDQ를 시도했으나 QAI Hub compile에서 실패했다. 오류 메시지는 아래와 같았다.

```
Unsupported input/output datatypes for HTP Op 'LayerNorm'
```

HTP(Hexagon Tensor Processor)는 `LayerNormalization` 연산의 full INT8 input/output 조합을 지원하지 않는다. Snapdragon 8 Gen 2 기준이며, HTP는 효율적 연산을 위해 연산별 지원 precision 조합이 제한된다.

이에 따라 MIXED_PR 전략으로 전환했다: Conv, MatMul, Gemm만 INT8로 양자화하고 LayerNorm과 그 외 연산은 FP32로 유지. 이 방식은 UNet에서 연산량 기준으로 지배적인 Conv/MatMul을 압축하면서 HTP compile을 통과할 수 있다.

`quant_runpod.py`에서 AIMET을 통해 op-type selective quantization을 적용했다. Linux 전용(`aimet_onnx`)이기 때문에 RunPod 환경에서 실행했다.

### W8A16 — qai-hub-models AIMET export

`qai-hub-models==0.47.0`의 `stable_diffusion_v1_5.export`를 사용해 AIMET W8A16(weight 8bit, activation FP16) 모델을 얻었다. Weight-only quantization이므로 ONNX에 QDQ 노드가 없고 별도 `.encodings` 파일에 scale/offset 정보가 저장된다. QAI Hub이 이 encodings를 compile 시 참조해 QNN context binary에 내장한다.

UNet LCM의 경우 `qai-hub-models` 자체 export가 없어 별도 스크립트(`export_sd_lcm_unet_w8a16.py`)에서 PyTorch에 LCM-LoRA fuse 후 AIMET W8A16 적용하는 경로로 진행했다.

---

## 4. QAI Hub Compile

양자화된 ONNX를 QAI Hub에서 target device용 QNN context binary로 컴파일했다. 핵심 옵션:

```
--target_runtime precompiled_qnn_onnx
--qairt_version 2.42
```

QAIRT 버전을 앱의 ORT QNN EP 빌드 버전(2.42.0)과 일치시키는 것이 중요하다. 버전 불일치 시 `.bin` 로딩 실패 또는 런타임 오류가 발생한다.

컴파일 결과물은 `.onnx` stub과 `.bin` QNN context binary의 쌍으로 구성된다. Stub의 내부 `ep_cache_context` 경로 필드가 실제 device에서의 `.bin` 경로를 가리키도록 수정이 필요하다 — `push_models_to_device.sh`에서 처리한다.

**컴파일 결과 요약 (bin 크기 기준)**

| 컴포넌트 | FP16 | W8A8 / MIXED_PR | W8A16 |
|---|---|---|---|
| text_encoder | 237MB | — | 156MB |
| vae_encoder | 76MB | 39MB | — |
| vae_decoder | 109MB | 57MB | 62MB |
| unet_base | 1,651MB | 1,151MB (MIXED_PR) | 842MB |
| unet_lcm | 1,651MB | 1,150MB (MIXED_PR) | 834MB |

UNet MIXED_PR이 1,151MB인 이유는 Conv/MatMul만 INT8이고 나머지(LayerNorm 포함)가 FP32이기 때문이다. Weight-only인 W8A16(842MB)보다 크다 — INT8로 압축된 weight보다 FP32로 남은 비중이 많아서다.

---

## 5. On-Device Profile

QAI Hub profile job을 통해 Galaxy S23(Snapdragon 8 Gen 2)에서 컴포넌트별 실행 시간을 측정했다.

| 컴포넌트 | Variant | Latency | Memory |
|---|---|---|---|
| text_encoder | FP16 | 5.9ms | 2MB |
| text_encoder | W8A16 | 3.3ms | 2MB |
| vae_encoder | FP16 | 201ms | 82MB |
| vae_encoder | W8A8 | 54ms | 41MB |
| vae_decoder | FP16 | 440ms | 119MB |
| vae_decoder | W8A8 | 111ms | 69MB |
| vae_decoder | W8A16 | 270ms | 3MB |
| unet_base | FP16 | — (미측정) | — |
| unet_base | MIXED_PR | 265ms | 1,151MB |
| unet_lcm | FP16 | 333ms | 1,793MB |
| unet_lcm | MIXED_PR | 264ms | 1,150MB |
| unet_lcm | W8A16 | 214ms | 902MB |

몇 가지 주목할 포인트:

**VAE Decoder W8A16 memory가 3MB인 이유**: W8A16은 weight-only quantization이고 activation은 FP16으로 흐른다. Profile job에서 측정하는 memory가 실행 중 peak activation memory를 기준으로 하기 때문에 weight 크기(62MB)와 독립적으로 낮게 나올 수 있다.

**unet_lcm FP16 memory 1,793MB**: 단일 UNet inference에서 activation이 1.7GB 이상 요구된다. UNet이 메모리 병목인 이유가 weight 크기(1.6GB)뿐 아니라 intermediate activation이 크기 때문이다.

---

## 6. 품질 평가 방법론

Compile된 모델을 full pipeline으로 실행해 이미지로 비교하는 것이 직관적이나, 이 방법은 어느 컴포넌트가 품질 열화의 원인인지 분리하기 어렵다. 컴포넌트가 3개(Text Encoder, UNet, VAE Decoder)이고 UNet은 N step 반복이라 디버깅 복잡도가 높다.

이를 해결하기 위해 **component-level CosSim** 방식을 설계했다.

```
FP32 ORT CPU (reference)
     ↓
[Component 단독 실행 — 고정 입력]
     ↓
FP32 출력 벡터

quantized model
     ↓
[동일 입력으로 단독 실행]
     ↓
양자화 출력 벡터

CosSim(FP32 출력, 양자화 출력) → 품질 지표
```

고정 입력은 calibration data에서 4개 샘플을 사용했다. CosSim은 방향 유사도를 측정하므로 출력 분포의 형태가 보존되는지를 본다. RMSE, PSNR, MaxErr도 함께 측정하여 outlier 여부를 추가 확인했다.

**평가 방식 분기**: W8A8(QDQ 노드 포함 ONNX)은 로컬 ORT CPU로 직접 추론 가능하다. W8A16은 QDQ 노드가 없고 `.encodings`만 있어 on-device QNN 실행이 필요하다 — QAI Hub inference job을 제출해 결과를 받아 비교했다.

평가 스크립트: `scripts/sd/eval_sd_quant_quality.py`
결과 파일: `exp_outputs/quantization/sd_quant_quality.txt`

---

## 7. 결과 분석

### 7.1 VAE Decoder W8A16: CosSim 0.9999

가장 높은 품질을 기록했다. VAE Decoder는 구조적으로 Conv 중심이고 decoder 방향(잠재→픽셀)이라 각 layer의 출력이 비교적 부드러운 분포를 가진다. Weight-only quantization(W8A16)이 이런 구조에서 특히 효과적임을 확인했다.

초기 평가 시 CosSim이 -0.117로 나와 이상값이 발생했다. 디버깅 과정에서 원인을 파악했다.

FP32 ONNX VAE Decoder의 출력 범위: `[-1.22, 1.27]` — 학습 관례상 raw 픽셀을 [-1,1]로 normalize한 값이다.
W8A16 컴파일 모델의 출력 범위: `[0.0, 1.0]` — qai-hub-models가 export 시 `/Div + /Clip` 연산을 모델 내부에 포함시켜, 출력이 자동으로 [0,1]로 변환된다.

두 모델의 출력 scale 자체가 달랐던 것이다. FP32 참조 출력에 `clip(x/2+0.5, 0, 1)`을 적용해 맞추자 CosSim 0.9999가 나왔다. 이 발견은 앱 코드에도 직결된다 — `ImagePreprocessor.postprocess()`에서 W8A16 precision 여부에 따라 postproc 분기를 추가했다(`normalized=true` 시 `x * 255`, 그 외 `(x+1) * 127.5`).

### 7.2 UNet base MIXED_PR: CosSim 0.9968

INT8 QDQ full compile이 HTP LayerNorm 제약으로 실패한 이후 설계한 MIXED_PR 전략이 Good 등급을 달성했다. Conv/MatMul/Gemm을 INT8로 압축하고 LayerNorm을 FP32로 유지하는 것이 UNet의 denoising 품질에서 합리적 트레이드오프임을 확인했다.

전체 UNet weight 대비 LayerNorm이 차지하는 비중은 작으나, denoising loop에서 LayerNorm은 feature 분포를 안정화하는 역할을 한다. Activation precision에 민감한 이 연산을 FP32로 유지한 것이 품질 보존에 결정적이었을 것으로 판단한다.

### 7.3 UNet LCM W8A16: CosSim 0.7292 (Poor)

같은 W8A16 전략인데 VAE Decoder(0.9999)와 극단적으로 다른 결과다. 원인을 단정하기는 어려우나 몇 가지 가설이 있다.

첫째, LCM-LoRA는 base UNet에 LoRA weight를 fuse한 모델이다. LoRA weight는 low-rank 행렬 두 개의 곱으로 구성되어 있어 weight 분포가 일반 Conv weight와 다를 수 있다. AIMET calibration이 이 분포를 충분히 반영하지 못했을 가능성이 있다.

둘째, AIMET의 W8A16 calibration이 base UNet용으로 설계된 파이프라인을 LCM UNet에 그대로 적용했다. LCM-LoRA 특유의 weight 분포 특성이 고려되지 않았을 수 있다.

이 모델은 deploy_config에는 포함시켰으나(실제 이미지 품질 확인 목적) 품질 보증 불가로 분류했다.

### 7.4 UNet base W8A16: OOM

S23에서 inference job이 메모리 초과로 실패했다.

```
Failed to profile the model because memory usage exceeded device limits.
```

컴파일된 bin 크기는 842MB지만 실행 중 activation memory까지 합산하면 S23 12GB RAM 기준에서도 한계를 초과한다. 샘플 수를 4→1로 줄여도 동일하게 실패했다. 따라서 배포 불가로 처리하고 on-device 품질 평가도 수행하지 못했다.

---

## 8. 배포 결정 및 근거

평가 결과를 바탕으로 `scripts/deploy/deploy_config.json`에 배포 가능 모델을 정의했다.

| 컴포넌트 | Variant | 배포 결정 | 근거 |
|---|---|---|---|
| text_encoder | FP16 | ✅ | baseline |
| text_encoder | W8A16 | ✅ | CosSim 0.9849, 크기 237→156MB |
| vae_encoder | FP16 | ✅ | baseline |
| vae_encoder | W8A8 | ✅ | CosSim 0.9810, 76→39MB, 201→54ms |
| vae_decoder | FP16 | ✅ | baseline |
| vae_decoder | W8A8 | ✅ | CosSim 0.9827, 109→57MB |
| vae_decoder | W8A16 | ✅ | CosSim 0.9999, 최고 품질 |
| unet_base | FP16 | ✅ | baseline |
| unet_base | MIXED_PR | ✅ | CosSim 0.9968, 1,651→1,151MB |
| unet_lcm | FP16 | ✅ | baseline |
| unet_lcm | MIXED_PR | ✅ | CosSim 0.9875, 조건부 |
| unet_lcm | W8A16 | ⚠️ | CosSim 0.7292, 이미지 품질 확인 목적으로만 포함 |

**제외 항목**:
- `unet_base_w8a16`: S23 inference OOM — 실행 불가
- `unet_base_int8_qdq`: HTP LayerNorm INT8 미지원 — compile 실패
- `unet_lcm_int8_qdq`: 동일 이유

---

## 9. 주요 발견 요약

**하드웨어 제약 발견**: Snapdragon 8 Gen 2 HTP는 LayerNorm의 full INT8 input/output 조합을 지원하지 않는다. Transformer 계열 모델(UNet의 attention block)을 HTP에서 INT8로 실행하려면 op-selective mixed precision이 필요하다.

**Postproc 불일치 디버깅**: qai-hub-models export 모델은 내부에 postprocessing 연산을 포함할 수 있다. 동일 모델이라도 export 경로에 따라 출력 range가 다를 수 있어, 비교 평가 시 reference model과의 출력 space 정합이 선행되어야 한다.

**Component-level 평가의 효용**: Full pipeline 이미지 비교만으로는 어느 컴포넌트가 품질 열화의 원인인지 파악하기 어렵다. 컴포넌트를 분리해 고정 입력으로 단독 평가하는 방식이 디버깅 효율을 크게 높인다. CosSim은 magnitude-independent하게 방향 유사도를 보므로 scale 차이가 있는 경우 보완 지표(RMSE, MaxErr)와 함께 해석해야 한다.

**Weight-only quantization의 특성**: W8A16은 weight만 8bit으로 압축하고 activation은 FP16으로 유지한다. VAE Decoder처럼 Conv 중심 구조에서는 품질 손실이 거의 없으나(0.9999), LCM-LoRA fused UNet처럼 non-standard weight 분포를 가진 경우 calibration이 실패할 수 있다.
