# Mobile Inference KPI Lab — Interview Guide

> SD v1.5 text-to-image 파이프라인의 on-device feasibility 프로젝트를 면접에서 설명할 때 사용할 핵심 질문 정리.

---

## 1. 프로젝트 핵심 정의

SD v1.5 기반 text-to-image 파이프라인을 Galaxy S23(Snapdragon 8 Gen 2) NPU에서 직접 실행해 latency·power·thermal·quality를 정량 측정하고, 어떤 조건에서 on-device가 현재 cloud 실행을 대체할 수 있는지 그 boundary를 도출한 feasibility study입니다.

---

## 2. 1분 설명

Galaxy S26 Creative Studio의 이미지 생성 기능은 cloud에서 실행되어 cold start 기준 약 8초 내외의 latency를 보입니다. 이 프로젝트는 "왜 on-device로 못 가는가"를 정량적으로 검증하기 위해 시작했습니다.

Snapdragon NPU를 타겟으로 ONNX Runtime + QNN EP 기반 Android 벤치마크 앱을 구축하고, SD v1.5와 LCM 모델을 quantization 전략과 NPU performance mode 조건을 달리하며 latency, quality, thermal, power를 측정했습니다.

종합적으로 서비스 마지노선을 10초 정도로 봤을 때, SD v1.5는 cold latency 16초로 기준을 초과하고, LCM은 8초로 기준을 만족하지만 quality가 떨어집니다. 또한 NPU clock에 따라 최대 15% 이상 latency가 커질 수 있음을 고려할 때, quality와 latency를 동시에 만족하는 모델이 없어서 cloud serving이 합리적이라는 결론을 확인했습니다.

### 1분 설명 직후 예상 꼬리질문

**Stable diffusion, LCM 모델에 대해서 설명**
SD v1.5는 텍스트 프롬프트를 받아 이미지를 생성하는 모델입니다. 내부적으로 Text Encoder, UNet, VAE Decoder 세 컴포넌트가 존재합니다. UNet이 핵심인데, 순수 노이즈에서 시작해 step마다 조금씩 노이즈를 제거하는 방식으로 동작합니다. 프롬프트를 잘 반영하기 위해 매 step에서 UNet을 두 번 (프롬프트 있을 때, 없을 때) 실행하고 그 차이를 증폭하는 방식을 씁니다.

LCM은 이미지 생성 과정에 긴 step이 필요하다는 단점을 개선하기 위해서 SD v1.5 UNet에 LoRA weight를 fuse한 모델입니다. 어느 노이즈 상태에서 출발해도 최종 이미지를 바로 예측하도록 학습되었고, 훨씬 적은 step(4~8)만으로 이미지를 생성할 수 있습니다. 프롬프트 없는 UNet 호출도 필요 없어져 step당 UNet이 한 번만 실행됩니다.

**VAE에 대해서 설명**
VAE는 이미지를 압축하고 복원하는 Encoder-Decoder 구조입니다. SD에서는 512×512 픽셀 이미지를 64×64 크기의 작은 latent로 압축하고, 반대로 복원할 수 있습니다. UNet이 이 압축된 latent 공간에서 노이즈 제거를 수행하기 때문에 픽셀 공간에서 직접 작업하는 것보다 연산량이 크게 줄어듭니다. txt2img에서는 최종 denoised latent를 이미지로 복원하는 VAE Decoder만 사용합니다.

**LoRA에 대해서 설명**
LoRA는 이미 학습된 모델을 특정 task나 조건에 맞게 변형할 때 전체 재학습 비용을 줄이기 위한 fine-tuning 기법입니다. 핵심 아이디어는 weight를 직접 학습하는 대신, d×r과 r×d 두 행렬의 곱 B·A을 학습하는 것입니다. r을 d보다 훨씬 작게 설정하면 파라미터 수가 d²에서 2dr로 학습 파라미터가 줄어듭니다. 학습 후에는 원본 weight에 값을 fuse하기 때문에 inference graph는 원본과 동일합니다.

**npu performance mode에 대해서 설명**
Qualcomm NPU는 clock 수준을 앱에서 직접 지정하는 performance mode를 제공합니다. burst는 최대 clock으로 가장 빠르지만 전력과 발열이 높고, balanced는 성능과 전력을 균형 있게 맞춘 모드입니다. 이 값은 앱에서 명시적으로 설정하는 값이지만 SoC 온도 등에 따라 Android thermal framework가 클럭을 강제로 낮출 수 있습니다.

연속으로 이미지를 생성하거나 발열이 누적된 환경에서는 burst를 요청해도 thermal throttling이 개입할 수 있고, background 추론 시에는 기기가 기본적으로 낮은 clock 상태에 있을 수 있습니다. 서비스 가능 여부를 판단하려면 이런 상황에서의 하한 latency도 KPI 안에 들어와야 합니다. 실측 결과는 balanced에서 warm latency가 burst 대비 약 15% 증가, 전력은 −20% 감소했습니다.

**latency, power, thermal, quality 등을 어떻게 측정했는지?**
latency는 wall-clock 기준으로 측정했고, cold/warm latency를 분리해서 측정했습니다. 

power는 Android BatteryManager API로 시스템 전체 소비전력을 측정했습니다. NPU 단독 분리는 어렵기 때문에 충전기 분리·airplane mode 등 변인을 통제해 실험 간 상대 차이를 비교하는 방식으로 활용했습니다. thermal은 SoC 온도를 시스템 thermal zone에서 직접 읽었습니다.

quality는 각 모델 output을 CosSim로 1차 비교를 거친 뒤, E2E output의 LPIPS(CNN 중간 feature L2-norm)와 CLIP Score를 측정했습니다.

**quantization은 어떻게 수행했고, 영향도는 어떻게 평가했는지?**
초기에는 모든 모델을 QAI Hub를 통해서 Int8 quantization을 시도했습니다. 그러나 UNet은 중간의 LayerNorm 연산의 미지원으로 인해 quantization을 실패했고, Qualcomm에서 제공하는 AIMET 라이브러리를 통해 Conv, Attention, GEMM만 quantization하는 mixed precision 기반 quantization을 수행했습니다.

**이 프로젝트의 의의는 무엇인가요?**
"왜 현재 cloud에 둘 수밖에 없는가"라는 질문에 정량적 근거를 제공합니다. 10초라는 가상의 kpi를 두기는 했지만, quality를 만족하면서 on device에서 돌 수 있는 stable diffusion model이 없음을 증명했고, 그 병목이 UNet 구조에 있으며, performance mode에 따라 더 높은 latency를 보일 수 있음까지 보였습니다. 이를 기반으로 SD 수준의 quality와 LCM 수준의 step 효율을 갖춘 모델이 등장하면 현재 SoC에서도 on-device 전환이 가능하다는 방향성을 제시합니다.

**왜 SD v1.5인가요? 다른 모델이 아닌 이유는?**
첫째, 모바일 사용 패턴과의 적합성입니다. SD v2.1은 상세한 프롬프트를 입력해야 원하는 결과를 얻기 쉬운 반면, v1.5는 짧은 프롬프트에서도 품질 하한이 잘 보존됩니다. 모바일 사용자는 프롬프트를 짧게 입력하는 경향이 있어서 v1.5가 더 적합한 선택이었습니다. 

둘째, 메모리 제약입니다. SD v1.5도 naive하게는 S23에서 실행 자체가 안 됐고, SDXL은 UNet이 두 개로 훨씬 커서 on-device 실행이 더욱 어렵습니다. 

---

## 2. 꼬리질문 답변

### 프로젝트 배경 · 기여

#### 1. 이 프로젝트를 왜 시작했나요?

---

### 시스템 구조 (Android · ORT · QNN)

#### 1. 앱의 layer 구조와 추론 흐름을 설명해주세요.

#### 2. ONNX Runtime, QNN Execution Provider, QAIRT는 각각 무엇이고 어떻게 연결되나요?

#### 3. 왜 ONNX Runtime을 선택했나요? PyTorch Mobile, LiteRT, native QNN과 비교하면?

#### 4. Cold start의 비용은 어디서 오나요? 한 번 발생한 뒤 왜 줄어드나요?

---

### 실험 방법론

#### 1. 실험 시작 전 통제 조건들(충전기 분리, airplane, brightness 등) — 각각 왜 필요한가요?

#### 2. 측정값을 믿을 수 있게 만들기 위해 어떤 장치를 두었나요? (warm trial mean, wall-clock, seed 고정 등)

---

### 도메인 지식

#### 1. CFG(Classifier-Free Guidance)란 무엇인가요? 왜 step당 UNet을 두 번 호출하나요?

#### 2. LPIPS와 CLIP Score는 각각 무엇을 측정하나요? 왜 둘 다 써야 하나요?

---

## 5. 약점 / 한계

### 단일 디바이스 측정

Galaxy S23 한 대만으로 결과를 일반화할 수 있는가에 대한 한계가 있다. Snapdragon 8 Gen 2 기준 결과이며, 다른 SoC(Dimensity, Exynos, Apple Silicon)에서의 동작은 확인하지 않았다.

### Cloud baseline 비대칭

S26 Ultra의 cloud latency(1024×1024)를 S23 on-device(512×512)와 비교하는 구조적 한계가 있다. 512+SR 파이프라인이 합리적 대안으로 제시되었으나, SR 모델의 실제 품질과 latency는 검증하지 않았다.

### W8A16 평가 누락

`unet_base_w8a16`이 S23 inference OOM으로 on-device 품질 평가가 불가했다. bin 크기(842MB)는 작지만 activation memory까지 합산하면 한계를 초과한다. 이 config가 품질·latency 양쪽에서 좋은 후보일 수 있었으나 평가를 완료하지 못했다.

### 통계적 신뢰도

trial 5회, prompt set 다양성에 한계가 있다. 더 많은 trial과 다양한 프롬프트로 반복 측정하면 결과의 신뢰구간을 좁힐 수 있다.

### Pipeline overlap 미적용

preprocess/inference/postprocess overlap으로 추가 throughput 개선 여지가 있다. 현재는 single-frame sequential 실행 기준이다.

### 전력 측정 한계

BatteryManager 기반 시스템 전체 소비전력으로, NPU 단독 전력 분리 불가. 다만 충전기 분리·airplane mode·brightness 고정 등 변인을 통제해 모든 실험을 동일 조건에서 측정했으므로, 절대값보다 실험 간 상대 차이(burst vs balanced −22%, precision 변경 −19% 등) 해석에 활용했다.

---

## 6. Behavioral 질문 대비

### 가장 크게 배운 점은?

### 당면했던 문제화 해결 방안?
