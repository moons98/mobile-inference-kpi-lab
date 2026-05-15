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

## 3. 꼬리질문 답변

### 시스템 구조 (Android · ORT · QNN)

#### 1. 앱 구조에 대해서 설명
앱은 Kotlin으로 구성되어 있고, 실험 세트를 JSON으로 선언적으로 정의해 precision·step·perf mode 같은 파라미터를 코드 수정 없이 추가할 수 있도록 했습니다. 사용자가 UI에서 세트를 선택하면 벤치마크 로직이 루프를 돌면서 cold/warm latency, power, thermal 등의 정보들을 CSV로 저장합니다.

그 아래 추론 스택은 ONNX Runtime + QNN Execution Provider 구조입니다. ORT가 graph 실행과 I/O를 관리하고, QNN EP가 연산을 Qualcomm Hexagon HTP(NPU)로 라우팅하며 perf mode 같은 NPU 제어도 담당합니다.

#### 2. 왜 ONNX Runtime을 선택했나요? native QNN을 직접 쓰면 안 되나요?
먼저 inference 성능 차이는 거의 없습니다. ORT + QNN EP도 결국 같은 QNN backend를 통해 HTP에서 graph를 실행하기 때문에, 모델 실행 시간 자체는 native QNN과 동일하고 ORT의 wrapper overhead는 측정 가능한 수준이 아닙니다.

ORT + QNN EP의 장점은 fallback 구조에 있습니다. native QNN은 graph 전체가 NPU에서 동작 가능해야 실행되지만, ORT는 QNN EP가 처리하지 못하는 연산을 자동으로 CPU EP로 fallback해 실행을 이어갑니다. 덕분에 어떤 layer가 NPU에서 동작하지 못했는지, 그로 인한 latency 비용은 어느 정도인지 운영 환경에서 그대로 측정할 수 있습니다. 실제 서비스를 가정하더라도 unsupported op 처리를 위한 fallback 경로는 어차피 필요하기 때문에, 이를 runtime이 표준 방식으로 담당하게 두는 것이 합리적이라고 판단했습니다.

#### 3. Cold start의 비용은 어디서 오나요? 한 번 발생한 뒤 왜 줄어드나요?
Cold start 비용은 거의 전부 첫 세션을 띄울 때 발생합니다. ONNX Runtime이 컴파일된 모델 binary를 읽어 NPU에 graph로 등록하고, 그 graph에 묶일 입출력 buffer를 준비하는 과정인데, Text Encoder · UNet · VAE Decoder 세 모델을 각각 띄우다 보니 실험에서 5~7초가 소요됐습니다. 그 중 UNet binary가 가장 커서 cold 비용의 대부분을 차지했고, 그래서 UNet을 양자화한 실험에서 cold latency도 가장 짧게 나왔습니다.

첫 추론 자체에도 NPU 캐시가 cold한 상태라 warm 대비 100ms 정도 추가 비용이 붙지만, 세션 로드에 비하면 미미한 수준입니다.

---

### 실험 방법론

#### 1. 실험 시작 전 통제 조건들이 어떤게 있었는지?
충전기 분리 및 airplance mode 설정을 수행했습니다. 충전 중에는 BatteryManager API가 충전 전류와 소비 전류를 혼재해 읽어 전력 측정값이 오염되고, Airplane mode는 셀룰러/Wi-Fi background 트래픽이 CPU와 전력에 주는 영향을 차단해주기 때문입니다.

화면 밝기 고정은 디스플레이가 소비전력의 큰 비중을 차지하므로 실험 간 일정하게 유지해야 상대 비교가 유효하고, 배터리 40% 이상 + 35°C 이하 유지는 저배터리 혹은 thermal로 인한 자체 throttling 방지를 위해 체크했습니다.

---

### 도메인 지식

#### 1. LPIPS와 CLIP Score는 각각 무엇을 측정하나요? 왜 둘 다 써야 하나요?
LPIPS는 두 이미지의 차이를 사람 시각과 가깝게 측정하는 지표입니다. 사전 학습된 모델로 두 이미지의 feature를 뽑아 그 차이를 계산하기 때문에 색이나 텍스처처럼 사람이 느끼는 차이를 잘 반영하고, 낮을수록 두 이미지가 비슷하다는 의미입니다.

CLIP Score는 이미지와 프롬프트가 얼마나 잘 맞는지를 점수로 보여줍니다. CLIP이 이미지와 텍스트를 같은 공간으로 임베딩하고 그 유사도를 측정하는데, 높을수록 프롬프트의 의미를 이미지가 잘 반영했다는 뜻입니다.

둘 다 필요한 이유는, diffusion은 매 step마다 noise를 조금씩 제거하면서 이미지를 만드는데, UNet에 양자화 같은 변화가 들어가면 step마다 미세한 오차가 누적되어 같은 seed로 출발해도 최종 이미지가 다른 모습이 되기 때문입니다. 이때 LPIPS는 "이미지가 다르다"는 건 잘 잡아내지만, 그게 품질이 나쁜 건지 아니면 단순히 다른 그림이 나온 건지를 구분하지 못합니다. CLIP Score를 함께 봐야 프롬프트 반영도가 유지되는지로 둘을 가를 수 있습니다. 반대로 CLIP Score만 보면 흐릿하거나 깨진 이미지여도 의미만 맞으면 비슷한 점수가 나올 수 있어, 시각적 품질은 LPIPS로 봐야 합니다.

---

## 4. Behavioral 질문 대비

### 가장 크게 배운 점은?
가장 크게 배운 건 NPU 기반 추론을 end-to-end로 굴려본 경험 자체입니다. 일반적인 GPU 추론과 달리, NPU에서 모델을 돌리려면 먼저 QAI Hub 같은 vendor 컴파일 파이프라인을 거쳐 device 전용 binary로 변환해야 했고, 그 과정에서 HTP가 지원하지 않는 ops(예: LayerNorm INT8)이 발견되면 mixed precision 양자화 같은 우회 전략을 직접 설계해야 했습니다.

그리고 실제 추론에서는 NPU clock 설정(burst/balanced)이나 thermal 상태에 따라 같은 모델이라도 latency가 15% 안팎으로 달라졌고, 발열이 누적되면 burst 요청 자체가 무력화되는 경우도 있었습니다. 그래서 모델 latency를 측정한다는 것이 단순히 inference 시간만 재는 게 아니라, compile · ops 호환성 · 운영 환경 통제까지 묶어서 봐야 의미 있는 수치가 된다는 점을 체감했습니다.

### 당면했던 문제와 해결 방안?
UNet 양자화 과정에서 문제를 경험했습니다. 처음에는 QAI Hub의 quantize API로 UNet INT8 양자화를 시도했는데, 모델 크기가 커서 cloud 측 메모리 한도를 초과하며 실패했습니다. 그래서 직접 Linux 서버를 띄워 Qualcomm AIMET 라이브러리로 양자화를 다시 진행했는데, 이번에는 compile 단계에서 막혔습니다. 문서상으로는 지원된다고 되어 있던 LayerNorm INT8이 Snapdragon 8 Gen 2 HTP에서는 실제로는 지원되지 않았고, 에러 메시지도 단순 "Unsupported input/output datatypes" 수준이라 원인을 특정하기 어려웠습니다. 결국 QAI Hub Slack 채널에서 담당자들과 직접 소통하며 원인을 좁혔고, Conv·MatMul·Gemm만 INT8로 양자화하고 LayerNorm은 FP32로 남겨두는 MIXED_PR 전략으로 우회했습니다. 이 일련의 과정에서 NPU 활용 개발 생태계가 GPU만큼 성숙하지 않다는 점, 문서·toolchain·에러 메시지 어디에서도 직접 trouble-shooting 부담이 크다는 점을 체감했습니다.

### 이 프로젝트의 약점/한계는?
첫째, 단일 디바이스 측정입니다. Galaxy S23 한 대(Snapdragon 8 Gen 2)로만 측정했기 때문에 결과를 그대로 다른 환경에 일반화하기는 어렵습니다. 본 프로젝트의 reference인 Galaxy S26은 8 Gen 3 기반이라 NPU 성능이 더 높고, 실제 운영 환경에서는 본 실험보다 latency가 짧게 나올 가능성이 큽니다.

둘째, 전력 측정의 정확도 한계입니다. 전력은 Android BatteryManager API로 읽은 값이라 NPU 단독 전력을 분리할 수 없고, 절대값 자체의 정확도도 제한적입니다. 다만 충전기 분리·airplane mode·brightness 고정 등 변인을 통제해 모든 실험을 동일 조건에서 측정했기 때문에, 절대값보다 실험 간 상대 차이(burst vs balanced −22%, precision 변경 −19% 등) 해석에 활용했습니다.
