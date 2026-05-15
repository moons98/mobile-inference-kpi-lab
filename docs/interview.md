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

### 시스템 구조 (Android · ORT · QNN)

#### 1. 앱 구조와 추론 흐름을 설명해주세요.
앱은 Kotlin으로 구성되어 있고, UI에서 실험 세트를 선택하면 벤치마크 로직이 warmup·cooldown을 포함한 trial 루프를 돌면서 latency·전력·온도 결과를 CSV로 저장합니다. 그 아래에 실제 이미지 생성 파이프라인이 있고, ONNX Runtime을 통해 NPU로 연산을 위임하는 구조입니다.

추론 흐름은 입력 텍스트를 토크나이징한 뒤 Text Encoder가 임베딩을 만들고, 순수 노이즈에서 출발해 UNet이 N번 반복하며 노이즈를 제거합니다. 마지막으로 VAE Decoder가 이 결과를 512×512 이미지로 복원합니다. 세 모델은 모두 NPU에서 실행되고, step 사이 스케줄러 연산만 CPU에서 처리됩니다.

#### 2. ONNX Runtime, QNN Execution Provider, QAIRT는 각각 무엇이고 어떻게 연결되나요?
ONNX Runtime은 ONNX 모델을 실행하는 cross-platform inference 엔진입니다. QNN Execution Provider(QNN EP)는 ONNX Runtime의 플러그인으로, 연산을 Qualcomm 하드웨어로 라우팅합니다. QAIRT(Qualcomm AI Runtime)는 QNN EP가 Hexagon HTP와 통신하는 저수준 런타임으로, precompiled QNN context binary(.bin)를 NPU에서 실행합니다.

연결 흐름: Android App → ONNX Runtime → QNN EP → QAIRT → Hexagon HTP. `.onnx` stub 파일이 graph 구조를 담고, `.bin` binary가 HTP에서 실행되는 precompiled graph입니다.

#### 3. 왜 ONNX Runtime을 선택했나요? PyTorch Mobile, LiteRT, native QNN과 비교하면?
Qualcomm이 QNN EP를 ONNX Runtime 위에 공식 제공하고, QAI Hub compile의 input이 ONNX format이라 ecosystem이 일치합니다. PyTorch Mobile은 QNN EP가 없어 HTP 가속이 불가합니다. LiteRT(TFLite)는 TensorFlow 생태계 기반으로 Qualcomm의 공식 QNN EP가 ONNX Runtime에만 있습니다. native QNN API는 ONNX Runtime 없이 직접 HTP를 제어할 수 있지만, graph 로드·I/O 관리·세션 수명 등을 직접 구현해야 해서 개발 비용이 높습니다.

#### 4. Cold start의 비용은 어디서 오나요? 한 번 발생한 뒤 왜 줄어드나요?
Cold start는 session_init(세션 로드)과 first_inference(첫 추론) 두 부분으로 구성됩니다. session_init 단계에서 ONNX Runtime이 .bin QNN context binary를 HTP 메모리에 로드하고 graph를 초기화하는 비용이 발생합니다(Text Encoder + UNet + VAE Decoder 세 모델 합산). first_inference에서는 HTP 내부 JIT 최적화와 buffer 할당, 캐시 적재로 steady-state보다 latency가 높습니다.

한 번 발생한 뒤 줄어드는 이유는 graph가 이미 HTP에 올라가 있고, JIT 최적화와 캐시가 완료되어 이후 추론은 순수 NPU 연산만 수행하기 때문입니다. 이것이 warmup 2회 후 warm trial mean을 측정 기준으로 삼는 이유입니다.

---

### 실험 방법론

#### 1. 실험 시작 전 통제 조건들(충전기 분리, airplane, brightness 등) — 각각 왜 필요한가요?
충전기 분리는 가장 중요한 조건입니다. 충전 중에는 BatteryManager API가 충전 전류와 소비 전류를 혼재해 읽어 전력 측정값이 오염됩니다. Airplane mode는 셀룰러/Wi-Fi background 트래픽이 CPU와 전력에 영향을 주기 때문입니다. 화면 밝기 고정은 디스플레이가 소비전력의 큰 비중을 차지하므로 실험 간 일정하게 유지해야 상대 비교가 유효합니다. 백그라운드 앱 종료는 CPU/메모리 경합으로 latency가 불안정해지는 것을 막습니다. 배터리 40% 이상 유지는 저배터리 시 기기가 자체 throttling을 적용하기 때문이고, 35°C 이하 확인은 이미 thermal throttle 상태에서 시작하면 trial 1부터 결과가 왜곡되기 때문입니다.

#### 2. 측정값을 믿을 수 있게 만들기 위해 어떤 장치를 두었나요? (warm trial mean, wall-clock, seed 고정 등)
warmup 2회를 결과에서 제외하고 이후 5회 warm trial의 평균을 사용했습니다. warmup 중에는 HTP 내부 JIT 최적화와 캐시 적재로 latency가 높고, steady-state 성능을 대표하지 않기 때문입니다. latency는 `System.nanoTime()` 기반 wall-clock으로 측정해 NPU profiler가 아닌 사용자 관점의 E2E 시간을 확보했습니다. seed를 42로 고정해 모든 trial이 동일한 초기 noise z_T에서 출발하도록 했고, 이렇게 하면 이미지 차이가 noise가 아닌 모델(양자화 등)의 영향임을 보장할 수 있습니다. trial 간에는 35°C 이하로 냉각 후 시작(최소 60s, 최대 180s)해 직전 trial의 thermal 상태가 다음 trial에 영향을 주지 않도록 했습니다.

---

### 도메인 지식

#### 1. CFG(Classifier-Free Guidance)란 무엇인가요? 왜 step당 UNet을 두 번 호출하나요?
CFG는 텍스트 프롬프트의 영향을 강하게 반영하기 위한 기법입니다. 매 denoising step에서 UNet을 두 번 실행합니다. 한 번은 텍스트 임베딩과 함께(조건부), 한 번은 null prompt와 함께(비조건부)입니다. 최종 noise prediction은 `uncond + guidance_scale × (cond - uncond)`로 계산되고, guidance_scale(기본 7.5)이 클수록 프롬프트를 더 강하게 반영합니다.

step당 UNet이 두 번 호출되는 것이 SD v1.5에서 UNet이 E2E latency의 95%를 차지하는 핵심 이유입니다. LCM은 consistency distillation으로 CFG 없이도 프롬프트를 잘 반영하도록 학습되어 guidance_scale=1.0으로 step당 한 번만 호출합니다.

#### 2. LPIPS와 CLIP Score는 각각 무엇을 측정하나요? 왜 둘 다 써야 하나요?
LPIPS는 AlexNet 중간 레이어의 feature를 추출해 L2 거리를 계산합니다. 낮을수록 두 이미지가 지각적으로 유사합니다. PSNR이 픽셀 단위 수치 오차만 보는 것과 달리 구조·텍스처 차이를 사람 시각과 유사하게 반영합니다. CLIP Score는 CLIP 모델로 이미지-텍스트 semantic alignment를 측정하며, 높을수록 프롬프트를 잘 반영합니다.

둘 다 필요한 이유는, UNet quantization처럼 denoising trajectory가 달라지면 LPIPS가 높아져도(이미지가 달라도) 프롬프트를 잘 반영한 품질이 동등할 수 있기 때문입니다. LPIPS만으로는 "이미지가 다른 것"과 "품질이 나쁜 것"을 구분할 수 없고, CLIP Score만으로는 시각적 품질을 알 수 없습니다.

---

## 5. 약점 / 한계

### 단일 디바이스 측정

Galaxy S23 한 대만으로 결과를 일반화할 수 있는가에 대한 한계가 있다. Snapdragon 8 Gen 2 기준 결과이며, 다른 SoC에서의 동작은 확인하지 않았다.

### 전력 측정 한계

BatteryManager 기반 시스템 전체 소비전력으로, NPU 단독 전력 분리 불가. 다만 충전기 분리·airplane mode·brightness 고정 등 변인을 통제해 모든 실험을 동일 조건에서 측정했으므로, 절대값보다 실험 간 상대 차이(burst vs balanced −22%, precision 변경 −19% 등) 해석에 활용했다.

---

## 6. Behavioral 질문 대비

### 가장 크게 배운 점은?
모바일 환경에서 측정 조건 통제의 중요성입니다. 충전 상태, thermal 상태, background process 하나로도 latency와 전력이 크게 달라졌고, burst mode를 요청해도 발열이 누적되면 thermal throttling이 개입해 balanced 수준으로 떨어질 수 있음을 실측으로 확인했습니다. 단순히 "실행해보고 숫자 기록하는 것"이 아니라 변인 통제 없이는 수치 자체를 신뢰할 수 없다는 점을 체감했습니다.

### 당면했던 문제와 해결 방안?
두 가지 주요 문제가 있었습니다. 첫 번째는 UNet INT8 full quantization 실패입니다. QAI Hub를 통해 UNet을 W8A8로 quantization하려 했으나, HTP가 LayerNorm INT8 연산을 지원하지 않아 compile 단계에서 실패했습니다. Qualcomm AIMET 라이브러리를 직접 사용해 Conv, MatMul, Gemm만 INT8로 quantization하는 MIXED_PR 방식으로 우회했습니다.

두 번째는 VAE Decoder W8A16의 CosSim이 초기에 -0.117로 측정된 것입니다. 완전히 품질이 망가졌다고 판단해 원인을 추적하니, FP32 기준 모델과 quantized 모델 간에 Div+Clip 연산의 처리 순서가 달라지는 문제였습니다. 수정 후 CosSim 0.9999로 회복됐습니다.
