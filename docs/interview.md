# Mobile Inference KPI Lab — Interview Guide

> SD v1.5 text-to-image 파이프라인의 on-device feasibility 프로젝트를 면접에서 설명할 때 사용할 핵심 질문 정리.
> 상세 자료/근거는 [`report/`](report/) 폴더 참조.

---

## 1. 프로젝트 핵심 정의

SD v1.5 기반 text-to-image 파이프라인을 Galaxy S23(Snapdragon 8 Gen 2) NPU에서 직접 실행해 latency·memory·power·thermal·quality를 정량 측정하고, 어떤 조건에서 on-device가 현재 cloud 실행을 대체할 수 있는지 그 boundary를 도출한 feasibility study입니다.

---

## 2. 1분 설명

Samsung Galaxy S26의 Creative Studio 이미지 생성 기능은 현재 SD 계열 모델을 cloud에서 실행해 ~8–25초 latency를 보입니다. 이 기능을 on-device로 옮기면 서버 비용과 네트워크 의존성은 줄어들지만, 대신 device 자원(메모리·전력·발열)을 직접 소비해야 합니다. 이 trade-off가 실제로 어느 조건에서 성립하는지를 정량적으로 확인하기 위한 프로젝트입니다.

저는 Galaxy S23(Snapdragon 8 Gen 2)를 타겟으로, ONNX Runtime + QNN EP 기반 Android 벤치마크 앱을 구축하고, SD v1.5와 LCM-LoRA 두 variant를 4 Phase 실험(precision · step sweep · perf mode · sustained)으로 측정했습니다. 컴포넌트별로 다른 precision(UNet MIXED_PR, VAE W8A8 등)을 QAI Hub으로 compile해 NPU에서 실행하고, latency·power·thermal과 함께 LPIPS·CLIP score로 품질까지 함께 추적했습니다.

핵심 결과는 UNet이 E2E의 95%를 점유한다는 것이고, SD는 quality는 충분(CLIP 35.97)하지만 11.3초로 cloud 기준(~10초)에 미달, LCM은 1.7초로 latency는 충분하지만 CLIP 30.42로 backbone 품질이 부족했습니다. 즉 현재 핸드폰에서 diffusion이 cloud로 가는 이유는 하드웨어 한계가 아니라 quality·latency를 동시에 만족하는 모델 아키텍처가 아직 없다는 것이며, on-device 전환의 전제 조건은 아키텍처 효율 개선이라는 결론을 도출했습니다.

### 1분 설명 직후 예상 꼬리질문

**"feasibility 판정이 구체적으로 무슨 의미인가요?"**
단일 KPI가 아니라 latency·memory·power·thermal·quality 다섯 가지를 동시에 만족하는 운용 가능 영역을 의미합니다. Latency만 보면 LCM은 cloud 기준을 충족하지만 CLIP score가 낮아 다양한 task의 backbone으로 쓰기엔 quality가 부족하고, SD는 quality는 충분하지만 latency가 초과합니다. 이렇게 어느 한 축이라도 무너지면 on-device로 옮길 수 없다는 관점에서 "feasibility"라는 단어를 썼고, 그 boundary를 정량화하는 것이 프로젝트의 목표였습니다.

**"결론이 cloud 유지인데, 이 프로젝트의 의의는 무엇인가요?"**
세 가지입니다. 첫째, "왜 현재 cloud에 둘 수밖에 없는가"라는 질문에 정량적 근거를 제공합니다. 둘째, UNet 95% 지배라는 병목 구조를 정량화함으로써 향후 아키텍처 개선의 leverage point를 명확히 보여줍니다. 셋째, LCM이 1.7초까지 내려간 결과는 하드웨어 성능이 아니라 모델 아키텍처가 병목이라는 것을 증명하는 proof-of-concept입니다. 즉 "SD 수준 quality + LCM 수준 step 효율"을 갖춘 모델이 등장하면 현재 SoC에서도 on-device 전환이 가능하다는 방향성을 제시합니다.

---

## 3. 3분 상세 설명

이 프로젝트는 Galaxy S26 Creative Studio의 이미지 생성 기능이 현재 cloud에서 실행되는 상황에서, 같은 SD 계열 파이프라인을 on-device로 옮길 수 있는지에 대한 feasibility를 정량 분석하기 위해 진행했습니다. 단일 기능 구현이 아니라 "실행 위치 결정의 근거를 만든다"는 관점에서 시작했습니다.

먼저 시스템부터 구축했습니다. Android(Kotlin) 기반 벤치마크 앱에서 ONNX Runtime 1.24.3 + QNN Execution Provider로 NPU 추론을 실행하고, Text Encoder · UNet · VAE Decoder를 각각 별도 OrtRunner 세션으로 관리했습니다. 모델은 PyTorch에서 ONNX(opset 18)로 export한 뒤 QAI Hub으로 Galaxy S23 타겟 QNN context binary(.bin)로 compile했고, 앱에서는 .onnx stub과 .bin을 함께 로드해 HTP에서 실행합니다. 동시에 KpiCollector가 thermal zone 온도, BatteryManager 기반 전력, VmRSS와 NPU 메모리를 1초 간격으로 폴링해 trial 단위로 기록합니다.

양자화는 컴포넌트별로 다르게 접근했습니다. UNet은 INT8 QDQ full compile이 HTP의 LayerNorm INT8 미지원 제약으로 실패해서 Conv·MatMul·Gemm만 INT8로 압축하고 LayerNorm은 FP32로 유지하는 MIXED_PR 전략으로 우회했습니다. VAE Decoder는 Conv 중심 구조라 W8A8(QAI Hub)이 안정적이었고, Text Encoder는 latency 기여가 작아 FP16을 유지했습니다. 품질 평가는 두 단계로 분리했습니다 — 컴포넌트 단위 CosSim으로 빠르게 스크리닝하고, 통과한 config만 E2E 파이프라인을 돌려 LPIPS·CLIP score로 최종 판정했습니다. 이 과정에서 VAE Decoder W8A16의 출력 range가 FP32 reference와 달랐던(qai-hub-models가 Div+Clip postproc을 내장) 디버깅 같은 사례를 거치며 두 단계 평가의 효용을 확인했습니다.

실험은 Phase 1~4로 나눠 진행했습니다. Phase 1에서 precision config을 sweep해 best precision을 선정하고, Phase 2에서 그 best precision으로 step sweep을 통해 step-quality tradeoff를 도출, Phase 3에서 burst와 balanced perf mode를 비교, Phase 4에서 10회 sustained 추론으로 thermal drift를 확인했습니다. 각 trial은 warmup 2회 후 5회 측정의 warm mean을 사용했고, trial 간 cooldown은 최소 60초, 온도 35°C 이상이면 최대 180초까지 대기하도록 했습니다. 충전기 분리·airplane mode·고정 brightness·배터리 40% 이상·시작 온도 35°C 이하 같은 통제 조건을 모두 점검해 측정 신뢰성을 확보했습니다.

결과는 UNet이 E2E의 94~97%를 지배한다는 점이 가장 컸습니다. SD는 CFG로 step당 UNet을 두 번 호출하기 때문에 step 수의 영향이 직접적으로 누적되고, LCM은 guidance_scale=1.0으로 한 번만 호출하면서 step 수도 20→4로 줄어 E2E를 85% 단축할 수 있었습니다(11.3s → 1.7s). SD에서는 best config(MIXED_PR + VAE W8A8)이 E2E −22%, memory −29%, power −19%를 달성했고, balanced mode는 latency +13~14% 손해와 power −22~24% 절감의 trade-off를 보였습니다. Sustained 10회에서도 LCM은 +2.1%, SD는 +0.07% drift로 thermal throttling은 발생하지 않았습니다.

다만 quality 측면에서 두 모델 모두 한계가 있었습니다. SD는 CLIP 35.97로 backbone 품질은 충분하지만 11.3초로 cloud 기준에 미달, LCM은 1.7초로 latency는 충분하지만 CLIP 30.42로 다양한 task의 backbone으로 쓰기엔 quality가 부족합니다. 결과적으로 현재 cloud 전략은 quality·latency를 동시에 만족하는 on-device 모델이 부재한 상황의 합리적 선택이며, on-device 전환의 전제는 "SD 수준 quality + LCM 수준 step 효율"을 갖춘 아키텍처 개선이라는 결론을 도출했습니다.

---

## 4. 꼬리질문

### 4.1 프로젝트 개요

1. 이 프로젝트를 왜 했나요? 어떤 문제에서 출발했나요?
2. 단일 기능 구현이 아니라 "feasibility 분석"으로 정의한 이유는?
3. 왜 SD v1.5인가요? SDXL이나 다른 모델이 아닌 이유는?
4. 본인의 기여 범위는 어디까지인가요?
5. 한 줄로 결론을 말하면?

---

### 4.2 시스템 구조 (Android · ORT · QNN)

1. 앱의 layer 구조와 추론 흐름을 설명해주세요.
2. ONNX Runtime, QNN Execution Provider, QAIRT는 각각 무엇이고 어떻게 연결되나요?
3. 왜 ONNX Runtime을 선택했나요? PyTorch Mobile, TFLite, native QNN과 비교하면?
4. QNN context binary(`.bin`)와 ONNX stub(`.onnx`)을 분리해서 배포하는 이유는?
5. Cold start의 비용은 어디서 오나요? 한 번 발생한 뒤 왜 줄어드나요?

---

### 4.3 모델 준비 · 양자화 전략

1. 왜 컴포넌트별로 다른 precision을 적용했나요? (UNet vs VAE vs Text Encoder)
2. W8A8, MIXED_PR, W8A16의 차이를 설명해주세요.
3. UNet INT8 QDQ full 양자화는 왜 실패했고, 어떻게 우회했나요? (HTP LayerNorm 제약)
4. MIXED_PR이 W8A16보다 bin 크기가 더 큰 이유는?
5. QAI Hub compile job은 어떤 역할을 하나요? TensorRT engine build와 비교하면?

---

### 4.4 품질 평가 방법론 (2단계)

1. 왜 단계 1(컴포넌트 CosSim)과 단계 2(E2E LPIPS/CLIP)를 분리해서 평가했나요?
2. VAE Decoder W8A16 디버깅 스토리를 설명해주세요. (CosSim -0.117 → 0.9999, Div+Clip 발견)
3. CosSim Good 등급인데 실제 on-device 추론은 품질이 무너진 W8A16 사례 — 원인 가설은?
4. LPIPS와 CLIP Score는 각각 무엇을 측정하나요? 왜 둘 다 써야 하나요?
5. LPIPS 0.67인데 "품질 동등"으로 판정한 근거는?

---

### 4.5 실험 방법론

1. Phase 1~4로 나눈 이유와 각 Phase의 목적은?
2. Warmup 2회와 cooldown 60s/180s/35°C 기준은 어떻게 정했나요?
3. 실험 시작 전 통제 조건들(충전기 분리, airplane, brightness 등) — 각각 왜 필요한가요?
4. 측정값을 믿을 수 있게 만들기 위해 어떤 장치를 두었나요? (warm trial mean, GPU vs wall-clock, seed 고정 등)

---

### 4.6 결과 해석

1. 가장 큰 발견 한 가지를 꼽으면? (UNet이 E2E 95% 지배)
2. UNet이 E2E의 95%를 점유하는 이유는? (CFG로 step당 2회 호출 + step 수)
3. SD 11.3s vs LCM 1.7s — LCM이 85% 단축된 메커니즘은?
4. Step sweep에서 UNet per-step이 step 수에 무관하게 일정한 이유는? (NPU 처리량 포화)
5. "SD는 quality 충분 / latency 미달, LCM은 반대" 딜레마의 본질과, 왜 cloud 유지가 합리적이라는 결론을 내렸나요?

---

### 4.7 Performance Mode · Thermal

1. QNN HTP의 burst / balanced / power_saver 모드는 어떻게 다른가요?
2. Balanced를 비교 대상으로 선택한 이유는?
3. Android의 thermal 제어는 어떤 layer에서 일어나나요? (kernel thermal framework, OEM thermal daemon)

---

### 4.8 도메인 지식 (Diffusion · NPU 생태계)

1. Stable Diffusion 파이프라인(Text Encoder → UNet → VAE Decoder)이 어떻게 동작하나요?
2. CFG(Classifier-Free Guidance)가 무엇인가요? guidance_scale 7.5의 의미는?
3. LCM-LoRA의 원리는? (consistency distillation, few-step 가능한 이유)
4. LPIPS가 PSNR보다 사람 지각에 가까운 이유는? (AlexNet feature 기반)
5. Hexagon HTP(NPU)란 어떤 하드웨어인가요? CUDA Tensor Core와 비교?
6. QAIRT란 무엇인가요? CUDA Runtime / TensorRT와 비교?

---

### 4.9 확장 · 심화

1. 고해상도(720/1024) 출력을 하려면? 왜 native가 아닌 512+SR을 권장했나요?
2. Multi-service 확장 시 Shared UNet Backbone + Lightweight Adapter 구조의 아이디어는?
3. 이 경험을 LLM inference 최적화로 옮긴다면 어떻게 접근하겠나요?

---

## 5. 약점 / 한계

1. **단일 디바이스 측정** — Galaxy S23 한 대만으로 결과를 일반화할 수 있는가
2. **Cloud baseline 비대칭** — S26 Ultra의 cloud latency(1024)를 S23 on-device(512)와 비교하는 한계, SR 파이프라인은 검증 안 됨
3. **W8A16 평가 누락** — `unet_base_w8a16`이 S23 inference OOM으로 on-device 품질 평가 불가
4. **통계적 신뢰도** — trial 5회 / prompt set 다양성 한계
5. **Pipeline overlap 미적용** — preprocess/inference/postprocess overlap으로 추가 throughput 개선 여지

---

## 6. Behavioral 질문 대비

1. 이 프로젝트에서 가장 힘들었던 순간은?
2. 가장 크게 배운 점은?
3. 측정값을 믿을 수 없었던 순간과 해결 과정은?
4. 더 하고 싶었지만 못한 것은?
5. 만약 다시 한다면 무엇을 다르게 할 건가요?
