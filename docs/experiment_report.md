# On-Device Generative Image Editing: Feasibility Report (Draft)

---

## 0. Context

### 0.1 문제 설정

본 프로젝트는 diffusion 기반 이미지 편집의 실행 위치를 다음 두 범주로 나누어 분석한다.

- **Global generative editing**: full-image scope (주로 cloud)
- **Local generative editing**: ROI scope (on-device 잠재 영역)

핵심 질문은 다음이다.

- Cloud-class diffusion editing을 mobile on-device에서 어디까지 수행할 수 있는가?
- ROI와 full-image의 비용 차이는 어느 정도인가?
- SD v1.5 대비 LCM이 mobile KPI를 얼마나 개선하는가?

### 0.2 실험 축

- Task Scope: ROI vs Full-image
- Model Variant: SD v1.5 vs LCM
- Resolution: 256 / 512 / 768 / 1024
- Precision: FP16 vs INT8(mixed 포함)
- Backend: CPU / GPU / NPU

### 0.3 측정 KPI

- Latency
- Memory
- Power
- Thermal
- Quality

---

## 1. Current Status

- Design 문서 재정의 완료: feasibility-study framing으로 전환
- 실행 스크립트 정리 진행 중: SD/LCM 비교 파이프라인 점검
- On-device Phase 1/2 실측 결과: 수집 예정

---

## 2. Planned Evaluation

### 2.1 Phase 1 — Single-Run

목적: 조건별 baseline/profile 확보

1. SD v1.5 vs LCM 비교
2. ROI vs Full-image 비교
3. Resolution sweep
4. Precision/backend sweep
5. Step sweep (SD mid/high vs LCM few-step)

### 2.2 Phase 2 — Sustained

목적: 실제 연속 사용에서 안정성 평가

- 상위 config 2~3개 선택
- 연속 실행으로 thermal/power/latency drift 측정

---

## 3. Reporting Structure (to be filled)

### 3.1 Key Findings

- UNet loop 비중
- LCM의 실효 개선폭
- ROI/full-image 비용 배율
- 해상도별 feasibility 경계
- on-device vs cloud decision boundary

### 3.2 Decision Guidance

- on-device feasible region
- offload 권장 region
- 품질/성능 trade-off 권장 운영점

---

## 4. Notes

- 기존 AI Eraser 단일 기능 중심 서술은 본 보고서에서 사용하지 않는다.
- 기능명 대신 scope(global/local)와 시스템 조건(해상도/step/precision/backend) 중심으로 기술한다.
- task-specific adapter(LoRA/conditioning)는 공통 diffusion backbone 위 확장 레이어로 해석한다.
