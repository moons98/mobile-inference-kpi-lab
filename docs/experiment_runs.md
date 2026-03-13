# Experiment Runs

실험 ID별 출력 파일 및 진행 상태 추적.
저장 위치: `outputs/exp/`

실행 순서: Phase 1 → (best precision 선정) → Phase 2 → Phase 3

---

## Phase 1 — Precision Burst

목적: best precision config 선정
설정: steps 고정 (SD 20 / LCM 4), 5 trials, 2 warmup, cooldown 있음

| ID | Model | Steps | Precision | Backend | 상태 | 실제 파일 |
|---|---|---|---|---|---|---|
| A1 | SD v1.5 | 20 | fp16 | QNN NPU | ✅ | txt2img_sd15_fp16_qnn_npu_s20_single_20260313_184651.csv |
| A2 | SD v1.5 | 20 | unet (mixed_pr) | QNN NPU | ✅ | txt2img_sd15_mixed_tfp16_umixed-pr_vfp16_qnn_npu_s20_single_20260313_193258.csv |
| A3 | SD v1.5 | 20 | vae (w8a8) | QNN NPU | ✅ | txt2img_sd15_mixed_tfp16_ufp16_vw8a8_qnn_npu_s20_single_20260313_194200.csv |
| A4 | SD v1.5 | 20 | unet (mixed_pr) + vae (w8a8) | QNN NPU | ✅ | txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_single_20260313_195034.csv |
| A5 | LCM | 4 | fp16 | QNN NPU | ✅ | txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823.csv |
| A6 | LCM | 4 | vae (w8a8) | QNN NPU | ✅ | txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_single_20260313_200423.csv |

품질 평가 (Phase 1 완료 후):

| 대상 실험 | base image | 목적 | 상태 |
|---|---|---|---|
| A1, A2, A3, A4 | A1 출력 | Precision 열화 확인 (SD) | ⬜ |
| A5, A6 | A5 출력 | Precision 열화 확인 (LCM) | ⬜ |

---

## Phase 2 — Step Sweep Burst

목적: step-quality tradeoff 곡선 도출
설정: best precision 고정, 5 trials, 2 warmup, cooldown 있음
> best precision은 Phase 1 결과 후 결정

| ID | Model | Steps | Precision | Backend | 상태 | 실제 파일 |
|---|---|---|---|---|---|---|
| A1 | SD v1.5 | 20 | (Phase 1 공유) | QNN NPU | ⬜ | |
| B1 | SD v1.5 | 30 | best precision | QNN NPU | ⬜ | |
| B2 | SD v1.5 | 50 | best precision | QNN NPU | ⬜ | |
| A5 | LCM | 4 | (Phase 1 공유) | QNN NPU | ⬜ | |
| B3 | LCM | 8 | best precision | QNN NPU | ⬜ | |

품질 평가 (Phase 2 완료 후):

| 대상 실험 | base image | 목적 | 상태 |
|---|---|---|---|
| A1, B1, B2 | B2 출력 | SD step sweep quality | ⬜ |
| A5, B3 | B2 출력 | LCM step sweep quality | ⬜ |
| A1 vs A5 | B2 출력 | SD vs LCM 직접 비교 | ⬜ |

---

## Phase 3 — Balanced Perf Mode

목적: burst 대비 balanced perf mode에서의 latency/power 차이 측정
설정: best config 고정, 5 trials, 2 warmup, cooldown 있음, perf mode = balanced
> Phase 1/2와 동일 조건에서 perf mode만 변경 — background 추론 시나리오
> 대상 config는 Phase 1/2 결과 후 결정

| ID | Model | Steps | Precision | Perf Mode | 상태 | 실제 파일 |
|---|---|---|---|---|---|---|
| C1 | SD v1.5 | best step | best precision | balanced | ⬜ | |
| C2 | LCM | best step | best precision | balanced | ⬜ | |

---

## 분석 커맨드

### Phase 1 — Precision 비교

**전체 (A1~A6)**
```bash
conda run -n mobile python analysis/parse_txt2img_csv.py \
  logs/phase1/txt2img_sd15_fp16_qnn_npu_s20_single_20260313_184651.csv \
  logs/phase1/txt2img_sd15_mixed_tfp16_umixed-pr_vfp16_qnn_npu_s20_single_20260313_193258.csv \
  logs/phase1/txt2img_sd15_mixed_tfp16_ufp16_vw8a8_qnn_npu_s20_single_20260313_194200.csv \
  logs/phase1/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_single_20260313_195034.csv \
  logs/phase1/txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823.csv \
  logs/phase1/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_single_20260313_200423.csv \
  --labels A1 A2 A3 A4 A5 A6 --compare
```

---

### Phase 2 — Step Sweep 비교

> B1~B3 파일은 실험 완료 후 채워넣기

**SD v1.5 step sweep (A1, B1, B2)**
```bash
conda run -n mobile python analysis/parse_txt2img_csv.py \
  logs/phase1/txt2img_sd15_fp16_qnn_npu_s20_single_20260313_184651.csv \
  logs/phase2/<B1_파일명>.csv \
  logs/phase2/<B2_파일명>.csv \
  --labels A1 B1 B2 --compare
```

**LCM step sweep (A5, B3)**
```bash
conda run -n mobile python analysis/parse_txt2img_csv.py \
  logs/phase1/txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823.csv \
  logs/phase2/<B3_파일명>.csv \
  --labels A5 B3 --compare
```

**SD vs LCM 직접 비교 (A1 vs A5)**
```bash
conda run -n mobile python analysis/parse_txt2img_csv.py \
  logs/phase1/txt2img_sd15_fp16_qnn_npu_s20_single_20260313_184651.csv \
  logs/phase1/txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823.csv \
  --labels A1 A5 --compare
```
