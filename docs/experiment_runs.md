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
| A1, A2, A3, A4 | A1 출력 | Precision 열화 확인 (SD) | ✅ |
| A5, A6 | A5 출력 | Precision 열화 확인 (LCM) | ✅ |

결과 파일: `outputs/exp/quality_phase1_20260313_203830.txt`

**Best Precision 선정:**
- SD v1.5: **A4** (unet mixed_pr + vae w8a8) — E2E -22%, NPU mem -32%, 전력 -19%, CLIP score 유지 (+0.21)
- LCM: **A6** (vae w8a8) — E2E -16%, 전력 -12%, LPIPS 0.03 (사실상 동일)

→ Phase 2 기준 precision: SD = mixed_pr + w8a8 / LCM = vae w8a8

---

## Phase 2 — Step Sweep Burst

목적: step-quality tradeoff 곡선 도출
설정: best precision 고정, 5 trials, 2 warmup, cooldown 있음
> best precision: SD = unet mixed_pr + vae w8a8 (A4), LCM = vae w8a8 (A6)

| ID | Model | Steps | Precision | Backend | 상태 | 실제 파일 |
|---|---|---|---|---|---|---|
| A4 | SD v1.5 | 20 | unet mixed_pr + vae w8a8 (Phase 1 공유) | QNN NPU | ✅ | txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_single_20260313_195034.csv |
| B1 | SD v1.5 | 30 | unet mixed_pr + vae w8a8 | QNN NPU | ✅ | txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s30_single_20260313_210149.csv |
| B2 | SD v1.5 | 50 | unet mixed_pr + vae w8a8 | QNN NPU | ✅ | txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s50_single_20260313_211043.csv |
| A5 | LCM | 4 | fp16 (Phase 1 공유) | QNN NPU | ✅ | txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823.csv |
| B3 | LCM | 8 | vae w8a8 | QNN NPU | ✅ | txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s8_single_20260313_212940.csv |

> SD s20 기준점: precision 통일을 위해 A1(fp16) 대신 A4(mixed_pr+w8a8) 이미지 사용.

품질 평가 (Phase 2 완료 후):

| 대상 실험 | base image | 목적 | 상태 |
|---|---|---|---|
| A4, B1, B2 | B2 출력 | SD step sweep quality | ✅ |
| A5, B3 | B2 출력 | LCM step sweep quality | ✅ |
| A4 vs A5 | B2 출력 | SD vs LCM 직접 비교 | ✅ |

결과 파일: `outputs/exp/quality_phase2_20260313_213141.txt` / `outputs/exp/txt2img_comparison_20260313_213247.txt`

**Phase 2 결과 요약:**

SD step sweep (precision 고정: mixed_pr + vae w8a8):
- E2E: A4(s20) 11,286ms / B1(s30) 16,535ms / B2(s50) 27,441ms — step당 ~540ms 선형 증가
- CLIP: A4=35.97, B1=34.06, B2=35.69 — 모두 >33(양호), 유의미한 차이 없음
- → **s20이 품질 유지하면서 가장 빠름**

LCM step sweep:
- E2E: A5(s4 fp16) 2,023ms / B3(s8 vae w8a8) 3,040ms (+1,017ms)
- CLIP: A5=30.13, B3=30.91 — 차이 0.78 (±2 이내, 유의미하지 않음)
- VAEDec: A5 468ms → B3 127ms (vae w8a8 효과)
- → **s4가 s8 대비 절반 latency에 품질 동등**

SD vs LCM 비교 (B2 기준 LPIPS):
- LCM 전체가 LPIPS >0.70 (완전히 다른 이미지) — 모델·scheduler 차이로 trajectory 분기
- CLIP: SD 35~36 vs LCM 30~31 — SD가 prompt 반영도 높음, LCM은 "보통(27~33)" 수준

**Best Config 선정 (Phase 2):**
- SD: **A4** (s20, mixed_pr + vae w8a8) — step 늘려도 품질 이득 없음
- LCM: **A6** (s4, vae w8a8) — s8 대비 절반 latency, 품질 동등

---

## Phase 3 — Balanced Perf Mode

목적: burst 대비 balanced perf mode에서의 latency/power 차이 측정
설정: best config 고정, 5 trials, 2 warmup, cooldown 있음, perf mode = balanced
> Phase 1/2와 동일 조건에서 perf mode만 변경 — background 추론 시나리오
> 대상 config는 Phase 1/2 결과 후 결정

| ID | Model | Steps | Precision | Perf Mode | 상태 | 실제 파일 |
|---|---|---|---|---|---|---|
| C1 | SD v1.5 | 20 | unet mixed_pr + vae w8a8 | balanced | ✅ | txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_single_20260313_214205.csv |
| C2 | LCM | 4 | vae w8a8 | balanced | ✅ | txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_single_20260313_214749.csv |

결과 파일: `outputs/exp/txt2img_comparison_20260313_214943.txt`

**Phase 3 결과 — Burst vs Balanced 비교:**

SD v1.5 s20 mixed_pr+w8a8 (A4 burst → C1 balanced):
- E2E: 11,286ms → 12,789ms (+13%)
- UNet/step: 546.7ms → 618.5ms (+13%)
- 전력 delta: +4,031mW → +2,929mW (−27%)
- 온도 end: 46.5°C → 45.0°C (−1.5°C)

LCM s4 vae w8a8 (A6 burst → C2 balanced):
- E2E: 1,693ms → 1,931ms (+14%)
- UNet/step: 341.0ms → 395.1ms (+16%)
- 전력 delta: +3,389mW → +2,482mW (−27%)
- 온도 end: 37.5°C → 37.9°C (거의 동일)

→ **balanced mode: latency +13~14% 증가, 전력 −27% 절감. background 추론 시나리오에서 trade-off 유효.**

---

## Phase 4 — Sustained Stability

목적: best config(LCM/SD)를 burst 모드로 쿨다운 없이 10회 연속 실행해 latency 안정성과 thermal drift 측정
설정: trials=10, cooldown 없음, burst mode

| ID | Model | Steps | Precision | Perf Mode | 상태 | 실제 파일 |
|---|---|---|---|---|---|---|
| D1 | LCM | 4 | vae w8a8 | burst | ✅ | txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_sustained_20260313_222148.csv |
| D2 | SD v1.5 | 20 | unet mixed_pr + vae w8a8 | burst | ✅ | txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_sustained_20260313_221930.csv |

결과 파일: `outputs/exp/sustained_phase4_20260313_222755.txt`

**Phase 4 결과 — Sustained 10회 Drift:**

LCM s4 vae_w8a8 burst (D1):
- E2E: trial1=1,637.6ms → trial10=1,672.9ms (drift=+2.15%, +35ms)
- UNet/step: 339ms → 343ms (+1.2%) — HTP burst clock 미세 감소
- 온도: 44.6°C → 54.0°C (+9.4°C)

SD s20 mixed_pr+w8a8 burst (D2):
- E2E: trial1=11,108.2ms → trial10=11,116.0ms (drift=+0.07%, +8ms) — 사실상 flat
- 온도: 45.0°C → 53.2°C (+8.2°C)

→ **두 config 모두 10회 연속 burst 추론에서 full thermal throttling 없이 실용적 latency 유지.**

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

**SD v1.5 step sweep (A4, B1, B2)**
```bash
conda run -n mobile python analysis/parse_txt2img_csv.py \
  logs/phase1/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_single_20260313_195034.csv \
  logs/phase2/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s30_single_20260313_210149.csv \
  logs/phase2/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s50_single_20260313_211043.csv \
  --labels A4 B1 B2 --compare
```

**LCM step sweep (A6, B3)**
```bash
conda run -n mobile python analysis/parse_txt2img_csv.py \
  logs/phase1/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_single_20260313_200423.csv \
  logs/phase2/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s8_single_20260313_212940.csv \
  --labels A6 B3 --compare
```

**SD vs LCM 직접 비교 (A4 vs A6)**
```bash
conda run -n mobile python analysis/parse_txt2img_csv.py \
  logs/phase1/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_single_20260313_195034.csv \
  logs/phase1/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_single_20260313_200423.csv \
  --labels A4 A6 --compare
```

---

### Phase 4 — Sustained Stability 분석

```bash
conda run -n mobile python analysis/parse_phase4_sustained.py
```
