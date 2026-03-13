#!/usr/bin/env python3
"""
Phase 4 연속 추론 안정성 분석 — SUSTAINED 10회 per-trial 추세 리포트.
D1: LCM s4 vae_w8a8 / burst
D2: SD  s20 mixed_pr+w8a8 / burst
결과를 outputs/exp/ 에 저장.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from parse_txt2img_csv import parse_csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "exp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRIES = [
    ("D1", "LCM s4  vae_w8a8  burst",
     "logs/phase4/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_sustained_20260313_222148.csv"),
    ("D2", "SD  s20 mixed_pr+w8a8  burst",
     "logs/phase4/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_sustained_20260313_221930.csv"),
]

W   = 108
sep  = "=" * W
sep2 = "-" * W

lines = []
lines.append(sep)
lines.append("Phase 4 — Sustained Inference Stability (10 consecutive trials, burst mode)")
lines.append(sep)
lines.append("")
lines.append("  목적: best config(LCM/SD)을 쿨다운 없이 10회 연속 실행해 latency 안정성과 thermal drift 측정")
lines.append("  측정: E2E latency, UNet/step latency, 온도(시작→종료), drift = (trial10 - trial1) / trial1 × 100")
lines.append("")

for eid, label, rel_path in ENTRIES:
    csv_path = PROJECT_ROOT / rel_path
    bench = parse_csv(csv_path)
    df = bench.generate_summary
    udf = bench.unet_step_detail

    trials = sorted(df["trial_id"].unique())
    n = len(trials)

    lines.append(sep2)
    lines.append(f"  [{eid}] {label}  (n={n} trials)")
    lines.append(sep2)

    # per-trial table
    lines.append(f"  {'Trial':>6}  {'E2E (ms)':>10}  {'UNet/step (ms)':>16}  {'Temp start':>11}  {'Temp end':>9}  {'Avg Power (mW)':>15}")
    lines.append(f"  {'------':>6}  {'----------':>10}  {'----------------':>16}  {'-----------':>11}  {'---------':>9}  {'---------------':>15}")

    e2e_vals  = []
    step_vals = []
    for tid in trials:
        row = df[df["trial_id"] == tid].iloc[0]
        e2e   = row["generate_e2e_ms"]
        unet  = row["unet_per_step_mean_ms"]
        ts    = row["start_temp_c"]
        te    = row["end_temp_c"]
        pwr   = row["avg_power_mw"]
        e2e_vals.append(e2e)
        step_vals.append(unet)
        lines.append(f"  {tid:>6}  {e2e:>10.1f}  {unet:>16.2f}  {ts:>10.1f}°  {te:>8.1f}°  {pwr:>14.1f}")

    lines.append("")

    # drift summary
    e2e_drift_ms  = e2e_vals[-1]  - e2e_vals[0]
    e2e_drift_pct = e2e_drift_ms  / e2e_vals[0]  * 100
    step_drift_ms = step_vals[-1] - step_vals[0]
    step_drift_pct= step_drift_ms / step_vals[0] * 100

    temp_start = df[df["trial_id"] == trials[0]].iloc[0]["start_temp_c"]
    temp_end   = df[df["trial_id"] == trials[-1]].iloc[0]["end_temp_c"]
    temp_rise  = temp_end - temp_start

    e2e_mean  = sum(e2e_vals)  / n
    e2e_min   = min(e2e_vals)
    e2e_max   = max(e2e_vals)
    e2e_range = e2e_max - e2e_min

    lines.append(f"  E2E        : trial1={e2e_vals[0]:.1f}ms  trial{n}={e2e_vals[-1]:.1f}ms  "
                 f"drift={e2e_drift_ms:+.1f}ms ({e2e_drift_pct:+.2f}%)")
    lines.append(f"  UNet/step  : trial1={step_vals[0]:.2f}ms  trial{n}={step_vals[-1]:.2f}ms  "
                 f"drift={step_drift_ms:+.2f}ms ({step_drift_pct:+.2f}%)")
    lines.append(f"  E2E range  : min={e2e_min:.1f}ms  max={e2e_max:.1f}ms  spread={e2e_range:.1f}ms")
    lines.append(f"  Temp rise  : {temp_start:.1f}°C → {temp_end:.1f}°C  (Δ{temp_rise:+.1f}°C)")
    lines.append("")

lines.append(sep2)
lines.append("")
lines.append(sep)
lines.append("")
lines.append("Interpretation")
lines.append(sep2)
lines.append("  [LCM D1]")
lines.append("  - 10회 연속 burst에서 +2.1% E2E drift 확인 (1,638ms → 1,673ms, Δ+35ms)")
lines.append("  - 온도 +9.4°C 상승(44.6→54.0°C)에도 불구, 최종 latency는 cloud 대비 ×0.21 유지")
lines.append("  - drift는 UNet step 시간 증가(339→343ms, +1.2%)에서 기인 — HTP burst clock 미세 감소")
lines.append("  - 결론: 연속 추론 환경에서도 실용적 latency 유지 ✅")
lines.append("")
lines.append("  [SD D2]")
lines.append("  - 10회 연속 burst에서 +0.07% E2E drift (11,108ms → 11,116ms, Δ+8ms) — 사실상 flat")
lines.append("  - SD는 11s 단위 장시간 추론 → NPU가 sustained clock으로 이미 안정, thermal 영향 미미")
lines.append("  - SD 자체는 cloud(8s) 대비 미달(×1.39)이므로 sustained 안정성은 참고 데이터")
lines.append(sep2)

report = "\n".join(lines)
print("\n" + report)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = OUTPUT_DIR / f"sustained_phase4_{ts}.txt"
out_path.write_text(report, encoding="utf-8")
print(f"\nSaved to {out_path}")
