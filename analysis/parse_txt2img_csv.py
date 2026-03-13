#!/usr/bin/env python3
"""
Parse txt2img benchmark CSV files exported from the Android app.
Supports: GENERATE_SUMMARY, UNET_STEP_DETAIL, COLD_START sections.

Usage:
    python parse_txt2img_csv.py <csv_file_or_directory>
    python parse_txt2img_csv.py results/ --compare
    python parse_txt2img_csv.py results/ -p   # print only, no save
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from io import StringIO
import sys
import argparse


@dataclass
class ParsedBenchmark:
    """Parsed benchmark result from one CSV file."""
    filepath: str
    metadata: dict = field(default_factory=dict)
    generate_summary: Optional[pd.DataFrame] = None
    unet_step_detail: Optional[pd.DataFrame] = None
    cold_start: Optional[pd.DataFrame] = None


SECTION_NAMES = {"GENERATE_SUMMARY", "UNET_STEP_DETAIL", "COLD_START"}


def parse_csv(filepath: Path) -> ParsedBenchmark:
    """Parse a multi-section CSV file into dataframes."""
    result = ParsedBenchmark(filepath=str(filepath))

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract metadata (# key,value lines at top)
    for line in lines:
        line = line.strip()
        if line.startswith("# ") and "," in line:
            tag = line[2:].split(",")[0].strip()
            if tag not in SECTION_NAMES:
                parts = line[2:].split(",", 1)
                if len(parts) == 2:
                    result.metadata[parts[0].strip()] = parts[1].strip()

    # Find section boundaries
    sections = {}
    current_section = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and stripped[2:] in SECTION_NAMES:
            if current_section and current_lines:
                sections[current_section] = current_lines
            current_section = stripped[2:]
            current_lines = []
        elif current_section and stripped and not stripped.startswith("#"):
            current_lines.append(stripped)

    if current_section and current_lines:
        sections[current_section] = current_lines

    # Parse each section
    for name, data_lines in sections.items():
        if len(data_lines) < 2:
            continue
        csv_text = "\n".join(data_lines)
        df = pd.read_csv(StringIO(csv_text))

        if name == "GENERATE_SUMMARY":
            result.generate_summary = df
        elif name == "UNET_STEP_DETAIL":
            result.unet_step_detail = df
        elif name == "COLD_START":
            result.cold_start = df

    return result


# ---------------------------------------------------------------------------
# Single-file report
# ---------------------------------------------------------------------------

def format_report(bench: ParsedBenchmark) -> str:
    lines = []
    W = 110
    sep = "=" * W
    sep2 = "-" * W
    m = bench.metadata

    lines.append(sep)
    lines.append("Txt2Img KPI Report")
    lines.append(sep)

    # --- [1] Experiment Overview ---
    lines.append("")
    lines.append("[1] Experiment Overview")
    lines.append(sep2)
    lines.append(f"  File:        {Path(bench.filepath).name}")
    lines.append(f"  Device:      {m.get('device_model','?')} ({m.get('soc_model','?')})")
    lines.append(f"  Runtime:     {m.get('runtime','?')}")
    lines.append(f"  Phase:       {m.get('phase','?')}")
    lines.append(f"  Variant:     {m.get('model_variant','?')}")
    lines.append(f"  Backend:     {m.get('sd_backend','?')}  |  Precision: {m.get('sd_precision','?')}")
    if m.get('sd_precision_per_component'):
        lines.append(f"  Per-comp:    {m.get('sd_precision_per_component','')}")
    lines.append(f"  Steps:       {m.get('steps','?')}  |  Guidance: {m.get('guidance_scale','?')}")
    if m.get('thermal_zone_type'):
        lines.append(f"  Thermal:     {m.get('thermal_zone_type')}")
    if m.get('is_charging', '').lower() == 'true':
        lines.append(f"  ⚠ WARNING:   충전 중 측정 — 전력 데이터 신뢰도 낮음")

    # --- [2] Cold Start ---
    if bench.cold_start is not None and not bench.cold_start.empty:
        cs = bench.cold_start.iloc[0]
        lines.append("")
        lines.append("[2] Cold Start Breakdown")
        lines.append(sep2)
        lines.append(f"  {'Component':<25} {'Time (ms)':>12}")
        lines.append(f"  {'-'*25} {'-'*12}")

        components = [
            ("Text Encoder", "text_enc_load_ms"),
            ("UNet", "unet_load_ms"),
            ("VAE Decoder", "vae_dec_load_ms"),
        ]
        for label, col in components:
            val = cs.get(col, 0)
            if pd.notna(val) and val > 0:
                lines.append(f"  {label:<25} {val:>12,.0f}")

        lines.append(f"  {'='*25} {'='*12}")
        lines.append(f"  {'Total (sum)':<25} {cs.get('total_load_ms', 0):>12,.0f}")

        # Init wall-clock vs sum comparison
        init_wc = cs.get('init_wall_clock_ms', None)
        if pd.notna(init_wc) and init_wc > 0:
            parallel = cs.get('parallel_init', False)
            mode = "parallel" if parallel else "sequential"
            lines.append(f"  {'Init wall-clock':<25} {init_wc:>12,.0f}  ({mode})")
            total_sum = cs.get('total_load_ms', 0)
            if total_sum > 0:
                savings = total_sum - init_wc
                pct = savings / total_sum * 100 if total_sum > 0 else 0
                if abs(savings) > 50:  # only show if meaningful difference
                    lines.append(f"  {'Parallel savings':<25} {savings:>12,.0f}  ({pct:.1f}%)")

        # Memory baseline vs after load
        idle_mem = cs.get('idle_memory_mb', 0)
        peak_mem = cs.get('peak_memory_after_load_mb', 0)
        if pd.notna(idle_mem) and idle_mem > 0:
            mem_delta = cs.get('memory_delta_mb', peak_mem - idle_mem)
            lines.append(f"  Idle memory (before load):  {idle_mem} MB")
            lines.append(f"  Memory after load:          {peak_mem} MB  (model delta: +{mem_delta} MB)")
        else:
            lines.append(f"  Peak memory after load:     {peak_mem} MB")

        # First inference + cold start total
        first_inf = cs.get('first_inference_wall_clock_ms', None)
        if pd.notna(first_inf) and first_inf > 0:
            lines.append(f"  First inference (cold):     {first_inf:,.0f} ms")
            cs_total = cs.get('cold_start_total_ms', 0)
            if pd.notna(cs_total) and cs_total > 0:
                lines.append(f"  Cold start total:           {cs_total:,.0f} ms  (init wall-clock + first inference)")
        warmup_ms = cs.get('warmup_total_ms', None)
        if pd.notna(warmup_ms) and warmup_ms > 0:
            lines.append(f"  Warmup total:               {warmup_ms:,.0f} ms")

        # Idle thermal/power
        idle_thermal = cs.get('idle_thermal_c', 0)
        idle_power = cs.get('idle_power_mw', 0)
        if pd.notna(idle_thermal) and idle_thermal > 0:
            lines.append(f"  Idle thermal:               {idle_thermal:.1f} C")
        if pd.notna(idle_power) and idle_power > 0:
            lines.append(f"  Idle power:                 {idle_power:.0f} mW")

    # --- [3] Generation Summary ---
    if bench.generate_summary is not None and not bench.generate_summary.empty:
        df = bench.generate_summary
        n = len(df)
        lines.append("")
        lines.append(f"[3] Generation E2E Performance ({n} trials)")
        lines.append(sep2)

        # Prompt (first trial)
        prompt = df["prompt"].iloc[0] if "prompt" in df.columns else "?"
        if len(prompt) > 80:
            prompt = prompt[:77] + "..."
        lines.append(f"  Prompt: \"{prompt}\"")
        lines.append("")

        # Stage breakdown table
        stage_cols = {
            "tokenize_ms": "Tokenize",
            "text_enc_ms": "Text Encoder",
            "unet_total_ms": "UNet Total",
            "vae_dec_ms": "VAE Decoder",
            "generate_e2e_ms": "Generate E2E",
        }

        lines.append(f"  {'Stage':<25} {'Mean':>10} {'P50':>10} {'P95':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
        lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for col, label in stage_cols.items():
            if col in df.columns and df[col].mean() > 0.001:
                s = df[col]
                lines.append(f"  {label:<25} {s.mean():>10.1f} {s.median():>10.1f} "
                             f"{s.quantile(0.95):>10.1f} {s.min():>10.1f} {s.max():>10.1f} {s.std():>10.1f}")

        # Sum check
        component_cols = ["tokenize_ms", "text_enc_ms", "unet_total_ms", "vae_dec_ms"]
        available_cols = [c for c in component_cols if c in df.columns]
        if available_cols and "generate_e2e_ms" in df.columns:
            component_sum = df[available_cols].sum(axis=1)
            gap = df["generate_e2e_ms"] - component_sum
            lines.append(f"\n  Sum check:  E2E={df['generate_e2e_ms'].mean():.1f}  "
                         f"Components={component_sum.mean():.1f}  "
                         f"Gap={gap.mean():.1f}ms ({gap.mean()/df['generate_e2e_ms'].mean()*100:.1f}%)")

        # Wall-clock vs component-sum gap
        if "pipeline_wall_clock_ms" in df.columns and df["pipeline_wall_clock_ms"].mean() > 0:
            wc = df["pipeline_wall_clock_ms"]
            e2e = df["generate_e2e_ms"]
            gap = wc - e2e
            lines.append(f"\n  Wall-clock check:  Pipeline={wc.mean():.1f}  "
                         f"ComponentSum={e2e.mean():.1f}  "
                         f"Overhead={gap.mean():.1f}ms ({gap.mean()/wc.mean()*100:.1f}%)")

        if "trial_wall_clock_ms" in df.columns and df["trial_wall_clock_ms"].mean() > 0:
            twc = df["trial_wall_clock_ms"]
            lines.append(f"  Trial wall-clock:  {twc.mean():.1f}ms (includes metric collection overhead)")

        # UNet per-step stats
        if "unet_per_step_mean_ms" in df.columns:
            lines.append("")
            lines.append(f"  [3b] UNet Per-Step Statistics")
            lines.append(f"  {'-'*70}")
            lines.append(f"  Per-step mean:       {df['unet_per_step_mean_ms'].mean():.2f} ms")
            if "unet_per_step_p95_ms" in df.columns:
                lines.append(f"  Per-step P95:        {df['unet_per_step_p95_ms'].mean():.2f} ms")
            if "scheduler_overhead_ms" in df.columns:
                lines.append(f"  Scheduler overhead:  {df['scheduler_overhead_ms'].mean():.2f} ms (total across all steps)")
            steps = df["actual_steps"].iloc[0] if "actual_steps" in df.columns else int(m.get("steps", 0))
            if steps > 0:
                unet_ms = df["unet_total_ms"].mean()
                lines.append(f"  UNet total / steps:  {unet_ms:.1f} / {steps} = {unet_ms/steps:.1f} ms/step")

        lines.append("")
        lines.append("    Tokenize:      CLIP tokenizer (prompt -> input_ids)")
        lines.append("    Text Encoder:  CLIP text model (input_ids -> embeddings)")
        lines.append("    UNet Total:    N denoising steps (4ch txt2img UNet)")
        lines.append("    VAE Decoder:   latent -> RGB 512x512 image")

    # --- [4] UNet Step Detail ---
    if bench.unet_step_detail is not None and not bench.unet_step_detail.empty:
        df = bench.unet_step_detail
        lines.append("")
        lines.append(f"[4] UNet Step Detail ({len(df)} total steps across all trials)")
        lines.append(sep2)

        _append_stat_table(lines, df, {
            "session_run_ms": "Session Run",
            "input_create_ms": "Input Create",
            "output_copy_ms": "Output Copy",
            "scheduler_step_ms": "Scheduler Step",
            "step_total_ms": "Step Total",
        })

        # Per-trial step time trend (first vs last step)
        if len(df) > 1:
            trials = df["trial_id"].unique()
            if len(trials) > 0:
                t1 = df[df["trial_id"] == trials[0]]
                if len(t1) >= 2:
                    first_step = t1["session_run_ms"].iloc[0]
                    last_step = t1["session_run_ms"].iloc[-1]
                    lines.append(f"\n  Trial 1 step trend:  first={first_step:.1f}ms  last={last_step:.1f}ms  "
                                 f"delta={last_step - first_step:+.1f}ms")

        lines.append("")
        lines.append("    Session Run:     ORT session.run() for one UNet step")
        lines.append("    Input Create:    latent concat -> OnnxTensor")
        lines.append("    Output Copy:     OnnxTensor -> FloatArray")
        lines.append("    Scheduler Step:  EulerDiscrete noise scheduling")

    # --- [5] System Resources ---
    if bench.generate_summary is not None and not bench.generate_summary.empty:
        df = bench.generate_summary
        has_thermal = "start_temp_c" in df.columns and df["start_temp_c"].mean() > 0
        has_power = "avg_power_mw" in df.columns and df["avg_power_mw"].mean() > 0
        has_memory = "peak_memory_mb" in df.columns

        if has_thermal or has_power or has_memory:
            lines.append("")
            lines.append("[5] System Resources")
            lines.append(sep2)

            if has_thermal:
                lines.append(f"  Thermal start:    {df['start_temp_c'].iloc[0]:.1f} C  (trial 1)")
                lines.append(f"  Thermal end:      {df['end_temp_c'].iloc[-1]:.1f} C  (trial {len(df)})")
                delta = df['end_temp_c'].iloc[-1] - df['start_temp_c'].iloc[0]
                lines.append(f"  Thermal delta:    {delta:+.1f} C")
                if len(df) > 1:
                    lines.append(f"  Per-trial temps:  " +
                                 "  ".join(f"T{i+1}:{r['start_temp_c']:.0f}->{r['end_temp_c']:.0f}"
                                           for i, r in df.iterrows()))

            if has_power:
                avg_power = df['avg_power_mw'].mean()
                lines.append(f"  Avg power:        {avg_power:.0f} mW")
                # Idle baseline + delta power
                idle_base_power = float(m.get('idle_baseline_power_mw', 0))
                if idle_base_power > 0:
                    delta_power = avg_power - idle_base_power
                    lines.append(f"  Idle baseline:    {idle_base_power:.0f} mW  (5s median)")
                    lines.append(f"  Delta power:      {delta_power:+.0f} mW  (inference - idle)")
                if m.get('is_charging', '').lower() == 'true':
                    lines.append(f"  ⚠ 충전 중 측정 — 전력 데이터 신뢰도 낮음")

            if has_memory:
                lines.append(f"  Peak VmRSS:       {df['peak_memory_mb'].max()} MB")
            if "native_heap_mb" in df.columns and df["native_heap_mb"].max() > 0:
                lines.append(f"  Peak Native Heap: {df['native_heap_mb'].max():.1f} MB")
            if "pss_mb" in df.columns and df["pss_mb"].max() > 0:
                lines.append(f"  Peak PSS:         {df['pss_mb'].max():.1f} MB")

            # Thermal zone type
            zone_type = m.get('thermal_zone_type', '')
            if zone_type:
                lines.append(f"  Thermal zone:     {zone_type}")

    # --- [6] Graph Partitioning ---
    ort_total = int(m.get('ort_total_nodes', 0))
    ort_qnn = int(m.get('ort_qnn_nodes', 0))
    ort_cpu = int(m.get('ort_cpu_nodes', 0))
    ort_fallback = m.get('ort_fallback_ops', '')

    lines.append("")
    lines.append("[6] Graph Partitioning & Coverage")
    lines.append(sep2)
    if ort_total > 0:
        coverage = (ort_qnn / ort_total * 100) if ort_total > 0 else 0
        lines.append(f"  Total nodes:     {ort_total}")
        lines.append(f"  QNN nodes:       {ort_qnn}")
        lines.append(f"  CPU fallback:    {ort_cpu}")
        lines.append(f"  Coverage:        {coverage:.1f}%")
        if ort_fallback:
            lines.append(f"  Fallback ops:    {ort_fallback.replace(';', ', ')}")
        else:
            lines.append(f"  Fallback ops:    None (100% QNN offload)")
    else:
        lines.append(f"  No ORT graph partitioning data captured.")
        lines.append(f"  (ORT verbose logging이 캡처되지 않았거나 logcat 버퍼 overflow)")

    # --- Methodology Notes ---
    lines.append("")
    _append_methodology_notes(lines)

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def _append_methodology_notes(lines):
    """Append measurement methodology notes."""
    lines.append("─" * 110)
    lines.append("Measurement Notes")
    lines.append("─" * 110)
    lines.append("  [Latency]")
    lines.append("    - System.nanoTime() 기반 wall-clock 측정 (monotonic clock)")
    lines.append("    - Tokenize/TextEnc/UNet/VAEDec 각 구간은 ORT session.run() 호출 전후 시간차")
    lines.append("    - E2E (component sum) = Tokenize + TextEnc + NoiseGen + UNet Total + VAEDec")
    lines.append("    - Pipeline wall-clock: generate() 함수 전체 실행 시간 (System.nanoTime)")
    lines.append("    -   E2E(합산)와의 차이 = 버퍼 할당, GC, thread scheduling 등 overhead")
    lines.append("    - Trial wall-clock: 온도/전력 측정 포함 전체 trial 시간")
    lines.append("    - Warmup trial은 통계에서 제외됨 (default 2회)")
    lines.append("")
    lines.append("  [Memory]")
    lines.append("    - VmRSS: /proc/self/status Resident Set Size (물리 메모리, shared lib 포함)")
    lines.append("    - Native Heap: Debug.getNativeHeapAllocatedSize() (ORT/QNN native 할당 포함)")
    lines.append("    - PSS: ActivityManager.getProcessMemoryInfo() totalPss (공유 라이브러리 비례 배분)")
    lines.append("    -   PSS는 측정 비용이 높아 trial 단위로만 측정 (step 단위 미측정)")
    lines.append("    - Idle: 모델 로드 전 측정 / Peak after load: 세션 생성 직후 측정")
    lines.append("")
    lines.append("  [Thermal]")
    lines.append("    - /sys/class/thermal/thermal_zone{N}/temp 읽기 (millidegree → °C 변환)")
    lines.append("    - 읽기 가능한 첫 번째 thermal zone 사용, zone type 기록 (예: cpu-0-0)")
    lines.append("    - Fallback: BatteryManager 배터리 온도 (SoC 온도보다 낮고 지연됨)")
    lines.append("    - Trial 간 cooldown (60s) 적용 시 start_temp은 쿨다운 후 온도")
    lines.append("")
    lines.append("  [Power]")
    lines.append("    - BatteryManager.BATTERY_PROPERTY_CURRENT_NOW × 배터리 전압으로 계산")
    lines.append("    - 순간 소비 전력 (mW) = |전류(μA)| × 전압(mV) / 10^6")
    lines.append("    - Idle baseline: 모델 로드 전 5초간 500ms 간격 10회 샘플 중위값")
    lines.append("    - Delta power: 추론 평균 전력 - idle baseline (순수 추론 소비 전력 추정)")
    lines.append("    - 충전 상태 감지: BatteryManager.EXTRA_STATUS 확인, 충전 시 경고 표시")
    lines.append("    - ⚠ 기기별 전류 단위 불일치 (μA vs mA), 시스템 전체 소비 전력")
    lines.append("")
    lines.append("  [Cold Start]")
    lines.append("    - 각 컴포넌트(TextEnc/UNet/VAEDec) ORT 세션 생성 시간")
    lines.append("    - QNN EP 사용 시 HTP 그래프 컴파일 시간이 포함됨")
    lines.append("    - First inference: 모델 로드 직후 첫 generate() 호출 wall-clock")
    lines.append("    -   JIT 컴파일, 캐시 warming 등 일회성 오버헤드 포함")
    lines.append("    - Cold start total = 세션 생성 + 첫 추론")
    lines.append("    - 앱 최초 실행 또는 모델 교체 시에만 발생 (세션 재사용 시 skip)")
    lines.append("")
    lines.append("  [Graph Partitioning]")
    lines.append("    - ORT verbose 로그에서 파싱 (logcat 캡처 필요)")
    lines.append("    - QNN 노드: NPU/GPU에서 실행되는 연산 수")
    lines.append("    - CPU fallback: QNN EP가 지원하지 않아 CPU에서 실행되는 연산")
    lines.append("    - logcat 버퍼 overflow 시 캡처 실패 가능")


def _append_stat_table(lines, df, col_map):
    """Append a statistics table (Mean/P50/P95/Min/Max/Std)."""
    lines.append(f"  {'Metric':<25} {'Mean':>10} {'P50':>10} {'P95':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for col, label in col_map.items():
        if col in df.columns and df[col].mean() > 0.001:
            s = df[col]
            lines.append(f"  {label:<25} {s.mean():>10.2f} {s.median():>10.2f} "
                         f"{s.quantile(0.95):>10.2f} {s.min():>10.2f} {s.max():>10.2f} {s.std():>10.2f}")


# ---------------------------------------------------------------------------
# Multi-file comparison
# ---------------------------------------------------------------------------

def format_comparison(benchmarks: list) -> str:
    lines = []
    W = 110
    sep = "=" * W
    sep2 = "-" * W

    lines.append(f"Found {len(benchmarks)} file(s)")
    lines.append(sep)
    lines.append("Txt2Img KPI Comparison Report")
    lines.append(sep)

    # --- [1] Experiment Overview ---
    lines.append("")
    lines.append("[1] Experiment Overview")
    lines.append(sep2)
    lines.append(f"  {'#':<4} {'File':<50} {'Phase':<18} {'Variant':<10} {'Backend':<10} {'Prec':<8} {'Steps':>5}")
    lines.append(f"  {'-'*4} {'-'*50} {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")
    for i, b in enumerate(benchmarks, 1):
        m = b.metadata
        name = Path(b.filepath).stem[:49]
        lines.append(f"  {i:<4} {name:<50} {m.get('phase','?'):<18} {m.get('model_variant','?'):<10} "
                     f"{m.get('sd_backend','?'):<10} {m.get('sd_precision','?'):<8} {m.get('steps','?'):>5}")

    # --- [2] Generation E2E Comparison ---
    gen_benchmarks = [b for b in benchmarks if b.generate_summary is not None and not b.generate_summary.empty]
    if gen_benchmarks:
        lines.append("")
        lines.append("[2] Generation Latency (mean, ms)")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'E2E':>10} {'WallClk':>10} {'Tokenize':>10} {'TextEnc':>10} "
                     f"{'UNet':>10} {'VAEDec':>10} {'Steps':>6}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

        for i, b in enumerate(gen_benchmarks, 1):
            df = b.generate_summary
            name = Path(b.filepath).stem[:39]
            e2e = df["generate_e2e_ms"].mean()
            wc = df["pipeline_wall_clock_ms"].mean() if "pipeline_wall_clock_ms" in df.columns and df["pipeline_wall_clock_ms"].mean() > 0 else 0
            tok = df["tokenize_ms"].mean() if "tokenize_ms" in df.columns else 0
            textenc = df["text_enc_ms"].mean() if "text_enc_ms" in df.columns else 0
            unet = df["unet_total_ms"].mean() if "unet_total_ms" in df.columns else 0
            vaedec = df["vae_dec_ms"].mean() if "vae_dec_ms" in df.columns else 0
            steps = df["actual_steps"].iloc[0] if "actual_steps" in df.columns else "?"
            wc_str = f"{wc:>10.1f}" if wc > 0 else f"{'--':>10}"
            lines.append(f"  {i:<4} {name:<40} {e2e:>10.1f} {wc_str} {tok:>10.1f} {textenc:>10.1f} "
                         f"{unet:>10.1f} {vaedec:>10.1f} {steps:>6}")

        # UNet per-step comparison
        lines.append("")
        lines.append("[2b] UNet Per-Step Comparison")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'PerStep':>10} {'P95':>10} {'SchedOH':>10}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
        for i, b in enumerate(gen_benchmarks, 1):
            df = b.generate_summary
            name = Path(b.filepath).stem[:39]
            ps_mean = df["unet_per_step_mean_ms"].mean() if "unet_per_step_mean_ms" in df.columns else 0
            ps_p95 = df["unet_per_step_p95_ms"].mean() if "unet_per_step_p95_ms" in df.columns else 0
            sched = df["scheduler_overhead_ms"].mean() if "scheduler_overhead_ms" in df.columns else 0
            lines.append(f"  {i:<4} {name:<40} {ps_mean:>10.2f} {ps_p95:>10.2f} {sched:>10.2f}")

    # --- [3] Cold Start Comparison ---
    cold_benchmarks = [b for b in benchmarks if b.cold_start is not None and not b.cold_start.empty]
    if cold_benchmarks:
        lines.append("")
        lines.append("[3] Cold Start Breakdown (ms)")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'TextEnc':>10} {'UNet':>10} {'VAEDec':>10} "
                     f"{'LoadSum':>10} {'InitWC':>10} {'Par':>4} {'1stInfer':>10} {'ColdTotal':>10}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*4} {'-'*10} {'-'*10}")
        for i, b in enumerate(cold_benchmarks, 1):
            cs = b.cold_start.iloc[0]
            name = Path(b.filepath).stem[:39]
            init_wc = cs.get('init_wall_clock_ms', 0)
            init_wc_str = f"{init_wc:>10.0f}" if pd.notna(init_wc) and init_wc > 0 else f"{'--':>10}"
            parallel = cs.get('parallel_init', False)
            par_str = f"{'Y':>4}" if parallel else f"{'N':>4}"
            first_inf = cs.get('first_inference_wall_clock_ms', 0)
            first_inf_str = f"{first_inf:>10.0f}" if pd.notna(first_inf) and first_inf > 0 else f"{'--':>10}"
            cold_total = cs.get('cold_start_total_ms', 0)
            cold_total_str = f"{cold_total:>10.0f}" if pd.notna(cold_total) and cold_total > 0 else f"{'--':>10}"
            lines.append(f"  {i:<4} {name:<40} "
                         f"{cs.get('text_enc_load_ms',0):>10.0f} {cs.get('unet_load_ms',0):>10.0f} "
                         f"{cs.get('vae_dec_load_ms',0):>10.0f} {cs.get('total_load_ms',0):>10.0f} "
                         f"{init_wc_str} {par_str} {first_inf_str} {cold_total_str}")

        lines.append("")
        lines.append("    LoadSum: 컴포넌트별 시간 합산 | InitWC: 실제 초기화 wall-clock | Par: 병렬 초기화 여부")
        lines.append("    TextEnc~VAEDec: 각 컴포넌트 세션 생성 시간 (QNN EP = HTP 그래프 컴파일 포함)")

    # --- [4] System Resources ---
    if gen_benchmarks:
        lines.append("")
        lines.append("[4] System Resources")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'TempStart':>10} {'TempEnd':>10} {'Delta':>8} "
                     f"{'Power':>10} {'IdlePwr':>10} {'DeltaPwr':>10} {'VmRSS':>8} {'NatHeap':>8} {'PSS':>8}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*8} "
                     f"{'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for i, b in enumerate(gen_benchmarks, 1):
            df = b.generate_summary
            bm = b.metadata
            name = Path(b.filepath).stem[:39]
            t_start = df["start_temp_c"].iloc[0] if "start_temp_c" in df.columns else 0
            t_end = df["end_temp_c"].iloc[-1] if "end_temp_c" in df.columns else 0
            delta = t_end - t_start
            power = df["avg_power_mw"].mean() if "avg_power_mw" in df.columns else 0
            idle_pwr = float(bm.get('idle_baseline_power_mw', 0))
            delta_pwr = power - idle_pwr if idle_pwr > 0 else 0
            mem = df["peak_memory_mb"].max() if "peak_memory_mb" in df.columns else 0
            nat_heap = df["native_heap_mb"].max() if "native_heap_mb" in df.columns and df["native_heap_mb"].max() > 0 else 0
            pss = df["pss_mb"].max() if "pss_mb" in df.columns and df["pss_mb"].max() > 0 else 0
            idle_str = f"{idle_pwr:>10.0f}" if idle_pwr > 0 else f"{'--':>10}"
            delta_pwr_str = f"{delta_pwr:>+10.0f}" if idle_pwr > 0 else f"{'--':>10}"
            nat_str = f"{nat_heap:>8.0f}" if nat_heap > 0 else f"{'--':>8}"
            pss_str = f"{pss:>8.0f}" if pss > 0 else f"{'--':>8}"
            lines.append(f"  {i:<4} {name:<40} {t_start:>10.1f} {t_end:>10.1f} {delta:>+8.1f} "
                         f"{power:>10.0f} {idle_str} {delta_pwr_str} {mem:>8} {nat_str} {pss_str}")

    # --- [5] Graph Partitioning ---
    ort_benchmarks = [b for b in benchmarks if int(b.metadata.get('ort_total_nodes', 0)) > 0]
    if ort_benchmarks:
        lines.append("")
        lines.append("[5] Graph Partitioning")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'Total':>8} {'QNN':>8} {'CPU':>8} {'Coverage':>10}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for i, b in enumerate(ort_benchmarks, 1):
            m = b.metadata
            name = Path(b.filepath).stem[:39]
            total = int(m.get('ort_total_nodes', 0))
            qnn = int(m.get('ort_qnn_nodes', 0))
            cpu = int(m.get('ort_cpu_nodes', 0))
            cov = (qnn / total * 100) if total > 0 else 0
            lines.append(f"  {i:<4} {name:<40} {total:>8} {qnn:>8} {cpu:>8} {cov:>9.1f}%")

    # --- Methodology Notes ---
    lines.append("")
    _append_methodology_notes(lines)

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Force UTF-8 stdout on Windows
    import io, os
    if os.name == "nt":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Parse txt2img benchmark CSV files")
    parser.add_argument("paths", nargs="+", help="CSV files or directories to analyze")
    parser.add_argument("--compare", "-c", action="store_true", help="Show comparison table")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--output-dir", "-d", help="Output directory (default: outputs/exp in project root)")
    parser.add_argument("--print", "-p", action="store_true", help="Print to console only, do not save")

    args = parser.parse_args()

    # Collect CSV files
    csv_files = []
    input_dir = None
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            input_dir = path
            csv_files.extend(sorted(path.glob("*.csv")))
        elif path.exists():
            if input_dir is None:
                input_dir = path.parent
            csv_files.append(path)
        else:
            print(f"Warning: {p} not found")

    if not csv_files:
        print("No CSV files found")
        sys.exit(1)

    benchmarks = [parse_csv(f) for f in csv_files]

    if args.compare and len(benchmarks) > 1:
        report = format_comparison(benchmarks)
    else:
        report = "\n\n".join(format_report(b) for b in benchmarks)

    # Always print to console
    print(report)

    # Save to file unless --print only
    if not args.print:
        if args.output:
            out_path = Path(args.output)
        else:
            out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / "outputs" / "exp"
            out_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if len(csv_files) == 1:
                stem = csv_files[0].stem
                out_path = out_dir / f"{stem}.txt"
            else:
                out_path = out_dir / f"txt2img_comparison_{timestamp}.txt"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nSaved to {out_path}")
