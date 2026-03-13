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


# ---------------------------------------------------------------------------
# QAI Hub on-device NPU memory lookup  (source: docs/weights_inventory.md)
# W8A16 excluded — 실추론 성능 하락 확인으로 실험 대상 외
# ---------------------------------------------------------------------------
_NPU_MEM_MB = {
    # text_encoder  (QAI Hub profile 2MB = 가중치 미계상 오류 → compile bin 크기 237MB로 대체)
    ("text_enc", "fp32"):        237,
    # unet — SD_V15 base  (no QAI Hub profile; assumed same as unet_lcm fp32 — same architecture)
    ("unet_v15", "fp32"):        1793,
    ("unet_v15", "mixed_pr"):    1151,
    # unet — SD_LCM
    ("unet_lcm", "fp32"):        1793,
    ("unet_lcm", "mixed_pr"):    1150,
    # vae_decoder
    ("vae_dec",  "fp32"):        119,
    ("vae_dec",  "w8a8"):        69,
}


def _npu_mem_mb(variant: str, component: str, precision: str):
    """Return QAI Hub profiled NPU memory (MB) or None if unknown/excluded."""
    v = "lcm" if "lcm" in variant.lower() else "v15"
    p = precision.lower().replace("-", "_")
    # fp16 → QNN compiles FP32 ONNX with FP16 activations by default;
    # QAI Hub profiles were done on the same compiled binary → use fp32 values.
    if p == "fp16":
        p = "fp32"
    comp_map = {"text_encoder": "text_enc", "unet": f"unet_{v}", "vae_decoder": "vae_dec"}
    key = (comp_map.get(component), p)
    return _NPU_MEM_MB.get(key)


def _parse_pcp(bench) -> dict:
    """Parse sd_precision_per_component → {component: precision} dict."""
    raw = bench.metadata.get("sd_precision_per_component", "")
    if not raw:
        # fall back: apply global precision to all components
        p = bench.metadata.get("sd_precision", "fp32").lower()
        return {"text_encoder": p, "unet": p, "vae_decoder": p}
    result = {}
    for part in raw.split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            result[k.strip()] = v.strip().lower()
    return result


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
        lines.append("[2] Cold Start Breakdown  (ms)")
        lines.append(sep2)

        init_wc   = cs.get('init_wall_clock_ms', 0) or 0
        first_inf = cs.get('first_inference_wall_clock_ms', 0) or 0
        cs_total  = cs.get('cold_start_total_ms', 0) or 0
        total_sum = cs.get('total_load_ms', 0) or 0
        parallel  = cs.get('parallel_init', False)
        steady_mean = None
        if bench.generate_summary is not None and not bench.generate_summary.empty:
            steady_mean = bench.generate_summary["generate_e2e_ms"].mean()

        # [2a] Idle Baseline
        idle_thermal = cs.get('idle_thermal_c', 0) or 0
        idle_power   = cs.get('idle_power_mw', 0) or 0
        if idle_power <= 0:
            idle_power = float(m.get('idle_baseline_power_mw', 0) or 0)
        idle_mem_b   = cs.get('idle_memory_mb', 0) or 0
        if idle_thermal > 0 or idle_power > 0 or idle_mem_b > 0:
            lines.append("")
            lines.append("  [2a] Idle Baseline  (pre-load, 5s 10-sample median)")
            lines.append(f"  {'-'*70}")
            lines.append(f"  {'Thermal (°C)':>12} {'Power (mW)':>12} {'Memory (MB)':>12}")
            lines.append(f"  {'-'*12} {'-'*12} {'-'*12}")
            t_str = f"{idle_thermal:>12.1f}" if idle_thermal > 0 else f"{'--':>12}"
            p_str = f"{idle_power:>12.0f}"   if idle_power   > 0 else f"{'--':>12}"
            m_str = f"{idle_mem_b:>12.0f}"   if idle_mem_b   > 0 else f"{'--':>12}"
            lines.append(f"  {t_str} {p_str} {m_str}")
            lines.append("    Thermal: SoC temp before model load  |  Power: system-wide idle  |  Memory: VmRSS before model load")

        # [2b] Session Initialization
        te_load  = cs.get('text_enc_load_ms', 0) or 0
        un_load  = cs.get('unet_load_ms', 0) or 0
        vd_load  = cs.get('vae_dec_load_ms', 0) or 0
        overhead = (init_wc - total_sum) if init_wc > 0 and total_sum > 0 else None
        par_str  = "Y" if parallel else "N"
        lines.append("")
        lines.append("  [2b] Session Initialization")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'TextEncLoad':>12} {'UNetLoad':>10} {'VAEDecLoad':>11} {'LoadSum':>9} {'InitWC':>9} {'Overhead':>10} {'Par':>4}")
        lines.append(f"  {'-'*12} {'-'*10} {'-'*11} {'-'*9} {'-'*9} {'-'*10} {'-'*4}")
        init_wc_str  = f"{init_wc:>9.0f}"  if init_wc  > 0 else f"{'--':>9}"
        overhead_str = f"{overhead:>+10.0f}" if overhead is not None else f"{'--':>10}"
        lines.append(f"  {te_load:>12.0f} {un_load:>10.0f} {vd_load:>11.0f} {total_sum:>9.0f} {init_wc_str} {overhead_str} {par_str:>4}")
        lines.append("")
        lines.append("    TextEncLoad/UNetLoad/VAEDecLoad  ORT session creation per component (NPU: includes QNN/HTP graph compile)")
        lines.append("    LoadSum   Sum of component times")
        lines.append("    InitWC    Actual wall-clock until all sessions ready (≥ LoadSum due to inter-session overhead)")
        lines.append("    Overhead  InitWC − LoadSum  (inter-session scheduling cost)")
        lines.append("    Par       Y = concurrent init; N = sequential (default, required for QNN HTP)")

        # [2c] Cold First Inference Breakdown
        first_tok  = cs.get('first_tokenize_ms', 0) or 0
        first_te   = cs.get('first_text_enc_ms', 0) or 0
        first_unet = cs.get('first_unet_total_ms', 0) or 0
        first_vae  = cs.get('first_vae_dec_ms', 0) or 0
        first_pp   = cs.get('first_postprocess_ms', 0) or 0
        first_e2e  = cs.get('first_generate_e2e_ms', 0) or 0
        lines.append("")
        lines.append("  [2c] Cold First Inference Breakdown  (1st generate() — post-load, pre-warmup, ms)")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'Tokenize':>10} {'TextEnc':>9} {'UNet':>9} {'VAEDec':>9} {'Postproc':>9} {'E2E':>10} {'WC':>9}")
        lines.append(f"  {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*9}")
        tok_s  = f"{first_tok:>10.1f}"  if first_tok  > 0 else f"{'--':>10}"
        te_s   = f"{first_te:>9.1f}"   if first_te   > 0 else f"{'--':>9}"
        un_s   = f"{first_unet:>9.1f}" if first_unet > 0 else f"{'--':>9}"
        vd_s   = f"{first_vae:>9.1f}"  if first_vae  > 0 else f"{'--':>9}"
        pp_s   = f"{first_pp:>9.1f}"   if first_pp   > 0 else f"{'--':>9}"
        e2e_s  = f"{first_e2e:>10.1f}" if first_e2e  > 0 else f"{'--':>10}"
        wc_s   = f"{first_inf:>9.1f}"  if first_inf  > 0 else f"{'--':>9}"
        lines.append(f"  {tok_s} {te_s} {un_s} {vd_s} {pp_s} {e2e_s} {wc_s}")
        lines.append("    JIT compile + cold cache 포함 — warmup 이전 단 1회 측정")
        lines.append("    WC: generate() wall-clock  |  E2E: component sum (WC ≥ E2E — 버퍼 할당 등 overhead 포함)")

        # [2d] Cold Start E2E
        lines.append("")
        lines.append("  [2d] Cold Start E2E  (ms)")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'SessionInit':>12} {'1stInfer':>10} {'ColdE2E':>10}")
        lines.append(f"  {'-'*12} {'-'*10} {'-'*10}")
        init_s  = f"{init_wc:>12.0f}"  if init_wc  > 0 else f"{'--':>12}"
        finf_s  = f"{first_inf:>10.0f}" if first_inf > 0 else f"{'--':>10}"
        cold_s  = f"{cs_total:>10.0f}" if cs_total  > 0 else f"{'--':>10}"
        lines.append(f"  {init_s} {finf_s} {cold_s}")
        if steady_mean:
            diff = first_inf - steady_mean
            lines.append(f"    1stInfer vs steady-state: {steady_mean:,.0f} ms  ({diff:+,.0f} ms / {diff/steady_mean*100:+.1f}%)")
        lines.append("    SessionInit  ORT 세션 생성 wall-clock (QNN HTP graph compile 포함)")
        lines.append("    1stInfer     첫 번째 generate() wall-clock (JIT compile + cold cache 포함; Postproc 포함)")
        lines.append("    ColdE2E      SessionInit + 1stInfer  (앱 최초 실행 → 첫 이미지 완성)")

        # Warmup summary
        warmup_ms = cs.get('warmup_total_ms', None)
        if pd.notna(warmup_ms) and warmup_ms > 0:
            warmup_trials = 2
            per_trial = warmup_ms / warmup_trials
            lines.append("")
            lines.append(f"  Warmup ({warmup_trials} trials):  {warmup_ms:,.0f} ms  (~{per_trial:,.0f} ms/trial)"
                         + ("  [steady-state established]" if steady_mean else ""))

    # --- [3] Generation Summary ---
    if bench.generate_summary is not None and not bench.generate_summary.empty:
        df = bench.generate_summary
        n = len(df)
        lines.append("")
        lines.append(f"[3] Generation E2E Performance ({n} trials, warm)")
        lines.append(sep2)

        # Prompt (first trial)
        prompt = df["prompt"].iloc[0] if "prompt" in df.columns else "?"
        if len(prompt) > 80:
            prompt = prompt[:77] + "..."
        lines.append(f"  Prompt: \"{prompt}\"")

        # [3a] Stage breakdown table (transposed: stats as rows, pipeline stages as columns)
        lines.append("")
        lines.append(f"  [3a] Stage Breakdown")
        lines.append(f"  {'-'*70}")

        stage_cols = {
            "tokenize_ms":     "Tokenize",
            "text_enc_ms":     "Text Enc",
            "unet_total_ms":   "UNet Total",
            "vae_dec_ms":      "VAE Dec",
            "postprocess_ms":  "Postproc",
            "generate_e2e_ms": "E2E",
        }

        e2e_mean = df["generate_e2e_ms"].mean() if "generate_e2e_ms" in df.columns else 0
        active_stages = [(col, lbl) for col, lbl in stage_cols.items()
                         if col in df.columns and df[col].mean() > 0.001]
        if active_stages:
            col_w  = max(max(len(lbl) for _, lbl in active_stages), 9) + 2
            stat_w = 10
            header = f"  {'Stat':<{stat_w}}"
            for _, lbl in active_stages:
                header += f"  {lbl:>{col_w}}"
            lines.append(header)
            lines.append(f"  {'-'*stat_w}" + f"  {'-'*col_w}" * len(active_stages))
            for stat_label, func in [
                ("Mean (ms)", lambda s: s.mean()),
                ("P50",       lambda s: s.median()),
                ("P95",       lambda s: s.quantile(0.95)),
                ("Min",       lambda s: s.min()),
                ("Max",       lambda s: s.max()),
                ("Std",       lambda s: s.std()),
            ]:
                row = f"  {stat_label:<{stat_w}}"
                for col, _ in active_stages:
                    val = func(df[col])
                    row += f"  {'nan':>{col_w}}" if pd.isna(val) else f"  {val:>{col_w}.2f}"
                lines.append(row)
            if e2e_mean > 0:
                row = f"  {'%E2E':<{stat_w}}"
                for col, _ in active_stages:
                    if col == "generate_e2e_ms":
                        row += f"  {'100%':>{col_w}}"
                    else:
                        pct_str = f"{df[col].mean() / e2e_mean * 100:.1f}%"
                        row += f"  {pct_str:>{col_w}}"
                lines.append(row)

        lines.append("")
        lines.append("    Tokenize:    CLIP tokenizer — prompt text → token IDs")
        lines.append("    Text Enc:    CLIP text encoder — token IDs → prompt embeddings")
        lines.append("    UNet Total:  N-step denoising UNet (noise → latent)")
        lines.append("    VAE Dec:     VAE decoder ORT session.run() — latent → float[]  (OnnxTensor → FloatArray native→Java copy 포함)")
        lines.append("    Postproc:    float[] → Bitmap (pixel clamp + ARGB packing, CPU)")
        lines.append("    E2E:         Tokenize + Text Enc + UNet + VAE Dec + Postproc  (component sum)")

        # Diagnostic checks
        component_cols = ["tokenize_ms", "text_enc_ms", "unet_total_ms", "vae_dec_ms", "postprocess_ms"]
        available_cols = [c for c in component_cols if c in df.columns]
        if available_cols and "generate_e2e_ms" in df.columns:
            component_sum = df[available_cols].sum(axis=1)
            gap = df["generate_e2e_ms"] - component_sum
            lines.append("")
            lines.append(f"  Sum check:    E2E={df['generate_e2e_ms'].mean():.1f} ms  "
                         f"∑components={component_sum.mean():.1f} ms  "
                         f"gap={gap.mean():.1f} ms ({gap.mean()/df['generate_e2e_ms'].mean()*100:.1f}%)")
        if "pipeline_wall_clock_ms" in df.columns and df["pipeline_wall_clock_ms"].mean() > 0:
            wc  = df["pipeline_wall_clock_ms"]
            e2e = df["generate_e2e_ms"]
            gap = wc - e2e
            lines.append(f"  Wall-clock:   generate()={wc.mean():.1f} ms  "
                         f"E2E={e2e.mean():.1f} ms  "
                         f"overhead={gap.mean():.1f} ms ({gap.mean()/wc.mean()*100:.1f}%)")
        if "trial_wall_clock_ms" in df.columns and df["trial_wall_clock_ms"].mean() > 0:
            twc = df["trial_wall_clock_ms"]
            lines.append(f"  Trial clock:  {twc.mean():.1f} ms  (includes power/thermal sampling overhead)")

        # [3b] UNet per-step summary
        if "unet_per_step_mean_ms" in df.columns:
            lines.append("")
            lines.append(f"  [3b] UNet Per-Step Summary")
            lines.append(f"  {'-'*70}")
            lines.append(f"  Per-step mean:       {df['unet_per_step_mean_ms'].mean():.2f} ms")
            if "unet_per_step_p95_ms" in df.columns:
                lines.append(f"  Per-step P95:        {df['unet_per_step_p95_ms'].mean():.2f} ms")
            if "scheduler_overhead_ms" in df.columns:
                lines.append(f"  Scheduler overhead:  {df['scheduler_overhead_ms'].mean():.2f} ms  (EulerDiscrete, total across all steps)")
            steps = df["actual_steps"].iloc[0] if "actual_steps" in df.columns else int(m.get("steps", 0))
            if steps > 0:
                unet_ms = df["unet_total_ms"].mean()
                lines.append(f"  UNet total / steps:  {unet_ms:.1f} / {steps} = {unet_ms/steps:.1f} ms/step")

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
        has_power   = "avg_power_mw" in df.columns and df["avg_power_mw"].mean() > 0
        has_memory  = "peak_memory_mb" in df.columns

        if has_thermal or has_power or has_memory:
            lines.append("")
            lines.append("[5] System Resources")
            lines.append(sep2)

            # Memory (MB) — horizontal table
            if has_memory:
                lines.append("")
                lines.append("  Memory (MB)")
                has_cold_mem = bench.cold_start is not None and not bench.cold_start.empty
                idle_mem = peak_load = 0
                if has_cold_mem:
                    cs2 = bench.cold_start.iloc[0]
                    idle_mem  = cs2.get('idle_memory_mb', 0) or 0
                    peak_load = cs2.get('peak_memory_after_load_mb', 0) or 0
                nat_heap = df["native_heap_mb"].max() if "native_heap_mb" in df.columns and df["native_heap_mb"].max() > 0 else 0
                pss      = df["pss_mb"].max()          if "pss_mb"          in df.columns and df["pss_mb"].max()          > 0 else 0
                nat_str  = f"{nat_heap:>12.0f}" if nat_heap > 0 else f"{'--':>12}"
                pss_str  = f"{pss:>8.0f}"       if pss      > 0 else f"{'--':>8}"
                if has_cold_mem:
                    idle_str      = f"{idle_mem:>8.0f}"   if idle_mem  > 0 else f"{'--':>8}"
                    peak_load_str = f"{peak_load:>10.0f}" if peak_load > 0 else f"{'--':>10}"
                    lines.append(f"  {'Idle':>8} {'AfterLoad':>10} {'Peak(infer)':>12} {'NativeHeap':>12} {'PSS':>8}")
                    lines.append(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
                    lines.append(f"  {idle_str} {peak_load_str} {df['peak_memory_mb'].max():>12} {nat_str} {pss_str}")
                else:
                    lines.append(f"  {'Peak(infer)':>12} {'NativeHeap':>12} {'PSS':>8}")
                    lines.append(f"  {'-'*12} {'-'*12} {'-'*8}")
                    lines.append(f"  {df['peak_memory_mb'].max():>12} {nat_str} {pss_str}")
                lines.append("    Idle: before model load  |  AfterLoad: after session creation  |  Peak(infer): peak RSS during inference")
                lines.append("    NativeHeap: ORT/QNN native allocations (Debug.getNativeHeapAllocatedSize)  |  PSS: proportional set size")

            # Total Peak Memory  (NPU estimate + App RSS)  — QNN_NPU only
            if m.get("sd_backend", "").upper() == "QNN_NPU" and has_memory:
                variant = m.get("model_variant", "")
                pcp = _parse_pcp(bench)
                te_mb  = _npu_mem_mb(variant, "text_encoder", pcp.get("text_encoder", "fp32"))
                un_mb  = _npu_mem_mb(variant, "unet",         pcp.get("unet", "fp32"))
                vd_mb  = _npu_mem_mb(variant, "vae_decoder",  pcp.get("vae_decoder", "fp32"))
                app_mb = df["peak_memory_mb"].max()
                all_known = all(x is not None for x in [te_mb, un_mb, vd_mb])
                npu_sum = sum([te_mb, un_mb, vd_mb]) if all_known else None
                total   = (npu_sum + app_mb) if npu_sum is not None else None
                lines.append("")
                lines.append("  Total Peak Memory  (App RSS + NPU estimate, MB)")
                lines.append("  Source: NPU values from QAI Hub on-device profiling (docs/weights_inventory.md)")
                lines.append(f"  {'TextEnc':>9} {'UNet':>9} {'VAEDec':>9} {'NPU Sum':>9} {'App RSS':>9} {'Total':>9}")
                lines.append(f"  {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
                te_s   = f"{te_mb:>9}"   if te_mb  is not None else f"{'--':>9}"
                un_s   = f"{un_mb:>9}"   if un_mb  is not None else f"{'?':>9}"
                vd_s   = f"{vd_mb:>9}"   if vd_mb  is not None else f"{'--':>9}"
                npu_s  = f"{npu_sum:>9}" if npu_sum is not None else f"{'?':>9}"
                tot_s  = f"{total:>9.0f}" if total  is not None else f"{'?':>9}"
                lines.append(f"  {te_s} {un_s} {vd_s} {npu_s} {app_mb:>9.0f} {tot_s}")
                lines.append("    TextEnc/UNet/VAEDec: QAI Hub on-device profiling 결과 (Snapdragon 8 Gen 2, S23)")
                lines.append("      NPU(HTP) 메모리 = 모델 가중치가 상주하는 HTP 전용 메모리 영역")
                lines.append("      App RSS(VmRSS)에는 미포함 — Android /proc/self/status 로 관측 불가")
                lines.append("      실기기 on-device 직접 측정은 root 권한 + /sys/kernel/debug/ion/ 접근 필요 (stock Android SELinux 차단)")
                lines.append("    App RSS: peak VmRSS during inference (ORT runtime, Java heap, shared libs; NPU 제외)")
                lines.append("    Total: NPU Sum + App RSS  (추정치)")
                lines.append("    ?: no QAI Hub profile available for this component/precision combination")

            # Power (mW) — horizontal table
            if has_power:
                avg_power       = df['avg_power_mw'].mean()
                idle_base_power = float(m.get('idle_baseline_power_mw', 0))
                delta_power     = avg_power - idle_base_power if idle_base_power > 0 else 0
                idle_str        = f"{idle_base_power:>10.0f}" if idle_base_power > 0 else f"{'--':>10}"
                delta_str       = f"{delta_power:>+10.0f}"    if idle_base_power > 0 else f"{'--':>10}"
                lines.append("")
                lines.append("  Power (mW)")
                lines.append(f"  {'Idle':>10} {'AvgInfer':>10} {'Delta':>10}")
                lines.append(f"  {'-'*10} {'-'*10} {'-'*10}")
                lines.append(f"  {idle_str} {avg_power:>10.0f} {delta_str}")
                lines.append("    Idle: 5s median before model load  |  AvgInfer: mean during inference  |  Delta = AvgInfer − Idle (net inference power)")
                lines.append("    ⚠ BatteryManager.BATTERY_PROPERTY_CURRENT_NOW 기반 — 폰 전체 시스템 소비 전력 (SoC+디스플레이+라디오 포함)")
                lines.append("      앱 단독 / NPU 단독 전력 분리는 Snapdragon Profiler 또는 PMU 카운터 접근 필요")
                if m.get('is_charging', '').lower() == 'true':
                    lines.append("    ⚠ 충전 중 측정 — 전력 데이터 신뢰도 낮음")

            # Thermal (°C) — horizontal table
            if has_thermal:
                cs_row = bench.cold_start.iloc[0] if bench.cold_start is not None and not bench.cold_start.empty else {}
                idle_t      = cs_row.get('idle_thermal_c', 0) or 0
                post_infer_t = cs_row.get('first_infer_end_thermal_c', 0) or 0
                t_end   = df["end_temp_c"].iloc[-1]
                zone_type = m.get('thermal_zone_type', '')
                lines.append("")
                lines.append(f"  Thermal (°C){'  zone: ' + zone_type if zone_type else ''}")
                lines.append(f"  {'Idle':>8} {'Post1stInf':>11} {'End':>8}")
                lines.append(f"  {'-'*8} {'-'*11} {'-'*8}")
                idle_s   = f"{idle_t:>8.1f}"      if idle_t      > 0 else f"{'--':>8}"
                pinf_s   = f"{post_infer_t:>11.1f}" if post_infer_t > 0 else f"{'--':>11}"
                lines.append(f"  {idle_s} {pinf_s} {t_end:>8.1f}")
                lines.append("    Idle: before model load  |  Post1stInf: after cold first inference  |  End: after last warm trial")

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
    """Transposed stat table: stats (Mean/P50/…) as rows, metrics as columns (pipeline time order)."""
    active = [(col, label) for col, label in col_map.items()
              if col in df.columns and df[col].mean() > 0.001]
    if not active:
        return

    col_w = max(max(len(lbl) for _, lbl in active), 9) + 2
    stat_w = 10

    header = f"  {'Stat':<{stat_w}}"
    for _, lbl in active:
        header += f"  {lbl:>{col_w}}"
    lines.append(header)
    lines.append(f"  {'-'*stat_w}" + f"  {'-'*col_w}" * len(active))

    for stat_label, func in [
        ("Mean (ms)", lambda s: s.mean()),
        ("P50",       lambda s: s.median()),
        ("P95",       lambda s: s.quantile(0.95)),
        ("Min",       lambda s: s.min()),
        ("Max",       lambda s: s.max()),
        ("Std",       lambda s: s.std()),
    ]:
        row = f"  {stat_label:<{stat_w}}"
        for col, _ in active:
            val = func(df[col])
            row += f"  {'nan':>{col_w}}" if pd.isna(val) else f"  {val:>{col_w}.2f}"
        lines.append(row)


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
        prec = "mixed" if m.get('sd_precision_per_component') else m.get('sd_precision', '?')
        lines.append(f"  {i:<4} {name:<50} {m.get('phase','?'):<18} {m.get('model_variant','?'):<10} "
                     f"{m.get('sd_backend','?'):<10} {prec:<8} {m.get('steps','?'):>5}")
    mixed_entries = [(i, b) for i, b in enumerate(benchmarks, 1) if b.metadata.get('sd_precision_per_component')]
    if mixed_entries:
        lines.append(f"  Mixed precision breakdown:")
        for i, b in mixed_entries:
            lines.append(f"    #{i}: {b.metadata['sd_precision_per_component']}")

    gen_benchmarks = [b for b in benchmarks if b.generate_summary is not None and not b.generate_summary.empty]
    cold_benchmarks = [b for b in benchmarks if b.cold_start is not None and not b.cold_start.empty]

    # --- [2] Cold Start Breakdown ---
    if cold_benchmarks:
        lines.append("")
        lines.append("[2] Cold Start Breakdown  (ms)")
        lines.append(sep2)

        # [2a] Idle Baseline
        lines.append("")
        lines.append("  [2a] Idle Baseline  (pre-load, 5s 10-sample median)")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'#':<4} {'File':<40} {'Thermal (°C)':>12} {'Power (mW)':>12} {'Memory (MB)':>12}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*12} {'-'*12} {'-'*12}")
        for i, b in enumerate(cold_benchmarks, 1):
            cs   = b.cold_start.iloc[0]
            name = Path(b.filepath).stem[:39]
            idle_t = cs.get('idle_thermal_c', 0) or 0
            idle_p = cs.get('idle_power_mw', 0) or 0
            if idle_p <= 0:
                idle_p = float(b.metadata.get('idle_baseline_power_mw', 0) or 0)
            idle_m = cs.get('idle_memory_mb', 0) or 0
            t_str = f"{idle_t:>12.1f}" if idle_t > 0 else f"{'--':>12}"
            p_str = f"{idle_p:>12.0f}" if idle_p > 0 else f"{'--':>12}"
            m_str = f"{idle_m:>12.0f}" if idle_m > 0 else f"{'--':>12}"
            lines.append(f"  {i:<4} {name:<40} {t_str} {p_str} {m_str}")
        lines.append("    Thermal: SoC temp before model load  |  Power: system-wide idle  |  Memory: VmRSS before model load")

        # [2b] Session Initialization
        lines.append("")
        lines.append("  [2b] Session Initialization")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'#':<4} {'File':<40} {'TextEncLoad':>12} {'UNetLoad':>10} {'VAEDecLoad':>11} "
                     f"{'LoadSum':>9} {'InitWC':>9} {'Overhead':>10} {'Par':>4}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*12} {'-'*10} {'-'*11} {'-'*9} {'-'*9} {'-'*10} {'-'*4}")
        for i, b in enumerate(cold_benchmarks, 1):
            cs = b.cold_start.iloc[0]
            name = Path(b.filepath).stem[:39]
            text_enc_load = cs.get('text_enc_load_ms', 0) or 0
            unet_load     = cs.get('unet_load_ms', 0) or 0
            vae_dec_load  = cs.get('vae_dec_load_ms', 0) or 0
            load_sum      = cs.get('total_load_ms', 0) or 0
            init_wc       = cs.get('init_wall_clock_ms', 0)
            parallel      = cs.get('parallel_init', False)
            init_wc_str   = f"{init_wc:>9.0f}"  if pd.notna(init_wc) and init_wc > 0 else f"{'--':>9}"
            overhead      = (init_wc - load_sum) if (pd.notna(init_wc) and init_wc > 0 and load_sum > 0) else None
            overhead_str  = f"{overhead:>+10.0f}" if overhead is not None else f"{'--':>10}"
            par_str       = f"{'Y':>4}" if parallel else f"{'N':>4}"
            lines.append(f"  {i:<4} {name:<40} "
                         f"{text_enc_load:>12.0f} {unet_load:>10.0f} {vae_dec_load:>11.0f} "
                         f"{load_sum:>9.0f} {init_wc_str} {overhead_str} {par_str}")
        lines.append("")
        lines.append("    TextEncLoad/UNetLoad/VAEDecLoad  ORT session creation per component (NPU: includes QNN/HTP graph compile)")
        lines.append("    LoadSum   Sum of component times")
        lines.append("    InitWC    Actual wall-clock until all sessions ready (≥ LoadSum due to inter-session overhead)")
        lines.append("    Overhead  InitWC − LoadSum  (inter-session scheduling cost)")
        lines.append("    Par       Y = concurrent init; N = sequential (default, required for QNN HTP)")

        # [2c] Cold First Inference Breakdown
        lines.append("")
        lines.append("  [2c] Cold First Inference Breakdown  (1st generate() — post-load, pre-warmup, ms)")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'#':<4} {'File':<40} {'Tokenize':>10} {'TextEnc':>9} {'UNet':>9} {'VAEDec':>9} {'Postproc':>9} {'E2E':>10} {'WC':>9}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*9}")
        for i, b in enumerate(cold_benchmarks, 1):
            cs   = b.cold_start.iloc[0]
            name = Path(b.filepath).stem[:39]
            tok     = cs.get('first_tokenize_ms', 0) or 0
            textenc = cs.get('first_text_enc_ms', 0) or 0
            unet    = cs.get('first_unet_total_ms', 0) or 0
            vaedec  = cs.get('first_vae_dec_ms', 0) or 0
            postproc= cs.get('first_postprocess_ms', 0) or 0
            e2e     = cs.get('first_generate_e2e_ms', 0) or 0
            wc      = cs.get('first_inference_wall_clock_ms', 0) or 0
            tok_str     = f"{tok:>10.1f}"     if tok     > 0 else f"{'--':>10}"
            textenc_str = f"{textenc:>9.1f}"  if textenc > 0 else f"{'--':>9}"
            unet_str    = f"{unet:>9.1f}"     if unet    > 0 else f"{'--':>9}"
            vaedec_str  = f"{vaedec:>9.1f}"   if vaedec  > 0 else f"{'--':>9}"
            pp_str      = f"{postproc:>9.1f}" if postproc > 0 else f"{'--':>9}"
            e2e_str     = f"{e2e:>10.1f}"     if e2e     > 0 else f"{'--':>10}"
            wc_str      = f"{wc:>9.1f}"       if wc      > 0 else f"{'--':>9}"
            lines.append(f"  {i:<4} {name:<40} {tok_str} {textenc_str} {unet_str} {vaedec_str} {pp_str} {e2e_str} {wc_str}")
        lines.append("    JIT compile + cold cache 포함 — warmup 이전 단 1회 측정")
        lines.append("    WC: generate() wall-clock  |  E2E: component sum (WC ≥ E2E — 버퍼 할당 등 overhead 포함)")

        # [2d] Cold Start E2E Summary
        lines.append("")
        lines.append("  [2d] Cold Start E2E  (ms)")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'#':<4} {'File':<40} {'SessionInit':>12} {'1stInfer':>10} {'ColdE2E':>10}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*12} {'-'*10} {'-'*10}")
        for i, b in enumerate(cold_benchmarks, 1):
            cs   = b.cold_start.iloc[0]
            name = Path(b.filepath).stem[:39]
            init_wc    = cs.get('init_wall_clock_ms', 0)
            first_inf  = cs.get('first_inference_wall_clock_ms', 0)
            cold_total = cs.get('cold_start_total_ms', 0)
            init_str   = f"{init_wc:>12.0f}"   if pd.notna(init_wc)    and init_wc    > 0 else f"{'--':>12}"
            first_str  = f"{first_inf:>10.0f}"  if pd.notna(first_inf)  and first_inf  > 0 else f"{'--':>10}"
            cold_str   = f"{cold_total:>10.0f}" if pd.notna(cold_total) and cold_total > 0 else f"{'--':>10}"
            lines.append(f"  {i:<4} {name:<40} {init_str} {first_str} {cold_str}")
        lines.append("    SessionInit  ORT 세션 생성 wall-clock (QNN HTP graph compile 포함)")
        lines.append("    1stInfer     첫 번째 generate() wall-clock (JIT compile + cold cache 포함; Postproc 포함)")
        lines.append("    ColdE2E      SessionInit + 1stInfer  (앱 최초 실행 → 첫 이미지 완성)")

    # --- [3] Generation Latency ---
    if gen_benchmarks:
        lines.append("")
        lines.append("[3] Generation Latency  (warm, post-warmup mean, ms)")
        lines.append(sep2)

        # [3a] Stage breakdown
        lines.append(f"  [3a] Stage Breakdown")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'#':<4} {'File':<40} {'Steps':>6} {'Tokenize':>10} {'TextEnc':>10} "
                     f"{'UNet':>10} {'VAEDec':>10} {'Postproc':>10} {'E2E':>10} {'WallClk':>10}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for i, b in enumerate(gen_benchmarks, 1):
            df = b.generate_summary
            name = Path(b.filepath).stem[:39]
            e2e        = df["generate_e2e_ms"].mean()
            wc         = df["pipeline_wall_clock_ms"].mean() if "pipeline_wall_clock_ms" in df.columns and df["pipeline_wall_clock_ms"].mean() > 0 else 0
            tok        = df["tokenize_ms"].mean() if "tokenize_ms" in df.columns else 0
            textenc    = df["text_enc_ms"].mean() if "text_enc_ms" in df.columns else 0
            unet       = df["unet_total_ms"].mean() if "unet_total_ms" in df.columns else 0
            vaedec     = df["vae_dec_ms"].mean() if "vae_dec_ms" in df.columns else 0
            postproc   = df["postprocess_ms"].mean() if "postprocess_ms" in df.columns and df["postprocess_ms"].mean() > 0 else 0
            steps      = df["actual_steps"].iloc[0] if "actual_steps" in df.columns else "?"
            wc_str     = f"{wc:>10.1f}" if wc > 0 else f"{'--':>10}"
            pp_str     = f"{postproc:>10.1f}" if postproc > 0 else f"{'--':>10}"
            lines.append(f"  {i:<4} {name:<40} {steps:>6} {tok:>10.1f} {textenc:>10.1f} "
                         f"{unet:>10.1f} {vaedec:>10.1f} {pp_str} {e2e:>10.1f} {wc_str}")
        lines.append("    E2E = Tokenize + TextEnc + UNet + VAEDec + Postproc  |  WallClk: generate() wall-clock")

        # [3b] UNet per-step comparison
        lines.append("")
        lines.append(f"  [3b] UNet Per-Step  (ms/step)")
        lines.append(f"  {'-'*70}")
        lines.append(f"  {'#':<4} {'File':<40} {'Mean':>10} {'P95':>10} {'SchedOH':>10}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
        for i, b in enumerate(gen_benchmarks, 1):
            df = b.generate_summary
            name = Path(b.filepath).stem[:39]
            ps_mean = df["unet_per_step_mean_ms"].mean() if "unet_per_step_mean_ms" in df.columns else 0
            ps_p95  = df["unet_per_step_p95_ms"].mean()  if "unet_per_step_p95_ms"  in df.columns else 0
            sched   = df["scheduler_overhead_ms"].mean()  if "scheduler_overhead_ms"  in df.columns else 0
            lines.append(f"  {i:<4} {name:<40} {ps_mean:>10.2f} {ps_p95:>10.2f} {sched:>10.2f}")
        lines.append("    PerStep: mean ORT session.run() per denoising step  |  P95: 95th-percentile step time  |  SchedOH: EulerDiscrete scheduler overhead (total)")

    # --- [4] System Resources ---
    if gen_benchmarks:
        lines.append("")
        lines.append("[4] System Resources")
        lines.append(sep2)

        # [4a] Memory (MB)
        lines.append("")
        lines.append("  Memory (MB)")
        has_cold_mem = any(b.cold_start is not None and not b.cold_start.empty for b in gen_benchmarks)
        if has_cold_mem:
            lines.append(f"  {'#':<4} {'File':<40} {'Idle':>8} {'AfterLoad':>10} {'Peak(infer)':>12} {'NativeHeap':>12} {'PSS':>8}")
            lines.append(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
        else:
            lines.append(f"  {'#':<4} {'File':<40} {'Peak(infer)':>12} {'NativeHeap':>12} {'PSS':>8}")
            lines.append(f"  {'-'*4} {'-'*40} {'-'*12} {'-'*12} {'-'*8}")
        for i, b in enumerate(gen_benchmarks, 1):
            df   = b.generate_summary
            name = Path(b.filepath).stem[:39]
            mem      = df["peak_memory_mb"].max() if "peak_memory_mb" in df.columns else 0
            nat_heap = df["native_heap_mb"].max() if "native_heap_mb" in df.columns and df["native_heap_mb"].max() > 0 else 0
            pss      = df["pss_mb"].max()          if "pss_mb"          in df.columns and df["pss_mb"].max()          > 0 else 0
            nat_str  = f"{nat_heap:>12.0f}" if nat_heap > 0 else f"{'--':>12}"
            pss_str  = f"{pss:>8.0f}"       if pss      > 0 else f"{'--':>8}"
            if has_cold_mem:
                idle_mem = peak_load = 0
                if b.cold_start is not None and not b.cold_start.empty:
                    cs = b.cold_start.iloc[0]
                    idle_mem  = cs.get('idle_memory_mb', 0) or 0
                    peak_load = cs.get('peak_memory_after_load_mb', 0) or 0
                idle_str      = f"{idle_mem:>8.0f}"   if idle_mem  > 0 else f"{'--':>8}"
                peak_load_str = f"{peak_load:>10.0f}" if peak_load > 0 else f"{'--':>10}"
                lines.append(f"  {i:<4} {name:<40} {idle_str} {peak_load_str} {mem:>12} {nat_str} {pss_str}")
            else:
                lines.append(f"  {i:<4} {name:<40} {mem:>12} {nat_str} {pss_str}")
        lines.append("    Idle: before model load  |  AfterLoad: after session creation  |  Peak(infer): peak RSS during inference")
        lines.append("    NativeHeap: ORT/QNN native allocations (Debug.getNativeHeapAllocatedSize)  |  PSS: proportional set size")

        # [4a-2] Total Peak Memory  (App + NPU estimate)
        npu_benchmarks = [b for b in gen_benchmarks if b.metadata.get("sd_backend", "").upper() == "QNN_NPU"]
        if npu_benchmarks:
            lines.append("")
            lines.append("  Total Peak Memory  (App RSS + NPU estimate, MB)")
            lines.append("  Source: NPU values from QAI Hub on-device profiling (docs/weights_inventory.md)")
            lines.append(f"  {'#':<4} {'File':<40} {'TextEnc':>9} {'UNet':>9} {'VAEDec':>9} {'NPU Sum':>9} {'App RSS':>9} {'Total':>9}")
            lines.append(f"  {'-'*4} {'-'*40} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
            for i, b in enumerate(npu_benchmarks, 1):
                df   = b.generate_summary
                name = Path(b.filepath).stem[:39]
                variant = b.metadata.get("model_variant", "")
                pcp = _parse_pcp(b)
                te_mb  = _npu_mem_mb(variant, "text_encoder", pcp.get("text_encoder", "fp32"))
                un_mb  = _npu_mem_mb(variant, "unet",         pcp.get("unet", "fp32"))
                vd_mb  = _npu_mem_mb(variant, "vae_decoder",  pcp.get("vae_decoder", "fp32"))
                app_mb = df["peak_memory_mb"].max() if "peak_memory_mb" in df.columns else 0
                te_str  = f"{te_mb:>9}"   if te_mb  is not None else f"{'--':>9}"
                un_str  = f"{un_mb:>9}"   if un_mb  is not None else f"{'?':>9}"
                vd_str  = f"{vd_mb:>9}"   if vd_mb  is not None else f"{'--':>9}"
                all_known = all(x is not None for x in [te_mb, un_mb, vd_mb])
                npu_sum = sum([te_mb, un_mb, vd_mb]) if all_known else None
                npu_str = f"{npu_sum:>9}" if npu_sum is not None else f"{'?':>9}"
                total   = (npu_sum + app_mb) if (npu_sum is not None and app_mb > 0) else None
                tot_str = f"{total:>9.0f}" if total is not None else f"{'?':>9}"
                app_str = f"{app_mb:>9.0f}" if app_mb > 0 else f"{'--':>9}"
                lines.append(f"  {i:<4} {name:<40} {te_str} {un_str} {vd_str} {npu_str} {app_str} {tot_str}")
            lines.append("    TextEnc/UNet/VAEDec: QAI Hub on-device profiling 결과 (Snapdragon 8 Gen 2, S23)")
            lines.append("      NPU(HTP) 메모리 = 모델 가중치가 상주하는 HTP 전용 메모리 영역")
            lines.append("      App RSS(VmRSS)에는 미포함 — Android /proc/self/status 로 관측 불가")
            lines.append("      실기기 on-device 직접 측정은 root 권한 + /sys/kernel/debug/ion/ 접근 필요 (stock Android SELinux 차단)")
            lines.append("    App RSS: peak VmRSS during inference (ORT runtime, Java heap, shared libs; NPU 제외)")
            lines.append("    Total: NPU Sum + App RSS  (추정치)")
            lines.append("    ?: no QAI Hub profile available for this component/precision combination")

        # [4b] Power (mW)
        has_power = any("avg_power_mw" in b.generate_summary.columns
                        and b.generate_summary["avg_power_mw"].mean() > 0 for b in gen_benchmarks)
        if has_power:
            lines.append("")
            lines.append("  Power (mW)")
            lines.append(f"  {'#':<4} {'File':<40} {'Idle':>10} {'AvgInfer':>10} {'Delta':>10}")
            lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
            for i, b in enumerate(gen_benchmarks, 1):
                df   = b.generate_summary
                bm   = b.metadata
                name = Path(b.filepath).stem[:39]
                power    = df["avg_power_mw"].mean() if "avg_power_mw" in df.columns else 0
                idle_pwr = float(bm.get('idle_baseline_power_mw', 0))
                delta_pwr = power - idle_pwr if idle_pwr > 0 else 0
                idle_str      = f"{idle_pwr:>10.0f}"  if idle_pwr > 0 else f"{'--':>10}"
                delta_pwr_str = f"{delta_pwr:>+10.0f}" if idle_pwr > 0 else f"{'--':>10}"
                lines.append(f"  {i:<4} {name:<40} {idle_str} {power:>10.0f} {delta_pwr_str}")
            lines.append("    Idle: 5s median before model load  |  AvgInfer: mean during inference  |  Delta = AvgInfer − Idle (net inference power)")
            lines.append("    ⚠ BatteryManager.BATTERY_PROPERTY_CURRENT_NOW 기반 — 폰 전체 시스템 소비 전력 (SoC+디스플레이+라디오 포함)")
            lines.append("      앱 단독 / NPU 단독 전력 분리는 Snapdragon Profiler 또는 PMU 카운터 접근 필요")

        # [4c] Thermal (°C)
        has_thermal = any("end_temp_c" in b.generate_summary.columns
                          and b.generate_summary["end_temp_c"].mean() > 0 for b in gen_benchmarks)
        if has_thermal:
            lines.append("")
            lines.append("  Thermal (°C)")
            lines.append(f"  {'#':<4} {'File':<40} {'Idle':>8} {'Post1stInf':>11} {'End':>8}")
            lines.append(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*11} {'-'*8}")
            for i, b in enumerate(gen_benchmarks, 1):
                df   = b.generate_summary
                name = Path(b.filepath).stem[:39]
                cs_row = b.cold_start.iloc[0] if b.cold_start is not None and not b.cold_start.empty else {}
                idle_t      = cs_row.get('idle_thermal_c', 0) or 0
                post_infer_t = cs_row.get('first_infer_end_thermal_c', 0) or 0
                t_end       = df["end_temp_c"].iloc[-1] if "end_temp_c" in df.columns else 0
                idle_s  = f"{idle_t:>8.1f}"       if idle_t      > 0 else f"{'--':>8}"
                pinf_s  = f"{post_infer_t:>11.1f}" if post_infer_t > 0 else f"{'--':>11}"
                end_s   = f"{t_end:>8.1f}"         if t_end       > 0 else f"{'--':>8}"
                lines.append(f"  {i:<4} {name:<40} {idle_s} {pinf_s} {end_s}")
            lines.append("    Idle: before model load  |  Post1stInf: after cold first inference  |  End: after last warm trial")

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
