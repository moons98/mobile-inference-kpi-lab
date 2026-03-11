#!/usr/bin/env python3
"""
Parse AI Eraser benchmark CSV files exported from the Android app.
Supports: ERASE_SUMMARY, UNET_STEP_DETAIL, COLD_START, YOLO_SEG_DETAIL sections.

Usage:
    python parse_eraser_csv.py <csv_file_or_directory>
    python parse_eraser_csv.py results/ --compare
    python parse_eraser_csv.py results/ -p   # print only, no save
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
    erase_summary: Optional[pd.DataFrame] = None
    unet_step_detail: Optional[pd.DataFrame] = None
    cold_start: Optional[pd.DataFrame] = None
    yolo_seg_detail: Optional[pd.DataFrame] = None


def parse_csv(filepath: Path) -> ParsedBenchmark:
    """Parse a multi-section CSV file into dataframes."""
    result = ParsedBenchmark(filepath=str(filepath))

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Extract metadata (# key,value lines at top)
    section_names = {"ERASE_SUMMARY", "UNET_STEP_DETAIL", "COLD_START", "YOLO_SEG_DETAIL"}
    for line in lines:
        line = line.strip()
        if line.startswith("# ") and "," in line:
            tag = line[2:].split(",")[0].strip()
            if tag not in section_names:
                parts = line[2:].split(",", 1)
                if len(parts) == 2:
                    result.metadata[parts[0].strip()] = parts[1].strip()

    # Find section boundaries
    sections = {}
    current_section = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and stripped[2:] in section_names:
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
        csv_text = data_lines[0] + "\n" + "\n".join(data_lines[1:])
        df = pd.read_csv(StringIO(csv_text))

        if name == "ERASE_SUMMARY":
            result.erase_summary = df
        elif name == "UNET_STEP_DETAIL":
            result.unet_step_detail = df
        elif name == "COLD_START":
            result.cold_start = df
        elif name == "YOLO_SEG_DETAIL":
            result.yolo_seg_detail = df

    return result


# ---------------------------------------------------------------------------
# Single-file report
# ---------------------------------------------------------------------------

def format_report(bench: ParsedBenchmark) -> str:
    lines = []
    W = 120
    sep = "=" * W
    sep2 = "-" * W

    lines.append(sep)
    lines.append("KPI Report")
    lines.append(sep)

    # --- [1] Experiment Overview ---
    lines.append("")
    lines.append("[1] Experiment Overview")
    lines.append(sep2)
    m = bench.metadata
    lines.append(f"  File:        {Path(bench.filepath).name}")
    lines.append(f"  Device:      {m.get('device_model','?')} ({m.get('soc_model','?')})")
    lines.append(f"  Runtime:     {m.get('runtime','?')}")
    lines.append(f"  Phase:       {m.get('phase','?')}")
    lines.append(f"  SD Backend:  {m.get('sd_backend','?')}  |  SD Precision:  {m.get('sd_precision','?')}")
    lines.append(f"  YOLO Backend:{m.get('yolo_backend','?')}  |  YOLO Precision:{m.get('yolo_precision','?')}")
    if m.get('phase') != 'YOLO_SEG_ONLY':
        lines.append(f"  Steps:       {m.get('steps','?')}  |  Strength:      {m.get('strength','?')}  |  ROI: {m.get('roi_size','?')}")

    # --- [2] Cold Start ---
    if bench.cold_start is not None and not bench.cold_start.empty:
        cs = bench.cold_start.iloc[0]
        lines.append("")
        lines.append("[2] Cold Start Breakdown")
        lines.append(sep2)
        lines.append(f"  {'Component':<25} {'Time (ms)':>12}")
        lines.append(f"  {'-'*25} {'-'*12}")

        components = [
            ("YOLO-seg load", "yolo_seg_load_ms"),
            ("VAE Encoder", "vae_enc_load_ms"),
            ("Text Encoder", "text_enc_load_ms"),
            ("UNet", "unet_load_ms"),
            ("VAE Decoder", "vae_dec_load_ms"),
        ]
        for label, col in components:
            val = cs.get(col, 0)
            if val > 0:
                lines.append(f"  {label:<25} {val:>12,.0f}")

        lines.append(f"  {'='*25} {'='*12}")
        lines.append(f"  {'Total':<25} {cs.get('total_load_ms', 0):>12,.0f}")
        lines.append(f"  Peak memory after load:  {cs.get('peak_memory_after_load_mb', 0)} MB")

    # --- [3] YOLO-seg Detail ---
    if bench.yolo_seg_detail is not None and not bench.yolo_seg_detail.empty:
        df = bench.yolo_seg_detail
        n = len(df)
        lines.append("")
        lines.append(f"[3] YOLO-seg Inference ({n} trials)")
        lines.append(sep2)
        lines.append(f"  Backend:   {df['backend'].iloc[0]}  |  Precision: {df['precision'].iloc[0]}  |  Image: {df['test_image'].iloc[0]}")
        lines.append("")

        _append_stat_table(lines, df, {
            "inference_ms": "ORT Inference",
            "input_create_ms": "Input Create",
            "nms_ms": "NMS",
            "mask_decode_ms": "Mask Decode",
            "output_process_ms": "Output Process",
        })

        # Total E2E
        if all(c in df.columns for c in ["inference_ms", "input_create_ms", "output_process_ms"]):
            total = df["inference_ms"] + df["input_create_ms"] + df["output_process_ms"]
            lines.append(f"  {'='*25} {'='*10} {'='*10} {'='*10} {'='*10} {'='*10} {'='*10}")
            lines.append(f"  {'Total E2E':<25} {total.mean():>10.2f} {total.median():>10.2f} "
                         f"{total.quantile(0.95):>10.2f} {total.min():>10.2f} {total.max():>10.2f} {total.std():>10.2f}")
            fps = 1000.0 / total.mean()
            lines.append(f"\n  Throughput: {fps:.1f} FPS (1000 / mean E2E)")

        if "mask_count" in df.columns:
            lines.append(f"  Avg masks detected:  {df['mask_count'].mean():.1f}")
        if "selected_mask_area_pct" in df.columns:
            lines.append(f"  Selected mask area:  {df['selected_mask_area_pct'].mean():.2f}%")

        lines.append("")
        lines.append("    ORT Inference: session.run() only (model forward pass)")
        lines.append("    Input Create:  Bitmap -> FloatBuffer -> OnnxTensor")
        lines.append("    Output Process: OnnxTensor copy + NMS + mask decode + post-processing")

    # --- [4] Inpainting Summary ---
    if bench.erase_summary is not None and not bench.erase_summary.empty:
        df = bench.erase_summary
        n = len(df)
        lines.append("")
        lines.append(f"[4] Inpainting E2E Performance ({n} trials)")
        lines.append(sep2)

        _append_stat_table(lines, df, {
            "full_e2e_ms": "Full E2E",
            "inpaint_e2e_ms": "Inpaint E2E",
        })

        if "inpaint_e2e_ms" in df.columns:
            fps = 1000.0 / df["inpaint_e2e_ms"].mean()
            lines.append(f"\n  Inpaint throughput: {fps:.2f} FPS")

        # --- [4b] Stage Breakdown ---
        lines.append("")
        lines.append(f"[4b] Inpainting Stage Breakdown (P50)")
        lines.append(sep2)

        stage_cols = {
            "yolo_seg_ms": "YOLO-seg",
            "roi_crop_ms": "ROI Crop",
            "tokenize_ms": "Tokenize",
            "text_enc_ms": "Text Encoder",
            "vae_enc_ms": "VAE Encoder (x2)",
            "masked_img_prep_ms": "Masked Img Prep",
            "unet_total_ms": "UNet Total",
            "vae_dec_ms": "VAE Decoder",
            "composite_ms": "Composite",
        }

        lines.append(f"  {'Stage':<25} {'P50':>10} {'Mean':>10} {'P95':>10} {'Min':>10} {'Max':>10}")
        lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for col, label in stage_cols.items():
            if col in df.columns and df[col].mean() > 0.01:
                s = df[col]
                lines.append(f"  {label:<25} {s.median():>10.1f} {s.mean():>10.1f} "
                             f"{s.quantile(0.95):>10.1f} {s.min():>10.1f} {s.max():>10.1f}")

        lines.append("")
        lines.append("    Text Encoder: CLIP text encoding (1x per inpaint)")
        lines.append("    VAE Encoder:  image -> latent + masked_image -> latent (2x)")
        lines.append("    UNet Total:   N denoising steps (9ch inpainting UNet)")
        lines.append("    VAE Decoder:  latent -> RGB image (1x)")

        # --- [4c] UNet Per-Step ---
        if "unet_per_step_mean_ms" in df.columns:
            lines.append("")
            lines.append(f"[4c] UNet Per-Step Statistics")
            lines.append(sep2)
            lines.append(f"  Per-step mean:   {df['unet_per_step_mean_ms'].mean():.2f} ms")
            lines.append(f"  Per-step P95:    {df['unet_per_step_p95_ms'].mean():.2f} ms")
            if "scheduler_overhead_ms" in df.columns:
                lines.append(f"  Scheduler overhead: {df['scheduler_overhead_ms'].mean():.2f} ms (total across all steps)")

    # --- [5] UNet Step Detail ---
    if bench.unet_step_detail is not None and not bench.unet_step_detail.empty:
        df = bench.unet_step_detail
        lines.append("")
        lines.append(f"[5] UNet Step Detail ({len(df)} total steps across all trials)")
        lines.append(sep2)

        _append_stat_table(lines, df, {
            "session_run_ms": "Session Run",
            "input_create_ms": "Input Create",
            "output_copy_ms": "Output Copy",
            "scheduler_step_ms": "Scheduler Step",
            "step_total_ms": "Step Total",
        })

        lines.append("")
        lines.append("    Session Run:    ORT session.run() for one UNet step")
        lines.append("    Input Create:   9ch latent concat -> OnnxTensor")
        lines.append("    Output Copy:    OnnxTensor -> FloatArray")
        lines.append("    Scheduler Step: PNDM/DDIM noise scheduling")

    # --- [6] System Resources ---
    if bench.erase_summary is not None and not bench.erase_summary.empty:
        df = bench.erase_summary
        has_thermal = "start_temp_c" in df.columns and df["start_temp_c"].mean() > 0
        has_power = "avg_power_mw" in df.columns and df["avg_power_mw"].mean() > 0
        has_memory = "peak_memory_mb" in df.columns

        if has_thermal or has_power or has_memory:
            lines.append("")
            lines.append("[6] System Resources")
            lines.append(sep2)

            if has_thermal:
                lines.append(f"  Thermal start:    {df['start_temp_c'].iloc[0]:.1f} C")
                lines.append(f"  Thermal end:      {df['end_temp_c'].iloc[-1]:.1f} C")
                delta = df['end_temp_c'].iloc[-1] - df['start_temp_c'].iloc[0]
                lines.append(f"  Thermal delta:    {delta:+.1f} C")

            if has_power:
                lines.append(f"  Avg power:        {df['avg_power_mw'].mean():.0f} mW")

            if has_memory:
                lines.append(f"  Peak memory:      {df['peak_memory_mb'].max()} MB")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def _append_stat_table(lines, df, col_map):
    """Append a statistics table (Mean/P50/P95/Min/Max/Std)."""
    lines.append(f"  {'Metric':<25} {'Mean':>10} {'P50':>10} {'P95':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for col, label in col_map.items():
        if col in df.columns:
            s = df[col]
            lines.append(f"  {label:<25} {s.mean():>10.2f} {s.median():>10.2f} "
                         f"{s.quantile(0.95):>10.2f} {s.min():>10.2f} {s.max():>10.2f} {s.std():>10.2f}")


# ---------------------------------------------------------------------------
# Multi-file comparison
# ---------------------------------------------------------------------------

def format_comparison(benchmarks: list) -> str:
    lines = []
    W = 120
    sep = "=" * W
    sep2 = "-" * W

    lines.append(f"Found {len(benchmarks)} file(s)")
    lines.append(sep)
    lines.append("KPI Comparison Report")
    lines.append(sep)

    # --- [1] Experiment Overview ---
    lines.append("")
    lines.append("[1] Experiment Overview")
    lines.append(sep2)
    lines.append(f"  {'#':<4} {'File':<50} {'Phase':<20} {'SD EP':<12} {'SD Prec':<8} {'YOLO EP':<12} {'YOLO Prec':<10}")
    lines.append(f"  {'-'*4} {'-'*50} {'-'*20} {'-'*12} {'-'*8} {'-'*12} {'-'*10}")
    for i, b in enumerate(benchmarks, 1):
        m = b.metadata
        name = Path(b.filepath).stem[:49]
        lines.append(f"  {i:<4} {name:<50} {m.get('phase','?'):<20} {m.get('sd_backend','?'):<12} "
                     f"{m.get('sd_precision','?'):<8} {m.get('yolo_backend','?'):<12} {m.get('yolo_precision','?'):<10}")

    # --- [2] YOLO-seg Comparison ---
    yolo_benchmarks = [b for b in benchmarks if b.yolo_seg_detail is not None and not b.yolo_seg_detail.empty]
    if yolo_benchmarks:
        lines.append("")
        lines.append("[2] YOLO-seg Latency (mean, ms)")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<45} {'Backend':<12} {'Prec':<6} "
                     f"{'Infer':>8} {'InpCr':>8} {'NMS':>8} {'MaskDec':>8} {'OutProc':>8} {'Total':>8} {'FPS':>6}")
        lines.append(f"  {'-'*4} {'-'*45} {'-'*12} {'-'*6} "
                     f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

        for i, b in enumerate(yolo_benchmarks, 1):
            df = b.yolo_seg_detail
            name = Path(b.filepath).stem[:44]
            backend = df["backend"].iloc[0]
            prec = df["precision"].iloc[0]
            infer = df["inference_ms"].mean()
            inp = df["input_create_ms"].mean()
            nms = df["nms_ms"].mean()
            mask = df["mask_decode_ms"].mean()
            outp = df["output_process_ms"].mean()
            total = infer + inp + outp
            fps = 1000.0 / total
            lines.append(f"  {i:<4} {name:<45} {backend:<12} {prec:<6} "
                         f"{infer:>8.2f} {inp:>8.2f} {nms:>8.2f} {mask:>8.2f} {outp:>8.2f} {total:>8.2f} {fps:>6.1f}")

        lines.append("")
        lines.append("    Infer:    session.run() only")
        lines.append("    InpCr:    Bitmap -> OnnxTensor")
        lines.append("    NMS:      Non-maximum suppression")
        lines.append("    MaskDec:  Instance mask decoding")
        lines.append("    OutProc:  Total output processing (NMS + MaskDec + copy)")
        lines.append("    Total:    Infer + InpCr + OutProc")
        lines.append("    FPS:      1000 / Total")

    # --- [3] Inpainting Comparison ---
    erase_benchmarks = [b for b in benchmarks if b.erase_summary is not None and not b.erase_summary.empty]
    if erase_benchmarks:
        lines.append("")
        lines.append("[3] Inpainting E2E Latency (mean, ms)")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'Backend':<10} {'Prec':<6} "
                     f"{'E2E':>10} {'TextEnc':>10} {'VAEEnc':>10} {'UNet':>10} {'VAEDec':>10} {'Steps':>6}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*6} "
                     f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

        for i, b in enumerate(erase_benchmarks, 1):
            df = b.erase_summary
            name = Path(b.filepath).stem[:39]
            backend = df["backend_sd"].iloc[0]
            prec = df["precision_sd"].iloc[0]
            e2e = df["inpaint_e2e_ms"].mean()
            textenc = df["text_enc_ms"].mean()
            vaeenc = df["vae_enc_ms"].mean()
            unet = df["unet_total_ms"].mean()
            vaedec = df["vae_dec_ms"].mean()
            steps = df["actual_steps"].iloc[0]
            lines.append(f"  {i:<4} {name:<40} {backend:<10} {prec:<6} "
                         f"{e2e:>10.1f} {textenc:>10.1f} {vaeenc:>10.1f} {unet:>10.1f} {vaedec:>10.1f} {steps:>6}")

        # UNet per-step comparison
        lines.append("")
        lines.append("[3b] UNet Per-Step Comparison")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'PerStep':>10} {'P95':>10} {'SchedOH':>10}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
        for i, b in enumerate(erase_benchmarks, 1):
            df = b.erase_summary
            name = Path(b.filepath).stem[:39]
            ps_mean = df["unet_per_step_mean_ms"].mean() if "unet_per_step_mean_ms" in df.columns else 0
            ps_p95 = df["unet_per_step_p95_ms"].mean() if "unet_per_step_p95_ms" in df.columns else 0
            sched = df["scheduler_overhead_ms"].mean() if "scheduler_overhead_ms" in df.columns else 0
            lines.append(f"  {i:<4} {name:<40} {ps_mean:>10.2f} {ps_p95:>10.2f} {sched:>10.2f}")

    # --- [4] Cold Start Comparison ---
    cold_benchmarks = [b for b in benchmarks if b.cold_start is not None and not b.cold_start.empty]
    if cold_benchmarks:
        lines.append("")
        lines.append("[4] Cold Start Breakdown (ms)")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'YOLO':>8} {'VAEEnc':>8} {'TextEnc':>8} "
                     f"{'UNet':>8} {'VAEDec':>8} {'Total':>8} {'Mem(MB)':>8}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for i, b in enumerate(cold_benchmarks, 1):
            cs = b.cold_start.iloc[0]
            name = Path(b.filepath).stem[:39]
            lines.append(f"  {i:<4} {name:<40} "
                         f"{cs.get('yolo_seg_load_ms',0):>8.0f} {cs.get('vae_enc_load_ms',0):>8.0f} "
                         f"{cs.get('text_enc_load_ms',0):>8.0f} {cs.get('unet_load_ms',0):>8.0f} "
                         f"{cs.get('vae_dec_load_ms',0):>8.0f} {cs.get('total_load_ms',0):>8.0f} "
                         f"{cs.get('peak_memory_after_load_mb',0):>8}")

        lines.append("")
        lines.append("    YOLO~VAEDec: 각 컴포넌트 세션 생성 시간 (QNN EP = HTP 그래프 컴파일 포함)")
        lines.append("    Total:       전체 로드 시간")
        lines.append("    Mem:         로드 완료 후 VmRSS (MB)")

    # --- [5] System Resources ---
    if erase_benchmarks:
        lines.append("")
        lines.append("[5] System Resources")
        lines.append(sep2)
        lines.append(f"  {'#':<4} {'File':<40} {'TempStart':>10} {'TempEnd':>10} {'Delta':>8} {'Power(mW)':>10} {'PeakMem':>8}")
        lines.append(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
        for i, b in enumerate(erase_benchmarks, 1):
            df = b.erase_summary
            name = Path(b.filepath).stem[:39]
            t_start = df["start_temp_c"].iloc[0] if "start_temp_c" in df.columns else 0
            t_end = df["end_temp_c"].iloc[-1] if "end_temp_c" in df.columns else 0
            delta = t_end - t_start
            power = df["avg_power_mw"].mean() if "avg_power_mw" in df.columns else 0
            mem = df["peak_memory_mb"].max() if "peak_memory_mb" in df.columns else 0
            lines.append(f"  {i:<4} {name:<40} {t_start:>10.1f} {t_end:>10.1f} {delta:>+8.1f} {power:>10.0f} {mem:>8}")

        lines.append("")
        lines.append("    TempStart/End: 시작/종료 시점 온도 (C)")
        lines.append("    Delta:         온도 변화량 (양수=발열)")
        lines.append("    Power:         평균 전력 소비 (mW)")
        lines.append("    PeakMem:       최대 메모리 사용량 (MB)")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse AI Eraser benchmark CSV files")
    parser.add_argument("paths", nargs="+", help="CSV files or directories to analyze")
    parser.add_argument("--compare", "-c", action="store_true", help="Show comparison table")
    parser.add_argument("--output", "-o", help="Output file path (default: auto-generate in analysis/outputs/)")
    parser.add_argument("--output-dir", "-d", help="Output directory (default: analysis/outputs/ in project root)")
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
            # Default: project root outputs/ directory
            out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent.parent / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if len(csv_files) == 1:
                stem = csv_files[0].stem
                out_path = out_dir / f"{stem}.txt"
            else:
                out_path = out_dir / f"eraser_comparison_{timestamp}.txt"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nSaved to {out_path}")
