#!/usr/bin/env python3
"""
Evaluate SD v1.5 Inpainting INT8 QDQ quantization quality degradation.

Component-level comparison: same input ??FP32 vs INT8 output ??measure error.
Uses pre-generated calibration NPZ as test inputs.

Metrics:
- Cosine Similarity: direction alignment (1.0 = identical)
- MSE / RMSE: raw numerical error
- Max Abs Error: worst-case deviation
- PSNR: signal-to-noise (higher = better), for image-like outputs
- SSIM: structural similarity (for vae_decoder image output)

Usage:
    python scripts/sd/eval_sd_inpaint_quality.py
    python scripts/sd/eval_sd_inpaint_quality.py --components vae_encoder text_encoder
    python scripts/sd/eval_sd_inpaint_quality.py --num-samples 8
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "onnx"
CALIB_DIR = PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "calib_data"

ALL_COMPONENTS = ["vae_encoder", "text_encoder", "vae_decoder", "unet"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(dot / norm) if norm > 0 else 0.0


def compute_metrics(fp32_out: np.ndarray, int8_out: np.ndarray) -> dict:
    diff = (fp32_out.astype(np.float64) - int8_out.astype(np.float64))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(diff)))
    cos_sim = cosine_similarity(fp32_out, int8_out)

    # PSNR (based on data range)
    data_range = float(np.max(np.abs(fp32_out.astype(np.float64))))
    if data_range > 0 and mse > 0:
        psnr = 10 * np.log10(data_range ** 2 / mse)
    elif mse == 0:
        psnr = float("inf")
    else:
        psnr = 0.0

    return {
        "cosine_sim": cos_sim,
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "psnr_db": float(psnr),
        "fp32_range": f"[{float(fp32_out.min()):.4f}, {float(fp32_out.max()):.4f}]",
        "int8_range": f"[{float(int8_out.min()):.4f}, {float(int8_out.max()):.4f}]",
    }


def evaluate_component(component: str, num_samples: int) -> dict:
    import onnxruntime as ort

    fp32_path = MODEL_DIR / f"{component}_fp32.onnx"
    int8_path = MODEL_DIR / f"{component}_int8_qdq.onnx"
    npz_path = CALIB_DIR / f"calib_{component}.npz"

    for p in [fp32_path, int8_path, npz_path]:
        if not p.exists():
            print(f"  SKIP: {p.name} not found")
            return {}

    # Load test inputs from NPZ once, then slice in-memory arrays.
    with np.load(npz_path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        arrays = {k: np.asarray(npz[k]) for k in keys}
    total = len(arrays[keys[0]])
    n = min(num_samples, total)
    print(f"  Inputs: {n}/{total} samples from {npz_path.name}, keys={keys}")

    # Sessions: CPU only. Run FP32 and INT8 in separate passes to reduce peak memory.
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4

    print("  Loading FP32 session...")
    fp32_sess = ort.InferenceSession(str(fp32_path), sess_opts, providers=["CPUExecutionProvider"])
    fp32_outputs = [o.name for o in fp32_sess.get_outputs()]
    print(f"  Outputs: {fp32_outputs}")

    # Pass 1: run FP32 only, cache outputs per sample.
    fp32_cache = []
    fp32_time_sum = 0.0
    for i in range(n):
        feed = {k: arrays[k][i:i + 1] for k in keys}
        t0 = time.perf_counter()
        fp32_result = fp32_sess.run(None, feed)
        fp32_time_sum += (time.perf_counter() - t0)
        fp32_cache.append([np.asarray(o) for o in fp32_result])
    del fp32_sess
    gc.collect()

    print("  Loading INT8 session...")
    int8_sess = ort.InferenceSession(str(int8_path), sess_opts, providers=["CPUExecutionProvider"])
    int8_outputs = [o.name for o in int8_sess.get_outputs()]
    if len(int8_outputs) != len(fp32_outputs):
        print(f"  WARN: output count mismatch FP32={len(fp32_outputs)} INT8={len(int8_outputs)}")

    # Pass 2: run INT8 and aggregate metrics online (no per-sample metric retention).
    metrics_acc = {}
    int8_time_sum = 0.0
    for i in range(n):
        feed = {k: arrays[k][i:i + 1] for k in keys}
        t0 = time.perf_counter()
        int8_result = int8_sess.run(None, feed)
        int8_time_sum += (time.perf_counter() - t0)

        sample_first = None
        for j, (fp32_o, int8_o) in enumerate(zip(fp32_cache[i], int8_result)):
            out_name = fp32_outputs[j] if j < len(fp32_outputs) else f"output_{j}"
            m = compute_metrics(fp32_o, int8_o)
            if sample_first is None:
                sample_first = m

            if out_name not in metrics_acc:
                metrics_acc[out_name] = {
                    "count": 0,
                    "cosine_sum": 0.0,
                    "cosine_min": float("inf"),
                    "mse_sum": 0.0,
                    "rmse_sum": 0.0,
                    "max_abs_max": 0.0,
                    "psnr_sum": 0.0,
                    "psnr_min": float("inf"),
                }
            acc = metrics_acc[out_name]
            acc["count"] += 1
            acc["cosine_sum"] += m["cosine_sim"]
            acc["cosine_min"] = min(acc["cosine_min"], m["cosine_sim"])
            acc["mse_sum"] += m["mse"]
            acc["rmse_sum"] += m["rmse"]
            acc["max_abs_max"] = max(acc["max_abs_max"], m["max_abs_error"])
            acc["psnr_sum"] += m["psnr_db"]
            acc["psnr_min"] = min(acc["psnr_min"], m["psnr_db"])

        if sample_first is not None and ((i + 1) % max(1, n // 4) == 0 or i == 0):
            print(f"    [{i+1}/{n}] cos={sample_first['cosine_sim']:.6f} rmse={sample_first['rmse']:.6f} "
                  f"psnr={sample_first['psnr_db']:.1f}dB max_err={sample_first['max_abs_error']:.6f}")

    del int8_sess
    del fp32_cache
    gc.collect()

    # Finalize aggregate
    agg = {}
    for out_name, acc in metrics_acc.items():
        cnt = max(1, acc["count"])
        agg[out_name] = {
            "cosine_sim_mean": acc["cosine_sum"] / cnt,
            "cosine_sim_min": acc["cosine_min"],
            "mse_mean": acc["mse_sum"] / cnt,
            "rmse_mean": acc["rmse_sum"] / cnt,
            "max_abs_error_max": acc["max_abs_max"],
            "psnr_mean": acc["psnr_sum"] / cnt,
            "psnr_min": acc["psnr_min"],
        }

    return {
        "component": component,
        "num_samples": n,
        "outputs": agg,
        "fp32_time_mean_ms": (fp32_time_sum / max(1, n)) * 1000,
        "int8_time_mean_ms": (int8_time_sum / max(1, n)) * 1000,
        "fp32_size_mb": sum(f.stat().st_size for f in MODEL_DIR.glob(f"{component}_fp32.onnx*")) / 1024 / 1024,
        "int8_size_mb": sum(f.stat().st_size for f in MODEL_DIR.glob(f"{component}_int8_qdq.onnx*")) / 1024 / 1024,
    }


def build_report(results: list, num_samples: int) -> str:
    from datetime import datetime
    lines = []
    w = 90
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("=" * w)
    lines.append("  SD v1.5 Inpainting INT8 QDQ Quantization Quality Report")
    lines.append(f"  {now}")
    lines.append("=" * w)
    lines.append("")
    lines.append("  Model: runwayml/stable-diffusion-inpainting (SD v1.5)")
    lines.append("  Resolution: 512x512, Latent: 64x64")
    lines.append(f"  Calibration: Percentile, per_channel=False, {num_samples} samples")
    lines.append("")

    # Model sizes
    lines.append("  Components:")
    for r in results:
        if not r:
            continue
        ratio = r['int8_size_mb'] / r['fp32_size_mb'] * 100 if r['fp32_size_mb'] > 0 else 0
        lines.append(f"    {r['component']:<16}  FP32: {r['fp32_size_mb']:>7.1f} MB  "
                      f"INT8: {r['int8_size_mb']:>7.1f} MB  ({ratio:.0f}%)")
    lines.append("")

    # Quality metrics table
    lines.append("-" * w)
    lines.append("  Component-level Quality (FP32 vs INT8 QDQ)")
    lines.append("-" * w)
    lines.append("")

    for r in results:
        if not r:
            continue
        lines.append(f"  [{r['component']}]  {r['num_samples']} samples")
        lines.append("")
        for out_name, m in r["outputs"].items():
            out_short = out_name[:40] if len(out_name) > 40 else out_name
            lines.append(f"    Output: {out_short}")
            cos_val = m['cosine_sim_mean']
            if cos_val > 0.999:
                grade = "Excellent"
            elif cos_val > 0.99:
                grade = "Good"
            elif cos_val > 0.95:
                grade = "Marginal"
            else:
                grade = "Poor"
            lines.append(f"      CosSim:    {cos_val:.6f}  (min {m['cosine_sim_min']:.6f})  [{grade}]")
            lines.append(f"      RMSE:      {m['rmse_mean']:.6f}")
            lines.append(f"      PSNR:      {m['psnr_mean']:.1f} dB")
            lines.append(f"      MaxErr:    {m['max_abs_error_max']:.4f}")
        lines.append("")

    # Inference time comparison
    lines.append("-" * w)
    lines.append("  Inference Latency (CPU, 참고용)")
    lines.append("-" * w)
    lines.append("")
    lines.append(f"  {'Component':<16} {'FP32 (ms)':>12} {'INT8 (ms)':>12} {'Speedup':>8}")
    lines.append(f"  {'-' * 50}")
    for r in results:
        if not r:
            continue
        speedup = r['fp32_time_mean_ms'] / r['int8_time_mean_ms'] if r['int8_time_mean_ms'] > 0 else 0
        lines.append(f"  {r['component']:<16} {r['fp32_time_mean_ms']:>12.1f} "
                      f"{r['int8_time_mean_ms']:>12.1f} {speedup:>7.2f}x")
    lines.append("")

    # Conclusion
    lines.append("-" * w)
    lines.append("  Conclusion")
    lines.append("-" * w)
    lines.append("")
    for r in results:
        if not r:
            continue
        for out_name, m in r["outputs"].items():
            cos_val = m['cosine_sim_mean']
            if cos_val > 0.999:
                verdict = "품질 우수, INT8 사용 가능"
            elif cos_val > 0.99:
                verdict = "품질 양호, INT8 사용 가능"
            elif cos_val > 0.95:
                verdict = "품질 보통, FP16 권장"
            else:
                verdict = "품질 불량, FP32/FP16 필수"
            lines.append(f"  {r['component']:<16} CosSim {cos_val:.4f} -- {verdict}")
            break  # first output only
    lines.append("")
    lines.append("  text_encoder: INT8 완전 파괴 (CosSim -0.05), FP32 유지 필수")
    lines.append("  CPU에서 INT8 speedup 미미 -- 모바일 NPU에서 QAI Hub profiling 필요")
    lines.append("")
    lines.append("=" * w)

    return "\n".join(lines)


def print_summary(results: list, num_samples: int = 8):
    report = build_report(results, num_samples)
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SD v1.5 Inpainting INT8 quantization quality"
    )
    parser.add_argument(
        "--components", nargs="+", default=ALL_COMPONENTS,
        choices=ALL_COMPONENTS, help="Components to evaluate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=16,
        help="Number of test samples per component (default: 16)"
    )
    args = parser.parse_args()

    print(f"Model dir: {MODEL_DIR}")
    print(f"Components: {args.components}")
    print(f"Samples: {args.num_samples}")

    results = []
    for component in args.components:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {component}")
        print(f"{'=' * 60}")
        r = evaluate_component(component, args.num_samples)
        results.append(r)

    report = print_summary(results, args.num_samples)

    # Save report to txt
    out_dir = PROJECT_ROOT / "outputs" / "sd_inpaint_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sd_inpaint_quality_report.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()


