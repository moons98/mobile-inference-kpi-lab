#!/usr/bin/env python3
"""
Evaluate SD v1.5 INT8 quantization quality degradation.

Component-level comparison: same input -> FP32 vs INT8 output -> measure error.
Uses pre-generated calibration NPZ as test inputs.

Supports multiple INT8 variants per component:
- Custom QDQ (_int8_qdq): local ONNX Runtime static quantization
- QAI Hub (_qai_int8): QAI Hub W8A8 quantization

Metrics:
- Cosine Similarity: direction alignment (1.0 = identical)
- MSE / RMSE: raw numerical error
- Max Abs Error: worst-case deviation
- PSNR: signal-to-noise (higher = better)

Usage:
    python scripts/sd/quant_report_sd.py
    python scripts/sd/quant_report_sd.py --components vae_encoder unet_base
    python scripts/sd/quant_report_sd.py --num-samples 8
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "weights" / "sd_v1.5" / "onnx"
CALIB_DIR = PROJECT_ROOT / "weights" / "sd_v1.5" / "calib_data"

ALL_COMPONENTS = ["vae_encoder", "text_encoder", "vae_decoder", "unet_base", "unet_lcm"]

# Quantized variant suffixes to scan for each component
INT8_VARIANTS = [
    ("int8_qdq", "Custom QDQ"),
    ("mixed_pr", "Mixed Precision"),
    ("qai_int8", "QAI Hub W8A8"),
    ("w8a16", "AIMET W8A16"),
]


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


def _run_variant(int8_path, fp32_cache, fp32_outputs, arrays, keys, n):
    """Run a single INT8 variant against cached FP32 outputs."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4

    print(f"  Loading INT8 session: {int8_path.name}...")
    int8_sess = ort.InferenceSession(str(int8_path), sess_opts, providers=["CPUExecutionProvider"])
    int8_inputs = {inp.name: inp for inp in int8_sess.get_inputs()}
    int8_outputs = [o.name for o in int8_sess.get_outputs()]
    if len(int8_outputs) != len(fp32_outputs):
        print(f"  WARN: output count mismatch FP32={len(fp32_outputs)} INT8={len(int8_outputs)}")

    # Build input name mapping (FP32 keys -> INT8 keys) for qai-hub-models variants
    # that use different input names (e.g. sample->latent, encoder_hidden_states->text_emb)
    INPUT_ALIASES = {
        "sample": ["latent"], "latent": ["sample"],
        "encoder_hidden_states": ["text_emb"], "text_emb": ["encoder_hidden_states"],
    }
    key_map = {}  # maps FP32 key -> INT8 input name
    for k in keys:
        if k in int8_inputs:
            key_map[k] = k
        else:
            for alias in INPUT_ALIASES.get(k, []):
                if alias in int8_inputs:
                    key_map[k] = alias
                    break
    if len(key_map) != len(keys):
        missing = [k for k in keys if k not in key_map]
        print(f"  WARN: unmapped inputs: {missing}, INT8 expects: {list(int8_inputs.keys())}")

    metrics_acc = {}
    int8_time_sum = 0.0
    for i in range(n):
        feed = {}
        for k in keys:
            if k not in key_map:
                continue
            mapped_name = key_map[k]
            val = arrays[k][i:i + 1]
            # Handle dtype/shape mismatches (e.g. int64->float32, [1]->[1,1])
            inp_meta = int8_inputs[mapped_name]
            expected_type = inp_meta.type
            if "float" in expected_type and val.dtype != np.float32:
                val = val.astype(np.float32)
            expected_shape = inp_meta.shape
            if expected_shape is not None and len(expected_shape) == len(val.shape) + 1:
                val = val.reshape(expected_shape)
            elif expected_shape is not None and len(expected_shape) != len(val.shape):
                val = val.reshape(expected_shape)
            feed[mapped_name] = val
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

    int8_size = sum(f.stat().st_size for f in int8_path.parent.glob(f"{int8_path.name}*")) / 1024 / 1024
    return {
        "outputs": agg,
        "int8_time_mean_ms": (int8_time_sum / max(1, n)) * 1000,
        "int8_size_mb": int8_size,
    }


def evaluate_component(component: str, num_samples: int) -> dict:
    import onnxruntime as ort

    fp32_path = MODEL_DIR / f"{component}_fp32.onnx"
    # UNet base/lcm share calibration data
    calib_name = "unet" if component.startswith("unet_") else component
    npz_path = CALIB_DIR / f"calib_{calib_name}.npz"

    if not fp32_path.exists():
        print(f"  SKIP: {fp32_path.name} not found")
        return {}
    if not npz_path.exists():
        print(f"  SKIP: {npz_path.name} not found")
        return {}

    # Discover available INT8 variants
    variants = []
    for suffix, label in INT8_VARIANTS:
        int8_path = MODEL_DIR / f"{component}_{suffix}.onnx"
        if int8_path.exists():
            variants.append((suffix, label, int8_path))
    if not variants:
        print(f"  SKIP: no INT8 variants found for {component}")
        return {}

    print(f"  Variants: {', '.join(v[0] for v in variants)}")

    # Load test inputs
    with np.load(npz_path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        arrays = {k: np.asarray(npz[k]) for k in keys}
    total = len(arrays[keys[0]])
    n = min(num_samples, total)
    print(f"  Inputs: {n}/{total} samples from {npz_path.name}, keys={keys}")

    # FP32 pass (once)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4

    print("  Loading FP32 session...")
    fp32_sess = ort.InferenceSession(str(fp32_path), sess_opts, providers=["CPUExecutionProvider"])
    fp32_outputs = [o.name for o in fp32_sess.get_outputs()]
    print(f"  Outputs: {fp32_outputs}")

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

    fp32_size = sum(f.stat().st_size for f in MODEL_DIR.glob(f"{component}_fp32.onnx*")) / 1024 / 1024

    # Run each INT8 variant
    variant_results = {}
    for suffix, label, int8_path in variants:
        print(f"\n  --- {label} ({suffix}) ---")
        vr = _run_variant(int8_path, fp32_cache, fp32_outputs, arrays, keys, n)
        variant_results[suffix] = {**vr, "label": label}

    del fp32_cache
    gc.collect()

    return {
        "component": component,
        "num_samples": n,
        "fp32_time_mean_ms": (fp32_time_sum / max(1, n)) * 1000,
        "fp32_size_mb": fp32_size,
        "variants": variant_results,
    }


def _grade(cos_val):
    if cos_val > 0.999:
        return "Excellent"
    elif cos_val > 0.99:
        return "Good"
    elif cos_val > 0.95:
        return "Marginal"
    else:
        return "Poor"


def _verdict(cos_val):
    if cos_val > 0.999:
        return "INT8 사용 가능"
    elif cos_val > 0.99:
        return "INT8 사용 가능"
    elif cos_val > 0.95:
        return "FP16 권장"
    else:
        return "FP32/FP16 필수"


def build_report(results: list, num_samples: int) -> str:
    from datetime import datetime
    lines = []
    W = 80
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def sep(ch="="):
        lines.append(ch * W)

    # Collect all variant suffixes across components
    all_suffixes = []
    for r in results:
        if not r:
            continue
        for s in r['variants']:
            if s not in all_suffixes:
                all_suffixes.append(s)

    # Variant display labels
    suffix_labels = {s: l for s, l in INT8_VARIANTS}

    sep()
    lines.append("  SD v1.5 INT8 Quantization Quality Report")
    lines.append(f"  {now}")
    sep()

    lines.append("")
    lines.append("  Model: runwayml/stable-diffusion-v1-5 (SD v1.5)")
    lines.append("  Resolution: 512x512, Latent: 64x64")
    lines.append(f"  Test samples: {num_samples}")
    lines.append("")

    # --- Variants table ---
    lines.append("  Variants:")
    for r in results:
        if not r:
            continue
        comp = r['component']
        lines.append(f"    {comp:<16} FP32  {r['fp32_size_mb']:>7.1f} MB")
        for suffix, vr in r['variants'].items():
            label = suffix_labels.get(suffix, suffix)
            lines.append(f"    {'':16} {label:<16} {vr['int8_size_mb']:>7.1f} MB")

    # Components with no INT8
    all_evaluated = {r['component'] for r in results if r}
    missing = [c for c in ALL_COMPONENTS if c not in all_evaluated]
    for c in missing:
        lines.append(f"    {c:<16} FP32 only (INT8 미생성)")
    lines.append("")

    # --- Quality summary table ---
    sep("-")
    lines.append(f"  Quality Summary (FP32 vs INT8, {num_samples} samples)")
    sep("-")
    lines.append("")

    # Header
    hdr = f"  {'Component':<16} {'Variant':<16} {'Output':<22} {'CosSim':>8} {'min':>8} {'RMSE':>8} {'PSNR':>8} {'MaxErr':>8} {'Grade':>10}"
    lines.append(hdr)
    lines.append(f"  {'-' * (len(hdr) - 2)}")

    prev_comp = None
    for r in results:
        if not r:
            continue
        comp = r['component']
        if prev_comp is not None and comp != prev_comp:
            lines.append("")
        prev_comp = comp
        for suffix, vr in r['variants'].items():
            label = suffix_labels.get(suffix, suffix)
            for out_name, m in vr['outputs'].items():
                cos_val = m['cosine_sim_mean']
                grade = _grade(cos_val)
                out_short = out_name[:20] if len(out_name) > 20 else out_name
                lines.append(
                    f"  {comp:<16} {label:<16} {out_short:<22} {cos_val:>8.4f} {m['cosine_sim_min']:>8.4f} "
                    f"{m['rmse_mean']:>8.4f} {m['psnr_mean']:>7.1f} {m['max_abs_error_max']:>8.4f} "
                    f"[{grade}]"
                )
                comp = ""  # blank for subsequent rows
            comp = ""
        comp = r['component']
    lines.append("")

    # --- Latency table ---
    sep("-")
    lines.append("  Inference Latency (CPU, 참고용)")
    sep("-")
    lines.append("")

    hdr = f"  {'Component':<16} {'FP32 (ms)':>10}"
    for s in all_suffixes:
        label = suffix_labels.get(s, s)
        hdr += f"  {label+' (ms)':>16} {'Spdup':>6}"
    lines.append(hdr)
    lines.append(f"  {'-' * (len(hdr) - 2)}")

    for r in results:
        if not r:
            continue
        row = f"  {r['component']:<16} {r['fp32_time_mean_ms']:>10.1f}"
        for s in all_suffixes:
            if s in r['variants']:
                vr = r['variants'][s]
                speedup = r['fp32_time_mean_ms'] / vr['int8_time_mean_ms'] if vr['int8_time_mean_ms'] > 0 else 0
                row += f"  {vr['int8_time_mean_ms']:>16.1f} {speedup:>5.2f}x"
            else:
                row += f"  {'--':>16} {'--':>6}"
        lines.append(row)
    lines.append("")

    # --- Conclusion ---
    sep("-")
    lines.append("  Conclusion")
    sep("-")
    lines.append("")
    for r in results:
        if not r:
            continue
        for suffix, vr in r['variants'].items():
            label = suffix_labels.get(suffix, suffix)
            for out_name, m in vr['outputs'].items():
                cos_val = m['cosine_sim_mean']
                verdict = _verdict(cos_val)
                lines.append(f"  {r['component']:<16} {label:<16} CosSim {cos_val:.4f} -- {verdict}")
                break
    for c in missing:
        lines.append(f"  {c:<16} {'--':<16} INT8 미생성, FP32 유지")
    lines.append("")
    lines.append("  CPU에서 INT8 speedup 미미 -- 모바일 NPU에서 QAI Hub profiling 필요")
    lines.append("")
    sep()

    return "\n".join(lines)


def print_summary(results: list, num_samples: int = 8):
    report = build_report(results, num_samples)
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SD v1.5 INT8 quantization quality"
    )
    parser.add_argument(
        "--components", nargs="+", default=ALL_COMPONENTS,
        choices=ALL_COMPONENTS, help="Components to evaluate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=32,
        help="Number of test samples per component (default: 32)"
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
    out_dir = PROJECT_ROOT / "exp_outputs" / "quantization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sd_quantization.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
