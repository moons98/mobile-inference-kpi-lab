#!/usr/bin/env python3
"""
On-device W8A16 quantization quality evaluation via QAI Hub inference jobs.

W8A16 ONNX has no QDQ nodes, so local ORT inference = FP32 identical.
Only way to evaluate real W8A16 quality is on-device QNN NPU execution.

Strategy:
  - FP32 baseline: local ORT CPU inference (fast, no job needed)
  - W8A8 (QDQ): local ORT CPU inference (QDQ nodes present in ONNX)
  - W8A16: QAI Hub inference job (on-device NPU, real QNN quantization)

Pipeline:
  1. --submit: Load calibration data, submit W8A16 inference jobs to QAI Hub
  2. --status: Check job completion
  3. --report: Run FP32/W8A8 locally + download W8A16 results → compare → report

Usage:
    # Submit W8A16 inference jobs (default 4 samples)
    python scripts/sd/eval_ondevice.py --submit
    python scripts/sd/eval_ondevice.py --submit --num-samples 8

    # Check job status
    python scripts/sd/eval_ondevice.py --status

    # Generate report (runs FP32/W8A8 locally, downloads W8A16 from QAI Hub)
    python scripts/sd/eval_ondevice.py --report
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "weights" / "sd_v1.5" / "onnx"
CALIB_DIR = PROJECT_ROOT / "weights" / "sd_v1.5" / "calib_data"
JOBS_FILE = PROJECT_ROOT / "outputs" / "quantization" / "sd_quant_eval_jobs.json"
REPORT_FILE = PROJECT_ROOT / "outputs" / "quantization" / "sd_quant_quality.txt"

DEVICE_NAME = "Samsung Galaxy S23"

# ============================================================
# Compile job IDs (W8A16 only — these need on-device inference)
# ============================================================
W8A16_COMPILE_JOBS = {
    "text_encoder": "j561l217p",
    "vae_decoder":  "jpev3ov05",   # uint16 I/O (qai-hub-models export)
    "unet_base":    "jgjl4dl8p",   # uint16 I/O (qai-hub-models export)
    "unet_lcm":     "jgz7k21xp",
}

# Components with uint16 quantized I/O (qai-hub-models export).
# Requires float32→uint16 quantization on input, uint16→float32 dequantization on output.
# Parameters from .encodings files (AIMET W8A16, asymmetric per-tensor).
# Formula: uint16 = clamp(round(float / scale) - offset, 0, 65535)
#          float  = (uint16 + offset) * scale
UINT16_IO_QUANT = {
    "vae_decoder": {
        "inputs": {
            "latent": {"scale": 0.00034003708, "offset": -34382.0, "nhwc": True},
        },
        "output": {"scale": 1.5259021896696422e-05, "offset": 0.0, "nhwc": True},
    },
    "unet_base": {
        "inputs": {
            "latent":   {"scale": 0.00024176309, "offset": -33983.0, "nhwc": True},
            "timestep": {"scale": 0.01477073, "offset": 0.0, "nhwc": False},
            "text_emb": {"scale": 0.00093315604, "offset": -30103.0, "nhwc": False},
        },
        "output": {"scale": 0.0001881735488753065, "offset": -32340.0, "nhwc": True},
    },
}

# Local (non-W8A16) ONNX variants — evaluated via ORT CPU
# {component: {variant_key: onnx_filename}}
LOCAL_ONNX = {
    "vae_encoder": {
        "w8a8": "vae_encoder_qai_int8.onnx",
    },
    "vae_decoder": {
        "w8a8": "vae_decoder_qai_int8.onnx",
    },
    "unet_base": {
        "int8_qdq": "unet_base_int8_qdq.onnx",
        "mixed_pr":  "unet_base_mixed_pr.onnx",
    },
    "unet_lcm": {
        "int8_qdq": "unet_lcm_int8_qdq.onnx",
        "mixed_pr":  "unet_lcm_mixed_pr.onnx",
    },
}

# Component calibration & input config
COMPONENT_INFO = {
    "vae_encoder": {
        "calib_file": "calib_vae_encoder.npz",
        "input_keys": ["sample"],
        "fp32_onnx": "vae_encoder_fp32.onnx",
    },
    "text_encoder": {
        "calib_file": "calib_text_encoder.npz",
        "input_keys": ["input_ids"],
        "fp32_onnx": "text_encoder_fp32.onnx",
    },
    "vae_decoder": {
        "calib_file": "calib_vae_decoder.npz",
        "input_keys": ["latent_sample"],
        "fp32_onnx": "vae_decoder_fp32.onnx",
    },
    "unet_base": {
        "calib_file": "calib_unet.npz",
        "input_keys": ["sample", "timestep", "encoder_hidden_states"],
        "fp32_onnx": "unet_base_fp32.onnx",
    },
    "unet_lcm": {
        "calib_file": "calib_unet.npz",
        "input_keys": ["sample", "timestep", "encoder_hidden_states"],
        "fp32_onnx": "unet_lcm_fp32.onnx",
    },
}

# Input name mapping: calib NPZ key -> compiled model's expected input name
# Discovered from QAI Hub compiled model input_spec
INPUT_NAME_MAP = {
    ("text_encoder", "w8a16"): {
        "input_ids": "tokens",
    },
    ("vae_decoder", "w8a16"): {
        "latent_sample": "latent",
    },
    ("unet_base", "w8a16"): {
        "sample": "latent",
        "encoder_hidden_states": "text_emb",
    },
}

# Compiled model expected input order (from input_spec)
# QAI Hub matches inputs by position, so order matters
INPUT_ORDER = {
    ("unet_lcm", "w8a16"): ["timestep", "sample", "encoder_hidden_states"],
    ("unet_base", "w8a16"): ["latent", "timestep", "text_emb"],
}

VARIANT_LABELS = {
    "fp32":     "FP32",
    "w8a8":     "W8A8 (QAI Hub INT8)",
    "int8_qdq": "INT8 QDQ",
    "mixed_pr": "Mixed Precision",
    "w8a16":    "W8A16 (AIMET)",
}

ALL_COMPONENTS = ["vae_encoder", "text_encoder", "vae_decoder", "unet_base", "unet_lcm"]

# Post-processing applied to FP32 output for W8A16 comparison.
# qai-hub-models compiled models include extra ops not in the raw FP32 ONNX.
# {component: callable(np.ndarray) -> np.ndarray}
FP32_POSTPROC_FOR_W8A16 = {
    # qai-hub-models adds /Div (÷2 + 0.5) + /Clip [0,1] after raw VAE output
    "vae_decoder": lambda x: np.clip(x.astype(np.float64) / 2 + 0.5, 0.0, 1.0).astype(np.float32),
}


# ============================================================
# Metrics
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(dot / norm) if norm > 0 else 0.0


def compute_metrics(fp32_out: np.ndarray, quant_out: np.ndarray) -> dict:
    diff = fp32_out.astype(np.float64) - quant_out.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(diff)))
    cos_sim = cosine_similarity(fp32_out, quant_out)

    data_range = float(np.max(np.abs(fp32_out.astype(np.float64))))
    if data_range > 0 and mse > 0:
        psnr = float(10 * np.log10(data_range ** 2 / mse))
    elif mse == 0:
        psnr = float("inf")
    else:
        psnr = 0.0

    return {
        "cosine_sim": cos_sim,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "psnr_db": psnr,
    }


def aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate per-sample metrics into summary."""
    if not metrics_list:
        return {}
    n = len(metrics_list)
    return {
        "cos_mean": sum(m["cosine_sim"] for m in metrics_list) / n,
        "cos_min": min(m["cosine_sim"] for m in metrics_list),
        "rmse_mean": sum(m["rmse"] for m in metrics_list) / n,
        "max_abs": max(m["max_abs_error"] for m in metrics_list),
        "psnr_mean": sum(m["psnr_db"] for m in metrics_list) / n,
        "psnr_min": min(m["psnr_db"] for m in metrics_list),
        "num_samples": n,
    }


# ============================================================
# Local inference (FP32, W8A8)
# ============================================================

def run_local_inference(onnx_path: Path, arrays: dict, input_keys: list, num_samples: int):
    """Run local ORT CPU inference, return list of output arrays per sample."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_opts.intra_op_num_threads = 4
    sess_opts.enable_mem_pattern = False

    print(f"    Loading: {onnx_path.name}...")
    sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    sess_inputs = {inp.name: inp for inp in sess.get_inputs()}
    output_names = [o.name for o in sess.get_outputs()]

    # Build input name/dtype mapping
    INPUT_ALIASES = {
        "sample": ["latent"], "latent": ["sample"],
        "encoder_hidden_states": ["text_emb"], "text_emb": ["encoder_hidden_states"],
    }

    key_map = {}
    for k in input_keys:
        if k in sess_inputs:
            key_map[k] = k
        else:
            for alias in INPUT_ALIASES.get(k, []):
                if alias in sess_inputs:
                    key_map[k] = alias
                    break

    results = []
    for i in range(num_samples):
        feed = {}
        for k in input_keys:
            if k not in key_map:
                continue
            mapped = key_map[k]
            val = arrays[k][i:i + 1]
            inp_meta = sess_inputs[mapped]
            expected_type = inp_meta.type
            if "float" in expected_type and val.dtype != np.float32:
                val = val.astype(np.float32)
            expected_shape = inp_meta.shape
            if expected_shape and len(expected_shape) != len(val.shape):
                val = val.reshape(expected_shape)
            feed[mapped] = val
        out = sess.run(None, feed)
        results.append({name: np.asarray(o) for name, o in zip(output_names, out)})

    del sess
    gc.collect()
    return results


# ============================================================
# QAI Hub job submission (W8A16 only)
# ============================================================

def load_calibration_data(component: str, num_samples: int):
    """Load calibration data from NPZ file."""
    info = COMPONENT_INFO[component]
    npz_path = CALIB_DIR / info["calib_file"]
    if not npz_path.exists():
        raise FileNotFoundError(f"Calibration data not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        arrays = {k: np.asarray(npz[k]) for k in keys}

    total = len(arrays[keys[0]])
    n = min(num_samples, total)
    return arrays, n


def _float_to_uint16(val: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Quantize float32 → uint16 using AIMET asymmetric params."""
    q = np.round(val.astype(np.float64) / scale) - offset
    return np.clip(q, 0, 65535).astype(np.uint16)


def _nchw_to_nhwc(val: np.ndarray) -> np.ndarray:
    """Transpose [N,C,H,W] → [N,H,W,C]."""
    if val.ndim == 4:
        return np.transpose(val, (0, 2, 3, 1))
    return val


def prepare_w8a16_inputs(component: str, arrays: dict, num_samples: int):
    """Prepare batched input dict for W8A16 QAI Hub inference job.
    Returns OrderedDict {input_name: [sample0, sample1, ...]} with correct order."""
    from collections import OrderedDict

    info = COMPONENT_INFO[component]
    name_map = INPUT_NAME_MAP.get((component, "w8a16"), {})
    uint16_params = UINT16_IO_QUANT.get(component, {}).get("inputs", {})

    # Build unordered first
    raw = {}
    for key in info["input_keys"]:
        mapped_name = name_map.get(key, key)
        samples = []
        for i in range(num_samples):
            val = arrays[key][i:i + 1]

            if mapped_name in uint16_params:
                # uint16 quantized I/O: float32 → NHWC → uint16
                qp = uint16_params[mapped_name]
                if qp["nhwc"]:
                    val = _nchw_to_nhwc(val.astype(np.float32))
                val = _float_to_uint16(val, qp["scale"], qp["offset"])
            else:
                # --truncate_64bit_io: int64 -> int32
                if val.dtype == np.int64:
                    val = val.astype(np.int32)

            samples.append(val)
        raw[mapped_name] = samples

    # Apply input order if specified
    order = INPUT_ORDER.get((component, "w8a16"))
    if order:
        feed = OrderedDict()
        for name in order:
            if name in raw:
                feed[name] = raw[name]
    else:
        feed = raw

    return feed


def submit_inference_jobs(components: list, num_samples: int):
    """Submit W8A16 inference jobs to QAI Hub (1 job per component, batched samples)."""
    import qai_hub as hub

    device = hub.Device(DEVICE_NAME)
    all_jobs = []

    for component in components:
        if component not in W8A16_COMPILE_JOBS:
            print(f"  {component}: no W8A16 compile job, skipping")
            continue

        compile_job_id = W8A16_COMPILE_JOBS[component]
        print(f"\n{'=' * 60}")
        print(f"{component} W8A16 (compile: {compile_job_id})")
        print(f"{'=' * 60}")

        arrays, n = load_calibration_data(component, num_samples)
        print(f"  Samples: {n}")

        try:
            compile_job = hub.get_job(compile_job_id)
            compiled_model = compile_job.get_target_model()
            if compiled_model is None:
                print(f"  ERROR: compiled model not available")
                continue
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        feed = prepare_w8a16_inputs(component, arrays, n)
        try:
            inf_job = hub.submit_inference_job(
                model=compiled_model,
                inputs=feed,
                device=device,
            )
            all_jobs.append({
                "component": component,
                "compile_job_id": compile_job_id,
                "inference_job_id": inf_job.job_id,
                "num_samples": n,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"  Job submitted: {inf_job.job_id} ({n} samples)")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Merge with existing jobs (don't overwrite previous entries)
    existing_jobs = []
    if JOBS_FILE.exists():
        with open(JOBS_FILE) as f:
            existing_jobs = json.load(f)
    # Replace entries for resubmitted components, keep others
    existing_comps = {j["component"] for j in all_jobs}
    merged = [j for j in existing_jobs if j["component"] not in existing_comps] + all_jobs
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(JOBS_FILE, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n{len(all_jobs)} W8A16 inference jobs submitted (1 per component).")
    print(f"Saved to {JOBS_FILE}")
    print(f"Use --status to check, --report to generate report.")
    return all_jobs


# ============================================================
# Status check
# ============================================================

def check_status():
    """Check status of submitted W8A16 inference jobs."""
    import qai_hub as hub

    if not JOBS_FILE.exists():
        print("No jobs file found. Run --submit first.")
        return

    with open(JOBS_FILE) as f:
        jobs = json.load(f)

    groups = {}
    for job in jobs:
        comp = job["component"]
        if comp not in groups:
            groups[comp] = []
        groups[comp].append(job)

    print(f"Total W8A16 inference jobs: {len(jobs)}\n")

    for comp, group in groups.items():
        counts = {"SUCCESS": 0, "RUNNING": 0, "FAILED": 0}
        for j in group:
            try:
                status = str(hub.get_job(j["inference_job_id"]).get_status())
                if "SUCCESS" in status:
                    counts["SUCCESS"] += 1
                elif "FAIL" in status:
                    counts["FAILED"] += 1
                else:
                    counts["RUNNING"] += 1
            except Exception:
                counts["FAILED"] += 1

        total = len(group)
        print(f"  {comp:<16} {counts['SUCCESS']}/{total} done, "
              f"{counts['RUNNING']} running, {counts['FAILED']} failed")


# ============================================================
# Report: local FP32/W8A8 + downloaded W8A16
# ============================================================

def generate_report(components: list, num_samples: int):
    """Run FP32/W8A8 locally, download W8A16 from QAI Hub, compare all."""
    import qai_hub as hub

    # Load W8A16 job results (batched: 1 job per component, N samples)
    w8a16_outputs = {}  # (component, sample_idx) -> {output_name: np.ndarray}
    if JOBS_FILE.exists():
        with open(JOBS_FILE) as f:
            jobs = json.load(f)
        print("Downloading W8A16 inference results from QAI Hub...")
        for j in jobs:
            comp = j["component"]
            if comp not in components:
                continue
            try:
                job = hub.get_job(j["inference_job_id"])
                if "SUCCESS" not in str(job.get_status()):
                    print(f"  {comp} not ready, skipping")
                    continue
                result = job.download_output_data()
                # result is {output_name: [sample0, sample1, ...]}
                output_names = list(result.keys())
                # Use actual result length (may differ from recorded num_samples)
                n = len(result[output_names[0]]) if output_names else 0
                for i in range(n):
                    sample_outputs = {}
                    for name in output_names:
                        sample_outputs[name] = result[name][i]
                    w8a16_outputs[(comp, i)] = sample_outputs
                print(f"  {comp}: {n} samples downloaded")
            except Exception as e:
                print(f"  {comp} ERROR: {e}")
    else:
        print("No W8A16 jobs file. W8A16 results will be empty.")

    # Run local evaluations and collect all metrics
    # component -> variant -> [per-sample metrics]
    all_metrics = {}

    for component in components:
        info = COMPONENT_INFO[component]
        fp32_path = MODEL_DIR / info["fp32_onnx"]
        if not fp32_path.exists():
            print(f"\n{component}: FP32 ONNX not found, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {component}")
        print(f"{'=' * 60}")

        arrays, n = load_calibration_data(component, num_samples)
        print(f"  Samples: {n}")

        # FP32 baseline (local)
        print(f"  [FP32] local inference...")
        fp32_results = run_local_inference(fp32_path, arrays, info["input_keys"], n)

        comp_metrics = {}

        # Local quantized variants (W8A8, int8_qdq, mixed_pr, ...)
        for variant_key, onnx_filename in LOCAL_ONNX.get(component, {}).items():
            local_path = MODEL_DIR / onnx_filename
            if local_path.exists():
                print(f"  [{variant_key}] local inference...")
                try:
                    w8a8_results = run_local_inference(local_path, arrays, info["input_keys"], n)

                    sample_metrics = []
                    for i in range(n):
                        fp32_out = list(fp32_results[i].values())[0]
                        w8a8_out = list(w8a8_results[i].values())[0]
                        if fp32_out.dtype != w8a8_out.dtype:
                            fp32_out = fp32_out.astype(np.float32)
                            w8a8_out = w8a8_out.astype(np.float32)
                        m = compute_metrics(fp32_out, w8a8_out)
                        sample_metrics.append(m)
                        if i == 0:
                            print(f"    [1/{n}] cos={m['cosine_sim']:.6f} rmse={m['rmse']:.6f}")

                    comp_metrics[variant_key] = aggregate_metrics(sample_metrics)
                    del w8a8_results
                    gc.collect()
                except Exception as e:
                    print(f"    {variant_key} load failed: {e}")

        # W8A16 (from QAI Hub)
        if component in W8A16_COMPILE_JOBS:
            out_qp = UINT16_IO_QUANT.get(component, {}).get("output")
            fp32_postproc = FP32_POSTPROC_FOR_W8A16.get(component)
            w8a16_samples = []
            for i in range(n):
                if (component, i) not in w8a16_outputs:
                    continue
                w8a16_data = w8a16_outputs[(component, i)]
                fp32_out = list(fp32_results[i].values())[0]
                if fp32_postproc:
                    fp32_out = fp32_postproc(fp32_out)
                w8a16_out = list(w8a16_data.values())[0]

                # Handle uint16 quantized I/O models
                if out_qp:
                    # Dequantize uint16 if needed
                    if w8a16_out.dtype == np.uint16:
                        w8a16_out = (w8a16_out.astype(np.float64) + out_qp["offset"]) * out_qp["scale"]
                        w8a16_out = w8a16_out.astype(np.float32)
                    # NHWC → NCHW to match FP32 output layout
                    if out_qp.get("nhwc") and w8a16_out.ndim == 4:
                        w8a16_out = np.transpose(w8a16_out, (0, 3, 1, 2))

                if fp32_out.dtype != w8a16_out.dtype:
                    fp32_out = fp32_out.astype(np.float32)
                    w8a16_out = w8a16_out.astype(np.float32)
                m = compute_metrics(fp32_out, w8a16_out)
                w8a16_samples.append(m)
                if len(w8a16_samples) == 1:
                    print(f"  [W8A16] cos={m['cosine_sim']:.6f} rmse={m['rmse']:.6f}")

            if w8a16_samples:
                comp_metrics["w8a16"] = aggregate_metrics(w8a16_samples)
            else:
                print(f"  [W8A16] no results available")

        if comp_metrics:
            all_metrics[component] = comp_metrics

        del fp32_results, arrays
        gc.collect()

    # Build and print report
    report = build_report(all_metrics)
    print(report)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {REPORT_FILE}")


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
    if cos_val > 0.99:
        return "배포 가능"
    elif cos_val > 0.95:
        return "조건부 사용"
    else:
        return "사용 불가"


def build_report(all_metrics: dict) -> str:
    W = 110
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    def sep(ch="="):
        lines.append(ch * W)

    sep()
    lines.append("  SD v1.5 Quantization Quality Report (On-Device + Local)")
    lines.append(f"  {now}")
    lines.append(f"  Device: {DEVICE_NAME} (Snapdragon 8 Gen 2)")
    sep()

    lines.append("")
    lines.append("  평가 방법:")
    lines.append("    W8A8  — 로컬 ORT CPU (ONNX에 QDQ 노드 포함, 실제 양자화 시뮬레이션)")
    lines.append("    W8A16 — QAI Hub on-device NPU (ONNX에 QDQ 없음, QNN에서 .encodings 적용)")
    lines.append("  기준: FP32 로컬 ORT CPU 출력")
    lines.append("")

    # Quality summary
    sep("-")
    lines.append("  Quality Summary (FP32 vs Quantized)")
    sep("-")
    lines.append("")

    hdr = (f"  {'Component':<16} {'Variant':<22} "
           f"{'CosSim':>8} {'min':>8} {'RMSE':>8} {'PSNR':>8} {'MaxErr':>8} {'N':>3} {'Grade':>12}")
    lines.append(hdr)
    lines.append(f"  {'-' * (len(hdr) - 2)}")

    VARIANT_ORDER = list(VARIANT_LABELS.keys())  # fp32, w8a8, int8_qdq, mixed_pr, w8a16

    for comp in ALL_COMPONENTS:
        if comp not in all_metrics:
            continue
        first = True
        comp_variants = all_metrics[comp]
        for variant in VARIANT_ORDER:
            if variant not in comp_variants:
                continue
            m = comp_variants[variant]
            label = VARIANT_LABELS.get(variant, variant)
            grade = _grade(m["cos_mean"])
            psnr_str = f"{m['psnr_mean']:.1f}" if m['psnr_mean'] != float("inf") else "inf"
            comp_name = comp if first else ""
            lines.append(
                f"  {comp_name:<16} {label:<22} "
                f"{m['cos_mean']:>8.4f} {m['cos_min']:>8.4f} "
                f"{m['rmse_mean']:>8.4f} {psnr_str:>8} "
                f"{m['max_abs']:>8.4f} {m['num_samples']:>3} [{grade:>9}]"
            )
            first = False
        if not first:
            lines.append("")

    # Components without results
    missing = [c for c in ALL_COMPONENTS if c not in all_metrics]
    if missing:
        for c in missing:
            lines.append(f"  {c:<16} (양자화 모델 없음 또는 미평가)")
        lines.append("")

    # Conclusion
    sep("-")
    lines.append("  Conclusion")
    sep("-")
    lines.append("")

    for comp in ALL_COMPONENTS:
        if comp not in all_metrics:
            continue
        for variant in VARIANT_ORDER:
            if variant not in all_metrics[comp]:
                continue
            m = all_metrics[comp][variant]
            label = VARIANT_LABELS.get(variant, variant)
            verdict = _verdict(m["cos_mean"])
            grade = _grade(m["cos_mean"])
            lines.append(f"  {comp:<16} {label:<22} CosSim {m['cos_mean']:.4f} [{grade}] → {verdict}")

    lines.append("")
    sep()

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SD v1.5 quantization quality: W8A16 on-device + W8A8 local"
    )
    parser.add_argument("--submit", action="store_true",
                        help="Submit W8A16 inference jobs to QAI Hub")
    parser.add_argument("--components", nargs="+", default=ALL_COMPONENTS,
                        choices=ALL_COMPONENTS,
                        help="Components to evaluate (default: all)")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of test samples per component (default: 4)")
    parser.add_argument("--status", action="store_true",
                        help="Check W8A16 job status")
    parser.add_argument("--report", action="store_true",
                        help="Generate report (local FP32/W8A8 + downloaded W8A16)")

    args = parser.parse_args()

    if args.submit:
        submit_inference_jobs(args.components, args.num_samples)
    elif args.status:
        check_status()
    elif args.report:
        generate_report(args.components, args.num_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
