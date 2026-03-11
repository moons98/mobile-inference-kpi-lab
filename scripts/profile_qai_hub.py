#!/usr/bin/env python3
"""
Profile & quantize ONNX models on Qualcomm AI Hub (Snapdragon 8 Gen 2).

Supports all AI Eraser pipeline components:
  - SD v1.5 Inpainting: vae_encoder, text_encoder, unet, vae_decoder
  - YOLO-seg: yolo_seg

Operations:
  --profile       Compile + profile (pre-quantized ONNX)
  --quantize      FP32 ONNX → QAI Hub INT8 W8A8 quantize → compile → profile
  --check-jobs    Monitor submitted jobs
  --download      Download compiled QNN context binaries
  --summary       Print pipeline latency summary

Prerequisites:
    pip install qai-hub numpy
    qai-hub configure --api_token <your_token>

Usage:
    # List available devices
    python scripts/profile_qai_hub.py --list-devices

    # Profile pre-quantized INT8 models
    python scripts/profile_qai_hub.py --profile-all
    python scripts/profile_qai_hub.py --profile yolo_seg --onnx-dir weights/yolov8n_seg/onnx

    # Quantize FP32 via QAI Hub + profile
    python scripts/profile_qai_hub.py --quantize yolo_seg --calib-samples 100
    python scripts/profile_qai_hub.py --quantize vae_encoder unet --calib-samples 50

    # Check job status and results
    python scripts/profile_qai_hub.py --check-jobs
    python scripts/profile_qai_hub.py --summary
    python scripts/profile_qai_hub.py --result <job_id>

    # Download compiled binaries
    python scripts/profile_qai_hub.py --download
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import qai_hub as hub
    import numpy as np
except ImportError:
    print("Error: qai-hub not installed")
    print("Install with: pip install qai-hub numpy")
    print("Configure with: qai-hub configure --api_token <your_token>")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_ROOT / "outputs" / "qai_hub_jobs.json"
COCO_VAL_DIR = PROJECT_ROOT / "datasets" / "coco" / "val2017"

# Model directories
MODEL_DIRS = {
    "vae_encoder":  PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "onnx",
    "text_encoder": PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "onnx",
    "unet":         PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "onnx",
    "vae_decoder":  PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "onnx",
    "yolo_seg":     PROJECT_ROOT / "weights" / "yolov8n_seg" / "onnx",
}

# Target device: Snapdragon 8 Gen 2 (SM8550)
TARGET_DEVICE_KEYWORDS = ["8 Gen 2", "SM8550", "S23"]

# Component input specs
COMPONENTS = {
    "vae_encoder": {
        "input_specs": dict(sample=((1, 3, 512, 512), "float32")),
        "description": "VAE Encoder (image → latent)",
    },
    "text_encoder": {
        "input_specs": dict(input_ids=((1, 77), "int64")),
        "description": "CLIP ViT-L/14 Text Encoder (768-dim)",
    },
    "unet": {
        "input_specs": dict(
            sample=((1, 9, 64, 64), "float32"),
            timestep=((1,), "int64"),
            encoder_hidden_states=((1, 77, 768), "float32"),
        ),
        "description": "Inpainting UNet2D (9ch, single step)",
    },
    "vae_decoder": {
        "input_specs": dict(latent_sample=((1, 4, 64, 64), "float32")),
        "description": "VAE Decoder (latent → 512×512 image)",
    },
    "yolo_seg": {
        "input_specs": dict(images=((1, 3, 640, 640), "float32")),
        "description": "YOLOv8n-seg instance segmentation",
    },
}

# ONNX file naming convention (INT8 QDQ preferred, then FP32)
ONNX_FILENAMES = {
    "vae_encoder":  ["vae_encoder_int8_qdq.onnx", "vae_encoder_fp32.onnx"],
    "text_encoder": ["text_encoder_fp32.onnx"],  # INT8 broken
    "unet":         ["unet_int8_qdq.onnx", "unet_fp32.onnx"],
    "vae_decoder":  ["vae_decoder_int8_qdq.onnx", "vae_decoder_fp32.onnx"],
    "yolo_seg":     ["yolov8n-seg_int8_qdq.onnx", "yolov8n-seg_fp32.onnx"],
}

SD_COMPONENTS = ["vae_encoder", "text_encoder", "unet", "vae_decoder"]
ALL_COMPONENT_NAMES = list(COMPONENTS.keys())


# ============================================================
# Device management
# ============================================================

def find_device(keyword=None):
    """Find Snapdragon 8 Gen 2 device on AI Hub."""
    devices = hub.get_devices()
    if keyword:
        return [d for d in devices if keyword.lower() in d.name.lower()]

    matches = []
    seen = set()
    for kw in TARGET_DEVICE_KEYWORDS:
        for d in devices:
            if kw.lower() in d.name.lower() and d.name not in seen:
                seen.add(d.name)
                matches.append(d)
    return matches


def list_devices():
    """List available devices matching 8 Gen 2."""
    print("Searching for Snapdragon 8 Gen 2 devices...\n")
    matches = find_device()
    if matches:
        for d in matches:
            print(f"  {d.name} (OS: {d.os})")
            attrs = [a for a in d.attributes if 'chipset' in a or 'hexagon' in a or 'framework' in a]
            for a in attrs:
                print(f"    {a}")
            print()
    else:
        print("No 8 Gen 2 devices found. Listing Snapdragon devices:")
        devices = hub.get_devices()
        for d in devices:
            if "snapdragon" in d.name.lower() or "galaxy" in d.name.lower():
                print(f"  {d.name}")


# ============================================================
# Model file resolution
# ============================================================

def find_onnx_file(component: str, precision: str = None, onnx_dir: Path = None) -> Path | None:
    """Find ONNX file for a component.
    If precision is specified ('fp32' or 'int8'), only look for that variant."""
    filenames = list(ONNX_FILENAMES[component])
    if precision == "fp32":
        filenames = [f for f in filenames if "int8" not in f]
    elif precision == "int8":
        filenames = [f for f in filenames if "int8" in f]

    search_dirs = []
    if onnx_dir:
        search_dirs.append(onnx_dir)
    search_dirs.append(MODEL_DIRS[component])

    for d in search_dirs:
        for filename in filenames:
            path = d / filename
            if path.exists():
                return path
    return None


def _prepare_model_source(component: str, onnx_path: Path):
    """Prepare model source for upload.
    For models with external data (.onnx.data), creates a temp directory."""
    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        import tempfile, shutil
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"qai_{component}_"))
        shutil.copy2(onnx_path, tmp_dir / onnx_path.name)
        shutil.copy2(data_file, tmp_dir / data_file.name)
        print(f"  External data model: staged in {tmp_dir}")
        print(f"    {onnx_path.name}: {onnx_path.stat().st_size/1024/1024:.1f} MB")
        print(f"    {data_file.name}: {data_file.stat().st_size/1024/1024:.1f} MB")
        return str(tmp_dir)
    return str(onnx_path)


def _compile_options(component: str, target_runtime: str) -> str:
    """Build compile options string."""
    opts = f"--target_runtime {target_runtime} --qairt_version 2.42"
    if component in ("text_encoder", "unet"):
        opts += " --truncate_64bit_io"
    return opts


# ============================================================
# Calibration data
# ============================================================

def _prepare_calibration_data(component: str, num_samples: int = 100) -> dict:
    """Prepare calibration data dict for QAI Hub quantize job.
    Returns dict of {input_name: [np.array, ...]}."""
    from PIL import Image

    spec = COMPONENTS[component]
    input_specs = spec["input_specs"]

    calib = {}
    for input_name, (shape, dtype_str) in input_specs.items():
        samples = []

        # Image-based calibration (COCO)
        if component == "yolo_seg" and input_name == "images":
            image_files = sorted(COCO_VAL_DIR.glob("*.jpg"))
            if not image_files:
                raise FileNotFoundError(f"No COCO images in {COCO_VAL_DIR}")
            rng = np.random.default_rng(42)
            indices = rng.choice(len(image_files), size=min(num_samples, len(image_files)),
                                 replace=False)
            for idx in indices:
                img = Image.open(image_files[idx]).convert("RGB")
                img = img.resize((shape[3], shape[2]), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)[np.newaxis]
                samples.append(arr)
            print(f"  Calibration: {len(samples)} COCO images [0,1] for {input_name}")

        elif component == "vae_encoder" and input_name == "sample":
            image_files = sorted(COCO_VAL_DIR.glob("*.jpg"))
            if not image_files:
                raise FileNotFoundError(f"No COCO images in {COCO_VAL_DIR}")
            rng = np.random.default_rng(42)
            indices = rng.choice(len(image_files), size=min(num_samples, len(image_files)),
                                 replace=False)
            for idx in indices:
                img = Image.open(image_files[idx]).convert("RGB")
                img = img.resize((shape[3], shape[2]), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
                arr = arr.transpose(2, 0, 1)[np.newaxis]
                samples.append(arr)
            print(f"  Calibration: {len(samples)} COCO images [-1,1] for {input_name}")

        else:
            # Fallback: random data (latency-accurate, quality-meaningless)
            rng = np.random.default_rng(42)
            np_dtype = np.float32 if dtype_str == "float32" else np.int64
            for _ in range(num_samples):
                if np_dtype == np.int64:
                    samples.append(rng.integers(0, 1000, size=shape).astype(np_dtype))
                else:
                    samples.append(rng.standard_normal(shape).astype(np_dtype))
            print(f"  Calibration: {len(samples)} random samples for {input_name}")

        calib[input_name] = samples
    return calib


# ============================================================
# Job submission
# ============================================================

def submit_compile_only(component: str, device_name: str, model_source,
                        target_runtime: str = "precompiled_qnn_onnx"):
    """Submit compile job without waiting."""
    spec = COMPONENTS[component]
    print(f"\n--- {component}: {spec['description']} ---")
    device = hub.Device(device_name)

    options = _compile_options(component, target_runtime)
    print(f"  Options: {options}")
    print(f"  Uploading & submitting compile job...")
    compile_job = hub.submit_compile_job(
        model=model_source,
        device=device,
        input_specs=spec["input_specs"],
        options=options,
    )
    print(f"  Compile job: {compile_job.job_id}")
    print(f"  URL: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")

    return {
        "component": component,
        "description": spec["description"],
        "device": device_name,
        "runtime": target_runtime,
        "compile_job_id": compile_job.job_id,
        "profile_job_id": "",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": str(model_source),
    }


def profile_component(component: str, device_name: str, model_source,
                       target_runtime: str = "precompiled_qnn_onnx"):
    """Submit compile + profile job for one component."""
    spec = COMPONENTS[component]

    print(f"\n{'='*60}")
    print(f"Component: {component} - {spec['description']}")
    print(f"Device: {device_name}")
    print(f"Runtime: {target_runtime}")
    print(f"{'='*60}")

    device = hub.Device(device_name)
    options = _compile_options(component, target_runtime)

    print(f"Compiling {component}... (options: {options})")
    compile_job = hub.submit_compile_job(
        model=model_source,
        device=device,
        input_specs=spec["input_specs"],
        options=options,
    )
    print(f"  Compile job: {compile_job.job_id}")
    print(f"  Waiting for compilation...")

    compiled_model = compile_job.get_target_model()
    if compiled_model is None:
        print(f"  ERROR: Compile failed for {component}.")
        return None

    print(f"  Compile complete.")
    print(f"  Submitting profile job...")
    profile_job = hub.submit_profile_job(model=compiled_model, device=device)
    print(f"  Profile job: {profile_job.job_id}")

    return {
        "component": component,
        "description": spec["description"],
        "device": device_name,
        "runtime": target_runtime,
        "compile_job_id": compile_job.job_id,
        "profile_job_id": profile_job.job_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def submit_quantize_and_profile(component: str, device_name: str, onnx_path: Path,
                                 num_calib_samples: int = 100,
                                 submit_only: bool = False):
    """Submit QAI Hub quantize job (FP32 → INT8 W8A8), then compile + profile."""
    spec = COMPONENTS[component]
    print(f"\n{'='*60}")
    print(f"QAI Hub Quantize: {component} - {spec['description']}")
    print(f"  FP32 model: {onnx_path.name}")
    print(f"  Device: {device_name}")
    print(f"{'='*60}")

    print("Preparing calibration data...")
    calib_data = _prepare_calibration_data(component, num_calib_samples)

    model_source = _prepare_model_source(component, onnx_path)

    print(f"Submitting quantize job (W8A8)...")
    quantize_job = hub.submit_quantize_job(
        model=model_source,
        calibration_data=calib_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    print(f"  Quantize job: {quantize_job.job_id}")
    print(f"  URL: https://workbench.aihub.qualcomm.com/jobs/{quantize_job.job_id}/")

    if submit_only:
        return {
            "component": component,
            "description": spec["description"] + " [QAI Hub INT8]",
            "device": device_name,
            "runtime": "precompiled_qnn_onnx",
            "quantize_job_id": quantize_job.job_id,
            "compile_job_id": "",
            "profile_job_id": "",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_file": str(onnx_path),
            "quantize_method": "qai_hub_w8a8",
            "calib_samples": num_calib_samples,
        }

    # Wait for quantized model
    print("  Waiting for quantization...")
    quantized_model = quantize_job.get_target_model()
    if quantized_model is None:
        print(f"  ERROR: Quantize failed.")
        return None
    print(f"  Quantization complete.")

    # Compile
    device = hub.Device(device_name)
    options = _compile_options(component, "precompiled_qnn_onnx")
    print(f"  Compiling quantized model... (options: {options})")
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        options=options,
    )
    print(f"  Compile job: {compile_job.job_id}")
    print(f"  Waiting for compilation...")

    compiled_model = compile_job.get_target_model()
    if compiled_model is None:
        print(f"  ERROR: Compile failed.")
        return None

    # Profile
    print(f"  Submitting profile job...")
    profile_job = hub.submit_profile_job(model=compiled_model, device=device)
    print(f"  Profile job: {profile_job.job_id}")

    return {
        "component": component,
        "description": spec["description"] + " [QAI Hub INT8]",
        "device": device_name,
        "runtime": "precompiled_qnn_onnx",
        "quantize_job_id": quantize_job.job_id,
        "compile_job_id": compile_job.job_id,
        "profile_job_id": profile_job.job_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": str(onnx_path),
        "quantize_method": "qai_hub_w8a8",
        "calib_samples": num_calib_samples,
    }


# ============================================================
# Results & status
# ============================================================

def print_profile_results(job_id):
    """Print profile results from a completed job."""
    job = hub.get_job(job_id)
    profile = job.download_profile()
    summary = profile["execution_summary"]

    print(f"\n{'='*60}")
    print(f"Profile Results: {job_id}")
    print(f"{'='*60}")

    print(f"| {'Stage':<21} | {'Peak':<18} | {'Increase':<18} | {'Time':<10} |")
    print(f"|{'-'*23}|{'-'*20}|{'-'*20}|{'-'*12}|")

    for label, key in [("Compilation", "compile"), ("First App Load", "first_load"),
                        ("Subsequent Load", "warm_load"), ("Inference", "inference")]:
        peak = summary.get(f"{key}_memory_peak_range", [0, 0])
        incr = summary.get(f"{key}_memory_increase_range", [0, 0])
        time_us = summary.get(f"{key}_time", 0)
        print(f"| {label:<21} | {peak[0]/1024**2:>6.1f}-{peak[1]/1024**2:>6.1f} MB | "
              f"{incr[0]/1024**2:>6.1f}-{incr[1]/1024**2:>6.1f} MB | {time_us/1000:>7.1f} ms |")

    compute_units = len(profile.get("execution_detail", []))
    est_time = summary.get("estimated_inference_time", 0)
    print(f"\nCompute Units: {compute_units}")
    print(f"Estimated Inference Time: {est_time/1000:.3f} ms")

    return {
        "inference_time_ms": est_time / 1000,
        "first_load_ms": summary.get("first_load_time", 0) / 1000,
        "warm_load_ms": summary.get("warm_load_time", 0) / 1000,
        "compile_time_ms": summary.get("compile_time", 0) / 1000,
        "compute_units": compute_units,
    }


def print_pipeline_summary(results):
    """Print combined pipeline summary."""
    print(f"\n{'='*60}")
    print("Pipeline Latency Summary")
    print(f"{'='*60}")

    total_inference = sum(r.get("inference_time_ms", 0) for r in results.values())
    total_first_load = sum(r.get("first_load_ms", 0) for r in results.values())

    print(f"\n| {'Component':<16} | {'Inference':<12} | {'First Load':<12} | {'Warm Load':<12} |")
    print(f"|{'-'*18}|{'-'*14}|{'-'*14}|{'-'*14}|")

    for comp in ALL_COMPONENT_NAMES:
        if comp in results:
            r = results[comp]
            print(f"| {comp:<16} | {r['inference_time_ms']:>8.1f} ms | "
                  f"{r['first_load_ms']:>8.1f} ms | {r['warm_load_ms']:>8.1f} ms |")
    print(f"|{'-'*18}|{'-'*14}|{'-'*14}|{'-'*14}|")
    print(f"| {'TOTAL':<16} | {total_inference:>8.1f} ms | {total_first_load:>8.1f} ms |{' '*14}|")

    if "unet" in results:
        unet_ms = results["unet"]["inference_time_ms"]
        non_unet = total_inference - unet_ms
        print(f"\nEstimated E2E (20 steps): "
              f"{non_unet + unet_ms * 20:.0f} ms "
              f"({(non_unet + unet_ms * 20)/1000:.1f} s)")
        print(f"Estimated E2E (50 steps): "
              f"{non_unet + unet_ms * 50:.0f} ms "
              f"({(non_unet + unet_ms * 50)/1000:.1f} s)")


def save_job_info(jobs):
    """Save job IDs for later retrieval."""
    existing = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)

    existing.extend(jobs)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nJob info saved to {RESULTS_FILE}")


def download_compiled_models(component_filter: str, download_dir: str = None):
    """Download compiled QNN context binaries from completed compile jobs."""
    if not RESULTS_FILE.exists():
        print("No saved jobs found.")
        return

    with open(RESULTS_FILE) as f:
        jobs = json.load(f)

    out_dir = Path(download_dir) if download_dir else PROJECT_ROOT / "weights" / "qnn_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_jobs = {}
    for job_info in jobs:
        comp = job_info.get("component", "unknown")
        if component_filter != "all" and comp != component_filter:
            continue
        latest_jobs[comp] = job_info

    if not latest_jobs:
        print(f"No jobs found for '{component_filter}'")
        return

    print(f"Download directory: {out_dir}")
    print(f"Components: {list(latest_jobs.keys())}\n")

    for comp, job_info in latest_jobs.items():
        compile_job_id = job_info.get("compile_job_id")
        if not compile_job_id:
            print(f"[SKIP] {comp}: no compile job ID")
            continue

        print(f"[{comp}] Fetching compile job {compile_job_id}...")
        try:
            compile_job = hub.get_job(compile_job_id)
            status = str(compile_job.get_status())
            if "SUCCESS" not in status:
                print(f"  Status: {status} — not ready")
                continue

            compiled_model = compile_job.get_target_model()
            if compiled_model is None:
                print(f"  ERROR: compiled model not available")
                continue

            out_path = out_dir / f"qnn_{comp}.bin"
            compiled_model.download(str(out_path))
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {out_path.name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Files in: {out_dir}")


# ============================================================
# Main
# ============================================================

def _resolve_device(args):
    """Resolve device name from args or auto-detect."""
    if args.device:
        return args.device
    matches = find_device()
    if not matches:
        print("No 8 Gen 2 device found. Use --list-devices and --device <name>")
        return None
    print(f"Using device: {matches[0].name}")
    return matches[0].name


def main():
    parser = argparse.ArgumentParser(
        description="Profile & quantize ONNX models on Qualcomm AI Hub (Snapdragon 8 Gen 2)"
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List available 8 Gen 2 devices")
    parser.add_argument("--device", type=str, default=None,
                        help="Device name (from --list-devices)")

    # Profile mode
    parser.add_argument("--profile", type=str, nargs="+",
                        choices=ALL_COMPONENT_NAMES,
                        help="Profile specific components")
    parser.add_argument("--profile-all", action="store_true",
                        help="Profile all components")
    parser.add_argument("--precision", type=str, choices=["fp32", "int8"],
                        default=None,
                        help="Force FP32 or INT8 model (default: prefer INT8)")
    parser.add_argument("--onnx-dir", type=str, default=None,
                        help="Override ONNX directory for model lookup")

    # Quantize mode
    parser.add_argument("--quantize", type=str, nargs="+",
                        choices=ALL_COMPONENT_NAMES,
                        help="Quantize FP32 ONNX via QAI Hub (W8A8) with calibration, "
                             "then compile + profile")
    parser.add_argument("--calib-samples", type=int, default=100,
                        help="Number of calibration samples (default: 100)")

    # Common
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit jobs without waiting for completion")

    # Status & results
    parser.add_argument("--check-jobs", action="store_true",
                        help="Check status & results of saved jobs")
    parser.add_argument("--result", type=str,
                        help="Print results for a specific job ID")
    parser.add_argument("--summary", action="store_true",
                        help="Print pipeline summary from completed jobs")

    # Download
    parser.add_argument("--download", type=str, nargs="?", const="all",
                        help="Download compiled QNN context binaries")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory to save downloaded binaries")

    args = parser.parse_args()
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else None

    # --- List devices ---
    if args.list_devices:
        list_devices()
        return

    # --- Single result ---
    if args.result:
        print_profile_results(args.result)
        return

    # --- Check jobs / summary ---
    if args.check_jobs or args.summary:
        if not RESULTS_FILE.exists():
            print("No saved jobs found.")
            return
        with open(RESULTS_FILE) as f:
            jobs = json.load(f)

        component_results = {}
        for job_info in jobs:
            job_id = job_info.get("profile_job_id", "")
            if not job_id:
                # Quantize submit-only: show quantize job status
                qid = job_info.get("quantize_job_id", "")
                comp = job_info.get("component", "unknown")
                if qid:
                    try:
                        qjob = hub.get_job(qid)
                        print(f"\n[{qjob.get_status()}] {comp} (quantize) {qid}")
                    except Exception as e:
                        print(f"\n[ERROR] {comp}: {e}")
                continue

            comp = job_info.get("component", "unknown")
            try:
                job = hub.get_job(job_id)
                status_str = str(job.get_status())
                desc = job_info.get("description", comp)
                print(f"\n[{status_str}] {desc}")
                print(f"  Profile job: {job_id}")
                print(f"  Submitted: {job_info.get('timestamp', '?')}")

                if "SUCCESS" in status_str:
                    result = print_profile_results(job_id)
                    component_results[comp] = result
            except Exception as e:
                print(f"\n[ERROR] {comp}: {e}")

        if args.summary and len(component_results) >= 2:
            print_pipeline_summary(component_results)
        return

    # --- Download ---
    if args.download:
        download_compiled_models(args.download, args.download_dir)
        return

    # --- Quantize mode ---
    if args.quantize:
        device_name = _resolve_device(args)
        if not device_name:
            return

        jobs = []
        for component in args.quantize:
            fp32_path = find_onnx_file(component, precision="fp32", onnx_dir=onnx_dir)
            if not fp32_path:
                print(f"\nERROR: No FP32 ONNX file for {component}")
                continue
            print(f"\nUsing FP32 model: {fp32_path}")
            job = submit_quantize_and_profile(
                component, device_name, fp32_path,
                num_calib_samples=args.calib_samples,
                submit_only=args.submit_only,
            )
            if job:
                jobs.append(job)

        if jobs:
            save_job_info(jobs)
            print(f"\n{'='*60}")
            print(f"{len(jobs)} quantize job(s) submitted.")
            print(f"Use --check-jobs to monitor progress.")
        return

    # --- Profile mode ---
    components_to_profile = []
    if args.profile_all:
        components_to_profile = list(ALL_COMPONENT_NAMES)
    elif args.profile:
        components_to_profile = args.profile
    else:
        parser.print_help()
        return

    device_name = _resolve_device(args)
    if not device_name:
        return

    jobs = []
    for component in components_to_profile:
        onnx_path = find_onnx_file(component, precision=args.precision, onnx_dir=onnx_dir)
        if onnx_path:
            model_source = _prepare_model_source(component, onnx_path)
            print(f"\nUsing ONNX file: {onnx_path}")
        else:
            print(f"\nERROR: No ONNX file found for {component}")
            print(f"  Expected: {ONNX_FILENAMES[component]}")
            continue

        if args.submit_only:
            job = submit_compile_only(component, device_name, model_source)
        else:
            job = profile_component(component, device_name, model_source)
        if job:
            jobs.append(job)

    if jobs:
        save_job_info(jobs)
        print(f"\n{'='*60}")
        print(f"{len(jobs)} job(s) submitted.")
        if args.submit_only:
            print(f"Compile jobs submitted (no wait). Use --check-jobs to monitor.")
        else:
            print(f"Use --check-jobs to monitor progress.")
        print(f"Use --summary to see pipeline summary when all complete.")


if __name__ == "__main__":
    main()
