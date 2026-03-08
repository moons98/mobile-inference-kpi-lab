#!/usr/bin/env python3
"""
Profile YOLOv8n models on Qualcomm AI Hub.

Compares our ONNX models on real Snapdragon 8 Gen 2 hardware via cloud-hosted device.
This gives vendor-native QNN profiling numbers to compare against our ORT QNN EP results.

Usage:
    python profile_qai_hub.py --list-devices          # Find available 8 Gen 2 devices
    python profile_qai_hub.py --profile-fp32           # Profile FP32 model (QNN FP16 compile)
    python profile_qai_hub.py --profile-int8           # Profile INT8 QDQ model (QNN INT8 compile)
    python profile_qai_hub.py --profile-all            # Profile all variants
    python profile_qai_hub.py --check-jobs             # Check status of submitted jobs
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import qai_hub as hub
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: qai-hub not installed")
    print("Install with: pip install qai-hub")
    print("Configure with: qai-hub configure --api_token <your_token>")
    sys.exit(1)

ASSETS_DIR = Path(__file__).parent.parent / "android" / "app" / "src" / "main" / "assets"
CALIBRATION_DIR = Path(__file__).parent / "calibration_data" / "coco"
RESULTS_FILE = Path(__file__).parent.parent / "outputs" / "qai_hub_jobs.json"

# Target device: Snapdragon 8 Gen 2 (SM8550) - same as Galaxy S23 Ultra
# Use --list-devices to find exact name
TARGET_DEVICE_KEYWORDS = ["8 Gen 2", "SM8550", "S23"]

INPUT_SHAPE = (1, 3, 640, 640)
INPUT_SPEC = dict(images=INPUT_SHAPE)


def load_calibration_images(num_samples=10):
    """Load COCO calibration images with YOLOv8 letterbox preprocessing."""
    _, _, height, width = INPUT_SHAPE

    image_files = list(CALIBRATION_DIR.glob("*.JPEG")) + \
                  list(CALIBRATION_DIR.glob("*.jpg")) + \
                  list(CALIBRATION_DIR.glob("*.png"))
    image_files = image_files[:num_samples]

    if not image_files:
        print(f"WARNING: No calibration images in {CALIBRATION_DIR}, falling back to random data")
        return [np.random.randn(*INPUT_SHAPE).astype(np.float32) for _ in range(num_samples)]

    images = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")

        # Letterbox resize (preserve aspect ratio + gray padding)
        scale = min(width / img.width, height / img.height)
        new_w, new_h = int(img.width * scale), int(img.height * scale)
        img_resized = img.resize((new_w, new_h))
        canvas = Image.new("RGB", (width, height), (114, 114, 114))
        paste_x = (width - new_w) // 2
        paste_y = (height - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))

        # HWC → NCHW, [0,1] range
        img_array = np.array(canvas).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        images.append(img_array)

    print(f"Loaded {len(images)} calibration images from {CALIBRATION_DIR}")
    return images


def find_device(keyword=None):
    """Find Snapdragon 8 Gen 2 device on AI Hub."""
    devices = hub.get_devices()

    if keyword:
        matches = [d for d in devices if keyword.lower() in d.name.lower()]
    else:
        matches = []
        for kw in TARGET_DEVICE_KEYWORDS:
            matches.extend(d for d in devices if kw.lower() in d.name.lower())
        # Deduplicate
        seen = set()
        unique = []
        for d in matches:
            if d.name not in seen:
                seen.add(d.name)
                unique.append(d)
        matches = unique

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
        print("No 8 Gen 2 devices found. Listing all devices:")
        devices = hub.get_devices()
        for d in devices:
            if "snapdragon" in d.name.lower() or "galaxy" in d.name.lower():
                print(f"  {d.name}")


def profile_model(model_path: Path, device_name: str, target_runtime: str = "qnn_context_binary",
                   quantize_on_hub: bool = False, compile_options: str = ""):
    """Submit compile + profile job to AI Hub."""
    print(f"\n{'='*60}")
    print(f"Model: {model_path.name}")
    print(f"Device: {device_name}")
    print(f"Runtime: {target_runtime}")
    print(f"{'='*60}")

    device = hub.Device(device_name)

    if quantize_on_hub:
        # Upload FP32 model, let AI Hub quantize with its own pipeline
        print("Step 1: Compile FP32 → optimized ONNX...")
        compile_onnx_job = hub.submit_compile_job(
            model=str(model_path),
            device=device,
            input_specs=INPUT_SPEC,
            options="--target_runtime onnx",
        )
        onnx_model = compile_onnx_job.get_target_model()

        print("Step 2: Quantize W8A8 on AI Hub...")
        calibration_data = dict(
            images=load_calibration_images(num_samples=10)
        )
        quantize_job = hub.submit_quantize_job(
            model=onnx_model,
            calibration_data=calibration_data,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT8,
        )
        source_model = quantize_job.get_target_model()
    else:
        source_model = str(model_path)

    # Compile to target runtime
    print(f"Compiling to {target_runtime}...")
    options = f"--target_runtime {target_runtime}"
    if compile_options:
        options += f" {compile_options}"

    compile_job = hub.submit_compile_job(
        model=source_model,
        device=device,
        input_specs=INPUT_SPEC,
        options=options,
    )
    compiled_model = compile_job.get_target_model()
    if compiled_model is None:
        print("ERROR: Compile job failed. Check the link above for details.")
        return None

    # Profile
    print("Profiling on device...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
    )

    return {
        "model": model_path.name,
        "device": device_name,
        "runtime": target_runtime,
        "quantize_on_hub": quantize_on_hub,
        "compile_job_id": compile_job.job_id,
        "profile_job_id": profile_job.job_id,
    }


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

    stages = [
        ("Compilation", "compile"),
        ("First App Load", "first_load"),
        ("Subsequent Load", "warm_load"),
        ("Inference", "inference"),
    ]
    for label, key in stages:
        peak = summary.get(f"{key}_memory_peak_range", [0, 0])
        incr = summary.get(f"{key}_memory_increase_range", [0, 0])
        time_us = summary.get(f"{key}_time", 0)
        print(f"| {label:<21} | {peak[0]/1024**2:>6.1f}-{peak[1]/1024**2:>6.1f} MB | "
              f"{incr[0]/1024**2:>6.1f}-{incr[1]/1024**2:>6.1f} MB | {time_us/1000:>7.1f} ms |")

    compute_units = len(profile.get("execution_detail", []))
    print(f"\nCompute Units: {compute_units}")
    print(f"Estimated Inference Time: {summary.get('estimated_inference_time', 0)/1000:.3f} ms")


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


def main():
    parser = argparse.ArgumentParser(description="Profile YOLOv8n on Qualcomm AI Hub")
    parser.add_argument("--list-devices", action="store_true", help="List available 8 Gen 2 devices")
    parser.add_argument("--device", type=str, default=None, help="Device name (from --list-devices)")
    parser.add_argument("--profile-fp32", action="store_true", help="Profile FP32 model (QNN FP16)")
    parser.add_argument("--profile-int8", action="store_true", help="Profile INT8 QDQ model")
    parser.add_argument("--profile-int8-hub", action="store_true", help="Profile with AI Hub quantization (W8A8)")
    parser.add_argument("--profile-all", action="store_true", help="Profile all variants")
    parser.add_argument("--no-quantize-io", action="store_true",
                        help="Disable --quantize_io for INT8 compile (FP32 I/O, matches ORT QDQ behavior)")
    parser.add_argument("--check-jobs", action="store_true", help="Check status & results of saved jobs")
    parser.add_argument("--result", type=str, help="Print results for a specific job ID")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.result:
        print_profile_results(args.result)
        return

    if args.check_jobs:
        if not RESULTS_FILE.exists():
            print("No saved jobs found.")
            return
        with open(RESULTS_FILE) as f:
            jobs = json.load(f)
        for job_info in jobs:
            job_id = job_info["profile_job_id"]
            try:
                job = hub.get_job(job_id)
                status = job.get_status()
                print(f"\n[{status}] {job_info['model']} / {job_info['runtime']}")
                print(f"  Profile job: {job_id}")
                if str(status) == "JobStatus.SUCCESS" or "SUCCESS" in str(status):
                    print_profile_results(job_id)
            except Exception as e:
                print(f"\n[ERROR] {job_info['model']}: {e}")
        return

    # Determine device
    device_name = args.device
    if not device_name:
        matches = find_device()
        if not matches:
            print("No 8 Gen 2 device found. Use --list-devices and --device <name>")
            return
        device_name = matches[0].name
        print(f"Using device: {device_name}")

    jobs = []

    if args.profile_fp32 or args.profile_all:
        fp32_model = ASSETS_DIR / "yolov8n_clean.onnx"
        if not fp32_model.exists():
            fp32_model = ASSETS_DIR / "yolov8n.onnx"
        if fp32_model.exists():
            job = profile_model(fp32_model, device_name, "qnn_context_binary")
            if job:
                jobs.append(job)
        else:
            print(f"Model not found: {fp32_model}")

    if args.profile_int8 or args.profile_all:
        int8_model = ASSETS_DIR / "yolov8n_int8_qdq_clean.onnx"
        if not int8_model.exists():
            int8_model = ASSETS_DIR / "yolov8n_int8_qdq.onnx"
        if int8_model.exists():
            int8_compile_opts = "" if args.no_quantize_io else "--quantize_io"
            job = profile_model(int8_model, device_name, "qnn_context_binary", compile_options=int8_compile_opts)
            if job:
                jobs.append(job)
        else:
            print(f"Model not found: {int8_model}")

    if args.profile_int8_hub or args.profile_all:
        fp32_model = ASSETS_DIR / "yolov8n_clean.onnx"
        if not fp32_model.exists():
            fp32_model = ASSETS_DIR / "yolov8n.onnx"
        if fp32_model.exists():
            hub_compile_opts = "" if args.no_quantize_io else "--quantize_io"
            job = profile_model(fp32_model, device_name, "qnn_context_binary",
                                quantize_on_hub=True, compile_options=hub_compile_opts)
            if job:
                jobs.append(job)
        else:
            print(f"Model not found: {fp32_model}")

    if jobs:
        save_job_info(jobs)
        print("\nJobs submitted. Use --check-jobs to monitor progress.")


if __name__ == "__main__":
    main()
