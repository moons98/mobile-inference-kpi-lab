#!/usr/bin/env python3
"""
Profile SD v1.5 Inpainting pipeline components on Qualcomm AI Hub.

Submits compile + profile jobs for VAE Encoder, Text Encoder, UNet, VAE Decoder
on real Snapdragon 8 Gen 2 hardware via cloud-hosted device.

Prerequisites:
    pip install qai-hub numpy
    qai-hub configure --api_token <your_token>

Usage:
    # List available devices
    python scripts/sd/profile_sd_qai_hub.py --list-devices

    # Profile all 4 components from local ONNX files
    python scripts/sd/profile_sd_qai_hub.py --profile-all --onnx-dir weights/sd_v1.5_inpaint/onnx

    # Profile individual components
    python scripts/sd/profile_sd_qai_hub.py --profile vae_encoder --onnx-dir weights/sd_v1.5_inpaint/onnx
    python scripts/sd/profile_sd_qai_hub.py --profile unet --onnx-dir weights/sd_v1.5_inpaint/onnx

    # Check job status and results
    python scripts/sd/profile_sd_qai_hub.py --check-jobs
    python scripts/sd/profile_sd_qai_hub.py --result <job_id>
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

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
RESULTS_FILE = PROJECT_ROOT / "outputs" / "sd_qai_hub_jobs.json"
MODEL_DIR = PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "onnx"

# Target device: Snapdragon 8 Gen 2 (SM8550) ??Galaxy S23 Ultra
TARGET_DEVICE_KEYWORDS = ["8 Gen 2", "SM8550", "S23"]

# SD v1.5 Inpainting component input specs
# UNet: 9ch input (latent 4 + mask 1 + masked_image latent 4)
# Text Encoder: CLIP ViT-L/14, 768-dim hidden states
SD_COMPONENTS = {
    "vae_encoder": {
        "input_specs": dict(sample=((1, 3, 512, 512), "float32")),
        "description": "VAE Encoder (image ??latent)",
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
        "description": "VAE Decoder (latent ??512횞512 image)",
    },
}

# ONNX file naming convention (INT8 QDQ preferred, then FP32)
ONNX_FILENAMES = {
    "vae_encoder": ["vae_encoder_int8_qdq.onnx", "vae_encoder_fp32.onnx"],
    "text_encoder": ["text_encoder_int8_qdq.onnx", "text_encoder_fp32.onnx"],
    "unet": ["unet_int8_qdq.onnx", "unet_fp32.onnx"],
    "vae_decoder": ["vae_decoder_int8_qdq.onnx", "vae_decoder_fp32.onnx"],
}


def find_device(keyword=None):
    """Find Snapdragon 8 Gen 2 device on AI Hub."""
    devices = hub.get_devices()

    if keyword:
        matches = [d for d in devices if keyword.lower() in d.name.lower()]
    else:
        matches = []
        for kw in TARGET_DEVICE_KEYWORDS:
            matches.extend(d for d in devices if kw.lower() in d.name.lower())
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
        print("No 8 Gen 2 devices found. Listing Snapdragon devices:")
        devices = hub.get_devices()
        for d in devices:
            if "snapdragon" in d.name.lower() or "galaxy" in d.name.lower():
                print(f"  {d.name}")


def find_onnx_file(onnx_dir: Path, component: str) -> Path | None:
    """Find ONNX file for a component in the given directory."""
    for filename in ONNX_FILENAMES[component]:
        path = onnx_dir / filename
        if path.exists():
            return path
    return None


def profile_component(component: str, device_name: str, model_source,
                       target_runtime: str = "precompiled_qnn_onnx"):
    """Submit compile + profile job for one SD component."""
    spec = SD_COMPONENTS[component]

    print(f"\n{'='*60}")
    print(f"Component: {component} ??{spec['description']}")
    print(f"Device: {device_name}")
    print(f"Runtime: {target_runtime}")
    print(f"Input specs: {list(spec['input_specs'].keys())}")
    print(f"{'='*60}")

    device = hub.Device(device_name)

    # Compile
    print(f"Compiling {component}...")
    compile_job = hub.submit_compile_job(
        model=model_source,
        device=device,
        input_specs=spec["input_specs"],
        options=f"--target_runtime {target_runtime}",
    )
    print(f"  Compile job: {compile_job.job_id}")
    print(f"  Waiting for compilation...")

    compiled_model = compile_job.get_target_model()
    if compiled_model is None:
        print(f"  ERROR: Compile failed for {component}. Check AI Hub for details.")
        return None

    print(f"  Compile complete.")

    # Profile
    print(f"  Submitting profile job...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
    )
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
    """Print combined pipeline summary from all 4 component results."""
    print(f"\n{'='*60}")
    print("SD v1.5 Inpainting Pipeline Summary (single UNet step)")
    print(f"{'='*60}")

    total_inference = sum(r.get("inference_time_ms", 0) for r in results.values())
    total_first_load = sum(r.get("first_load_ms", 0) for r in results.values())

    print(f"\n| {'Component':<16} | {'Inference':<12} | {'First Load':<12} | {'Warm Load':<12} |")
    print(f"|{'-'*18}|{'-'*14}|{'-'*14}|{'-'*14}|")
    for comp in ["vae_encoder", "text_encoder", "unet", "vae_decoder"]:
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


def main():
    parser = argparse.ArgumentParser(
        description="Profile SD v1.5 Inpainting components on Qualcomm AI Hub (Snapdragon 8 Gen 2)"
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List available 8 Gen 2 devices")
    parser.add_argument("--device", type=str, default=None,
                        help="Device name (from --list-devices)")
    parser.add_argument("--profile", type=str,
                        choices=["vae_encoder", "text_encoder", "unet", "vae_decoder"],
                        help="Profile a single component")
    parser.add_argument("--profile-all", action="store_true",
                        help="Profile all 4 components")
    parser.add_argument("--onnx-dir", type=str, default=None,
                        help="Directory with ONNX files (default: weights/sd_v1.5_inpaint/onnx)")
    parser.add_argument("--check-jobs", action="store_true",
                        help="Check status & results of saved jobs")
    parser.add_argument("--result", type=str,
                        help="Print results for a specific job ID")
    parser.add_argument("--summary", action="store_true",
                        help="Print pipeline summary from completed jobs")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.result:
        print_profile_results(args.result)
        return

    if args.check_jobs or args.summary:
        if not RESULTS_FILE.exists():
            print("No saved jobs found.")
            return
        with open(RESULTS_FILE) as f:
            jobs = json.load(f)

        # Filter to latest run for each component
        component_results = {}
        for job_info in jobs:
            job_id = job_info["profile_job_id"]
            comp = job_info.get("component", "unknown")
            try:
                job = hub.get_job(job_id)
                status = job.get_status()
                status_str = str(status)
                print(f"\n[{status_str}] {comp} / {job_info.get('runtime', '?')}")
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

    # --- Profile mode ---
    components_to_profile = []
    if args.profile_all:
        components_to_profile = list(SD_COMPONENTS.keys())
    elif args.profile:
        components_to_profile = [args.profile]
    else:
        parser.print_help()
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

    # Determine ONNX directory
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else MODEL_DIR

    # Profile each component
    jobs = []
    for component in components_to_profile:
        onnx_path = find_onnx_file(onnx_dir, component)
        if onnx_path:
            model_source = str(onnx_path)
            print(f"\nUsing ONNX file: {onnx_path}")
        else:
            print(f"\nERROR: No ONNX file found for {component} in {onnx_dir}")
            print(f"  Expected: {ONNX_FILENAMES[component]}")
            continue

        job = profile_component(component, device_name, model_source)
        if job:
            jobs.append(job)

    if jobs:
        save_job_info(jobs)
        print(f"\n{'='*60}")
        print(f"{len(jobs)} job(s) submitted.")
        print(f"Use --check-jobs to monitor progress.")
        print(f"Use --summary to see pipeline summary when all complete.")


if __name__ == "__main__":
    main()


