#!/usr/bin/env python3
"""
Profile Stable Diffusion v2.1 components on Qualcomm AI Hub.

Submits compile + profile jobs for TextEncoder, UNet, VAE Decoder
on real Snapdragon 8 Gen 2 hardware via cloud-hosted device.

Prerequisites:
    pip install qai-hub
    pip install qai-hub-models          # for --from-hub-models (auto export)
    qai-hub configure --api_token <your_token>

Usage:
    # List available devices
    python profile_sd_qai_hub.py --list-devices

    # Profile all 3 components using qai_hub_models (auto export + compile + profile)
    python profile_sd_qai_hub.py --profile-all

    # Profile individual components
    python profile_sd_qai_hub.py --profile text_encoder
    python profile_sd_qai_hub.py --profile unet
    python profile_sd_qai_hub.py --profile vae_decoder

    # Profile from local ONNX files
    python profile_sd_qai_hub.py --profile-all --onnx-dir ./sd_models/

    # Check job status and results
    python profile_sd_qai_hub.py --check-jobs
    python profile_sd_qai_hub.py --result <job_id>
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

RESULTS_FILE = Path(__file__).parent.parent / "outputs" / "sd_qai_hub_jobs.json"

# Target device: Snapdragon 8 Gen 2 (SM8550) — Galaxy S23 Ultra
TARGET_DEVICE_KEYWORDS = ["8 Gen 2", "SM8550", "S23"]

# SD v2.1 component input specs
SD_COMPONENTS = {
    "text_encoder": {
        "input_specs": dict(tokens=((1, 77), "int32")),
        "description": "CLIP ViT-L/14 Text Encoder",
    },
    "unet": {
        "input_specs": dict(
            latent=((1, 4, 64, 64), "float32"),
            timestep=((1, 1), "float32"),
            text_emb=((1, 77, 1024), "float32"),
        ),
        "description": "UNet2D Conditional (single step)",
    },
    "vae_decoder": {
        "input_specs": dict(latent=((1, 4, 64, 64), "float32")),
        "description": "VAE Decoder (latent → 512×512 image)",
    },
}

# ONNX file naming convention
ONNX_FILENAMES = {
    "text_encoder": ["text_encoder_quantized.onnx", "text_encoder.onnx"],
    "unet": ["unet_quantized.onnx", "unet.onnx"],
    "vae_decoder": ["vae_decoder_quantized.onnx", "vae_decoder.onnx"],
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


_sd_model_cache = None

def _get_sd_model():
    """Load SD v2.1 Quantized model (cached)."""
    global _sd_model_cache
    if _sd_model_cache is None:
        try:
            from qai_hub_models.models.stable_diffusion_v2_1.model import (
                StableDiffusionV2_1_Quantized,
            )
        except ImportError:
            print("Error: qai-hub-models not installed or missing dependencies")
            print("Install with: pip install qai-hub-models onnxsim")
            print("Or provide ONNX files with --onnx-dir")
            sys.exit(1)
        print("Loading SD v2.1 Quantized from HuggingFace (first time may download ~2GB)...")
        _sd_model_cache = StableDiffusionV2_1_Quantized.from_pretrained()
    return _sd_model_cache


def export_from_hub_models(component: str):
    """Export model from qai_hub_models package. Returns model object for compile."""
    sd = _get_sd_model()
    component_map = {
        "text_encoder": sd.text_encoder,
        "unet": sd.unet,
        "vae_decoder": sd.vae_decoder,
    }
    return component_map[component]


def profile_component(component: str, device_name: str, model_source,
                       target_runtime: str = "precompiled_qnn_onnx"):
    """Submit compile + profile job for one SD component."""
    spec = SD_COMPONENTS[component]

    print(f"\n{'='*60}")
    print(f"Component: {component} — {spec['description']}")
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
    """Print combined pipeline summary from all 3 component results."""
    print(f"\n{'='*60}")
    print("SD v2.1 Pipeline Summary (single UNet step)")
    print(f"{'='*60}")

    total_inference = sum(r.get("inference_time_ms", 0) for r in results.values())
    total_first_load = sum(r.get("first_load_ms", 0) for r in results.values())

    print(f"\n| {'Component':<16} | {'Inference':<12} | {'First Load':<12} | {'Warm Load':<12} |")
    print(f"|{'-'*18}|{'-'*14}|{'-'*14}|{'-'*14}|")
    for comp in ["text_encoder", "unet", "vae_decoder"]:
        if comp in results:
            r = results[comp]
            print(f"| {comp:<16} | {r['inference_time_ms']:>8.1f} ms | "
                  f"{r['first_load_ms']:>8.1f} ms | {r['warm_load_ms']:>8.1f} ms |")
    print(f"|{'-'*18}|{'-'*14}|{'-'*14}|{'-'*14}|")
    print(f"| {'TOTAL':<16} | {total_inference:>8.1f} ms | {total_first_load:>8.1f} ms |{' '*14}|")

    if "unet" in results:
        unet_ms = results["unet"]["inference_time_ms"]
        print(f"\nEstimated E2E (20 steps): "
              f"{total_inference - unet_ms + unet_ms * 20:.0f} ms "
              f"({(total_inference - unet_ms + unet_ms * 20)/1000:.1f} s)")
        print(f"Estimated E2E (50 steps): "
              f"{total_inference - unet_ms + unet_ms * 50:.0f} ms "
              f"({(total_inference - unet_ms + unet_ms * 50)/1000:.1f} s)")


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
        description="Profile SD v2.1 components on Qualcomm AI Hub (Snapdragon 8 Gen 2)"
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List available 8 Gen 2 devices")
    parser.add_argument("--device", type=str, default=None,
                        help="Device name (from --list-devices)")
    parser.add_argument("--profile", type=str, choices=["text_encoder", "unet", "vae_decoder"],
                        help="Profile a single component")
    parser.add_argument("--profile-all", action="store_true",
                        help="Profile all 3 components")
    parser.add_argument("--onnx-dir", type=str, default=None,
                        help="Directory with pre-exported ONNX files")
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
        components_to_profile = ["text_encoder", "unet", "vae_decoder"]
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

    # Profile each component
    jobs = []
    for component in components_to_profile:
        # Determine model source
        model_source = None

        if args.onnx_dir:
            onnx_path = find_onnx_file(Path(args.onnx_dir), component)
            if onnx_path:
                model_source = str(onnx_path)
                print(f"\nUsing ONNX file: {onnx_path}")
            else:
                print(f"\nWARNING: No ONNX file found for {component} in {args.onnx_dir}")
                print(f"  Expected: {ONNX_FILENAMES[component]}")
                continue

        if model_source is None:
            print(f"\nExporting {component} from qai_hub_models...")
            model_source = export_from_hub_models(component)

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
