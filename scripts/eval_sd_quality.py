#!/usr/bin/env python3
"""
Evaluate Stable Diffusion v2.1 quantization quality degradation.

Compares generated image quality between precision variants:
- FP32 (PyTorch baseline on GPU - ground truth)
- INT8 QDQ (ONNX, static quantized for QNN EP / NPU)

Metrics:
- PSNR (Peak Signal-to-Noise Ratio): pixel-level fidelity (higher = better)
- SSIM (Structural Similarity Index): structural similarity (higher = better)
- LPIPS (Learned Perceptual Image Patch Similarity): perceptual quality (lower = better)
- MSE (Mean Squared Error): raw pixel difference

Methodology:
1. Generate N images with PyTorch FP16 pipeline on GPU (ground truth)
2. Generate same images with ONNX INT8 QDQ pipeline (same seed, scheduler, steps)
3. Compute per-image and aggregate quality metrics
4. Output comparison table + per-prompt breakdown

Usage:
    python eval_sd_quality.py --compare int8
    python eval_sd_quality.py --compare int8 --steps 20 --num-images 10
    python eval_sd_quality.py --status
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).parent
MODEL_DIR = SCRIPTS_DIR.parent / "weights" / "sd_v2.1" / "onnx"
OUTPUTS_DIR = SCRIPTS_DIR.parent / "outputs" / "sd_quality"

SD_MODEL_ID = "sd2-community/stable-diffusion-2-1"

# Preset prompts from experiment design (Section 0.4)
DEFAULT_PROMPTS = [
    "sunset beach with golden light",
    "snowy winter cityscape at night",
    "lush green forest with sunlight",
    "futuristic neon city",
    "soft watercolor painting style",
]

DEFAULT_SEED = 42
DEFAULT_STEPS = 20
DEFAULT_RESOLUTION = 512
DEFAULT_NUM_IMAGES = 5  # 1 per preset prompt


# ============================================================
# Image Quality Metrics
# ============================================================

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images (uint8, 0-255)."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute MSE between two images."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images. Requires scikit-image."""
    try:
        from skimage.metrics import structural_similarity
        # Convert to grayscale for SSIM or use multichannel
        return structural_similarity(
            img1, img2,
            channel_axis=2,  # HWC format
            data_range=255,
        )
    except ImportError:
        print("  Warning: scikit-image not installed, SSIM unavailable")
        return float("nan")


def compute_lpips(img1: np.ndarray, img2: np.ndarray, lpips_model=None) -> float:
    """Compute LPIPS between two images. Requires lpips + torch."""
    if lpips_model is None:
        return float("nan")
    try:
        import torch

        def to_tensor(img):
            # uint8 HWC → float32 CHW, normalize to [-1, 1]
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0
            return t.unsqueeze(0)

        with torch.no_grad():
            score = lpips_model(to_tensor(img1), to_tensor(img2))
        return float(score.item())
    except Exception as e:
        print(f"  Warning: LPIPS computation failed: {e}")
        return float("nan")


def load_lpips_model():
    """Load LPIPS model (VGG-based). Returns None if unavailable."""
    try:
        import lpips
        model = lpips.LPIPS(net="vgg", verbose=False)
        model.eval()
        return model
    except ImportError:
        print("  Note: lpips not installed. Skipping perceptual metric.")
        print("  Install: pip install lpips")
        return None


# ============================================================
# Image Generation - PyTorch Baseline (FP32)
# ============================================================

def generate_pytorch_baseline(
    prompts: list,
    seed: int,
    steps: int,
    resolution: int,
    output_dir: Path,
) -> list:
    """Generate baseline images using PyTorch pipeline on GPU (FP16) or CPU (FP32).

    Returns list of (prompt, image_path, generation_time_s) tuples.
    """
    import torch
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    from PIL import Image

    print("\n" + "=" * 60)
    print("Generating PyTorch Baseline Images")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"  Device: {device}, dtype: {dtype}")
    print(f"  Steps: {steps}, Resolution: {resolution}x{resolution}")

    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\"")
        generator = torch.Generator(device=device).manual_seed(seed + i)

        t_start = time.perf_counter()
        image = pipe(
            prompt,
            num_inference_steps=steps,
            height=resolution,
            width=resolution,
            generator=generator,
        ).images[0]
        t_elapsed = time.perf_counter() - t_start

        img_path = output_dir / f"baseline_fp32_{i:02d}.png"
        image.save(img_path)
        results.append((prompt, img_path, t_elapsed))
        print(f"    Time: {t_elapsed:.1f}s - Saved: {img_path.name}")

    del pipe
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ============================================================
# Image Generation - ONNX Runtime (INT8 QDQ)
# ============================================================

def generate_onnx_images(
    prompts: list,
    seed: int,
    steps: int,
    resolution: int,
    precision: str,
    output_dir: Path,
) -> list:
    """Generate images using ONNX Runtime pipeline.

    Implements the SD pipeline manually:
    1. Tokenize prompt
    2. Text Encoder (ONNX session)
    3. UNet denoising loop (ONNX session x N steps)
    4. VAE Decode (ONNX session)

    Returns list of (prompt, image_path, generation_time_s) tuples.
    """
    import onnxruntime as ort
    import torch
    from diffusers import EulerDiscreteScheduler
    from transformers import CLIPTokenizer
    from PIL import Image

    print(f"\n{'=' * 60}")
    print(f"Generating ONNX {precision.upper()} Images")
    print(f"{'=' * 60}")

    # Load ONNX sessions
    # INT8 QDQ files use _int8_qdq suffix, FP32 uses _fp32
    suffix = "int8_qdq" if precision == "int8" else precision
    text_enc_path = MODEL_DIR / f"text_encoder_{suffix}.onnx"
    unet_path = MODEL_DIR / f"unet_{suffix}.onnx"
    vae_dec_path = MODEL_DIR / f"vae_decoder_{suffix}.onnx"

    for p in [text_enc_path, unet_path, vae_dec_path]:
        if not p.exists():
            print(f"  Error: {p.name} not found. Run export_sd_to_onnx.py first.")
            return []

    print(f"  Loading ONNX sessions...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    print(f"  Providers: {providers}")

    text_enc_sess = ort.InferenceSession(str(text_enc_path), sess_opts, providers=providers)
    unet_sess = ort.InferenceSession(str(unet_path), sess_opts, providers=providers)
    vae_dec_sess = ort.InferenceSession(str(vae_dec_path), sess_opts, providers=providers)

    # Load tokenizer and scheduler
    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    scheduler = EulerDiscreteScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")

    latent_h = resolution // 8
    latent_w = resolution // 8

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\"")
        t_start = time.perf_counter()

        # 1. Tokenize
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        input_ids = text_input.input_ids.astype(np.int64)

        # Unconditional input for classifier-free guidance
        uncond_input = tokenizer(
            "",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        uncond_ids = uncond_input.input_ids.astype(np.int64)

        # 2. Text Encoder
        text_embeddings = text_enc_sess.run(None, {"input_ids": input_ids})[0]
        uncond_embeddings = text_enc_sess.run(None, {"input_ids": uncond_ids})[0]
        text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # 3. Prepare latents
        rng = np.random.default_rng(seed + i)
        latents = rng.standard_normal((1, 4, latent_h, latent_w)).astype(np.float32)

        scheduler.set_timesteps(steps)
        latents = latents * float(scheduler.init_noise_sigma)

        # 4. UNet denoising loop
        guidance_scale = 7.5
        for t_idx, t in enumerate(scheduler.timesteps):
            latent_model_input = np.concatenate([latents, latents])

            # Scale model input
            timestep_val = np.array([float(t)], dtype=np.float32)

            noise_pred = unet_sess.run(None, {
                "sample": latent_model_input.astype(np.float32),
                "timestep": timestep_val,
                "encoder_hidden_states": text_embeddings.astype(np.float32),
            })[0]

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            scheduler_output = scheduler.step(
                torch.from_numpy(noise_pred),
                t,
                torch.from_numpy(latents),
            )
            latents = scheduler_output.prev_sample.numpy()

        # 5. VAE Decode
        decoded = vae_dec_sess.run(None, {
            "latent_sample": latents.astype(np.float32),
        })[0]

        t_elapsed = time.perf_counter() - t_start

        # Post-process: [-1, 1] → [0, 255] uint8
        image = ((decoded[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        img_path = output_dir / f"onnx_{precision}_{i:02d}.png"
        Image.fromarray(image).save(img_path)
        results.append((prompt, img_path, t_elapsed))
        print(f"    Time: {t_elapsed:.1f}s - Saved: {img_path.name}")

    return results


# ============================================================
# Quality Comparison
# ============================================================

def compare_quality(
    baseline_results: list,
    variant_results: list,
    precision_label: str,
    lpips_model=None,
) -> dict:
    """Compare baseline vs variant images using quality metrics.

    Returns dict with per-image and aggregate metrics.
    """
    from PIL import Image

    print(f"\n{'=' * 60}")
    print(f"Quality Comparison: FP32 (baseline) vs {precision_label}")
    print(f"{'=' * 60}")

    per_image = []

    for (prompt, base_path, base_time), (_, var_path, var_time) in zip(
        baseline_results, variant_results
    ):
        img_base = np.array(Image.open(base_path))
        img_var = np.array(Image.open(var_path))

        # Ensure same shape
        if img_base.shape != img_var.shape:
            print(f"  Warning: Shape mismatch {img_base.shape} vs {img_var.shape}")
            continue

        psnr = compute_psnr(img_base, img_var)
        ssim = compute_ssim(img_base, img_var)
        mse = compute_mse(img_base, img_var)
        lpips_score = compute_lpips(img_base, img_var, lpips_model)

        record = {
            "prompt": prompt,
            "psnr": psnr,
            "ssim": ssim,
            "mse": mse,
            "lpips": lpips_score,
            "baseline_time_s": base_time,
            "variant_time_s": var_time,
        }
        per_image.append(record)

        print(f"\n  Prompt: \"{prompt}\"")
        print(f"    PSNR:  {psnr:.2f} dB")
        print(f"    SSIM:  {ssim:.4f}")
        print(f"    MSE:   {mse:.2f}")
        if not np.isnan(lpips_score):
            print(f"    LPIPS: {lpips_score:.4f}")
        print(f"    Time:  {base_time:.1f}s (FP32) → {var_time:.1f}s ({precision_label})")

    # Aggregate
    if not per_image:
        return {}

    agg = {
        "precision": precision_label,
        "num_images": len(per_image),
        "psnr_mean": np.mean([r["psnr"] for r in per_image if not np.isinf(r["psnr"])]),
        "psnr_min": np.min([r["psnr"] for r in per_image if not np.isinf(r["psnr"])]),
        "ssim_mean": np.nanmean([r["ssim"] for r in per_image]),
        "ssim_min": np.nanmin([r["ssim"] for r in per_image]),
        "mse_mean": np.mean([r["mse"] for r in per_image]),
        "mse_max": np.max([r["mse"] for r in per_image]),
        "lpips_mean": np.nanmean([r["lpips"] for r in per_image]),
        "lpips_max": np.nanmax([r["lpips"] for r in per_image]),
        "avg_time_baseline_s": np.mean([r["baseline_time_s"] for r in per_image]),
        "avg_time_variant_s": np.mean([r["variant_time_s"] for r in per_image]),
        "per_image": per_image,
    }

    return agg


def print_summary(all_results: dict, steps: int, resolution: int, seed: int = DEFAULT_SEED):
    """Print formatted summary table."""
    print(f"\n\n{'=' * 100}")
    print(f"SD v2.1 Quantization Quality Summary")
    print(f"Settings: {steps} steps, {resolution}x{resolution}, seed={seed}")
    print(f"{'=' * 100}")

    header = (f"{'Precision':<12} {'Images':>6} {'PSNR(dB)':>10} {'SSIM':>8} "
              f"{'MSE':>10} {'LPIPS':>8} {'Time(s)':>8} {'vs FP32':>8}")
    print(header)
    print("-" * 100)

    # FP32 baseline row
    fp32_time = None
    for label, result in all_results.items():
        if not result:
            continue
        fp32_time = result.get("avg_time_baseline_s", 0)
        break

    print(f"{'FP32 (ref)':<12} {'-':>6} {'inf':>10} {'1.0000':>8} "
          f"{'0.00':>10} {'0.0000':>8} {fp32_time:>8.1f} {'-':>8}")

    for label, result in all_results.items():
        if not result:
            continue
        speedup = fp32_time / result["avg_time_variant_s"] if result["avg_time_variant_s"] > 0 else 0
        lpips_str = f"{result['lpips_mean']:.4f}" if not np.isnan(result["lpips_mean"]) else "N/A"
        print(f"{label:<12} {result['num_images']:>6} "
              f"{result['psnr_mean']:>10.2f} {result['ssim_mean']:>8.4f} "
              f"{result['mse_mean']:>10.2f} {lpips_str:>8} "
              f"{result['avg_time_variant_s']:>8.1f} {speedup:>7.2f}x")

    print("-" * 100)
    print("\nInterpretation guide:")
    print("  PSNR > 30 dB  : Good fidelity (minor pixel differences)")
    print("  PSNR > 25 dB  : Acceptable (visible but not objectionable)")
    print("  PSNR < 20 dB  : Significant degradation")
    print("  SSIM > 0.95   : Excellent structural similarity")
    print("  SSIM > 0.90   : Good (minor structural changes)")
    print("  LPIPS < 0.05  : Perceptually very similar")
    print("  LPIPS < 0.10  : Perceptually acceptable")
    print()
    print("Note: SD is stochastic - even FP32 vs FP32 with tiny float differences")
    print("can diverge significantly due to iterative denoising. Low PSNR does NOT")
    print("necessarily mean bad quality - check images visually.")


def save_results(all_results: dict, steps: int, resolution: int, seed: int = DEFAULT_SEED):
    """Save results to JSON and text report."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON (full data)
    json_path = OUTPUTS_DIR / f"quality_comparison_{timestamp}.json"
    json_data = {
        "model": SD_MODEL_ID,
        "settings": {
            "steps": steps,
            "resolution": resolution,
            "seed": seed,
        },
        "timestamp": timestamp,
        "results": {},
    }
    for label, result in all_results.items():
        if result:
            # Convert numpy types for JSON serialization
            clean = {}
            for k, v in result.items():
                if k == "per_image":
                    clean[k] = [
                        {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                         for kk, vv in item.items()}
                        for item in v
                    ]
                elif isinstance(v, (np.floating, float)):
                    clean[k] = float(v) if not np.isnan(v) else None
                else:
                    clean[k] = v
            json_data["results"][label] = clean

    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")


def check_status():
    """Check available models for evaluation."""
    print("=" * 60)
    print("SD v2.1 Quality Evaluation Status")
    print("=" * 60)

    print(f"\nModel directory: {MODEL_DIR}")
    for component in ["vae_encoder", "text_encoder", "unet", "vae_decoder"]:
        print(f"\n  {component}:")
        for precision in ["fp32", "int8_qdq"]:
            path = MODEL_DIR / f"{component}_{precision}.onnx"
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"    [OK] {path.name} ({size_mb:.1f} MB)")
            else:
                print(f"    [--] {path.name}")

    # Check pipeline completeness
    print("\n  Pipeline readiness:")
    for precision, label in [("fp32", "FP32"), ("int8_qdq", "INT8 QDQ")]:
        components = ["text_encoder", "unet", "vae_decoder"]
        ready = all((MODEL_DIR / f"{c}_{precision}.onnx").exists() for c in components)
        status = "[OK]" if ready else "[--]"
        print(f"    {status} {label} pipeline - {'Ready' if ready else 'Missing components'}")

    # Check previous results
    if OUTPUTS_DIR.exists():
        results = list(OUTPUTS_DIR.glob("quality_comparison_*.json"))
        if results:
            print(f"\n  Previous evaluations ({len(results)}):")
            for r in sorted(results)[-3:]:
                print(f"    {r.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SD v2.1 quantization quality degradation"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        choices=["int8"],
        help="Precision variants to compare against FP32 baseline (int8 = INT8 QDQ)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of denoising steps (default: {DEFAULT_STEPS})"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Image resolution (default: {DEFAULT_RESOLUTION})"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of images to generate (default: {DEFAULT_NUM_IMAGES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Text file with custom prompts (one per line)"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline generation (use existing images)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check available models and previous results"
    )

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if not args.compare:
        parser.print_help()
        print("\nExamples:")
        print("  python eval_sd_quality.py --compare int8")
        print("  python eval_sd_quality.py --compare int8 --steps 20 --num-images 10")
        print("  python eval_sd_quality.py --status")
        return

    # Load prompts
    if args.prompts and args.prompts.exists():
        prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
        print(f"Loaded {len(prompts)} custom prompts from {args.prompts}")
    else:
        prompts = DEFAULT_PROMPTS[:args.num_images]

    print(f"\nSettings:")
    print(f"  Model:      {SD_MODEL_ID}")
    print(f"  Steps:      {args.steps}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Seed:       {args.seed}")
    print(f"  Prompts:    {len(prompts)}")
    print(f"  Compare:    FP32 vs {', '.join(p.upper() for p in args.compare)}")

    run_dir = OUTPUTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate FP32 baseline
    baseline_dir = run_dir / "baseline_fp32"
    if args.skip_baseline:
        # Find most recent baseline
        existing = sorted(OUTPUTS_DIR.glob("run_*/baseline_fp32"))
        if existing:
            baseline_dir = existing[-1]
            print(f"\nReusing baseline from: {baseline_dir}")
            from PIL import Image
            baseline_results = []
            for i, prompt in enumerate(prompts):
                img_path = baseline_dir / f"baseline_fp32_{i:02d}.png"
                if img_path.exists():
                    baseline_results.append((prompt, img_path, 0.0))
                else:
                    print(f"  Missing: {img_path.name}")
                    args.skip_baseline = False
                    break

    if not args.skip_baseline:
        baseline_results = generate_pytorch_baseline(
            prompts, args.seed, args.steps, args.resolution, baseline_dir
        )

    if not baseline_results:
        print("Error: Failed to generate baseline images")
        return

    # Step 2: Load LPIPS model (once)
    lpips_model = load_lpips_model()

    # Step 3: Generate and compare each variant
    all_results = {}

    for precision in args.compare:
        variant_dir = run_dir / f"onnx_{precision}"
        variant_results = generate_onnx_images(
            prompts, args.seed, args.steps, args.resolution, precision, variant_dir
        )

        if variant_results:
            result = compare_quality(
                baseline_results, variant_results,
                precision.upper(), lpips_model
            )
            all_results[precision.upper()] = result
        else:
            all_results[precision.upper()] = None
            print(f"\n  SKIPPED: {precision.upper()} - model files missing")

    # Step 4: Summary
    print_summary(all_results, args.steps, args.resolution, args.seed)
    save_results(all_results, args.steps, args.resolution, args.seed)


if __name__ == "__main__":
    main()
