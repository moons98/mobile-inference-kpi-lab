#!/usr/bin/env python3
"""
SD v2.1 PyTorch example inference - verify pipeline works correctly.

Downloads the model (if not cached), generates images with preset prompts,
and saves results to outputs/sd_examples/.

Usage:
    python example_sd_inference.py
    python example_sd_inference.py --prompt "a cat on the moon"
    python example_sd_inference.py --all-presets
    python example_sd_inference.py --steps 30 --resolution 768
    python example_sd_inference.py --save-weights   # save pipeline to ./weights/sd_v2.1/
"""

import argparse
import gc
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPTS_DIR.parent
WEIGHTS_DIR = PROJECT_DIR / "weights" / "sd_v2.1"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "sd_examples"

SD_MODEL_ID = "sd2-community/stable-diffusion-2-1"

PRESET_PROMPTS = [
    "sunset beach with golden light",
    "snowy winter cityscape at night",
    "lush green forest with sunlight",
    "futuristic neon city",
    "soft watercolor painting style",
]

DEFAULT_NEGATIVE = "low quality, blurry, distorted"


def run_inference(
    prompts: list,
    steps: int = 20,
    resolution: int = 512,
    seed: int = 42,
    save_weights: bool = False,
    device: str = None,
):
    import torch
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 60)
    print("SD v2.1 PyTorch Example Inference")
    print("=" * 60)
    print(f"  Model:      {SD_MODEL_ID}")
    print(f"  Device:     {device} ({dtype})")
    print(f"  Steps:      {steps}")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Seed:       {seed}")
    print(f"  Prompts:    {len(prompts)}")

    # Check for local weights first
    if WEIGHTS_DIR.exists() and (WEIGHTS_DIR / "model_index.json").exists():
        print(f"\n  Loading from local weights: {WEIGHTS_DIR}")
        pipe = StableDiffusionPipeline.from_pretrained(
            str(WEIGHTS_DIR),
            torch_dtype=dtype,
        )
    else:
        print(f"\n  Downloading from HuggingFace: {SD_MODEL_ID}")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
        )

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # Optionally save weights locally
    if save_weights:
        print(f"\n  Saving weights to: {WEIGHTS_DIR}")
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        pipe.save_pretrained(str(WEIGHTS_DIR))
        print("  Weights saved successfully")

    # Print model info
    print(f"\n  Pipeline components:")
    print(f"    Text Encoder: {pipe.text_encoder.__class__.__name__}")
    print(f"    UNet:         {pipe.unet.__class__.__name__}")
    print(f"    VAE:          {pipe.vae.__class__.__name__}")
    print(f"    Scheduler:    {pipe.scheduler.__class__.__name__}")

    unet_params = sum(p.numel() for p in pipe.unet.parameters())
    te_params = sum(p.numel() for p in pipe.text_encoder.parameters())
    vae_params = sum(p.numel() for p in pipe.vae.parameters())
    print(f"    UNet params:  {unet_params / 1e6:.1f}M")
    print(f"    TE params:    {te_params / 1e6:.1f}M")
    print(f"    VAE params:   {vae_params / 1e6:.1f}M")

    # Generate images
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\"")
        generator = torch.Generator(device=device).manual_seed(seed + i)

        t_start = time.perf_counter()
        result = pipe(
            prompt,
            negative_prompt=DEFAULT_NEGATIVE,
            num_inference_steps=steps,
            height=resolution,
            width=resolution,
            generator=generator,
            guidance_scale=7.5,
        )
        t_elapsed = time.perf_counter() - t_start

        image = result.images[0]
        filename = f"example_{i:02d}_{prompt[:30].replace(' ', '_')}.png"
        img_path = OUTPUTS_DIR / filename
        image.save(img_path)

        print(f"    Time:   {t_elapsed:.1f}s")
        print(f"    Saved:  {img_path.name}")
        print(f"    Size:   {image.size}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Done! {len(prompts)} images saved to: {OUTPUTS_DIR}")
    print("=" * 60)

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="SD v2.1 PyTorch example inference"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom prompt (default: first preset)"
    )
    parser.add_argument(
        "--all-presets", action="store_true",
        help="Generate images for all 5 preset prompts"
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of denoising steps (default: 20)"
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Image resolution (default: 512)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-weights", action="store_true",
        help="Save pipeline weights to ./weights/sd_v2.1/"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Device (default: auto-detect)"
    )

    args = parser.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    elif args.all_presets:
        prompts = PRESET_PROMPTS
    else:
        prompts = [PRESET_PROMPTS[0]]

    run_inference(
        prompts=prompts,
        steps=args.steps,
        resolution=args.resolution,
        seed=args.seed,
        save_weights=args.save_weights,
        device=args.device,
    )


if __name__ == "__main__":
    main()
