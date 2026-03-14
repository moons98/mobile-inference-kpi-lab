#!/usr/bin/env python3
"""
SD v1.5 / LCM-LoRA 해상도별 출력 품질 비교 (512, 720, 1024).

각 해상도 × 모델(SD, LCM) 조합으로 이미지를 생성하고
side-by-side 비교 이미지를 저장한다.

Usage:
  conda run -n mobile python scripts/inference/resolution_compare.py
  conda run -n mobile python scripts/inference/resolution_compare.py --prompt "a castle on a cliff" --style cinematic
  conda run -n mobile python scripts/inference/resolution_compare.py --resolutions 512 720
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resolution_compare"

DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"

PROMPT = "a photo of a cat sitting on a windowsill, watercolor painting, soft edges, delicate, artistic"
NEGATIVE = "blurry, low quality, artifacts, deformed, watermark, text"


def generate(pipe, prompt, negative, width, height, steps, guidance, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    img = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=gen,
    ).images[0]
    elapsed = time.perf_counter() - t0
    return img, elapsed


def make_grid(images, labels, out_path, thumb=384):
    """Create a grid image: rows=resolutions, cols=models."""
    n_rows = len(images)
    n_cols = len(images[0])
    pad = 6
    label_h = 28

    w = thumb * n_cols + pad * (n_cols - 1)
    h = (thumb + label_h) * n_rows + pad * (n_rows - 1)
    canvas = Image.new("RGB", (w, h), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)

    for r in range(n_rows):
        for c in range(n_cols):
            img = images[r][c].resize((thumb, thumb), Image.LANCZOS)
            x = c * (thumb + pad)
            y = r * (thumb + label_h + pad)
            draw.text((x + thumb // 2, y + 4), labels[r][c],
                      fill=(30, 30, 30), anchor="mt")
            canvas.paste(img, (x, y + label_h))

    canvas.save(str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="SD v1.5 / LCM resolution comparison")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative", default=NEGATIVE)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 720, 1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sd-steps", type=int, default=20)
    parser.add_argument("--sd-guidance", type=float, default=7.5)
    parser.add_argument("--lcm-steps", type=int, default=4)
    parser.add_argument("--lcm-guidance", type=float, default=1.0)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lcm-lora", default=DEFAULT_LCM_LORA)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    prompt = args.prompt
    negative = args.negative

    print(f"Device: {args.device}, dtype: {dtype}")
    print(f"Prompt: {prompt}")
    print(f"Resolutions: {args.resolutions}")
    print()

    from diffusers import DiffusionPipeline, LCMScheduler

    # --- SD v1.5 pipeline ---
    print("Loading SD v1.5 pipeline...")
    sd_pipe = DiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    sd_pipe = sd_pipe.to(args.device)

    # --- LCM-LoRA pipeline ---
    print("Loading LCM-LoRA pipeline...")
    lcm_pipe = DiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    lcm_pipe.load_lora_weights(args.lcm_lora)
    lcm_pipe.fuse_lora()
    lcm_pipe.scheduler = LCMScheduler.from_config(lcm_pipe.scheduler.config)
    lcm_pipe = lcm_pipe.to(args.device)

    # --- Generate ---
    grid_images = []  # [row][col]
    grid_labels = []
    results = []

    for res in args.resolutions:
        row_imgs = []
        row_labels = []

        # SD
        print(f"[SD  {res}x{res}] generating (steps={args.sd_steps})...", end=" ", flush=True)
        sd_img, sd_t = generate(
            sd_pipe, prompt, negative, res, res,
            args.sd_steps, args.sd_guidance, args.seed, args.device,
        )
        sd_path = out_dir / f"sd_{res}.png"
        sd_img.save(str(sd_path))
        print(f"{sd_t:.1f}s")
        row_imgs.append(sd_img)
        row_labels.append(f"SD s{args.sd_steps} {res}x{res}")
        results.append(("SD", res, args.sd_steps, sd_t))

        # LCM
        print(f"[LCM {res}x{res}] generating (steps={args.lcm_steps})...", end=" ", flush=True)
        lcm_img, lcm_t = generate(
            lcm_pipe, prompt, negative, res, res,
            args.lcm_steps, args.lcm_guidance, args.seed, args.device,
        )
        lcm_path = out_dir / f"lcm_{res}.png"
        lcm_img.save(str(lcm_path))
        print(f"{lcm_t:.1f}s")
        row_imgs.append(lcm_img)
        row_labels.append(f"LCM s{args.lcm_steps} {res}x{res}")
        results.append(("LCM", res, args.lcm_steps, lcm_t))

        grid_images.append(row_imgs)
        grid_labels.append(row_labels)

    # --- Grid ---
    grid_path = make_grid(grid_images, grid_labels, out_dir / "grid.png")
    print(f"\nGrid saved: {grid_path}")

    # --- Summary ---
    print(f"\n{'Model':<6} {'Res':>6} {'Steps':>5} {'Time':>8}")
    print("-" * 30)
    for model, res, steps, t in results:
        print(f"{model:<6} {res:>5}  {steps:>5} {t:>7.1f}s")

    print(f"\nAll outputs: {out_dir}")


if __name__ == "__main__":
    main()
