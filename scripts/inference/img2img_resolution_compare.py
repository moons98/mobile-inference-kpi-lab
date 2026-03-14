#!/usr/bin/env python3
"""
SD v1.5 / LCM-LoRA img2img 해상도별 출력 비교 (512, 720, 1024).

COCO val2017 이미지를 입력으로 받아 각 해상도로 img2img 변환 후
side-by-side 비교 이미지를 저장한다.

Usage:
  conda run -n mobile python scripts/inference/img2img_resolution_compare.py
  conda run -n mobile python scripts/inference/img2img_resolution_compare.py --num-images 3 --strength 0.6
  conda run -n mobile python scripts/inference/img2img_resolution_compare.py --resolutions 512 720
"""

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent.parent
COCO_DIR = PROJECT_ROOT / "datasets" / "coco" / "val2017"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "img2img_resolution_compare"

DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"

PROMPT = "watercolor painting, soft brushstrokes, artistic"
NEGATIVE = "blurry, low quality, artifacts, deformed"


def generate_img2img(pipe, image, prompt, negative, width, height,
                     steps, guidance, strength, seed, device):
    resized = image.resize((width, height), Image.LANCZOS)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=resized,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        width=width,
        height=height,
        generator=gen,
    ).images[0]
    elapsed = time.perf_counter() - t0
    return out, elapsed


def make_row_grid(orig, images, labels, out_path, thumb=320):
    """Create a single row: original + generated images."""
    n = len(images) + 1  # original + outputs
    pad = 6
    label_h = 24
    w = thumb * n + pad * (n - 1)
    h = thumb + label_h
    canvas = Image.new("RGB", (w, h), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)

    # Original
    orig_resized = orig.resize((thumb, thumb), Image.LANCZOS)
    draw.text((thumb // 2, 4), "Original", fill=(30, 30, 30), anchor="mt")
    canvas.paste(orig_resized, (0, label_h))

    # Generated
    for i, (img, label) in enumerate(zip(images, labels)):
        x = (i + 1) * (thumb + pad)
        resized = img.resize((thumb, thumb), Image.LANCZOS)
        draw.text((x + thumb // 2, 4), label, fill=(30, 30, 30), anchor="mt")
        canvas.paste(resized, (x, label_h))

    canvas.save(str(out_path))
    return out_path


def pick_coco_images(num_images, seed=42):
    """Pick diverse COCO images (different sizes/subjects)."""
    import random
    random.seed(seed)
    all_imgs = sorted(COCO_DIR.glob("*.jpg"))
    if not all_imgs:
        raise FileNotFoundError(f"No images in {COCO_DIR}")
    selected = random.sample(all_imgs, min(num_images, len(all_imgs)))
    return selected


def main():
    parser = argparse.ArgumentParser(description="SD v1.5 / LCM img2img resolution comparison")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative", default=NEGATIVE)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 720, 1024])
    parser.add_argument("--num-images", type=int, default=3, help="Number of COCO images to use")
    parser.add_argument("--strength", type=float, default=0.75, help="img2img strength (0=keep original, 1=fully generate)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sd-steps", type=int, default=30)
    parser.add_argument("--sd-guidance", type=float, default=7.5)
    parser.add_argument("--lcm-steps", type=int, default=8)
    parser.add_argument("--lcm-guidance", type=float, default=1.0)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lcm-lora", default=DEFAULT_LCM_LORA)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.device == "cuda" else torch.float32

    print(f"Device: {args.device}, dtype: {dtype}")
    print(f"Prompt: {args.prompt}")
    print(f"Strength: {args.strength}")
    print(f"Resolutions: {args.resolutions}")
    print(f"Steps: SD={args.sd_steps}, LCM={args.lcm_steps}")
    print()

    from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler

    # --- SD img2img pipeline ---
    print("Loading SD v1.5 img2img pipeline...")
    sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype
    ).to(args.device)

    # --- LCM img2img pipeline ---
    print("Loading LCM-LoRA img2img pipeline...")
    lcm_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype
    ).to(args.device)
    lcm_pipe.load_lora_weights(args.lcm_lora)
    lcm_pipe.fuse_lora()
    lcm_pipe.scheduler = LCMScheduler.from_config(lcm_pipe.scheduler.config)

    # --- Pick COCO images ---
    coco_images = pick_coco_images(args.num_images, args.seed)
    print(f"\nSelected {len(coco_images)} COCO images")

    results = []

    for idx, img_path in enumerate(coco_images):
        orig = Image.open(img_path).convert("RGB")
        img_name = img_path.stem
        print(f"\n{'='*60}")
        print(f"Image {idx+1}/{len(coco_images)}: {img_path.name} ({orig.size[0]}x{orig.size[1]})")
        print(f"{'='*60}")

        # Copy original to output dir
        shutil.copy2(img_path, out_dir / f"{img_name}_original.jpg")

        row_images = []
        row_labels = []

        for res in args.resolutions:
            # SD
            print(f"  [SD  {res}x{res}] ...", end=" ", flush=True)
            sd_img, sd_t = generate_img2img(
                sd_pipe, orig, args.prompt, args.negative, res, res,
                args.sd_steps, args.sd_guidance, args.strength, args.seed, args.device,
            )
            sd_path = out_dir / f"{img_name}_sd_{res}.png"
            sd_img.save(str(sd_path))
            print(f"{sd_t:.1f}s")
            row_images.append(sd_img)
            row_labels.append(f"SD s{args.sd_steps} {res}")
            results.append(("SD", img_name, res, sd_t))

            # LCM
            print(f"  [LCM {res}x{res}] ...", end=" ", flush=True)
            lcm_img, lcm_t = generate_img2img(
                lcm_pipe, orig, args.prompt, args.negative, res, res,
                args.lcm_steps, args.lcm_guidance, args.strength, args.seed, args.device,
            )
            lcm_path = out_dir / f"{img_name}_lcm_{res}.png"
            lcm_img.save(str(lcm_path))
            print(f"{lcm_t:.1f}s")
            row_images.append(lcm_img)
            row_labels.append(f"LCM s{args.lcm_steps} {res}")
            results.append(("LCM", img_name, res, lcm_t))

        # Save comparison grid for this image
        grid_path = make_row_grid(orig, row_images, row_labels,
                                   out_dir / f"{img_name}_compare.png")
        print(f"  Grid: {grid_path.name}")

    # --- Summary ---
    print(f"\n{'Model':<6} {'Image':<16} {'Res':>6} {'Time':>8}")
    print("-" * 42)
    for model, img_name, res, t in results:
        print(f"{model:<6} {img_name:<16} {res:>5}  {t:>7.1f}s")

    print(f"\nAll outputs: {out_dir}")


if __name__ == "__main__":
    main()
