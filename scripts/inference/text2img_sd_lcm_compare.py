#!/usr/bin/env python3
"""
Compare SD v1.5 vs LCM-LoRA text-to-image generation.

Features:
- Prompt + style preset selection
- Generate with SD and LCM-LoRA
- Save per-model outputs and a side-by-side comparison image

Usage:
  conda run -n mobile python scripts/inference/text2img_sd_lcm_compare.py ^
    --prompt "a cozy cafe street at sunset" ^
    --style cinematic
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "inference" / "outputs"

DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"

STYLE_PRESETS = {
    "none": {
        "suffix": "",
        "negative": "blurry, low quality, artifacts, deformed",
    },
    "cinematic": {
        "suffix": "cinematic lighting, film still, depth of field, highly detailed",
        "negative": "blurry, low quality, oversaturated, artifacts, watermark, text",
    },
    "anime": {
        "suffix": "anime style, clean lineart, vibrant colors, masterpiece",
        "negative": "realistic, photo, blurry, low quality, artifacts, watermark, text",
    },
    "oil_painting": {
        "suffix": "oil painting, textured brushstrokes, artistic composition, masterpiece",
        "negative": "photo, blurry, low quality, artifacts, watermark, text",
    },
    "watercolor": {
        "suffix": "watercolor painting, soft washes, delicate brush texture, paper grain, artistic",
        "negative": "photo, hard edges, blurry, low quality, artifacts, watermark, text",
    },
    "pixel_art": {
        "suffix": "pixel art, 8-bit style, retro game palette",
        "negative": "photo, blurry, smooth shading, artifacts, watermark, text",
    },
}


def build_prompt(prompt: str, style: str):
    preset = STYLE_PRESETS[style]
    if preset["suffix"]:
        return f"{prompt}, {preset['suffix']}", preset["negative"]
    return prompt, preset["negative"]


def make_compare(sd_img: Image.Image, lcm_img: Image.Image, out_path: Path, panel_size: int = 768):
    sd_p = sd_img.resize((panel_size, panel_size), Image.LANCZOS)
    lcm_p = lcm_img.resize((panel_size, panel_size), Image.LANCZOS)

    pad = 8
    label_h = 34
    canvas = Image.new("RGB", (panel_size * 2 + pad, panel_size + label_h), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)

    draw.text((panel_size // 2, 8), "SD v1.5", fill=(30, 30, 30), anchor="mt")
    draw.text((panel_size + pad + panel_size // 2, 8), "LCM-LoRA", fill=(30, 30, 30), anchor="mt")

    canvas.paste(sd_p, (0, label_h))
    canvas.paste(lcm_p, (panel_size + pad, label_h))
    canvas.save(str(out_path))


def main():
    parser = argparse.ArgumentParser(description="SD vs LCM-LoRA text2img comparison")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--style", default="cinematic", choices=sorted(STYLE_PRESETS.keys()))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="SD1.5 model id or local path")
    parser.add_argument("--lcm-lora", default=DEFAULT_LCM_LORA, help="LCM-LoRA id or local path")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sd-steps", type=int, default=30)
    parser.add_argument("--sd-guidance", type=float, default=7.5)
    parser.add_argument("--lcm-steps", type=int, default=8)
    parser.add_argument("--lcm-guidance", type=float, default=1.5)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt, negative = build_prompt(args.prompt, args.style)
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    from diffusers import DiffusionPipeline, LCMScheduler

    print(f"Device={args.device}, dtype={dtype}")
    print(f"Prompt: {prompt}")
    print(f"Negative: {negative}")

    # SD baseline
    t0 = time.perf_counter()
    sd_pipe = DiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype).to(args.device)
    gen_sd = torch.Generator(device=args.device).manual_seed(args.seed)
    sd_img = sd_pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=args.sd_steps,
        guidance_scale=args.sd_guidance,
        width=args.width,
        height=args.height,
        generator=gen_sd,
    ).images[0]
    sd_t = time.perf_counter() - t0

    # LCM-LoRA
    t1 = time.perf_counter()
    lcm_pipe = DiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype).to(args.device)
    lcm_pipe.load_lora_weights(args.lcm_lora)
    lcm_pipe.fuse_lora()
    lcm_pipe.scheduler = LCMScheduler.from_config(lcm_pipe.scheduler.config)
    gen_lcm = torch.Generator(device=args.device).manual_seed(args.seed)
    lcm_img = lcm_pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=args.lcm_steps,
        guidance_scale=args.lcm_guidance,
        width=args.width,
        height=args.height,
        generator=gen_lcm,
    ).images[0]
    lcm_t = time.perf_counter() - t1

    safe_tag = args.tag if args.tag else args.style
    sd_path = OUTPUT_DIR / f"sd_{safe_tag}.png"
    lcm_path = OUTPUT_DIR / f"lcm_{safe_tag}.png"
    cmp_path = OUTPUT_DIR / f"compare_{safe_tag}.png"

    sd_img.save(str(sd_path))
    lcm_img.save(str(lcm_path))
    make_compare(sd_img, lcm_img, cmp_path)

    print(f"SD saved: {sd_path} ({sd_t:.2f}s)")
    print(f"LCM saved: {lcm_path} ({lcm_t:.2f}s)")
    print(f"Compare: {cmp_path}")


if __name__ == "__main__":
    main()
