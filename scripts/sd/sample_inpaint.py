#!/usr/bin/env python3
"""
Run SD v1.5 Inpainting pipeline end-to-end with ONNX Runtime (CPU).
Generates FP32 vs INT8 comparison images side-by-side.

Usage:
    python scripts/sd/sample_inpaint.py
    python scripts/sd/sample_inpaint.py --steps 10 --strength 0.7
    python scripts/sd/sample_inpaint.py --configs fp32          # FP32 only
    python scripts/sd/sample_inpaint.py --configs int8          # INT8 only
"""

import argparse
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "onnx"
ASSETS_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sd_inpaint_quality"

SEED = 42


def load_tokenizer():
    """Load CLIP tokenizer from local assets (vocab.json + merges.txt)."""
    from transformers import CLIPTokenizer

    vocab_path = ASSETS_DIR / "vocab.json"
    merges_path = ASSETS_DIR / "merges.txt"
    if vocab_path.exists() and merges_path.exists():
        return CLIPTokenizer(str(vocab_path), str(merges_path))
    else:
        return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


def load_session(model_path, label=""):
    """Load ONNX Runtime InferenceSession (CPU)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    print(f"  Loading {label}: {model_path.name} ...", end="", flush=True)
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    dt = time.perf_counter() - t0
    print(f" ({dt:.1f}s)")
    return sess


def prepare_inputs(image_path, mask_path, resolution=512):
    """Load and preprocess image + mask for inpainting."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize((resolution, resolution), Image.NEAREST)

    img_np = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    img_np = img_np.transpose(2, 0, 1)[np.newaxis]  # (1,3,512,512)

    mask_np = np.array(mask, dtype=np.float32) / 255.0  # [0, 1]
    mask_np = (mask_np > 0.5).astype(np.float32)
    mask_np = mask_np[np.newaxis, np.newaxis]  # (1,1,512,512)

    # Masked image: zero out masked region
    masked_img = img_np * (1 - mask_np)

    return img_np, mask_np, masked_img, img, mask


def run_pipeline(config_name, model_files, tokenizer, image_np, mask_np,
                 masked_img_np, prompt, steps=20, strength=0.7, seed=SEED):
    """Run full inpainting pipeline: vae_enc -> text_enc -> UNet loop -> vae_dec."""
    from diffusers import PNDMScheduler

    print(f"\n{'='*60}")
    print(f"Running pipeline: {config_name}")
    print(f"  steps={steps}, strength={strength}, seed={seed}")
    print(f"{'='*60}")

    t_total = time.perf_counter()

    # 1. Tokenize
    tokens = tokenizer(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="np",
    )
    input_ids = tokens.input_ids.astype(np.int64)

    # Also encode empty prompt for classifier-free guidance (unconditional)
    uncond_tokens = tokenizer(
        "", padding="max_length", max_length=77,
        truncation=True, return_tensors="np",
    )
    uncond_input_ids = uncond_tokens.input_ids.astype(np.int64)

    # 2. Text Encoder
    text_enc = load_session(model_files["text_encoder"], "text_encoder")
    t0 = time.perf_counter()
    text_emb = text_enc.run(None, {"input_ids": input_ids})[0]
    uncond_emb = text_enc.run(None, {"input_ids": uncond_input_ids})[0]
    dt_text = time.perf_counter() - t0
    print(f"  text_encoder: {dt_text:.2f}s, output {text_emb.shape}")
    del text_enc

    # 3. VAE Encoder — encode masked image to latent
    vae_enc = load_session(model_files["vae_encoder"], "vae_encoder")
    t0 = time.perf_counter()
    masked_latent = vae_enc.run(None, {"sample": masked_img_np.astype(np.float32)})[0]
    dt_vae_enc = time.perf_counter() - t0
    print(f"  vae_encoder: {dt_vae_enc:.2f}s, output {masked_latent.shape}")
    del vae_enc

    # 4. Prepare latent mask (downsample to latent space)
    # mask: (1,1,512,512) -> (1,1,64,64)
    from PIL import Image
    mask_pil = Image.fromarray((mask_np[0, 0] * 255).astype(np.uint8))
    mask_latent = mask_pil.resize((64, 64), Image.NEAREST)
    mask_latent = np.array(mask_latent, dtype=np.float32) / 255.0
    mask_latent = (mask_latent > 0.5).astype(np.float32)[np.newaxis, np.newaxis]

    # 5. Setup scheduler
    scheduler = PNDMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        subfolder="scheduler",
    )
    scheduler.set_timesteps(steps)

    # Apply strength: skip initial timesteps
    init_timestep = min(int(steps * strength), steps)
    t_start = max(steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]
    print(f"  Scheduler: {len(timesteps)} actual steps (strength={strength})")

    # Initial noise
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal(masked_latent.shape).astype(np.float32)

    # Scale initial noise
    latents = latents * float(scheduler.init_noise_sigma)

    # 6. UNet denoising loop
    unet = load_session(model_files["unet"], "unet")
    guidance_scale = 7.5
    dt_unet_total = 0

    for i, t in enumerate(timesteps):
        t_val = np.array([int(t)], dtype=np.int64)

        # Concat for inpainting: [noisy_latent, mask, masked_image_latent]
        latent_input = np.concatenate([latents, mask_latent, masked_latent], axis=1)

        # Classifier-free guidance: run unconditional + conditional
        t0 = time.perf_counter()
        noise_pred_uncond = unet.run(None, {
            "sample": latent_input,
            "timestep": t_val,
            "encoder_hidden_states": uncond_emb,
        })[0]
        noise_pred_text = unet.run(None, {
            "sample": latent_input,
            "timestep": t_val,
            "encoder_hidden_states": text_emb,
        })[0]
        dt_unet_total += (time.perf_counter() - t0)

        # CFG
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Scheduler step
        import torch
        scheduler_output = scheduler.step(
            torch.from_numpy(noise_pred),
            int(t),
            torch.from_numpy(latents),
        )
        latents = scheduler_output.prev_sample.numpy()

        if (i + 1) % max(1, len(timesteps) // 4) == 0 or i == 0:
            print(f"    Step [{i+1}/{len(timesteps)}] t={int(t)}")

    del unet
    print(f"  unet: {dt_unet_total:.2f}s total ({dt_unet_total/len(timesteps):.2f}s/step)")

    # 7. VAE Decoder
    vae_dec = load_session(model_files["vae_decoder"], "vae_decoder")
    t0 = time.perf_counter()
    decoded = vae_dec.run(None, {"latent_sample": latents})[0]
    dt_vae_dec = time.perf_counter() - t0
    print(f"  vae_decoder: {dt_vae_dec:.2f}s, output {decoded.shape}")
    del vae_dec

    # Convert to image: [-1,1] -> [0,255]
    decoded = np.clip((decoded[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(decoded)

    dt_total = time.perf_counter() - t_total
    print(f"  Total: {dt_total:.1f}s")

    return result_img, {
        "text_encoder": dt_text,
        "vae_encoder": dt_vae_enc,
        "unet_total": dt_unet_total,
        "unet_per_step": dt_unet_total / len(timesteps),
        "vae_decoder": dt_vae_dec,
        "total": dt_total,
        "steps": len(timesteps),
    }


def make_comparison(input_img, mask_img, results, output_path):
    """Create side-by-side comparison image."""
    from PIL import Image, ImageDraw, ImageFont

    images = []
    labels = []

    # Input + mask overlay
    input_rgba = input_img.convert("RGBA")
    mask_rgba = Image.new("RGBA", input_img.size, (255, 0, 0, 0))
    mask_arr = np.array(mask_img)
    mask_overlay = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
    mask_overlay[mask_arr > 127] = [255, 0, 0, 100]
    mask_rgba = Image.fromarray(mask_overlay)
    input_with_mask = Image.alpha_composite(input_rgba, mask_rgba).convert("RGB")
    images.append(input_with_mask)
    labels.append("Input + Mask")

    for name, (img, timing) in results.items():
        images.append(img)
        labels.append(f"{name}\n({timing['total']:.1f}s)")

    # Compose
    w, h = images[0].size
    padding = 10
    label_h = 40
    total_w = len(images) * w + (len(images) - 1) * padding
    total_h = h + label_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * (w + padding)
        canvas.paste(img, (x, label_h))
        draw.text((x + w // 2, 5), label, fill=(0, 0, 0), anchor="mt")

    canvas.save(str(output_path))
    print(f"\nComparison saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SD v1.5 Inpainting sample inference")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--image", default=str(ASSETS_DIR / "test_images" / "scene_medium.jpg"))
    parser.add_argument("--mask", default=str(ASSETS_DIR / "test_masks" / "mask_medium.png"))
    parser.add_argument("--configs", nargs="+", default=["fp32", "int8"],
                        choices=["fp32", "int8"], help="Which configs to run")
    parser.add_argument("--prompt", default="remove the object and fill the background naturally")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    image_np, mask_np, masked_img_np, input_img, mask_img = prepare_inputs(
        args.image, args.mask
    )

    tokenizer = load_tokenizer()

    # Define model file sets
    configs = {}
    if "fp32" in args.configs:
        configs["FP32"] = {
            "vae_encoder": MODEL_DIR / "vae_encoder_fp32.onnx",
            "text_encoder": MODEL_DIR / "text_encoder_fp32.onnx",
            "unet": MODEL_DIR / "unet_fp32.onnx",
            "vae_decoder": MODEL_DIR / "vae_decoder_fp32.onnx",
        }
    if "int8" in args.configs:
        configs["INT8 (text_enc=FP32)"] = {
            "vae_encoder": MODEL_DIR / "vae_encoder_int8_qdq.onnx",
            "text_encoder": MODEL_DIR / "text_encoder_fp32.onnx",  # INT8 broken
            "unet": MODEL_DIR / "unet_int8_qdq.onnx",
            "vae_decoder": MODEL_DIR / "vae_decoder_int8_qdq.onnx",
        }

    # Check all files exist
    for config_name, files in configs.items():
        for comp, path in files.items():
            if not path.exists():
                print(f"ERROR: {path} not found")
                return

    # Run pipelines
    results = {}
    for config_name, model_files in configs.items():
        img, timing = run_pipeline(
            config_name, model_files, tokenizer,
            image_np, mask_np, masked_img_np,
            prompt=args.prompt,
            steps=args.steps,
            strength=args.strength,
            seed=args.seed,
        )
        results[config_name] = (img, timing)

        # Save individual result
        safe_name = config_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        img.save(str(OUTPUT_DIR / f"sample_{safe_name}.png"))
        print(f"  Saved: sample_{safe_name}.png")

    # Comparison image
    if len(results) > 0:
        make_comparison(input_img, mask_img, results, OUTPUT_DIR / "sample_comparison.png")

    # Print timing summary
    print(f"\n{'='*60}")
    print("Timing Summary")
    print(f"{'='*60}")
    for config_name, (_, timing) in results.items():
        print(f"\n  [{config_name}]")
        for k, v in timing.items():
            if k == "steps":
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v:.2f}s")


if __name__ == "__main__":
    main()
