#!/usr/bin/env python3
"""
Export SD v1.5 base UNet and LCM-LoRA fused UNet to ONNX (FP32).

Exports two UNet variants for the feasibility comparison:
  1. sd_v1.5 base UNet (4ch input) — baseline
  2. sd_v1.5 + LCM-LoRA fused UNet (4ch input) — few-step variant

VAE Encoder/Decoder and Text Encoder are shared between both variants
and are exported separately by export_sd_to_onnx.py.

Usage:
    # Export both variants
    python scripts/sd/export_sd_lcm_unet.py --export-all

    # Export only base UNet
    python scripts/sd/export_sd_lcm_unet.py --export base

    # Export only LCM-LoRA fused UNet
    python scripts/sd/export_sd_lcm_unet.py --export lcm

    # Check status
    python scripts/sd/export_sd_lcm_unet.py --status
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "weights" / "sd_v1.5" / "onnx"
LCM_LORA_LOCAL = PROJECT_ROOT / "weights" / "lcm_lora_sdv1_5"
CALIB_IMAGES_DIR = PROJECT_ROOT / "datasets" / "coco" / "val2017"
SEED = 42

# Diverse prompts for calibration
CALIB_PROMPTS = [
    "a photo of a dog playing in the park",
    "sunset beach with golden light",
    "snowy winter cityscape at night",
    "lush green forest with sunlight filtering through trees",
    "futuristic neon city street at night",
    "a photo of a cat sitting on a wooden chair",
    "mountain landscape with dramatic clouds",
    "modern architecture building with glass facade",
    "portrait of a woman in oil painting style",
    "colorful abstract geometric pattern",
    "a cozy coffee shop interior with warm lighting",
    "ocean waves crashing on rocky shore",
    "autumn leaves in a park pathway",
    "vintage car on a country road",
    "night sky with stars and milky way",
    "japanese garden with cherry blossoms",
    "underwater coral reef scene with tropical fish",
    "medieval castle on a hilltop at dawn",
    "tropical rainforest with waterfall",
    "minimalist scandinavian living room",
    "busy city intersection with pedestrians",
    "close-up of fresh fruits on a table",
    "snow-covered mountain peak at sunrise",
    "old brick building with ivy growing on walls",
    "a bicycle leaning against a fence",
    "crowded market street with colorful stalls",
    "calm lake reflecting mountains at dusk",
    "a person walking through a wheat field",
    "aerial view of a winding river through forest",
    "rainy street with reflections and umbrellas",
    "desert sand dunes under blue sky",
    "a bowl of ramen on a wooden table",
    "sunflower field stretching to the horizon",
    "foggy forest path in early morning",
    "a lighthouse on a rocky cliff",
    "children playing in a playground",
    "a train passing through countryside",
    "close-up of a butterfly on a flower",
    "an old library with tall bookshelves",
    "a sailboat on calm turquoise water",
    "graffiti art on an urban wall",
    "a campfire under starry night sky",
    "birds flying over a golden field at sunset",
    "a narrow alley in an italian village",
    "fresh snow on pine tree branches",
    "a cup of coffee with latte art",
    "horses grazing in a green meadow",
    "an abandoned factory with broken windows",
    "a surfer riding a big wave",
    "a cozy bedroom with soft natural light",
    "a busy kitchen with steam rising from pots",
    "autumn forest with orange and red leaves",
    "a dog running on a sandy beach",
    "rooftop view of a cityscape at golden hour",
    "a field of lavender in provence",
    "an old wooden door with peeling paint",
    "a waterfall in a tropical jungle",
    "a street musician playing guitar",
    "close-up of raindrops on a leaf",
    "a hot air balloon over rolling hills",
    "a snowy village with warm lit windows",
    "a bridge over a river in autumn",
    "an eagle soaring over mountain peaks",
    "a plate of sushi on a black slate",
]

def _update_output_dir(new_dir):
    global OUTPUT_DIR
    if new_dir:
        OUTPUT_DIR = new_dir


VARIANTS = {
    "base": {
        "description": "SD v1.5 Base UNet (4ch, txt2img/img2img)",
        "filename": "unet_base_fp32.onnx",
        "use_lcm_lora": False,
    },
    "lcm": {
        "description": "SD v1.5 + LCM-LoRA Fused UNet (4ch, few-step)",
        "filename": "unet_lcm_fp32.onnx",
        "use_lcm_lora": True,
    },
}

# UNet input spec (standard 4ch txt2img)
UNET_INPUT_SHAPES = {
    "sample": [1, 4, 64, 64],
    "timestep": [1],
    "encoder_hidden_states": [1, 77, 768],
}


def load_sd_pipeline(use_lcm_lora: bool = False):
    """Load SD v1.5 base pipeline, optionally fuse LCM-LoRA."""
    import torch
    from diffusers import StableDiffusionPipeline

    print(f"Loading base model: {SD_MODEL_ID}")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float32,
    )
    print("  Pipeline loaded")

    if use_lcm_lora:
        # Load from local if available, otherwise from HuggingFace
        lora_source = str(LCM_LORA_LOCAL) if LCM_LORA_LOCAL.exists() else LCM_LORA_ID
        print(f"  Loading LCM-LoRA from: {lora_source}")
        pipe.load_lora_weights(lora_source)
        print("  Fusing LoRA into UNet...")
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        print("  LCM-LoRA fused and unloaded (weights baked into UNet)")

    return pipe


def _merge_external_data(onnx_path: Path):
    """Merge ONNX model with scattered external data into .onnx + single .data file."""
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    parent = onnx_path.parent

    # Collect scattered external data locations before loading (load clears them)
    scattered = set()
    model_meta = onnx.load(str(onnx_path), load_external_data=False)
    for tensor in model_meta.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                scattered.add(entry.value)
    del model_meta

    model = onnx.load(str(onnx_path), load_external_data=True)

    ext_data_name = onnx_path.stem + ".onnx.data"
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=ext_data_name,
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.save(model, str(onnx_path))

    # Remove scattered external data files
    removed = 0
    for name in scattered:
        p = parent / name
        if p.exists() and name != ext_data_name:
            p.unlink()
            removed += 1

    data_path = parent / ext_data_name
    data_mb = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
    print(f"  Merged external data: {onnx_path.name} + {ext_data_name} ({data_mb:.1f} MB)")
    if removed:
        print(f"  Cleaned up {removed} scattered data files")


def export_unet(pipe, output_path: Path, variant_name: str) -> bool:
    """Export UNet to ONNX (4ch standard input)."""
    import torch

    desc = VARIANTS[variant_name]["description"]
    print(f"\n  Exporting {desc}...")

    unet = pipe.unet
    unet.eval()
    unet = unet.to("cpu")

    dummy_sample = torch.randn(*UNET_INPUT_SHAPES["sample"], dtype=torch.float32)
    dummy_timestep = torch.tensor([1], dtype=torch.long)
    dummy_encoder_hidden = torch.randn(
        *UNET_INPUT_SHAPES["encoder_hidden_states"], dtype=torch.float32
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

    wrapper = UNetWrapper(unet)

    torch.onnx.export(
        wrapper,
        (dummy_sample, dummy_timestep, dummy_encoder_hidden),
        str(output_path),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    _merge_external_data(output_path)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported: {output_path.name} ({size_mb:.1f} MB)")
    return True


def _load_coco_images(num_images: int, image_dir: Path):
    """Load and preprocess COCO images for calibration. Returns list of (1,3,512,512) arrays."""
    from PIL import Image

    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPEG images found in {image_dir}")

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(image_files), size=min(num_images, len(image_files)), replace=False)
    indices.sort()

    images = []
    for idx in indices:
        img = Image.open(image_files[idx]).convert("RGB")
        w, h = img.size
        crop_size = min(w, h)
        left, top = (w - crop_size) // 2, (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        img = img.resize((512, 512), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        images.append(arr.transpose(2, 0, 1)[np.newaxis])

    print(f"  Loaded {len(images)} COCO images from {image_dir.name}/")
    return images


def generate_calibration_data(num_samples: int, output_dir: Path, image_dir: Path = None):
    """Generate real calibration data for 4ch UNet (txt2img).

    Produces:
    - calib_unet.npz: noisy latents (4ch) + text embeddings + stratified timesteps

    Uses COCO images → VAE encode → add noise at stratified timesteps.
    Text embeddings from real diverse prompts.
    """
    import torch

    image_dir = image_dir or CALIB_IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Generating 4ch UNet calibration data ({num_samples} samples)")
    print("=" * 60)

    coco_images = _load_coco_images(num_samples, image_dir)
    num_samples = len(coco_images)

    pipe = load_sd_pipeline(use_lcm_lora=False)
    vae = pipe.vae.eval()
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler

    rng = np.random.default_rng(SEED)
    gen = torch.Generator().manual_seed(SEED)

    # 1. Text encoder: real prompts → hidden states
    print("\n  [1/3] Text encoder: encoding prompts...")
    text_enc_hidden_states = []
    with torch.no_grad():
        for i in range(num_samples):
            prompt = CALIB_PROMPTS[i % len(CALIB_PROMPTS)]
            tok = tokenizer(
                prompt, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt",
            )
            hidden = text_encoder(tok.input_ids)[0]
            text_enc_hidden_states.append(hidden.numpy())
    text_enc_hidden_states = np.concatenate(text_enc_hidden_states, axis=0)
    print(f"    {num_samples} prompts → hidden_states {text_enc_hidden_states.shape}")

    # 2. VAE encode: COCO images → real latents
    print("\n  [2/3] VAE encoder: images → latents...")
    latents = []
    with torch.no_grad():
        for img_np in coco_images:
            img_tensor = torch.from_numpy(img_np)
            latent = vae.encode(img_tensor).latent_dist.sample(gen)
            latent = latent * vae.config.scaling_factor
            latents.append(latent.numpy())
    latents = np.concatenate(latents, axis=0)
    print(f"    {num_samples} latents → {latents.shape}")

    # 3. UNet: noisy latents (4ch) at stratified timesteps + text embeddings
    print("\n  [3/3] UNet: assembling 4ch inputs...")
    timesteps = np.linspace(1, 999, num_samples, dtype=int)
    rng.shuffle(timesteps)

    unet_samples = []
    unet_timesteps = []
    unet_hidden = []
    scheduler.set_timesteps(1000)

    with torch.no_grad():
        for i in range(num_samples):
            latent = torch.from_numpy(latents[i:i + 1])
            t = int(timesteps[i])
            noise = torch.randn_like(latent)
            noisy = scheduler.add_noise(latent, noise, torch.tensor([t]))

            unet_samples.append(noisy.numpy())
            unet_timesteps.append(np.array([t], dtype=np.int64))
            unet_hidden.append(text_enc_hidden_states[i:i + 1])

    np.savez(
        output_dir / "calib_unet.npz",
        sample=np.concatenate(unet_samples, axis=0),
        timestep=np.concatenate(unet_timesteps, axis=0),
        encoder_hidden_states=np.concatenate(unet_hidden, axis=0),
    )
    print(f"    {num_samples} samples (4ch), timesteps [{int(timesteps.min())}-{int(timesteps.max())}]")

    del pipe
    gc.collect()

    # Report
    print("\n" + "-" * 40)
    npz_path = output_dir / "calib_unet.npz"
    sz = npz_path.stat().st_size / 1024 / 1024
    print(f"  calib_unet.npz: {sz:.1f} MB")
    print(f"  Saved to {output_dir}")


def export_variant(variant_name: str, force: bool = False) -> bool:
    """Export a single UNet variant."""
    info = VARIANTS[variant_name]
    output_path = OUTPUT_DIR / info["filename"]

    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  Reusing existing: {output_path.name} ({size_mb:.1f} MB)")
        return True

    if output_path.exists() and force:
        output_path.unlink()
        data_path = output_path.parent / (output_path.stem + ".onnx.data")
        if data_path.exists():
            data_path.unlink()
        print(f"  Removed existing: {output_path.name}")

    print("=" * 60)
    print(f"Exporting {info['description']}")
    print("=" * 60)

    pipe = load_sd_pipeline(use_lcm_lora=info["use_lcm_lora"])
    success = export_unet(pipe, output_path, variant_name)

    del pipe
    gc.collect()

    return success


def check_status():
    """Check which models are already exported."""
    print("=" * 60)
    print(f"SD v1.5 UNet Export Status (in {OUTPUT_DIR})")
    print("=" * 60)

    # UNet variants
    for name, info in VARIANTS.items():
        path = OUTPUT_DIR / info["filename"]
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            data_path = path.parent / (path.stem + ".onnx.data")
            total = size_mb
            if data_path.exists():
                total += data_path.stat().st_size / 1024 / 1024
            print(f"  [OK] {name}: {info['filename']} ({total:.1f} MB total)")
        else:
            print(f"  [--] {name}: {info['filename']} (not exported)")

    # Shared components (same directory)
    print(f"\n  Shared components:")
    for comp in ["vae_encoder_fp32.onnx", "text_encoder_fp32.onnx", "vae_decoder_fp32.onnx"]:
        path = OUTPUT_DIR / comp
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"    [OK] {comp} ({size_mb:.1f} MB)")
        else:
            print(f"    [--] {comp} (not found)")


def main():
    parser = argparse.ArgumentParser(
        description="Export SD v1.5 base and LCM-LoRA fused UNet to ONNX"
    )
    parser.add_argument(
        "--export",
        nargs="+",
        choices=list(VARIANTS.keys()),
        metavar="VARIANT",
        help="Variants to export: base, lcm",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all UNet variants",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check export status",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export even if output exists",
    )
    parser.add_argument(
        "--generate-calib-data",
        action="store_true",
        help="Generate real calibration NPZ for 4ch UNet (from COCO images + SD pipeline)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=64,
        help="Number of calibration samples (default: 64)",
    )
    parser.add_argument(
        "--calib-images-dir",
        type=Path,
        default=None,
        help=f"Directory with calibration images (default: {CALIB_IMAGES_DIR})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if args.output:
        _update_output_dir(args.output)

    if args.generate_calib_data:
        calib_dir = args.output or (PROJECT_ROOT / "weights" / "sd_v1.5" / "calib_data")
        generate_calibration_data(
            args.calib_samples, calib_dir, args.calib_images_dir
        )
        return

    if args.status:
        check_status()
        return

    variants = []
    if args.export_all:
        variants = list(VARIANTS.keys())
    elif args.export:
        variants = args.export

    if not variants:
        parser.print_help()
        print(f"\nVariants: {', '.join(VARIANTS.keys())}")
        print("\nExamples:")
        print("  python scripts/sd/export_sd_lcm_unet.py --export-all")
        print("  python scripts/sd/export_sd_lcm_unet.py --export base")
        print("  python scripts/sd/export_sd_lcm_unet.py --export lcm")
        print("  python scripts/sd/export_sd_lcm_unet.py --status")
        return

    results = []
    for variant in variants:
        success = export_variant(variant, force=args.force)
        results.append((variant, success))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for variant, success in results:
        info = VARIANTS[variant]
        path = OUTPUT_DIR / info["filename"]
        status = "[OK]" if success else "[FAIL]"
        if success and path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            data_path = path.parent / (path.stem + ".onnx.data")
            if data_path.exists():
                size_mb += data_path.stat().st_size / 1024 / 1024
            print(f"  {status} {variant}: {info['filename']} ({size_mb:.1f} MB)")
        else:
            print(f"  {status} {variant}: {info['filename']}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nNote: VAE/Text Encoder are shared between base and LCM variants.")
    print("  Export with: python scripts/sd/export_sd_to_onnx.py --export-all")


if __name__ == "__main__":
    main()
