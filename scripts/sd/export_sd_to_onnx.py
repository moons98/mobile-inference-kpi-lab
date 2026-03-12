#!/usr/bin/env python3
"""
Export SD v1.5 shared components (VAE Encoder/Decoder, Text Encoder) to ONNX.

These components are shared between SD v1.5 base and LCM-LoRA variants.
UNet export is handled separately by export_sd_lcm_unet.py.

Precision variants:
- FP32: PyTorch default export (baseline, also used with useNpuFp16 runtime option)
- INT8 QDQ: Static quantization with calibration (QNN EP / NPU compatible)

Usage:
    python scripts/sd/export_sd_to_onnx.py --export-all
    python scripts/sd/export_sd_to_onnx.py --export text_encoder --precision fp32
    python scripts/sd/export_sd_to_onnx.py --export vae_encoder --precision fp32 int8
    python scripts/sd/export_sd_to_onnx.py --status

    # Real-data calibration workflow:
    python scripts/sd/export_sd_to_onnx.py --generate-calib-data
    python scripts/sd/export_sd_to_onnx.py --export vae_encoder --precision int8 --calib-data weights/sd_v1.5/calib_data
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "weights" / "sd_v1.5" / "onnx"
CALIB_IMAGES_DIR = PROJECT_ROOT / "datasets" / "coco" / "val2017"
SEED = 42

# Representative prompts for calibration (diverse scenes/styles/objects)
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

# Component definitions (shared components only — UNet handled by export_sd_lcm_unet.py)
COMPONENTS = {
    "vae_encoder": {
        "description": "AutoencoderKL Encoder",
        "params": "~34M",
        "input_names": ["sample"],
        "output_names": ["latent_sample"],
        "input_shapes": {"sample": [1, 3, 512, 512]},
    },
    "text_encoder": {
        "description": "CLIP ViT-L/14 Text Encoder",
        "params": "~123M",
        "input_names": ["input_ids"],
        "output_names": ["last_hidden_state", "pooler_output"],
        "input_shapes": {"input_ids": [1, 77]},
    },
    "vae_decoder": {
        "description": "AutoencoderKL Decoder",
        "params": "~83M",
        "input_names": ["latent_sample"],
        "output_names": ["sample"],
        "input_shapes": {"latent_sample": [1, 4, 64, 64]},
    },
}

PRECISIONS = ["fp32", "int8"]

EXPECTED_SIZES_MB = {
    "vae_encoder":  {"fp32": 130, "int8": 35},
    "text_encoder": {"fp32": 470, "int8": 125},
    "vae_decoder":  {"fp32": 190, "int8": 50},
}

# Calibration settings
CALIBRATION_SAMPLES = 8
CALIBRATION_METHOD = "percentile"
CALIBRATION_STREAMING_CHUNK = 2


def _update_output_dir(new_dir):
    global OUTPUT_DIR
    if new_dir:
        OUTPUT_DIR = new_dir


def get_output_filename(component: str, precision: str) -> str:
    if precision == "int8":
        return f"{component}_int8_qdq.onnx"
    return f"{component}_{precision}.onnx"


def get_output_path(component: str, precision: str) -> Path:
    return OUTPUT_DIR / get_output_filename(component, precision)


def load_sd_pipeline():
    """Load SD v1.5 pipeline (FP32)."""
    import torch
    from diffusers import StableDiffusionPipeline

    print(f"Loading from HuggingFace: {SD_MODEL_ID}")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float32,
    )
    print("  Pipeline loaded successfully")
    return pipe


# ============================================================
# ONNX Export Functions
# ============================================================

def export_vae_encoder(pipe, output_path: Path, precision: str) -> bool:
    """Export VAE Encoder to ONNX."""
    import torch

    print(f"\n  Exporting VAE Encoder ({precision})...")
    vae = pipe.vae
    vae.eval()
    vae = vae.to("cpu")

    dummy_image = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, sample):
            h = self.vae.encoder(sample)
            moments = self.vae.quant_conv(h)
            mean, _ = torch.chunk(moments, 2, dim=1)
            return mean * self.vae.config.scaling_factor

    wrapper = VAEEncoderWrapper(vae)

    torch.onnx.export(
        wrapper,
        (dummy_image,),
        str(output_path),
        input_names=["sample"],
        output_names=["latent_sample"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported: {output_path.name} ({size_mb:.1f} MB)")
    return True


def export_text_encoder(pipe, output_path: Path, precision: str) -> bool:
    """Export CLIP Text Encoder to ONNX."""
    import torch

    print(f"\n  Exporting Text Encoder ({precision})...")
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    text_encoder = text_encoder.to("cpu")

    dummy_input = torch.zeros(1, 77, dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        text_encoder,
        (dummy_input,),
        str(output_path),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported: {output_path.name} ({size_mb:.1f} MB)")
    return True


def export_vae_decoder(pipe, output_path: Path, precision: str) -> bool:
    """Export VAE Decoder to ONNX."""
    import torch

    print(f"\n  Exporting VAE Decoder ({precision})...")
    vae = pipe.vae
    vae.eval()
    vae = vae.to("cpu")

    dummy_latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample):
            latent_sample = latent_sample / self.vae.config.scaling_factor
            return self.vae.decode(latent_sample, return_dict=False)[0]

    wrapper = VAEDecoderWrapper(vae)

    torch.onnx.export(
        wrapper,
        (dummy_latent,),
        str(output_path),
        input_names=["latent_sample"],
        output_names=["sample"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported: {output_path.name} ({size_mb:.1f} MB)")
    return True


EXPORT_FUNCTIONS = {
    "vae_encoder": export_vae_encoder,
    "text_encoder": export_text_encoder,
    "vae_decoder": export_vae_decoder,
}


# ============================================================
# Static INT8 QDQ Quantization
# ============================================================

def _get_total_size(onnx_path: Path) -> float:
    total = onnx_path.stat().st_size
    data_path = onnx_path.parent / (onnx_path.stem + ".onnx.data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total / 1024 / 1024


def _has_external_data(onnx_path: Path) -> bool:
    data_path = onnx_path.parent / (onnx_path.stem + ".onnx.data")
    return data_path.exists()


class SdCalibrationDataReader:
    """Calibration data reader loading pre-generated NPZ files."""

    def __init__(self, component: str, num_samples: int = 20, calib_data_dir=None):
        self._real_arrays = None
        self._keys = []
        self._num_samples = 0
        self._index = 0
        if not calib_data_dir:
            raise ValueError("Calibration NPZ directory is required. Use --calib-data <dir>.")

        npz_path = Path(calib_data_dir) / f"calib_{component}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Calibration data not found: {npz_path}")
        with np.load(npz_path) as npz:
            self._keys = list(npz.keys())
            self._real_arrays = {key: npz[key] for key in self._keys}
        if not self._keys:
            raise ValueError(f"No tensors found in calibration file: {npz_path}")
        available = len(self._real_arrays[self._keys[0]])
        self._num_samples = min(num_samples, available)
        print(
            f"    Loaded {self._num_samples}/{available} real calibration samples "
            f"from {npz_path.name} (cached arrays)"
        )

    def get_next(self):
        if self._index >= self._num_samples:
            return None
        i = self._index
        self._index += 1
        return {
            key: np.ascontiguousarray(self._real_arrays[key][i:i + 1])
            for key in self._keys
        }

    def rewind(self):
        self._index = 0


def _select_calibration_method(component: str, calibration_method: str):
    from onnxruntime.quantization import CalibrationMethod

    if component == "text_encoder":
        return CalibrationMethod.MinMax, "text_encoder has inf mask values with Percentile"
    if calibration_method == "minmax":
        return CalibrationMethod.MinMax, None
    return CalibrationMethod.Percentile, None


def _enable_histogram_streaming_patch(chunk_size: int = 1):
    """Patch ORT HistogramCalibrater to collect histogram incrementally."""
    import copy
    from onnxruntime.quantization import calibrate as ort_calib

    chunk_size = max(1, int(chunk_size))
    current = ort_calib.HistogramCalibrater.collect_data
    if getattr(current, "_streaming_patch", False):
        return

    def _collect_data_streaming(self, data_reader):
        input_names_set = {node_arg.name for node_arg in self.infer_session.get_inputs()}
        output_names = [node_arg.name for node_arg in self.infer_session.get_outputs()]

        if not self.collector:
            self.collector = ort_calib.HistogramCollector(
                method=self.method,
                symmetric=self.symmetric,
                num_bins=self.num_bins,
                num_quantized_bins=self.num_quantized_bins,
                percentile=self.percentile,
                scenario=self.scenario,
            )

        seen = 0
        pending = {}
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            outputs = self.infer_session.run(None, inputs)
            for output_index, output in enumerate(outputs):
                name = output_names[output_index]
                if name not in self.tensors_to_calibrate:
                    continue
                if name in input_names_set:
                    output = copy.copy(output)
                pending.setdefault(name, []).append(output)

            seen += 1
            if seen % chunk_size == 0:
                self.collector.collect(pending)
                pending = {}
            del outputs

        if pending:
            self.collector.collect(pending)
        if seen == 0:
            raise ValueError("No data is collected.")
        self.clear_collected_data()

    _collect_data_streaming._streaming_patch = True
    ort_calib.HistogramCalibrater.collect_data = _collect_data_streaming


def _load_coco_images(num_images: int, image_dir: Path, seed: int = SEED):
    """Load and preprocess COCO images for calibration.
    Returns list of numpy arrays in [-1, 1] range, shape (1, 3, 512, 512).
    """
    from PIL import Image

    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPEG images found in {image_dir}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(image_files), size=min(num_images, len(image_files)),
                         replace=False)
    indices.sort()

    images = []
    for idx in indices:
        img = Image.open(image_files[idx]).convert("RGB")
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        img = img.resize((512, 512), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]
        images.append(arr)

    print(f"  Loaded {len(images)} COCO images from {image_dir.name}/")
    return images


def generate_calibration_data(num_samples: int, output_dir: Path,
                              image_dir: Path = None):
    """Generate real calibration data for shared components (VAE/TextEncoder).

    Produces per-component NPZ files:
    - calib_vae_encoder.npz: real COCO images (512x512, [-1,1])
    - calib_text_encoder.npz: diverse real prompts (tokenized)
    - calib_vae_decoder.npz: VAE-encoded real image latents

    UNet calibration data is generated by export_sd_lcm_unet.py --generate-calib-data.
    """
    import torch

    image_dir = image_dir or CALIB_IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Generating shared component calibration data ({num_samples} samples)")
    print("=" * 60)

    coco_images = _load_coco_images(num_samples, image_dir)
    num_samples = len(coco_images)

    pipe = load_sd_pipeline()
    vae = pipe.vae.eval()
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer

    gen = torch.Generator().manual_seed(SEED)

    # 1. Text encoder: diverse real prompts
    print("\n  [1/3] Text encoder: tokenizing prompts...")
    text_enc_input_ids = []
    with torch.no_grad():
        for i in range(num_samples):
            prompt = CALIB_PROMPTS[i % len(CALIB_PROMPTS)]
            tok = tokenizer(
                prompt, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt",
            )
            text_enc_input_ids.append(tok.input_ids.numpy())

    text_enc_input_ids = np.concatenate(text_enc_input_ids, axis=0)
    np.savez(output_dir / "calib_text_encoder.npz",
             input_ids=text_enc_input_ids)
    print(f"    {num_samples} prompts -> input_ids {text_enc_input_ids.shape}")

    # 2. VAE encoder: real COCO images
    print("\n  [2/3] VAE encoder: real COCO images...")
    vae_enc_samples = np.concatenate(coco_images, axis=0)
    np.savez(output_dir / "calib_vae_encoder.npz", sample=vae_enc_samples)
    print(f"    {num_samples} images -> {vae_enc_samples.shape}")

    # 3. VAE decoder: encode real images -> real latents
    print("\n  [3/3] VAE decoder: encoding images to latents...")
    vae_dec_samples = []
    with torch.no_grad():
        for img_np in coco_images:
            img_tensor = torch.from_numpy(img_np)
            latent = vae.encode(img_tensor).latent_dist.sample(gen)
            latent = latent * vae.config.scaling_factor
            vae_dec_samples.append(latent.numpy())

    vae_dec_samples = np.concatenate(vae_dec_samples, axis=0)
    np.savez(output_dir / "calib_vae_decoder.npz", latent_sample=vae_dec_samples)
    print(f"    {num_samples} latents -> {vae_dec_samples.shape}")

    del pipe
    gc.collect()

    # Report
    print("\n" + "-" * 40)
    total_mb = 0
    for f in sorted(output_dir.glob("calib_*.npz")):
        sz = f.stat().st_size / 1024 / 1024
        total_mb += sz
        print(f"  {f.name}: {sz:.1f} MB")
    print(f"  Total: {total_mb:.1f} MB")
    print(f"\nCalibration data saved to {output_dir}")
    print("\nNote: UNet calibration -> use export_sd_lcm_unet.py --generate-calib-data")


def preprocess_for_quantization(input_path: Path) -> Path:
    """Preprocess model before quantization (shape inference + optimization)."""
    preprocessed_path = input_path.parent / (input_path.stem + ".preprocessed.onnx")

    if _has_external_data(input_path):
        print("  Skipping preprocessing (model has external data)")
        return input_path

    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(
            input_model_path=str(input_path),
            output_model_path=str(preprocessed_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
        )
        print("  [OK] Preprocessing complete (shape inference + optimization)")
        return preprocessed_path

    except Exception as e:
        print(f"  [!] Preprocessing failed: {e}")
        print("  Proceeding without preprocessing...")
        return input_path


def quantize_static_qdq(fp32_path: Path, output_path: Path, component: str,
                        calib_data_dir=None) -> bool:
    """Static INT8 QDQ quantization using quantize_static()."""
    try:
        from onnxruntime.quantization import (
            quantize_static, QuantFormat, QuantType, CalibrationMethod,
        )
    except ImportError:
        print("  Error: onnxruntime.quantization not available")
        return False

    print(f"\n  Static quantizing {component} to INT8 QDQ...")
    print(f"    Input:  {fp32_path.name}")
    print(f"    Output: {output_path.name}")

    preprocessed_path = preprocess_for_quantization(fp32_path)

    if not calib_data_dir:
        raise ValueError("INT8 quantization requires --calib-data <dir> with calib_*.npz files.")

    num_samples = CALIBRATION_SAMPLES
    print(f"    Calibration: {num_samples} samples ({CALIBRATION_METHOD}, real NPZ)")
    calibrator = SdCalibrationDataReader(component, num_samples, calib_data_dir)

    calib_method, reason = _select_calibration_method(component, CALIBRATION_METHOD)
    if reason:
        print(f"    Calibration override: {reason}")

    if calib_method == CalibrationMethod.Percentile:
        _enable_histogram_streaming_patch(CALIBRATION_STREAMING_CHUNK)
        print(f"    Histogram collection: streaming (chunk={CALIBRATION_STREAMING_CHUNK})")

    extra_options = {}
    if calib_method == CalibrationMethod.MinMax:
        extra_options["CalibMovingAverage"] = True
        extra_options["CalibMovingAverageConstant"] = 0.01
    else:
        extra_options["num_bins"] = 2048
        extra_options["percentile"] = 99.999

    use_external = _has_external_data(preprocessed_path)

    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        calibration_data_reader=calibrator,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=calib_method,
        use_external_data_format=use_external,
        extra_options=extra_options,
    )

    if preprocessed_path != fp32_path and preprocessed_path.exists():
        preprocessed_path.unlink()

    if not output_path.exists():
        print("  Error: Quantization produced no output")
        return False

    fp32_size = _get_total_size(fp32_path)
    int8_size = _get_total_size(output_path)
    ratio = int8_size / fp32_size * 100 if fp32_size > 0 else 0

    print(f"  Quantized: {output_path.name}")
    print(f"    FP32:  {fp32_size:.1f} MB")
    print(f"    INT8:  {int8_size:.1f} MB ({ratio:.0f}% of FP32)")
    return True


# ============================================================
# Export Orchestration
# ============================================================

def export_component(component: str, precisions: list, calib_data_dir=None) -> list:
    """Export a single SD component in specified precisions."""
    exported = []
    fp32_path = get_output_path(component, "fp32")
    need_fp32 = any(p in precisions for p in ["fp32", "int8"])

    if need_fp32 and not fp32_path.exists():
        print("=" * 60)
        print(f"Exporting {COMPONENTS[component]['description']} (FP32 base)")
        print("=" * 60)

        pipe = load_sd_pipeline()
        export_fn = EXPORT_FUNCTIONS[component]
        success = export_fn(pipe, fp32_path, "fp32")

        del pipe
        gc.collect()

        if not success:
            print(f"  FAILED: {component} FP32 export")
            return exported
    elif need_fp32 and fp32_path.exists():
        size_mb = fp32_path.stat().st_size / 1024 / 1024
        print(f"  Reusing existing FP32: {fp32_path.name} ({size_mb:.1f} MB)")

    if "fp32" in precisions:
        exported.append((component, "fp32"))

    if "int8" in precisions:
        int8_path = get_output_path(component, "int8")
        if int8_path.exists():
            size_mb = _get_total_size(int8_path)
            print(f"  Reusing existing INT8 QDQ: {int8_path.name} ({size_mb:.1f} MB)")
            exported.append((component, "int8"))
        else:
            print("=" * 60)
            print(f"Quantizing {COMPONENTS[component]['description']} to INT8 QDQ")
            print("=" * 60)
            if quantize_static_qdq(fp32_path, int8_path, component, calib_data_dir):
                exported.append((component, "int8"))

    return exported


def list_components():
    """List available components and precisions."""
    print("=" * 60)
    print(f"SD v1.5 Shared Components ({SD_MODEL_ID})")
    print("=" * 60)
    for name, comp in COMPONENTS.items():
        print(f"\n  {name}:")
        print(f"    Description: {comp['description']}")
        print(f"    Parameters:  {comp['params']}")
        print(f"    Inputs:      {comp['input_names']}")
        print(f"    Outputs:     {comp['output_names']}")
        sizes = EXPECTED_SIZES_MB[name]
        print(f"    Expected sizes: FP32 ~{sizes['fp32']}MB | INT8 QDQ ~{sizes['int8']}MB")

    print("\n  UNet export: use export_sd_lcm_unet.py (base + LCM-LoRA variants)")


def check_status():
    """Check which models are already exported."""
    print("=" * 60)
    print(f"SD v1.5 Shared Component Status (in {OUTPUT_DIR})")
    print("=" * 60)

    total_size = 0
    for component in COMPONENTS:
        print(f"\n  {component}:")
        for precision in PRECISIONS:
            path = get_output_path(component, precision)
            if path.exists():
                size_mb = _get_total_size(path)
                total_size += size_mb
                print(f"    [OK] {path.name} ({size_mb:.1f} MB)")
            else:
                expected = EXPECTED_SIZES_MB[component][precision]
                print(f"    [--] {get_output_filename(component, precision)} (expected ~{expected} MB)")

    if total_size > 0:
        print(f"\n  Total on disk: {total_size:.1f} MB")

    print(f"\n  UNet status: use export_sd_lcm_unet.py --status")


def main():
    global CALIBRATION_SAMPLES, CALIBRATION_METHOD, CALIBRATION_STREAMING_CHUNK

    parser = argparse.ArgumentParser(
        description="Export SD v1.5 shared components (VAE/TextEncoder) to ONNX"
    )
    parser.add_argument(
        "--export", nargs="+", choices=list(COMPONENTS.keys()),
        metavar="COMPONENT",
        help="Components to export (vae_encoder, text_encoder, vae_decoder)"
    )
    parser.add_argument(
        "--precision", nargs="+", choices=PRECISIONS, default=PRECISIONS,
        help="Precision variants to generate (default: all)"
    )
    parser.add_argument("--export-all", action="store_true",
                        help="Export all components in all precisions")
    parser.add_argument("--list", action="store_true",
                        help="List available components")
    parser.add_argument("--status", action="store_true",
                        help="Check which models are already exported")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if output file already exists")
    parser.add_argument("--calibration-samples", type=int, default=CALIBRATION_SAMPLES,
                        help=f"Number of calibration samples (default: {CALIBRATION_SAMPLES})")
    parser.add_argument("--calibration-method", choices=["minmax", "percentile"],
                        default=CALIBRATION_METHOD)
    parser.add_argument("--calibration-streaming-chunk", type=int,
                        default=CALIBRATION_STREAMING_CHUNK)
    parser.add_argument("--generate-calib-data", action="store_true",
                        help="Generate real calibration NPZ files from COCO images + SD pipeline")
    parser.add_argument("--calib-images-dir", type=Path, default=None,
                        help=f"Directory with calibration images (default: {CALIB_IMAGES_DIR})")
    parser.add_argument("--calib-data", type=Path, default=None,
                        help="Directory containing calib_*.npz files for quantization")

    args = parser.parse_args()

    if args.output:
        _update_output_dir(args.output)

    CALIBRATION_SAMPLES = args.calibration_samples
    CALIBRATION_METHOD = args.calibration_method
    CALIBRATION_STREAMING_CHUNK = max(1, args.calibration_streaming_chunk)

    if args.generate_calib_data:
        calib_dir = args.output or (PROJECT_ROOT / "weights" / "sd_v1.5" / "calib_data")
        generate_calibration_data(
            CALIBRATION_SAMPLES, calib_dir, args.calib_images_dir
        )
        return

    if args.list:
        list_components()
        return

    if args.status:
        check_status()
        return

    if "int8" in args.precision and args.calib_data is None:
        parser.error(
            "INT8 quantization requires --calib-data <dir>. "
            "Generate first with --generate-calib-data."
        )
    if args.calib_data is not None and not args.calib_data.exists():
        parser.error(f"--calib-data directory does not exist: {args.calib_data}")

    if args.force:
        for comp in (args.export or list(COMPONENTS.keys()) if args.export_all else []):
            for prec in args.precision:
                path = get_output_path(comp, prec)
                if path.exists():
                    path.unlink()
                    print(f"  Removed: {path.name}")

    components_to_export = []
    if args.export_all:
        components_to_export = list(COMPONENTS.keys())
    elif args.export:
        components_to_export = args.export

    if not components_to_export:
        parser.print_help()
        print(f"\nAvailable components: {', '.join(COMPONENTS.keys())}")
        print("\nExamples:")
        print("  python scripts/sd/export_sd_to_onnx.py --export-all")
        print("  python scripts/sd/export_sd_to_onnx.py --export text_encoder --precision fp32")
        print("  python scripts/sd/export_sd_to_onnx.py --export vae_encoder --precision fp32 int8")
        print("  python scripts/sd/export_sd_to_onnx.py --status")
        return

    all_exported = []
    all_failed = []

    for component in components_to_export:
        results = export_component(component, args.precision, args.calib_data)
        if results:
            all_exported.extend(results)
        else:
            all_failed.append(component)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if all_exported:
        print("Exported:")
        for comp, prec in all_exported:
            path = get_output_path(comp, prec)
            size_mb = _get_total_size(path) if path.exists() else 0
            print(f"  {comp} [{prec}] - {size_mb:.1f} MB")

    if all_failed:
        print(f"Failed: {', '.join(all_failed)}")

    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
