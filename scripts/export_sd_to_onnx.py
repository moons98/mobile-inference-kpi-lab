#!/usr/bin/env python3
"""
Export Stable Diffusion v2.1 sub-models to ONNX and quantize.

Exports four pipeline components independently (img2img pipeline):
- VAE Encoder (AutoencoderKL encoder) - image -> latent
- Text Encoder (CLIP ViT-L/14)
- UNet (UNet2DConditionModel)
- VAE Decoder (AutoencoderKL decoder)

Precision variants:
- FP32: PyTorch default export (baseline, also used with useNpuFp16 runtime option)
- INT8 QDQ: Static quantization with calibration (QNN EP / NPU compatible)
  Uses get_qnn_qdq_config() + quantize() - same approach as YOLO quantization.
  W8A16 vs W8A8 distinction is handled by QAI Hub at QNN compilation level.

Note: FP16 is NOT a separate ONNX export. On-device FP16 inference is achieved
by running the FP32 model with the QNN EP `useNpuFp16` runtime option.

Usage:
    python export_sd_to_onnx.py --export-all
    python export_sd_to_onnx.py --export text_encoder --precision fp32
    python export_sd_to_onnx.py --export unet --precision int8
    python export_sd_to_onnx.py --export vae_encoder --precision fp32 int8
    python export_sd_to_onnx.py --status
    python export_sd_to_onnx.py --list
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

SD_MODEL_ID = "sd2-community/stable-diffusion-2-1"
SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR.parent / "weights" / "sd_v2.1" / "onnx"

# Component definitions
COMPONENTS = {
    "vae_encoder": {
        "description": "AutoencoderKL Encoder (img2img)",
        "params": "~34M",
        "input_names": ["sample"],
        "output_names": ["latent_sample"],
        "input_shapes": {"sample": [1, 3, 512, 512]},
        "dynamic_axes": None,
    },
    "text_encoder": {
        "description": "CLIP ViT-L/14 Text Encoder",
        "params": "~340M",
        "input_names": ["input_ids"],
        "output_names": ["last_hidden_state", "pooler_output"],
        "input_shapes": {"input_ids": [1, 77]},
        "dynamic_axes": None,
    },
    "unet": {
        "description": "UNet2D Conditional Denoiser",
        "params": "~860M",
        "input_names": ["sample", "timestep", "encoder_hidden_states"],
        "output_names": ["out_sample"],
        "input_shapes": {
            "sample": [1, 4, 64, 64],
            "timestep": [1],
            "encoder_hidden_states": [1, 77, 1024],
        },
        "dynamic_axes": None,
    },
    "vae_decoder": {
        "description": "AutoencoderKL Decoder",
        "params": "~83M",
        "input_names": ["latent_sample"],
        "output_names": ["sample"],
        "input_shapes": {"latent_sample": [1, 4, 64, 64]},
        "dynamic_axes": None,
    },
}

PRECISIONS = ["fp32", "int8"]

# Expected file sizes (approximate, for status check)
EXPECTED_SIZES_MB = {
    "vae_encoder":  {"fp32": 130, "int8": 35},
    "text_encoder": {"fp32": 1300, "int8": 350},
    "unet":         {"fp32": 3400, "int8": 900},
    "vae_decoder":  {"fp32": 190, "int8": 50},
}

# Calibration settings
CALIBRATION_SAMPLES = 20
CALIBRATION_METHOD = "percentile"  # "minmax" or "percentile"


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


WEIGHTS_DIR = SCRIPTS_DIR.parent / "weights" / "sd_v2.1"


def load_sd_pipeline():
    """Load SD v2.1 pipeline (FP32) from local weights or HuggingFace."""
    import torch
    from diffusers import StableDiffusionPipeline

    if WEIGHTS_DIR.exists() and (WEIGHTS_DIR / "model_index.json").exists():
        source = str(WEIGHTS_DIR)
        print(f"Loading from local weights: {WEIGHTS_DIR}")
    else:
        source = SD_MODEL_ID
        print(f"Loading from HuggingFace: {SD_MODEL_ID}")

    pipe = StableDiffusionPipeline.from_pretrained(
        source,
        torch_dtype=torch.float32,
    )
    print("  Pipeline loaded successfully")
    return pipe


# ============================================================
# ONNX Export Functions
# ============================================================

def export_vae_encoder(pipe, output_path: Path, precision: str) -> bool:
    """Export VAE Encoder to ONNX (for img2img pipeline)."""
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
        opset_version=17,
        do_constant_folding=True,
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
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported: {output_path.name} ({size_mb:.1f} MB)")
    return True


def export_unet(pipe, output_path: Path, precision: str) -> bool:
    """Export UNet to ONNX."""
    import torch

    print(f"\n  Exporting UNet ({precision})...")
    unet = pipe.unet
    unet.eval()
    unet = unet.to("cpu")

    dummy_sample = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    dummy_timestep = torch.tensor([1], dtype=torch.long)
    dummy_encoder_hidden = torch.randn(1, 77, 1024, dtype=torch.float32)

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
        opset_version=17,
        do_constant_folding=True,
    )

    # UNet is >2GB so torch exports with external data files.
    _merge_external_data(output_path)

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
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported: {output_path.name} ({size_mb:.1f} MB)")
    return True


def _merge_external_data(onnx_path: Path):
    """Merge ONNX model with scattered external data into .onnx + single .data file."""
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    model = onnx.load(str(onnx_path), load_external_data=True)

    parent = onnx_path.parent
    for f in list(parent.iterdir()):
        if f.is_file() and not f.suffix and f.name != onnx_path.name:
            f.unlink()

    ext_data_name = onnx_path.stem + ".onnx.data"
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=ext_data_name,
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.save(model, str(onnx_path))
    data_path = parent / ext_data_name
    data_mb = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
    print(f"  Merged external data: {onnx_path.name} + {ext_data_name} ({data_mb:.1f} MB)")


EXPORT_FUNCTIONS = {
    "vae_encoder": export_vae_encoder,
    "text_encoder": export_text_encoder,
    "unet": export_unet,
    "vae_decoder": export_vae_decoder,
}


# ============================================================
# Static INT8 QDQ Quantization (QNN EP compatible)
# Same approach as YOLO quantization using get_qnn_qdq_config()
# ============================================================

def _get_total_size(onnx_path: Path) -> float:
    """Get total size of ONNX model including external data files (MB)."""
    total = onnx_path.stat().st_size
    data_path = onnx_path.parent / (onnx_path.stem + ".onnx.data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total / 1024 / 1024


def _has_external_data(onnx_path: Path) -> bool:
    """Check if ONNX model uses external data."""
    data_path = onnx_path.parent / (onnx_path.stem + ".onnx.data")
    return data_path.exists()


class SdCalibrationDataReader:
    """Calibration data reader for SD pipeline components.

    Generates representative input data for static quantization calibration.
    Uses random data with appropriate distributions for each component type.
    """

    def __init__(self, component: str, num_samples: int = 20):
        self.data = self._generate_data(component, num_samples)
        self.iter = iter(self.data)

    def get_next(self):
        return next(self.iter, None)

    def rewind(self):
        self.iter = iter(self.data)

    def _generate_data(self, component, n):
        rng = np.random.default_rng(42)
        samples = []

        if component == "vae_encoder":
            # Normalized images in [-1, 1] range (SD VAE input)
            for _ in range(n):
                img = rng.uniform(-1, 1, size=(1, 3, 512, 512)).astype(np.float32)
                samples.append({"sample": img})

        elif component == "text_encoder":
            # Use real tokenized prompts for stable calibration (avoids inf outputs)
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
            calib_prompts = [
                "sunset beach with golden light",
                "snowy winter cityscape at night",
                "lush green forest with sunlight",
                "futuristic neon city",
                "soft watercolor painting style",
                "a photo of a cat sitting on a chair",
                "mountain landscape with clouds",
                "modern architecture building",
                "portrait of a woman in oil painting style",
                "colorful abstract geometric pattern",
                "a cozy coffee shop interior",
                "ocean waves crashing on rocks",
                "autumn leaves in a park",
                "vintage car on a country road",
                "night sky with stars and milky way",
                "japanese garden with cherry blossoms",
                "underwater coral reef scene",
                "medieval castle on a hilltop",
                "tropical rainforest with waterfall",
                "minimalist scandinavian living room",
            ]
            for i in range(n):
                prompt = calib_prompts[i % len(calib_prompts)]
                tok = tokenizer(
                    prompt, padding="max_length", max_length=77,
                    truncation=True, return_tensors="np",
                )
                samples.append({"input_ids": tok.input_ids.astype(np.int64)})

        elif component == "unet":
            # Latents (Gaussian), timesteps (uniform 1-1000), text embeddings (Gaussian)
            for _ in range(n):
                samples.append({
                    "sample": rng.standard_normal((1, 4, 64, 64)).astype(np.float32),
                    "timestep": np.array([rng.integers(1, 1000)], dtype=np.int64),
                    "encoder_hidden_states": rng.standard_normal((1, 77, 1024)).astype(np.float32),
                })

        elif component == "vae_decoder":
            # Latent samples (Gaussian, typical denoised latent range)
            for _ in range(n):
                samples.append({
                    "latent_sample": rng.standard_normal((1, 4, 64, 64)).astype(np.float32),
                })

        return samples


def preprocess_for_quantization(input_path: Path) -> Path:
    """Preprocess model before quantization (shape inference + optimization)."""
    preprocessed_path = input_path.parent / (input_path.stem + ".preprocessed.onnx")

    # Skip preprocessing for models with external data (UNet >2GB)
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


def quantize_static_qdq(fp32_path: Path, output_path: Path, component: str) -> bool:
    """Static INT8 QDQ quantization using QNN-optimized config.

    Uses get_qnn_qdq_config() + quantize() for NPU-compatible quantization.
    Same approach as YOLO quantization in this project.
    """
    try:
        from onnxruntime.quantization import quantize, CalibrationMethod
        from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
    except ImportError:
        print("  Error: onnxruntime.quantization with QNN support not available")
        print("  Install: pip install onnxruntime")
        return False

    print(f"\n  Static quantizing {component} to INT8 QDQ...")
    print(f"    Input:  {fp32_path.name}")
    print(f"    Output: {output_path.name}")

    # Preprocess model
    preprocessed_path = preprocess_for_quantization(fp32_path)

    # Calibration data
    num_samples = 10 if component == "unet" else CALIBRATION_SAMPLES
    print(f"    Calibration: {num_samples} samples ({CALIBRATION_METHOD})")
    calibrator = SdCalibrationDataReader(component, num_samples)

    # Calibration method
    if CALIBRATION_METHOD == "minmax":
        calib_method = CalibrationMethod.MinMax
    else:
        calib_method = CalibrationMethod.Percentile

    # Build QNN-optimized QDQ config
    qnn_config = get_qnn_qdq_config(
        model_input=str(preprocessed_path),
        calibration_data_reader=calibrator,
        calibrate_method=calib_method,
        activation_symmetric=False,
    )

    quantize(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        quant_config=qnn_config,
    )

    # Cleanup preprocessed file
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

def export_component(component: str, precisions: list) -> list:
    """Export a single SD component in specified precisions.

    Strategy:
    - FP32: Export from PyTorch (base model, also for useNpuFp16)
    - INT8: Static QDQ quantization from FP32 base (for NPU INT8)
    """
    exported = []
    fp32_path = get_output_path(component, "fp32")
    need_fp32 = any(p in precisions for p in ["fp32", "int8"])

    # Step 1: Export FP32 (base model)
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

    # Step 2: Static INT8 QDQ quantization
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
            if quantize_static_qdq(fp32_path, int8_path, component):
                exported.append((component, "int8"))

    return exported


def list_components():
    """List available components and precisions."""
    print("=" * 60)
    print(f"SD v2.1 Components ({SD_MODEL_ID})")
    print("=" * 60)
    for name, comp in COMPONENTS.items():
        print(f"\n  {name}:")
        print(f"    Description: {comp['description']}")
        print(f"    Parameters:  {comp['params']}")
        print(f"    Inputs:      {comp['input_names']}")
        print(f"    Outputs:     {comp['output_names']}")
        sizes = EXPECTED_SIZES_MB[name]
        print(f"    Expected sizes: FP32 ~{sizes['fp32']}MB | INT8 QDQ ~{sizes['int8']}MB")

    print("\n  Precisions:")
    print("    fp32  - Full precision (baseline, also for useNpuFp16 runtime FP16)")
    print("    int8  - Static INT8 QDQ (QNN EP / NPU compatible, calibrated)")
    print("\n  On-device precision mapping:")
    print("    FP16  = FP32 model + useNpuFp16 EP option")
    print("    W8A16 = QAI Hub compilation of FP32 or INT8 model")
    print("    W8A8  = INT8 QDQ model on QNN EP")


def check_status():
    """Check which models are already exported."""
    print("=" * 60)
    print(f"SD v2.1 Model Status (in {OUTPUT_DIR})")
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


def main():
    global CALIBRATION_SAMPLES, CALIBRATION_METHOD

    parser = argparse.ArgumentParser(
        description="Export Stable Diffusion v2.1 sub-models to ONNX with quantization"
    )
    parser.add_argument(
        "--export",
        nargs="+",
        choices=list(COMPONENTS.keys()),
        metavar="COMPONENT",
        help="Components to export (vae_encoder, text_encoder, unet, vae_decoder)"
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=PRECISIONS,
        default=PRECISIONS,
        help="Precision variants to generate (default: all)"
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all components in all precisions"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available components"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check which models are already exported"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export even if output file already exists"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=CALIBRATION_SAMPLES,
        help=f"Number of calibration samples for INT8 quantization (default: {CALIBRATION_SAMPLES})"
    )
    parser.add_argument(
        "--calibration-method",
        choices=["minmax", "percentile"],
        default=CALIBRATION_METHOD,
        help="Calibration method (default: percentile)"
    )

    args = parser.parse_args()

    if args.output:
        _update_output_dir(args.output)

    CALIBRATION_SAMPLES = args.calibration_samples
    CALIBRATION_METHOD = args.calibration_method

    if args.list:
        list_components()
        return

    if args.status:
        check_status()
        return

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
        print("  python export_sd_to_onnx.py --export-all")
        print("  python export_sd_to_onnx.py --export text_encoder unet --precision int8")
        print("  python export_sd_to_onnx.py --export vae_encoder --precision fp32 int8")
        print("  python export_sd_to_onnx.py --status")
        return

    all_exported = []
    all_failed = []

    for component in components_to_export:
        results = export_component(component, args.precision)
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
    print("\nNext steps:")
    print("  - Run eval_sd_quality.py to compare FP32 vs INT8 QDQ image quality")
    print("  - Use analyze_ops.py to check NPU compatibility of each component")
    print("  - FP16 on-device: use FP32 model with useNpuFp16 QNN EP option")


if __name__ == "__main__":
    main()
