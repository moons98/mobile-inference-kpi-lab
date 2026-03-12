#!/usr/bin/env python3
"""
Export SD v1.5 + LCM-LoRA fused UNet with W8A16 quantization via AIMET,
then compile on QAI Hub with QAIRT 2.42.

Pipeline:
  1. Load SD v1.5 base + fuse LCM-LoRA (PyTorch)
  2. Export to ONNX (opset 18) with external data
  3. Simplify with onnxsim (load from disk with external data support)
  4. Apply AIMET W8A16 quantization (QuantSimOnnx)
  5. Export quantized ONNX
  6. Compile on QAI Hub (QAIRT 2.42, Samsung Galaxy S23)

Usage (on Linux with aimet-onnx):
    python export_sd_lcm_unet_w8a16.py
"""

import gc
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

UNET_INPUT_SHAPES = {
    "sample": [1, 4, 64, 64],
    "timestep": [1],
    "encoder_hidden_states": [1, 77, 768],
}

WORK_DIR = Path("/root/unet_lcm_w8a16_work")
OUTPUT_DIR = WORK_DIR / "output"


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


def step1_export_and_quantize():
    """Export UNet to ONNX, load into memory, delete disk files, then apply AIMET W8A16."""
    import onnx
    from diffusers import StableDiffusionPipeline
    from aimet_common.defs import QuantScheme
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

    quantized_onnx = OUTPUT_DIR / "unet_lcm_w8a16.onnx"
    if quantized_onnx.exists():
        print(f"[Step 1] Reusing existing: {quantized_onnx}")
        return quantized_onnx

    # --- Phase 1: PyTorch → ONNX export ---
    fp32_path = WORK_DIR / "unet_lcm_fp32.onnx"

    if not fp32_path.exists():
        print("[Step 1a] Loading SD v1.5 + LCM-LoRA...")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID, torch_dtype=torch.float32
        )
        pipe.load_lora_weights(LCM_LORA_ID)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        print("  LCM-LoRA fused")

        unet = pipe.unet.eval().to("cpu")
        wrapper = UNetWrapper(unet)

        dummy_sample = torch.randn(*UNET_INPUT_SHAPES["sample"])
        dummy_timestep = torch.tensor([1], dtype=torch.long)
        dummy_hidden = torch.randn(*UNET_INPUT_SHAPES["encoder_hidden_states"])

        WORK_DIR.mkdir(parents=True, exist_ok=True)

        print("  Exporting to ONNX (opset 18)...")
        torch.onnx.export(
            wrapper,
            (dummy_sample, dummy_timestep, dummy_hidden),
            str(fp32_path),
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

        del pipe, unet, wrapper
        gc.collect()
        print(f"  Exported: {fp32_path.name}")
    else:
        print(f"[Step 1a] Reusing existing ONNX: {fp32_path}")

    # --- Phase 2: Load into memory, then delete disk files to free space ---
    print("[Step 1b] Loading ONNX into memory...")
    model = onnx.load(str(fp32_path), load_external_data=True)
    print(f"  Model loaded: ir_version={model.ir_version}, opset={model.opset_import[0].version}")
    print(f"  {len(model.graph.initializer)} initializers")

    # Delete ALL step1 files from disk to free ~3.2GB before step2 writes ~6.5GB
    import shutil
    for f in WORK_DIR.iterdir():
        if f.name != "output" and not f.is_dir():
            f.unlink()
    # Also clean HF cache
    hf_cache = Path("/root/.cache/huggingface")
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
    print("  Disk files cleaned (model is in memory)")

    # --- Phase 3: AIMET W8A16 quantization ---
    print("[Step 1c] Applying AIMET W8A16 quantization...")

    # Find AIMET default config
    import qai_hub_models
    qai_hub_models_dir = Path(qai_hub_models.__file__).parent
    config_path = qai_hub_models_dir / "aimet" / "default_per_tensor_config_v69.json"
    if not config_path.exists():
        # Try alternative paths
        for candidate in qai_hub_models_dir.rglob("default_per_tensor_config*.json"):
            config_path = candidate
            break
    print(f"  AIMET config: {config_path}")

    # Create QuantSim with W8A16 config
    # W8A16: weights int8, activations fp16 (int16 in AIMET terms)
    sim = QuantSimOnnx(
        model=model,
        quant_scheme=QuantScheme.post_training_tf,
        default_param_bw=8,
        default_activation_bw=16,
        config_file=str(config_path),
    )

    # For W8A16 (weight-only), no calibration needed
    # Just need to compute encodings based on weight values
    # Use dummy forward pass to initialize
    dummy_inputs = {
        "sample": np.random.randn(*UNET_INPUT_SHAPES["sample"]).astype(np.float32),
        "timestep": np.array([1], dtype=np.int64),
        "encoder_hidden_states": np.random.randn(
            *UNET_INPUT_SHAPES["encoder_hidden_states"]
        ).astype(np.float32),
    }

    def dummy_forward(session, args):
        session.run(None, dummy_inputs)

    print("  Computing quantization encodings...")
    sim.compute_encodings(dummy_forward, None)

    # Export quantized model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("  Exporting quantized model...")
    sim.export(str(OUTPUT_DIR), "unet_lcm_w8a16")

    # The exported file should be at OUTPUT_DIR/unet_lcm_w8a16.onnx
    exported = OUTPUT_DIR / "unet_lcm_w8a16.onnx"
    if exported.exists():
        size_mb = exported.stat().st_size / 1024 / 1024
        # Check for external data
        data_file = OUTPUT_DIR / "unet_lcm_w8a16.onnx.data"
        if data_file.exists():
            data_mb = data_file.stat().st_size / 1024 / 1024
            print(f"  Quantized: {exported.name} ({size_mb:.1f} MB) + .data ({data_mb:.0f} MB)")
        else:
            print(f"  Quantized: {exported.name} ({size_mb:.1f} MB)")
    else:
        print(f"  WARNING: Expected output at {exported}, checking alternatives...")
        for f in OUTPUT_DIR.glob("unet_lcm_w8a16*"):
            print(f"    Found: {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    del sim, model
    gc.collect()

    return exported


def step2_compile_qai_hub(quantized_path: Path):
    """Compile on QAI Hub with QAIRT 2.42."""
    import qai_hub

    print("[Step 2] Compiling on QAI Hub (QAIRT 2.42, Galaxy S23)...")

    # .encodings must be in the same directory as the ONNX for QNN to apply quantization
    enc_file = quantized_path.parent / (quantized_path.stem + ".encodings")
    if enc_file.exists():
        print(f"  .encodings found: {enc_file.name}")
    else:
        print("  WARNING: .encodings not found — QNN will compile as FP32!")

    # Check for external data (.data or .onnx.data)
    data_file = quantized_path.parent / (quantized_path.stem + ".data")
    data_file2 = quantized_path.parent / (quantized_path.stem + ".onnx.data")
    if data_file.exists() or data_file2.exists():
        # Upload directory (onnx + data in same folder)
        print(f"  External data found, uploading directory: {quantized_path.parent}")
        model_input = str(quantized_path.parent)
    else:
        model_input = str(quantized_path)

    compile_job = qai_hub.submit_compile_job(
        model=model_input,
        device=qai_hub.Device("Samsung Galaxy S23"),
        options="--target_runtime precompiled_qnn_onnx --qairt_version 2.42 --truncate_64bit_io",
        input_specs=dict(
            sample=(tuple(UNET_INPUT_SHAPES["sample"]), "float32"),
            timestep=(tuple(UNET_INPUT_SHAPES["timestep"]), "int64"),
            encoder_hidden_states=(tuple(UNET_INPUT_SHAPES["encoder_hidden_states"]), "float32"),
        ),
        name="unet_lcm_w8a16_qairt242",
    )

    print(f"  Compile job submitted: {compile_job.job_id}")
    print(f"  URL: {compile_job.url}")
    print("  Waiting for completion...")

    compile_job.wait()

    if compile_job.get_status().success:
        print(f"  Compile SUCCESS: {compile_job.job_id}")
        # Download compiled model
        compiled_dir = OUTPUT_DIR / f"compiled_{compile_job.job_id}"
        compiled_dir.mkdir(parents=True, exist_ok=True)
        target = compile_job.get_target_model()
        target.download(str(compiled_dir / "model.bin"))
        bin_size = (compiled_dir / "model.bin").stat().st_size / 1024 / 1024
        print(f"  Downloaded: {compiled_dir}/model.bin ({bin_size:.0f} MB)")
    else:
        print(f"  Compile FAILED: {compile_job.job_id}")
        print(f"  Check: {compile_job.url}")

    return compile_job


def step3_profile_qai_hub(compile_job):
    """Profile compiled model on QAI Hub."""
    import qai_hub

    print("[Step 3] Profiling on QAI Hub (Galaxy S23)...")

    target_model = compile_job.get_target_model()
    profile_job = qai_hub.submit_profile_job(
        model=target_model,
        device=qai_hub.Device("Samsung Galaxy S23"),
        name="unet_lcm_w8a16_profile",
    )

    print(f"  Profile job submitted: {profile_job.job_id}")
    print(f"  URL: {profile_job.url}")
    print("  Waiting for completion...")

    profile_job.wait()

    if profile_job.get_status().success:
        profile = profile_job.download_profile()
        exec_summary = profile.get("execution_summary", {})
        total_inf_time = exec_summary.get("estimated_inference_time", "N/A")
        peak_memory = exec_summary.get("inference_memory_peak_range", "N/A")
        print(f"  Profile SUCCESS: {profile_job.job_id}")
        print(f"  Inference time: {total_inf_time}")
        print(f"  Peak memory: {peak_memory}")
    else:
        print(f"  Profile FAILED: {profile_job.job_id}")
        print(f"  Check: {profile_job.url}")

    return profile_job.job_id


def main():
    print("=" * 60)
    print("UNet LCM W8A16 Quantization Pipeline")
    print("=" * 60)

    quantized_path = step1_export_and_quantize()
    compile_job = step2_compile_qai_hub(quantized_path)
    if not compile_job.get_status().success:
        print("Compile failed, skipping profile.")
        return
    profile_job_id = step3_profile_qai_hub(compile_job)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Quantized ONNX: {OUTPUT_DIR}/unet_lcm_w8a16.onnx")
    print(f"  Compile job: {compile_job.job_id}")
    print(f"  Profile job: {profile_job_id}")
    print(f"  Compiled binary: {OUTPUT_DIR}/compiled_{compile_job.job_id}/model.bin")
    print("=" * 60)


if __name__ == "__main__":
    main()
