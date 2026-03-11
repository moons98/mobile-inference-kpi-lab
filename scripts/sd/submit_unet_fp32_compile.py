"""Submit UNet FP32 (opset 18) compile job to QAI Hub.

FP32 UNet is 3.2GB with external data. Uses QAI Hub's directory upload
format (.onnx + .data in a directory) to handle external weights.
"""
import sys
from pathlib import Path
import qai_hub as hub

device = hub.Device("Samsung Galaxy S23")
options = "--target_runtime precompiled_qnn_onnx --qairt_version 2.42 --truncate_64bit_io"
input_specs = dict(
    sample=((1, 9, 64, 64), "float32"),
    timestep=((1,), "int64"),
    encoder_hidden_states=((1, 77, 768), "float32"),
)

# Use directory format for external data support
model_dir = Path("weights/sd_v1.5_inpaint/onnx/unet_fp32_dir")
print(f"Model dir: {model_dir}")
for f in sorted(model_dir.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")

print(f"\nUploading directory {model_dir}...")
hub_model = hub.upload_model(str(model_dir))
print(f"Model uploaded: {hub_model}")

print(f"\nSubmitting FP32 compile job...")
print(f"Options: {options}")
compile_job = hub.submit_compile_job(
    model=hub_model,
    device=device,
    input_specs=input_specs,
    options=options,
)
print(f"Job ID: {compile_job.job_id}")
print(f"URL: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
