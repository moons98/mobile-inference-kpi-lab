"""Submit UNet mixed precision (Conv/MatMul INT8, LayerNorm FP32) compile job to QAI Hub.

Output: precompiled QNN context binary (.bin)
"""
import shutil
import tempfile
import time
from pathlib import Path

import qai_hub as hub

ONNX_DIR = Path("weights/sd_v1.5_inpaint/onnx")
ONNX_FILE = ONNX_DIR / "unet_mixed_pr.onnx"
DATA_FILE = ONNX_DIR / "unet_mixed_pr.onnx.data"

# Stage model directory for external data upload
tmp_dir = Path(tempfile.mkdtemp(prefix="qai_unet_mixed_"))
shutil.copy2(ONNX_FILE, tmp_dir / ONNX_FILE.name)
shutil.copy2(DATA_FILE, tmp_dir / DATA_FILE.name)
print(f"Staged: {tmp_dir}")
print(f"  {ONNX_FILE.name}: {ONNX_FILE.stat().st_size/1024/1024:.1f} MB")
print(f"  {DATA_FILE.name}: {DATA_FILE.stat().st_size/1024/1024:.1f} MB")

# Upload
print("\nUploading model...")
model = hub.upload_model(str(tmp_dir))
print(f"  Uploaded: {model}")

# Compile
device = hub.Device("Samsung Galaxy S23 (Family)")
options = "--target_runtime precompiled_qnn_onnx --qairt_version 2.42 --truncate_64bit_io"
input_specs = dict(
    sample=((1, 9, 64, 64), "float32"),
    timestep=((1,), "int64"),
    encoder_hidden_states=((1, 77, 768), "float32"),
)

print(f"\nSubmitting compile job...")
print(f"  Device: {device}")
print(f"  Options: {options}")
compile_job = hub.submit_compile_job(
    model=model,
    device=device,
    input_specs=input_specs,
    options=options,
)
print(f"\nCompile job: {compile_job.job_id}")
print(f"URL: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")

# Cleanup
shutil.rmtree(tmp_dir, ignore_errors=True)
