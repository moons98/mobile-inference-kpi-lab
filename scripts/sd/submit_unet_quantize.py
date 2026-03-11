"""Submit UNet FP32 (opset 18) quantize job to QAI Hub.

Reuses already-uploaded FP32 model. Generates random calibration data.
"""
import numpy as np
import qai_hub as hub

# Upload with directory format for external data support
from pathlib import Path
model_dir = Path("weights/sd_v1.5_inpaint/onnx/unet_fp32_dir")
print(f"Uploading model directory: {model_dir}")
hub_model = hub.upload_model(str(model_dir))
print(f"Model uploaded: {hub_model}")

# Load real calibration data
npz_path = "weights/sd_v1.5_inpaint/calib_data/calib_unet.npz"
print(f"Loading calibration data from {npz_path}...")
d = np.load(npz_path)
# Format: dict of lists, each element is a single sample with batch=1
n = len(d["sample"])
calibration_data = dict(
    sample=[d["sample"][i:i+1] for i in range(n)],
    timestep=[d["timestep"][i:i+1] for i in range(n)],
    encoder_hidden_states=[d["encoder_hidden_states"][i:i+1] for i in range(n)],
)
print(f"  {n} samples, shapes: sample={calibration_data['sample'][0].shape}, "
      f"timestep={calibration_data['timestep'][0].shape}, "
      f"encoder_hidden_states={calibration_data['encoder_hidden_states'][0].shape}")

print(f"\nSubmitting UNet FP32 quantize job...")
quantize_job = hub.submit_quantize_job(
    model=hub_model,
    calibration_data=calibration_data,
)
print(f"Job ID: {quantize_job.job_id}")
print(f"URL: https://workbench.aihub.qualcomm.com/jobs/{quantize_job.job_id}/")
