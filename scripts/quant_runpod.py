"""
Lightweight INT8 QDQ quantization script for RunPod.

Only requires: numpy, onnxruntime
Does NOT require: torch, diffusers, transformers

Quantizes one component at a time from pre-exported FP32 ONNX + pre-generated NPZ.

Usage:
    python3.13 quant_runpod.py --component vae_encoder --dir /workspace/weights/sd_v1.5_inpaint/onnx
    python3.13 quant_runpod.py --component text_encoder --dir /workspace/weights/sd_v1.5_inpaint/onnx
    python3.13 quant_runpod.py --component vae_decoder --dir /workspace/weights/sd_v1.5_inpaint/onnx
    python3.13 quant_runpod.py --component unet --dir /workspace/weights/sd_v1.5_inpaint/onnx
"""

import argparse
import gc
import glob
import numpy as np
from pathlib import Path


def quantize_component(component: str, work_dir: Path):
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType, CalibrationMethod,
    )
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    fp32_path = work_dir / f"{component}_fp32.onnx"
    int8_path = work_dir / f"{component}_int8_qdq.onnx"
    npz_path = work_dir / f"calib_{component}.npz"

    if not fp32_path.exists():
        print(f"ERROR: {fp32_path} not found")
        return
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found")
        return

    # Check if already done
    if int8_path.exists():
        sz = sum(Path(f).stat().st_size for f in glob.glob(str(int8_path) + "*")) / 1024 / 1024
        print(f"Already exists: {int8_path.name} ({sz:.0f} MB), skipping")
        return

    # Load calibration data
    print(f"Loading calibration data from {npz_path.name}...")
    npz = np.load(npz_path)
    keys = list(npz.keys())
    n = len(npz[keys[0]])
    data = []
    for i in range(n):
        sample = {k: npz[k][i:i+1] for k in keys}
        data.append(sample)
    print(f"  Loaded {n} samples, keys: {keys}")
    del npz
    gc.collect()

    class NpzCalibReader(CalibrationDataReader):
        def __init__(self, samples):
            self.data = samples
            self.iter = iter(self.data)
        def get_next(self):
            return next(self.iter, None)
        def rewind(self):
            self.iter = iter(self.data)

    reader = NpzCalibReader(data)

    # Preprocessing (skip for models with external data like UNet)
    data_file = Path(str(fp32_path) + ".data")
    has_external = data_file.exists()
    input_path = fp32_path

    if not has_external:
        preprocessed = work_dir / f"{component}_fp32.preprocessed.onnx"
        if preprocessed.exists():
            print(f"  Using existing preprocessed: {preprocessed.name}")
            input_path = preprocessed
        else:
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process
                print("  Preprocessing (shape inference)...")
                quant_pre_process(
                    input_model_path=str(fp32_path),
                    output_model_path=str(preprocessed),
                )
                input_path = preprocessed
                print("  Preprocessing done.")
            except Exception as e:
                print(f"  Preprocessing failed ({e}), using raw FP32")
    else:
        print("  Skipping preprocessing (external data)")

    # MinMax for all: Percentile stores all activations → OOM
    calib_method = CalibrationMethod.MinMax

    print(f"Quantizing {component} ({calib_method.name}, {n} samples)...")
    quantize_static(
        model_input=str(input_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=calib_method,
        use_external_data_format=has_external,
    )

    # Cleanup
    del data, reader
    gc.collect()
    if not has_external and input_path != fp32_path and input_path.exists():
        input_path.unlink()
        print(f"  Cleaned up {input_path.name}")

    # Report
    int8_files = glob.glob(str(int8_path) + "*")
    total_int8 = sum(Path(f).stat().st_size for f in int8_files) / 1024 / 1024
    fp32_files = glob.glob(str(fp32_path) + "*")
    total_fp32 = sum(Path(f).stat().st_size for f in fp32_files) / 1024 / 1024

    print(f"\nDone: {component}")
    print(f"  FP32: {total_fp32:.0f} MB")
    print(f"  INT8: {total_int8:.0f} MB ({total_int8 / total_fp32 * 100:.0f}% of FP32)")
    for f in sorted(int8_files):
        print(f"    {Path(f).name}: {Path(f).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight INT8 QDQ quantization (RunPod)")
    parser.add_argument("--component", required=True,
                        choices=["vae_encoder", "text_encoder", "vae_decoder", "unet"],
                        help="Component to quantize")
    parser.add_argument("--dir", required=True, type=str,
                        help="Directory with FP32 ONNX + calib NPZ files")
    args = parser.parse_args()

    work_dir = Path(args.dir)
    print("=" * 60)
    print(f"INT8 QDQ Quantization: {args.component}")
    print(f"Working dir: {work_dir}")
    print("=" * 60)

    quantize_component(args.component, work_dir)
