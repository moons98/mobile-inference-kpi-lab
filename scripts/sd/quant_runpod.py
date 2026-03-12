"""
Lightweight INT8 QDQ quantization script for RunPod.

Only requires: numpy, onnxruntime
Does NOT require: torch, diffusers, transformers

Quantizes one component at a time from pre-exported FP32 ONNX + pre-generated NPZ.

Usage:
    # Full INT8 QDQ (all ops)
    python3.13 scripts/sd/quant_runpod.py --component vae_encoder --dir /workspace/weights/sd_v1.5/onnx
    python3.13 scripts/sd/quant_runpod.py --component unet_base --dir /workspace/weights/sd_v1.5/onnx

    # Mixed precision (Conv/MatMul/Gemm INT8, LayerNorm/Softmax FP32)
    python3.13 scripts/sd/quant_runpod.py --component unet_lcm --dir /workspace/weights/sd_v1.5/onnx \
        --op-types Conv MatMul Gemm --output-suffix mixed_pr
"""

import argparse
import copy
import gc
import glob
import numpy as np
from pathlib import Path


def _enable_histogram_streaming_patch(chunk_size: int = 1):
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


def quantize_component(
    component: str,
    work_dir: Path,
    calibration_samples: int = 8,
    calibration_method: str = "minmax",
    calibration_streaming_chunk: int = 1,
    op_types_to_quantize: list[str] | None = None,
    output_suffix: str = "int8_qdq",
    calib_dir: Path | None = None,
):
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType, CalibrationMethod,
    )
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    fp32_path = work_dir / f"{component}_fp32.onnx"
    int8_path = work_dir / f"{component}_{output_suffix}.onnx"
    # UNet base/lcm share the same calibration data (calib_unet.npz)
    calib_name = "unet" if component.startswith("unet_") else component
    calib_search = calib_dir or work_dir
    npz_path = calib_search / f"calib_{calib_name}.npz"

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

    # Load calibration arrays once and reuse per-sample slices.
    print(f"Loading calibration data from {npz_path.name}...")
    with np.load(npz_path) as meta:
        keys = list(meta.keys())
        total = len(meta[keys[0]])
        n = min(max(1, int(calibration_samples)), total)
        arrays = {k: meta[k] for k in keys}
    print(f"  Loaded {n}/{total} samples, keys: {keys} (cached arrays)")

    class NpzCalibReader(CalibrationDataReader):
        def __init__(self, arrays_dict: dict[str, np.ndarray], num_samples: int):
            self._arrays = arrays_dict
            self._keys = list(self._arrays.keys())
            self._num_samples = num_samples
            self._index = 0
        def get_next(self):
            if self._index >= self._num_samples:
                return None
            i = self._index
            self._index += 1
            return {
                k: np.ascontiguousarray(self._arrays[k][i:i+1])
                for k in self._keys
            }
        def rewind(self):
            self._index = 0

    reader = NpzCalibReader(arrays, n)

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

    # MinMax for all: Percentile stores all activations ??OOM
    if calibration_method == "percentile":
        calib_method = CalibrationMethod.Percentile
        _enable_histogram_streaming_patch(calibration_streaming_chunk)
        print(f"  Histogram collection: streaming (chunk={max(1, int(calibration_streaming_chunk))})")
    else:
        calib_method = CalibrationMethod.MinMax

    print(f"Quantizing {component} ({calib_method.name}, {n} samples)...")
    extra_options = {}
    if calib_method == CalibrationMethod.MinMax:
        extra_options["CalibMovingAverage"] = True
        extra_options["CalibMovingAverageConstant"] = 0.01
    else:
        extra_options["num_bins"] = 2048
        extra_options["percentile"] = 99.999
    quant_kwargs = dict(
        model_input=str(input_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=calib_method,
        use_external_data_format=has_external,
        extra_options=extra_options,
    )
    if op_types_to_quantize:
        quant_kwargs["op_types_to_quantize"] = op_types_to_quantize
        print(f"  Op types to quantize: {op_types_to_quantize}")
    quantize_static(**quant_kwargs)

    # Cleanup
    del reader
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
                        choices=["vae_encoder", "text_encoder", "vae_decoder",
                                 "unet_base", "unet_lcm"],
                        help="Component to quantize")
    parser.add_argument("--dir", required=True, type=str,
                        help="Directory with FP32 ONNX files")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory with calib NPZ files (default: same as --dir)")
    parser.add_argument("--calibration-method", choices=["minmax", "percentile"], default="minmax",
                        help="Calibration method (default: minmax)")
    parser.add_argument("--calibration-samples", type=int, default=8,
                        help="Number of calibration samples to use (default: 8)")
    parser.add_argument("--calibration-streaming-chunk", type=int, default=2,
                        help="Histogram calibration streaming chunk size for percentile (default: 2)")
    parser.add_argument("--op-types", nargs="+", default=None,
                        help="Op types to quantize (e.g. Conv MatMul Gemm). Default: all ops")
    parser.add_argument("--output-suffix", default="int8_qdq",
                        help="Output filename suffix (default: int8_qdq -> {component}_int8_qdq.onnx)")
    args = parser.parse_args()

    work_dir = Path(args.dir)
    label = args.output_suffix if args.op_types else "INT8 QDQ"
    print("=" * 60)
    print(f"Quantization ({label}): {args.component}")
    print(f"Working dir: {work_dir}")
    print("=" * 60)

    quantize_component(
        args.component,
        work_dir,
        calibration_samples=args.calibration_samples,
        calibration_method=args.calibration_method,
        calibration_streaming_chunk=args.calibration_streaming_chunk,
        op_types_to_quantize=args.op_types,
        output_suffix=args.output_suffix,
        calib_dir=Path(args.calib_dir) if args.calib_dir else None,
    )

