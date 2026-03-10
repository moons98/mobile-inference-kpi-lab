#!/usr/bin/env python3
"""
Export YOLOv8n-seg to ONNX and quantize to INT8 QDQ.

For the AI Eraser pipeline: YOLO-seg detects objects and generates segmentation
masks, which are fed to the SD v2.1 inpainting UNet.

Precision variants:
- FP32: PyTorch default export
- INT8 QDQ (noh): Static quantization with head excluded
  Detection/segmentation heads are kept in FP32 to preserve bbox/mask accuracy.
- INT8 QDQ (full): All nodes quantized including head
  Maximum compression, but may degrade bbox/mask accuracy.

Calibration uses COCO val2017 images for representative data.

Usage:
    python scripts/yolo/export_yolo_seg.py --export-all
    python scripts/yolo/export_yolo_seg.py --export fp32
    python scripts/yolo/export_yolo_seg.py --export int8
    python scripts/yolo/export_yolo_seg.py --export int8_full
    python scripts/yolo/export_yolo_seg.py --status
    python scripts/yolo/export_yolo_seg.py --export int8 --calib-data path/to/images/
"""

import argparse
import copy
import glob
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "weights" / "yolov8n_seg" / "onnx"
MODEL_NAME = "yolov8n-seg"
INPUT_SIZE = 640

# Expected sizes
EXPECTED_SIZES_MB = {"fp32": 13, "int8_noh": 7, "int8_full": 7}
CALIBRATION_STREAMING_CHUNK = 1


def _enable_histogram_streaming_patch(chunk_size: int = 1):
    """Patch ORT HistogramCalibrater to collect histogram incrementally."""
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


def export_fp32(output_dir: Path, force: bool = False) -> Path:
    """Export YOLOv8n-seg to FP32 ONNX."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / f"{MODEL_NAME}.onnx"

    if fp32_path.exists() and not force:
        size_mb = fp32_path.stat().st_size / 1024 / 1024
        print(f"  Reusing existing: {fp32_path.name} ({size_mb:.1f} MB)")
        return fp32_path

    print(f"Exporting {MODEL_NAME} to FP32 ONNX...")
    model = YOLO(f"{MODEL_NAME}.pt")
    model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        opset=17,
        simplify=True,
    )

    # ultralytics exports to same dir as .pt, move to output_dir
    exported = Path(f"{MODEL_NAME}.onnx")
    if exported.exists() and exported != fp32_path:
        exported.replace(fp32_path)

    size_mb = fp32_path.stat().st_size / 1024 / 1024
    print(f"  Exported: {fp32_path.name} ({size_mb:.1f} MB)")
    return fp32_path


def get_head_nodes(fp32_path: Path) -> list:
    """Identify detection and segmentation head nodes to exclude from quantization."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    model = sess._model_meta

    # YOLOv8 head nodes contain "Detect" or "Segment" or "model.22" (last module)
    # We need to inspect the ONNX graph to find them
    import onnx
    onnx_model = onnx.load(str(fp32_path))

    head_nodes = []
    for node in onnx_model.graph.node:
        # YOLOv8n-seg: model.22 is the Segment head
        if "model.22" in node.name or "/model.22/" in node.name:
            head_nodes.append(node.name)

    print(f"  Found {len(head_nodes)} head nodes to exclude from quantization")
    return head_nodes


def quantize_int8(fp32_path: Path, output_dir: Path,
                  calib_data_dir: Path = None, num_samples: int = 20,
                  force: bool = False) -> Path:
    """Quantize YOLOv8n-seg to INT8 QDQ with head excluded."""
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType, CalibrationMethod,
    )
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    int8_path = output_dir / f"{MODEL_NAME}_int8_qdq_noh.onnx"

    if int8_path.exists() and not force:
        size_mb = int8_path.stat().st_size / 1024 / 1024
        print(f"  Reusing existing: {int8_path.name} ({size_mb:.1f} MB)")
        return int8_path

    if calib_data_dir is None:
        raise ValueError("INT8 quantization requires --calib-data <image_dir>.")
    if not Path(calib_data_dir).exists():
        raise FileNotFoundError(f"Calibration image directory not found: {calib_data_dir}")

    # Get head nodes to exclude
    head_nodes = get_head_nodes(fp32_path)

    class YoloCalibReader(CalibrationDataReader):
        def __init__(self, data_dir, n, input_size):
            self.input_size = input_size
            from PIL import Image
            self._Image = Image
            image_files = sorted(glob.glob(str(Path(data_dir) / "*.jpg")))
            image_files += sorted(glob.glob(str(Path(data_dir) / "*.png")))
            self.image_files = image_files[:n]
            print(f"  Using {len(self.image_files)} real images from {data_dir}")
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.image_files):
                return None
            img = self._Image.open(self.image_files[self.idx]).convert("RGB").resize((self.input_size, self.input_size))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
            self.idx += 1
            return {"images": arr}

        def rewind(self):
            self.idx = 0

    print(f"\n  Quantizing {MODEL_NAME} to INT8 QDQ (head excluded)...")
    reader = YoloCalibReader(calib_data_dir, num_samples, INPUT_SIZE)
    _enable_histogram_streaming_patch(CALIBRATION_STREAMING_CHUNK)
    print(f"  Histogram collection: streaming (chunk={CALIBRATION_STREAMING_CHUNK})")

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.Percentile,
        nodes_to_exclude=head_nodes,
        extra_options={"num_bins": 2048, "percentile": 99.999},
    )

    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    int8_size = int8_path.stat().st_size / 1024 / 1024
    print(f"  FP32:  {fp32_size:.1f} MB")
    print(f"  INT8 (noh): {int8_size:.1f} MB ({int8_size / fp32_size * 100:.0f}%)")
    return int8_path


def quantize_int8_full(fp32_path: Path, output_dir: Path,
                       calib_data_dir: Path = None, num_samples: int = 20,
                       force: bool = False) -> Path:
    """Quantize YOLOv8n-seg to INT8 QDQ (all nodes including head)."""
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType, CalibrationMethod,
    )
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    int8_full_path = output_dir / f"{MODEL_NAME}_int8_qdq.onnx"

    if int8_full_path.exists() and not force:
        size_mb = int8_full_path.stat().st_size / 1024 / 1024
        print(f"  Reusing existing: {int8_full_path.name} ({size_mb:.1f} MB)")
        return int8_full_path

    if calib_data_dir is None:
        raise ValueError("INT8 quantization requires --calib-data <image_dir>.")
    if not Path(calib_data_dir).exists():
        raise FileNotFoundError(f"Calibration image directory not found: {calib_data_dir}")

    class YoloCalibReader(CalibrationDataReader):
        def __init__(self, data_dir, n, input_size):
            self.input_size = input_size
            from PIL import Image
            self._Image = Image
            image_files = sorted(glob.glob(str(Path(data_dir) / "*.jpg")))
            image_files += sorted(glob.glob(str(Path(data_dir) / "*.png")))
            self.image_files = image_files[:n]
            print(f"  Using {len(self.image_files)} real images from {data_dir}")
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.image_files):
                return None
            img = self._Image.open(self.image_files[self.idx]).convert("RGB").resize((self.input_size, self.input_size))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
            self.idx += 1
            return {"images": arr}

        def rewind(self):
            self.idx = 0

    print(f"\n  Quantizing {MODEL_NAME} to INT8 QDQ (full, head included)...")
    reader = YoloCalibReader(calib_data_dir, num_samples, INPUT_SIZE)
    _enable_histogram_streaming_patch(CALIBRATION_STREAMING_CHUNK)
    print(f"  Histogram collection: streaming (chunk={CALIBRATION_STREAMING_CHUNK})")

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_full_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"num_bins": 2048, "percentile": 99.999},
        # No nodes_to_exclude: quantize everything
    )

    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    int8_size = int8_full_path.stat().st_size / 1024 / 1024
    print(f"  FP32:  {fp32_size:.1f} MB")
    print(f"  INT8 (full): {int8_size:.1f} MB ({int8_size / fp32_size * 100:.0f}%)")
    return int8_full_path


def check_status():
    """Check exported model status."""
    print("=" * 60)
    print(f"YOLOv8n-seg Model Status (in {OUTPUT_DIR})")
    print("=" * 60)

    model_files = {
        "fp32": OUTPUT_DIR / f"{MODEL_NAME}.onnx",
        "int8_noh": OUTPUT_DIR / f"{MODEL_NAME}_int8_qdq_noh.onnx",
        "int8_full": OUTPUT_DIR / f"{MODEL_NAME}_int8_qdq.onnx",
    }
    for name, path in model_files.items():
        expected = EXPECTED_SIZES_MB.get(name, "?")
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  [OK] {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  [--] {path.name} (expected ~{expected} MB)")


def main():
    global CALIBRATION_STREAMING_CHUNK
    parser = argparse.ArgumentParser(
        description="Export YOLOv8n-seg to ONNX with INT8 quantization"
    )
    parser.add_argument(
        "--export",
        nargs="+",
        choices=["fp32", "int8", "int8_full", "all"],
        help="Precision variants to export",
    )
    parser.add_argument("--export-all", action="store_true", help="Export all precisions")
    parser.add_argument("--status", action="store_true", help="Check model status")
    parser.add_argument("--force", action="store_true", help="Re-export even if exists")
    parser.add_argument(
        "--calib-data",
        type=Path,
        default=None,
        help="Directory with calibration images (e.g., COCO val2017)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=20,
        help="Number of calibration samples (default: 20)",
    )
    parser.add_argument(
        "--calibration-streaming-chunk",
        type=int,
        default=CALIBRATION_STREAMING_CHUNK,
        help="Histogram calibration streaming chunk size for percentile (default: 1)",
    )

    args = parser.parse_args()
    CALIBRATION_STREAMING_CHUNK = max(1, args.calibration_streaming_chunk)

    if args.status:
        check_status()
        return

    if args.calib_data is not None and not args.calib_data.exists():
        parser.error(f"--calib-data directory does not exist: {args.calib_data}")

    precisions = []
    if args.export_all or (args.export and "all" in args.export):
        precisions = ["fp32", "int8", "int8_full"]
    elif args.export:
        precisions = [p for p in args.export if p != "all"]

    if not precisions:
        parser.print_help()
        print(f"\nExamples:")
        print(f"  python scripts/yolo/export_yolo_seg.py --export-all")
        print(f"  python scripts/yolo/export_yolo_seg.py --export int8 int8_full")
        print(f"  python scripts/yolo/export_yolo_seg.py --export int8 --calib-data path/to/coco/val2017/")
        print(f"  python scripts/yolo/export_yolo_seg.py --status")
        return

    if ("int8" in precisions or "int8_full" in precisions) and args.calib_data is None:
        parser.error("INT8 quantization requires --calib-data <image_dir>.")

    # FP32 is always needed for INT8 variants
    fp32_path = None
    needs_fp32 = "fp32" in precisions or "int8" in precisions or "int8_full" in precisions
    if needs_fp32:
        fp32_path = export_fp32(OUTPUT_DIR, args.force)

    if "int8" in precisions and fp32_path:
        quantize_int8(
            fp32_path, OUTPUT_DIR,
            calib_data_dir=args.calib_data,
            num_samples=args.calib_samples,
            force=args.force,
        )

    if "int8_full" in precisions and fp32_path:
        quantize_int8_full(
            fp32_path, OUTPUT_DIR,
            calib_data_dir=args.calib_data,
            num_samples=args.calib_samples,
            force=args.force,
        )

    print("\nDone!")
    check_status()


if __name__ == "__main__":
    main()


