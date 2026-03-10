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
    python export_yolo_seg.py --export-all
    python export_yolo_seg.py --export fp32
    python export_yolo_seg.py --export int8
    python export_yolo_seg.py --export int8_full
    python export_yolo_seg.py --status
    python export_yolo_seg.py --export int8 --calib-data path/to/images/
"""

import argparse
import glob
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR.parent / "weights" / "yolov8n_seg" / "onnx"
MODEL_NAME = "yolov8n-seg"
INPUT_SIZE = 640

# Expected sizes
EXPECTED_SIZES_MB = {"fp32": 13, "int8_noh": 7, "int8_full": 7}


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

    # Get head nodes to exclude
    head_nodes = get_head_nodes(fp32_path)

    class YoloCalibReader(CalibrationDataReader):
        def __init__(self, data_dir, n, input_size):
            self.data = []
            if data_dir and Path(data_dir).exists():
                # Use real images from directory
                from PIL import Image
                image_files = sorted(glob.glob(str(Path(data_dir) / "*.jpg")))
                image_files += sorted(glob.glob(str(Path(data_dir) / "*.png")))
                image_files = image_files[:n]
                print(f"  Using {len(image_files)} real images from {data_dir}")
                for img_path in image_files:
                    img = Image.open(img_path).convert("RGB").resize((input_size, input_size))
                    arr = np.array(img).astype(np.float32) / 255.0
                    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
                    self.data.append({"images": arr})
            else:
                # Fallback: random data
                print(f"  Using {n} random calibration samples (no image dir specified)")
                rng = np.random.default_rng(42)
                for _ in range(n):
                    self.data.append({
                        "images": rng.random((1, 3, input_size, input_size)).astype(np.float32),
                    })
            self.iter = iter(self.data)

        def get_next(self):
            return next(self.iter, None)

        def rewind(self):
            self.iter = iter(self.data)

    print(f"\n  Quantizing {MODEL_NAME} to INT8 QDQ (head excluded)...")
    reader = YoloCalibReader(calib_data_dir, num_samples, INPUT_SIZE)

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
    )

    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    int8_size = int8_path.stat().st_size / 1024 / 1024
    print(f"  FP32:  {fp32_size:.1f} MB")
    print(f"  INT8 (noh): {int8_size:.1f} MB ({int8_size / fp32_size * 100:.0f}%)")
    return int8_path


def quantize_int8_full(fp32_path: Path, output_dir: Path,
                       calib_data_dir: Path = None, num_samples: int = 20,
                       force: bool = False) -> Path:
    """Quantize YOLOv8n-seg to INT8 QDQ — all nodes including head."""
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType, CalibrationMethod,
    )
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    int8_full_path = output_dir / f"{MODEL_NAME}_int8_qdq.onnx"

    if int8_full_path.exists() and not force:
        size_mb = int8_full_path.stat().st_size / 1024 / 1024
        print(f"  Reusing existing: {int8_full_path.name} ({size_mb:.1f} MB)")
        return int8_full_path

    class YoloCalibReader(CalibrationDataReader):
        def __init__(self, data_dir, n, input_size):
            self.data = []
            if data_dir and Path(data_dir).exists():
                from PIL import Image
                image_files = sorted(glob.glob(str(Path(data_dir) / "*.jpg")))
                image_files += sorted(glob.glob(str(Path(data_dir) / "*.png")))
                image_files = image_files[:n]
                print(f"  Using {len(image_files)} real images from {data_dir}")
                for img_path in image_files:
                    img = Image.open(img_path).convert("RGB").resize((input_size, input_size))
                    arr = np.array(img).astype(np.float32) / 255.0
                    arr = arr.transpose(2, 0, 1)[np.newaxis]
                    self.data.append({"images": arr})
            else:
                print(f"  Using {n} random calibration samples (no image dir specified)")
                rng = np.random.default_rng(42)
                for _ in range(n):
                    self.data.append({
                        "images": rng.random((1, 3, input_size, input_size)).astype(np.float32),
                    })
            self.iter = iter(self.data)

        def get_next(self):
            return next(self.iter, None)

        def rewind(self):
            self.iter = iter(self.data)

    print(f"\n  Quantizing {MODEL_NAME} to INT8 QDQ (full — head included)...")
    reader = YoloCalibReader(calib_data_dir, num_samples, INPUT_SIZE)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_full_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.Percentile,
        # No nodes_to_exclude — quantize everything
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

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    precisions = []
    if args.export_all or (args.export and "all" in args.export):
        precisions = ["fp32", "int8", "int8_full"]
    elif args.export:
        precisions = [p for p in args.export if p != "all"]

    if not precisions:
        parser.print_help()
        print(f"\nExamples:")
        print(f"  python export_yolo_seg.py --export-all")
        print(f"  python export_yolo_seg.py --export int8 int8_full")
        print(f"  python export_yolo_seg.py --export int8 --calib-data path/to/coco/val2017/")
        print(f"  python export_yolo_seg.py --status")
        return

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
