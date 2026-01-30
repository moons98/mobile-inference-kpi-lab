#!/usr/bin/env python3
"""
Export models from torchvision and ultralytics to ONNX format.

Supported models:
- MobileNetV2 (torchvision) - ImageNet classification
- YOLOv8n (ultralytics) - Object detection

Quantization methods:
- dynamic: Fast, no calibration data needed, but NOT supported by QNN EP (CPU fallback)
- static: Requires calibration data, fully supported by QNN EP (NPU acceleration)

Usage:
    python export_to_onnx.py --export-mobilenetv2
    python export_to_onnx.py --export-mobilenetv2-quantized
    python export_to_onnx.py --export-mobilenetv2-quantized --quant-method static
    python export_to_onnx.py --export-yolov8n
    python export_to_onnx.py --export-all
    python export_to_onnx.py --list
    python export_to_onnx.py --status
"""

import argparse
import sys
from pathlib import Path

# Global settings (will be overridden by CLI args)
QUANT_METHOD = "dynamic"
CALIBRATION_DATA_PATH = None
CALIBRATION_SAMPLES = 100

MODELS = {
    "mobilenetv2": {
        "source": "torchvision",
        "filename": "mobilenetv2_torchvision.onnx",
        "description": "MobileNetV2 (FP32) - torchvision export",
        "input_shape": [1, 3, 224, 224],  # NCHW
        "output": "1000 classes (ImageNet)",
        "dtype": "FP32",
    },
    "mobilenetv2-quantized": {
        "source": "torchvision",
        "filename": "mobilenetv2_torchvision_quantized.onnx",
        "description": "MobileNetV2 (INT8) - torchvision + onnxruntime quantization",
        "input_shape": [1, 3, 224, 224],
        "output": "1000 classes (ImageNet)",
        "dtype": "INT8",
        "quantize": True,
    },
    "yolov8n": {
        "source": "ultralytics",
        "filename": "yolov8n_ultralytics.onnx",
        "description": "YOLOv8n (FP32) - ultralytics export",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "FP32",
    },
    "yolov8n-quantized": {
        "source": "ultralytics",
        "filename": "yolov8n_ultralytics_quantized.onnx",
        "description": "YOLOv8n (INT8) - ultralytics + onnxruntime quantization",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
    },
}

ASSETS_DIR = Path(__file__).parent.parent / "android" / "app" / "src" / "main" / "assets"


def export_mobilenetv2(output_path: Path, quantize: bool = False) -> bool:
    """Export MobileNetV2 from torchvision to ONNX."""
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("Error: torch and torchvision packages not installed")
        print("Install with: pip install torch torchvision")
        return False

    try:
        print("Loading MobileNetV2 from torchvision...")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)

        # Export to ONNX
        fp32_path = output_path if not quantize else output_path.with_suffix(".fp32.onnx")
        print(f"Exporting to ONNX: {fp32_path}")

        # Use legacy JIT exporter to avoid dynamo issues with opset conversion
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(fp32_path),
                export_params=True,
                opset_version=17,  # Use higher opset to avoid version conversion issues
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamo=False,  # Use legacy exporter for stability
            )

        print(f"Exported: {fp32_path.name} ({fp32_path.stat().st_size / 1024 / 1024:.2f} MB)")

        if quantize:
            return quantize_onnx_model(fp32_path, output_path, input_shape=[1, 3, 224, 224])

        return True

    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_yolov8n(output_path: Path, quantize: bool = False) -> bool:
    """Export YOLOv8n from ultralytics to ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Install with: pip install ultralytics")
        return False

    try:
        print("Loading YOLOv8n from ultralytics...")
        model = YOLO("yolov8n.pt")

        # Export to ONNX
        fp32_path = output_path if not quantize else output_path.with_suffix(".fp32.onnx")
        print(f"Exporting to ONNX: {fp32_path}")

        model.export(
            format="onnx",
            imgsz=640,
            opset=17,  # Use higher opset for better compatibility
            simplify=True,
            dynamic=False,  # Fixed shape for NPU compatibility
        )

        # ultralytics saves to yolov8n.onnx in current directory
        exported_file = Path("yolov8n.onnx")
        if exported_file.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            exported_file.rename(fp32_path)
            print(f"Exported: {fp32_path.name} ({fp32_path.stat().st_size / 1024 / 1024:.2f} MB)")

            # Cleanup
            for cleanup_path in [Path("yolov8n.pt")]:
                if cleanup_path.exists():
                    cleanup_path.unlink()

            if quantize:
                return quantize_onnx_model(fp32_path, output_path, input_shape=[1, 3, 640, 640])

            return True
        else:
            print("Error: ONNX export file not found")
            return False

    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


class RandomCalibrationDataReader:
    """Calibration data reader using random/synthetic data."""

    def __init__(self, input_name: str, input_shape: list, num_samples: int = 100):
        import numpy as np
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.current = 0
        # Pre-generate random data (normalized to [0, 1] range typical for images)
        np.random.seed(42)
        self.data = [
            np.random.rand(*input_shape).astype(np.float32)
            for _ in range(num_samples)
        ]

    def get_next(self):
        if self.current >= self.num_samples:
            return None
        data = {self.input_name: self.data[self.current]}
        self.current += 1
        return data

    def rewind(self):
        self.current = 0


class ImageCalibrationDataReader:
    """Calibration data reader using real images from directory."""

    def __init__(self, calibration_dir: str, input_name: str, input_shape: list, num_samples: int = 100):
        import numpy as np
        from PIL import Image

        self.input_name = input_name
        self.input_shape = input_shape
        self.current = 0

        # Get image size from input shape (NCHW format)
        _, channels, height, width = input_shape

        # Find image files
        calibration_path = Path(calibration_dir)
        image_files = list(calibration_path.glob("*.JPEG")) + \
                      list(calibration_path.glob("*.jpg")) + \
                      list(calibration_path.glob("*.png"))
        image_files = image_files[:num_samples]

        print(f"  Loading {len(image_files)} calibration images from {calibration_dir}")

        self.data = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((width, height))
                img_array = np.array(img).astype(np.float32) / 255.0

                # NHWC to NCHW
                img_array = np.transpose(img_array, (2, 0, 1))
                img_array = np.expand_dims(img_array, axis=0)

                # Normalize with ImageNet mean/std (common for pretrained models)
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
                img_array = (img_array - mean) / std

                self.data.append(img_array.astype(np.float32))
            except Exception as e:
                print(f"  Warning: Failed to load {img_path.name}: {e}")

        print(f"  Loaded {len(self.data)} images successfully")

    def get_next(self):
        if self.current >= len(self.data):
            return None
        data = {self.input_name: self.data[self.current]}
        self.current += 1
        return data

    def rewind(self):
        self.current = 0


def quantize_onnx_model(input_path: Path, output_path: Path, input_shape: list = None) -> bool:
    """Quantize ONNX model to INT8 using onnxruntime.

    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path to save quantized model
        input_shape: Input shape for static quantization calibration

    Returns:
        True if successful, False otherwise
    """
    global QUANT_METHOD

    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
    except ImportError:
        print("Error: onnxruntime package not installed")
        print("Install with: pip install onnxruntime")
        return False

    try:
        if QUANT_METHOD == "static":
            return quantize_static_onnx(input_path, output_path, input_shape)
        else:
            return quantize_dynamic_onnx(input_path, output_path)

    except Exception as e:
        print(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quantize_dynamic_onnx(input_path: Path, output_path: Path) -> bool:
    """Dynamic quantization (NOT supported by QNN EP - will fallback to CPU)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"Quantizing (dynamic) to INT8: {output_path}")
    print("  [!] Warning: Dynamic quantization NOT supported by QNN EP")
    print("  [!] Use --quant-method static for NPU acceleration")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
    )

    print(f"Quantized: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Remove intermediate FP32 file
    if input_path != output_path and input_path.exists():
        input_path.unlink()
        print(f"Removed intermediate file: {input_path.name}")

    return True


def quantize_static_onnx(input_path: Path, output_path: Path, input_shape: list = None) -> bool:
    """Static quantization (QDQ format - supported by QNN EP)."""
    global CALIBRATION_DATA_PATH, CALIBRATION_SAMPLES

    from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod
    import onnx

    print(f"Quantizing (static/QDQ) to INT8: {output_path}")
    print("  [OK] QDQ format supported by QNN EP for NPU acceleration")

    # Get input name and shape from model if not provided
    model = onnx.load(str(input_path))
    input_info = model.graph.input[0]
    input_name = input_info.name

    if input_shape is None:
        input_shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            input_shape.append(dim.dim_value if dim.dim_value > 0 else 1)

    print(f"  Input: {input_name} {input_shape}")

    # Check if real calibration data is available
    if CALIBRATION_DATA_PATH and Path(CALIBRATION_DATA_PATH).exists():
        print(f"  Using real calibration data: {CALIBRATION_DATA_PATH}")
        calibration_reader = ImageCalibrationDataReader(
            calibration_dir=str(CALIBRATION_DATA_PATH),
            input_name=input_name,
            input_shape=input_shape,
            num_samples=CALIBRATION_SAMPLES
        )
    else:
        # Check default calibration data locations
        default_imagenet = Path(__file__).parent / "calibration_data" / "imagenet"
        default_coco = Path(__file__).parent / "calibration_data" / "coco"

        if input_shape[2] == 224 and default_imagenet.exists():
            print(f"  Using ImageNet calibration data: {default_imagenet}")
            calibration_reader = ImageCalibrationDataReader(
                calibration_dir=str(default_imagenet),
                input_name=input_name,
                input_shape=input_shape,
                num_samples=CALIBRATION_SAMPLES
            )
        elif input_shape[2] == 640 and default_coco.exists():
            print(f"  Using COCO calibration data: {default_coco}")
            calibration_reader = ImageCalibrationDataReader(
                calibration_dir=str(default_coco),
                input_name=input_name,
                input_shape=input_shape,
                num_samples=CALIBRATION_SAMPLES
            )
        else:
            print(f"  Using synthetic calibration data ({CALIBRATION_SAMPLES} samples)")
            print("  [!] For better accuracy, run: python setup_calibration_data.py --download-imagenet")
            calibration_reader = RandomCalibrationDataReader(
                input_name=input_name,
                input_shape=input_shape,
                num_samples=CALIBRATION_SAMPLES
            )

    # Quantize with static method
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=__import__('onnxruntime.quantization', fromlist=['QuantFormat']).QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    print(f"Quantized: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Remove intermediate FP32 file
    if input_path != output_path and input_path.exists():
        input_path.unlink()
        print(f"Removed intermediate file: {input_path.name}")

    return True


def export_model(model_key: str, output_dir: Path) -> bool:
    """Export a model by key."""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        return False

    model = MODELS[model_key]
    dest = output_dir / model["filename"]
    quantize = model.get("quantize", False)

    print("=" * 60)
    print(f"Exporting {model['description']}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    if model["source"] == "torchvision":
        return export_mobilenetv2(dest, quantize=quantize)
    elif model["source"] == "ultralytics":
        return export_yolov8n(dest, quantize=quantize)
    else:
        print(f"Unknown source: {model['source']}")
        return False


def list_models():
    """List available models."""
    print("=" * 60)
    print("Available Models for Export")
    print("=" * 60)
    print()
    for key, model in MODELS.items():
        print(f"  {key}:")
        print(f"    Description: {model['description']}")
        print(f"    Source: {model['source']}")
        print(f"    Input: {model['input_shape']} (NCHW)")
        print(f"    Output: {model['output']}")
        print(f"    Data type: {model['dtype']}")
        print(f"    Filename: {model['filename']}")
        print()


def check_assets():
    """Check which models are already exported."""
    print("=" * 60)
    print(f"Model Status (in {ASSETS_DIR})")
    print("=" * 60)
    print()

    print("Exported models (torchvision/ultralytics):")
    for model in MODELS.values():
        path = ASSETS_DIR / model["filename"]
        if path.exists():
            size = path.stat().st_size / 1024 / 1024
            print(f"  [OK] {model['filename']} ({size:.2f} MB)")
        else:
            print(f"  [--] {model['filename']} (not found)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export models from torchvision/ultralytics to ONNX for mobile inference"
    )
    parser.add_argument(
        "--export-mobilenetv2",
        action="store_true",
        help="Export MobileNetV2 FP32 from torchvision"
    )
    parser.add_argument(
        "--export-mobilenetv2-quantized",
        action="store_true",
        help="Export MobileNetV2 INT8 quantized"
    )
    parser.add_argument(
        "--export-yolov8n",
        action="store_true",
        help="Export YOLOv8n FP32 from ultralytics"
    )
    parser.add_argument(
        "--export-yolov8n-quantized",
        action="store_true",
        help="Export YOLOv8n INT8 quantized"
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all available models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check which models are already exported"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=ASSETS_DIR,
        help=f"Output directory (default: {ASSETS_DIR})"
    )
    parser.add_argument(
        "--quant-method",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization method: dynamic (fast, CPU fallback) or static (QDQ, NPU supported)"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples for static quantization (default: 100)"
    )
    parser.add_argument(
        "--calibration-data",
        type=Path,
        default=None,
        help="Path to calibration data directory (images for static quantization)"
    )

    args = parser.parse_args()

    # Set global settings
    global QUANT_METHOD, CALIBRATION_DATA_PATH, CALIBRATION_SAMPLES
    QUANT_METHOD = args.quant_method
    CALIBRATION_DATA_PATH = args.calibration_data
    CALIBRATION_SAMPLES = args.calibration_samples

    if args.list:
        list_models()
        return

    if args.status:
        check_assets()
        return

    exports = []
    if args.export_all:
        exports = list(MODELS.keys())
    else:
        if args.export_mobilenetv2:
            exports.append("mobilenetv2")
        if args.export_mobilenetv2_quantized:
            exports.append("mobilenetv2-quantized")
        if args.export_yolov8n:
            exports.append("yolov8n")
        if args.export_yolov8n_quantized:
            exports.append("yolov8n-quantized")

    if not exports:
        parser.print_help()
        print("\nExamples:")
        print("  python export_to_onnx.py --export-mobilenetv2")
        print("  python export_to_onnx.py --export-mobilenetv2-quantized")
        print("  python export_to_onnx.py --export-mobilenetv2-quantized --quant-method static  # NPU supported")
        print("  python export_to_onnx.py --export-yolov8n")
        print("  python export_to_onnx.py --export-yolov8n-quantized --quant-method static")
        print("  python export_to_onnx.py --export-all")
        print("  python export_to_onnx.py --list")
        print("  python export_to_onnx.py --status")
        print("\nQuantization methods:")
        print("  dynamic: Fast, no calibration needed, but CPU fallback on QNN EP")
        print("  static:  Uses QDQ format, fully supported by QNN EP for NPU acceleration")
        return

    exported = []
    failed = []

    for model_key in exports:
        if export_model(model_key, args.output):
            exported.append(model_key)
        else:
            failed.append(model_key)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if exported:
        print(f"Exported: {', '.join(exported)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Output directory: {args.output}")
    if any("quantized" in e for e in exported):
        print(f"Quantization method: {args.quant_method}")
        if args.quant_method == "dynamic":
            print("  [!] Note: Dynamic quantization will fallback to CPU on QNN EP")
        else:
            print("  [OK] Static (QDQ) quantization supported by QNN EP")

    if exported:
        print()
        print("Next steps:")
        print("  - Use analyze_ops.py to check NPU compatibility")
        print("  - Use graph_transform.py to optimize if needed")
        print("  - Copy models to android/app/src/main/assets/")


if __name__ == "__main__":
    main()
