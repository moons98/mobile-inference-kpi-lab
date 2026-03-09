#!/usr/bin/env python3
"""
Export YOLOv8 models from ultralytics to ONNX format.

Supported models:
- YOLOv8n (ultralytics) - Object detection (nano)
- YOLOv8s (ultralytics) - Object detection (small)
- YOLOv8m (ultralytics) - Object detection (medium)

Quantization: static/QDQ format, fully supported by QNN EP (NPU acceleration)

Usage:
    python export_to_onnx.py --export-yolov8n
    python export_to_onnx.py --export-yolov8s
    python export_to_onnx.py --export-yolov8m
    python export_to_onnx.py --export-yolov8n-quantized
    python export_to_onnx.py --export-yolov8s-quantized
    python export_to_onnx.py --export-yolov8m-quantized
    python export_to_onnx.py --export-all
    python export_to_onnx.py --list
    python export_to_onnx.py --status
"""

import argparse
import shutil
from pathlib import Path

# Global settings (will be overridden by CLI args)
CALIBRATION_DATA_PATH = None
CALIBRATION_SAMPLES = 100
CALIBRATION_METHOD = "percentile"  # "minmax" or "percentile"

MODELS = {
    "yolov8n": {
        "source": "ultralytics",
        "variant": "n",
        "filename": "yolov8n.onnx",
        "description": "YOLOv8n (FP32) - ultralytics export",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "FP32",
    },
    "yolov8n-quantized": {
        "source": "ultralytics",
        "variant": "n",
        "filename_static": "yolov8n_int8_qdq.onnx",
        "description": "YOLOv8n INT8 QDQ baseline (per-tensor, full graph)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "baseline",
    },
    "yolov8n-quantized-pc": {
        "source": "ultralytics",
        "variant": "n",
        "filename_static": "yolov8n_int8_qdq_pc.onnx",
        "description": "YOLOv8n INT8 QDQ per-channel (full graph)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "per_channel",
    },
    "yolov8n-quantized-noh": {
        "source": "ultralytics",
        "variant": "n",
        "filename_static": "yolov8n_int8_qdq_noh.onnx",
        "description": "YOLOv8n INT8 QDQ per-tensor + head excluded (FP32 head)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "exclude_head",
    },
    "yolov8n-quantized-pc-noh": {
        "source": "ultralytics",
        "variant": "n",
        "filename_static": "yolov8n_int8_qdq_pc_noh.onnx",
        "description": "YOLOv8n INT8 QDQ per-channel + head excluded (FP32 head)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "per_channel_exclude_head",
    },
    "yolov8s": {
        "source": "ultralytics",
        "variant": "s",
        "filename": "yolov8s.onnx",
        "description": "YOLOv8s (FP32) - ultralytics export",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "FP32",
    },
    "yolov8s-quantized": {
        "source": "ultralytics",
        "variant": "s",
        "filename_static": "yolov8s_int8_qdq.onnx",
        "description": "YOLOv8s INT8 QDQ baseline (per-tensor, full graph)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "baseline",
    },
    "yolov8s-quantized-pc": {
        "source": "ultralytics",
        "variant": "s",
        "filename_static": "yolov8s_int8_qdq_pc.onnx",
        "description": "YOLOv8s INT8 QDQ per-channel (full graph)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "per_channel",
    },
    "yolov8s-quantized-noh": {
        "source": "ultralytics",
        "variant": "s",
        "filename_static": "yolov8s_int8_qdq_noh.onnx",
        "description": "YOLOv8s INT8 QDQ per-tensor + head excluded (FP32 head)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "exclude_head",
    },
    "yolov8s-quantized-pc-noh": {
        "source": "ultralytics",
        "variant": "s",
        "filename_static": "yolov8s_int8_qdq_pc_noh.onnx",
        "description": "YOLOv8s INT8 QDQ per-channel + head excluded (FP32 head)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "per_channel_exclude_head",
    },
    "yolov8m": {
        "source": "ultralytics",
        "variant": "m",
        "filename": "yolov8m.onnx",
        "description": "YOLOv8m (FP32) - ultralytics export",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "FP32",
    },
    "yolov8m-quantized": {
        "source": "ultralytics",
        "variant": "m",
        "filename_static": "yolov8m_int8_qdq.onnx",
        "description": "YOLOv8m INT8 QDQ baseline (per-tensor, full graph)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "baseline",
    },
    "yolov8m-quantized-pc": {
        "source": "ultralytics",
        "variant": "m",
        "filename_static": "yolov8m_int8_qdq_pc.onnx",
        "description": "YOLOv8m INT8 QDQ per-channel (full graph)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "per_channel",
    },
    "yolov8m-quantized-noh": {
        "source": "ultralytics",
        "variant": "m",
        "filename_static": "yolov8m_int8_qdq_noh.onnx",
        "description": "YOLOv8m INT8 QDQ per-tensor + head excluded (FP32 head)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "exclude_head",
    },
    "yolov8m-quantized-pc-noh": {
        "source": "ultralytics",
        "variant": "m",
        "filename_static": "yolov8m_int8_qdq_pc_noh.onnx",
        "description": "YOLOv8m INT8 QDQ per-channel + head excluded (FP32 head)",
        "input_shape": [1, 3, 640, 640],
        "output": "Object detection boxes",
        "dtype": "INT8",
        "quantize": True,
        "quant_strategy": "per_channel_exclude_head",
    },
}

ASSETS_DIR = Path(__file__).parent.parent / "android" / "app" / "src" / "main" / "assets"


def export_yolov8(variant: str, output_path: Path, quantize: bool = False,
                  quant_strategy: str = "baseline") -> bool:
    """Export YOLOv8 variant from ultralytics to ONNX.

    Args:
        variant: Model variant letter (e.g., "n", "m", "l", "x")
        output_path: Destination path for the ONNX model
        quantize: Whether to quantize to INT8
        quant_strategy: baseline / per_channel / per_channel_exclude_head
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Install with: pip install ultralytics")
        return False

    model_name = f"yolov8{variant}"
    pt_file = f"{model_name}.pt"

    try:
        # For quantization, reuse existing FP32 model if available in assets
        if quantize:
            existing_fp32 = output_path.parent / f"{model_name}.onnx"
            if existing_fp32.exists():
                print(f"Reusing existing FP32 model: {existing_fp32}")
                fp32_path = output_path.with_suffix(".fp32.onnx")
                shutil.copy(str(existing_fp32), str(fp32_path))
                return quantize_onnx_model(fp32_path, output_path,
                                           input_shape=[1, 3, 640, 640],
                                           strategy=quant_strategy)

        from ultralytics import YOLO
        print(f"Loading {model_name} from ultralytics...")
        model = YOLO(pt_file)

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

        # ultralytics saves to <model_name>.onnx in current directory
        exported_file = Path(f"{model_name}.onnx")
        if exported_file.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported_file), str(fp32_path))
            print(f"Exported: {fp32_path.name} ({fp32_path.stat().st_size / 1024 / 1024:.2f} MB)")

            # Cleanup downloaded .pt file
            pt_path = Path(pt_file)
            if pt_path.exists():
                pt_path.unlink()

            if quantize:
                return quantize_onnx_model(fp32_path, output_path,
                                           input_shape=[1, 3, 640, 640],
                                           strategy=quant_strategy)

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
        rng = np.random.default_rng(42)
        self.data = [
            rng.random(input_shape, dtype=np.float32)
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
        _, _, height, width = input_shape

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

                # Letterbox resize (preserve aspect ratio + gray padding)
                # to match YOLOv8 inference preprocessing
                scale = min(width / img.width, height / img.height)
                new_w, new_h = int(img.width * scale), int(img.height * scale)
                img_resized = img.resize((new_w, new_h))
                canvas = Image.new("RGB", (width, height), (114, 114, 114))
                paste_x = (width - new_w) // 2
                paste_y = (height - new_h) // 2
                canvas.paste(img_resized, (paste_x, paste_y))

                img_array = np.array(canvas).astype(np.float32) / 255.0

                # NHWC to NCHW
                img_array = np.transpose(img_array, (2, 0, 1))
                img_array = np.expand_dims(img_array, axis=0)

                # YOLOv8 uses [0,1] range only (no ImageNet mean/std normalization)
                self.data.append(img_array)
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


def quantize_onnx_model(input_path: Path, output_path: Path,
                        input_shape: list = None, strategy: str = "baseline") -> bool:
    """Quantize ONNX model to INT8 (QDQ) using QNN-optimized config.

    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path to save quantized model
        input_shape: Input shape for static quantization calibration
        strategy: baseline / per_channel / per_channel_exclude_head

    Returns:
        True if successful, False otherwise
    """
    try:
        import onnxruntime.quantization  # noqa: F401
        from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config  # noqa: F401
    except ImportError:
        print("Error: onnxruntime.quantization with QNN support not available")
        print("Install with: pip install onnxruntime")
        return False

    try:
        return quantize_static_onnx(input_path, output_path, input_shape, strategy=strategy)

    except Exception as e:
        print(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_for_quantization(input_path: Path) -> Path:
    """Preprocess model before quantization for better results.

    Steps:
    1. Shape inference - provides tensor shape information
    2. Model optimization - merges Conv+BN, etc.

    Returns:
        Path to preprocessed model
    """
    import onnx
    from onnx import shape_inference

    print("  Preprocessing model for quantization...")

    preprocessed_path = input_path.with_suffix(".preprocessed.onnx")

    try:
        # Try using onnxruntime's preprocessing
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

    except ImportError:
        # Fallback: just use onnx shape inference
        print("  [!] onnxruntime.quantization.shape_inference not available")
        print("  Using ONNX shape inference only...")

        model = onnx.load(str(input_path))
        model = shape_inference.infer_shapes(model)
        onnx.save(model, str(preprocessed_path))
        print("  [OK] Shape inference complete")
        return preprocessed_path

    except Exception as e:
        print(f"  [!] Preprocessing failed: {e}")
        print("  Proceeding without preprocessing...")
        return input_path


def get_yolov8_head_nodes(model_path: str) -> list:
    """Extract YOLOv8 detection head node names (model.22/*) from ONNX graph.

    The detection head contains box regression (cv2), classification (cv3),
    DFL processing, and output concat/reshape. Excluding these from quantization
    preserves confidence score precision while keeping backbone+neck in INT8.
    """
    import onnx
    model = onnx.load(str(model_path))
    head_nodes = [node.name for node in model.graph.node if "/model.22/" in node.name]
    return head_nodes


def create_calibration_reader(input_name: str, input_shape: list):
    """Create calibration data reader from available sources."""
    global CALIBRATION_DATA_PATH, CALIBRATION_SAMPLES

    if CALIBRATION_DATA_PATH and Path(CALIBRATION_DATA_PATH).exists():
        print(f"  Using real calibration data: {CALIBRATION_DATA_PATH}")
        return ImageCalibrationDataReader(
            calibration_dir=str(CALIBRATION_DATA_PATH),
            input_name=input_name,
            input_shape=input_shape,
            num_samples=CALIBRATION_SAMPLES
        )

    default_coco = Path(__file__).parent / "coco_val2017" / "val2017"
    if default_coco.exists() and any(default_coco.glob("*.jpg")):
        print(f"  Using COCO val2017 calibration data: {default_coco}")
        return ImageCalibrationDataReader(
            calibration_dir=str(default_coco),
            input_name=input_name,
            input_shape=input_shape,
            num_samples=CALIBRATION_SAMPLES
        )

    print(f"  Using synthetic calibration data ({CALIBRATION_SAMPLES} samples)")
    print("  [!] For better accuracy, run: python eval_accuracy.py --setup")
    return RandomCalibrationDataReader(
        input_name=input_name,
        input_shape=input_shape,
        num_samples=CALIBRATION_SAMPLES
    )


def quantize_static_onnx(input_path: Path, output_path: Path,
                         input_shape: list = None, strategy: str = "baseline") -> bool:
    """Static quantization using QNN-optimized config.

    Strategies:
        baseline:                  per-tensor, full graph quantization
        per_channel:               per-channel Conv weights, full graph
        exclude_head:              per-tensor + detection head (model.22) in FP32
        per_channel_exclude_head:  per-channel + detection head (model.22) in FP32
    """
    global CALIBRATION_DATA_PATH, CALIBRATION_SAMPLES, CALIBRATION_METHOD

    from onnxruntime.quantization import quantize, CalibrationMethod
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
    import onnx

    strategy_labels = {
        "baseline": "per-tensor, full graph",
        "per_channel": "per-channel Conv, full graph",
        "exclude_head": "per-tensor, head excluded (FP32)",
        "per_channel_exclude_head": "per-channel Conv, head excluded (FP32)",
    }
    print(f"Quantizing (QNN QDQ) to INT8: {output_path}")
    print(f"  Strategy: {strategy_labels.get(strategy, strategy)}")

    # Preprocess model for better quantization
    preprocessed_path = preprocess_for_quantization(input_path)

    # Get input info from model
    model = onnx.load(str(preprocessed_path))
    input_info = model.graph.input[0]
    input_name = input_info.name
    if input_shape is None:
        input_shape = [
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in input_info.type.tensor_type.shape.dim
        ]
    print(f"  Input: {input_name} {input_shape}")

    # Create calibration reader
    calibration_reader = create_calibration_reader(input_name, input_shape)

    # Calibration method
    if CALIBRATION_METHOD == "minmax":
        calib_method = CalibrationMethod.MinMax
        print("  Calibration: MinMax")
    else:
        calib_method = CalibrationMethod.Percentile
        print("  Calibration: Percentile (99.99th)")

    # Strategy-specific options
    per_channel = strategy in ("per_channel", "per_channel_exclude_head")
    nodes_to_exclude = None
    if strategy in ("exclude_head", "per_channel_exclude_head"):
        nodes_to_exclude = get_yolov8_head_nodes(str(preprocessed_path))
        print(f"  Excluding {len(nodes_to_exclude)} head nodes (model.22/*)")

    print(f"  Per-channel: {per_channel}")

    # Build QNN-optimized config
    # get_qnn_qdq_config auto-handles:
    #   - QDQ format with QUInt8 activations
    #   - Per-channel axis for Conv weights (axis 0)
    #   - MatMul forced to per-tensor (QNN limitation)
    #   - Proper weight symmetry settings
    qnn_config = get_qnn_qdq_config(
        model_input=str(preprocessed_path),
        calibration_data_reader=calibration_reader,
        calibrate_method=calib_method,
        per_channel=per_channel,
        nodes_to_exclude=nodes_to_exclude,
        activation_symmetric=False,
    )

    quantize(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        quant_config=qnn_config,
    )

    if not output_path.exists():
        print("Error: Quantization produced no output")
        return False

    print(f"Quantized: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Remove intermediate files
    if preprocessed_path != input_path and preprocessed_path.exists():
        preprocessed_path.unlink()
    if input_path != output_path and input_path.exists():
        input_path.unlink()

    return True


def export_model(model_key: str, output_dir: Path) -> list:
    """Export a model by key.

    Returns:
        List of (model_key, quant_method) tuples that were successfully exported.
    """
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        return []

    model = MODELS[model_key]
    quantize = model.get("quantize", False)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []

    if quantize:
        filename = model.get("filename_static", model.get("filename"))
    else:
        filename = model["filename"]

    dest = output_dir / filename

    print("=" * 60)
    print(f"Exporting {model['description']}")
    if quantize:
        print(f"Quantization: static/QDQ → {filename}")
    print("=" * 60)

    success = False
    if model["source"] == "ultralytics":
        variant = model.get("variant", "n")
        quant_strategy = model.get("quant_strategy", "baseline")
        success = export_yolov8(variant, dest, quantize=quantize,
                                quant_strategy=quant_strategy)
    else:
        print(f"Unknown source: {model['source']}")

    if success:
        exported.append((model_key, "static" if quantize else None))

    print()  # Add spacing between exports

    return exported


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
        if model.get("quantize"):
            print(f"    Filename: {model['filename_static']} [static/QDQ, NPU supported]")
        else:
            print(f"    Filename: {model['filename']}")
        print()


def check_assets():
    """Check which models are already exported."""
    print("=" * 60)
    print(f"Model Status (in {ASSETS_DIR})")
    print("=" * 60)
    print()

    print("FP32 models:")
    for model in MODELS.values():
        if not model.get("quantize"):
            path = ASSETS_DIR / model["filename"]
            dtype_tag = f"[{model['dtype']}]"
            if path.exists():
                size = path.stat().st_size / 1024 / 1024
                print(f"  [OK] {model['filename']} ({size:.2f} MB) {dtype_tag}")
            else:
                print(f"  [--] {model['filename']} (not found) {dtype_tag}")

    print()
    print("INT8 quantized models:")
    for model in MODELS.values():
        if model.get("quantize"):
            path_static = ASSETS_DIR / model["filename_static"]
            if path_static.exists():
                size = path_static.stat().st_size / 1024 / 1024
                print(f"  [OK] {model['filename_static']} ({size:.2f} MB) [static/NPU]")
            else:
                print(f"  [--] {model['filename_static']} (not found) [static/NPU]")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export models from ultralytics to ONNX for mobile inference"
    )
    parser.add_argument(
        "--export",
        nargs="+",
        metavar="MODEL",
        help="Model keys to export (e.g., yolov8n yolov8n-quantized yolov8n-quantized-pc)"
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
    parser.add_argument(
        "--calibration-method",
        choices=["minmax", "percentile"],
        default="percentile",
        help="Calibration method: minmax (fast, outlier-sensitive) or percentile (robust, default)"
    )
    args = parser.parse_args()

    # Set global settings
    global CALIBRATION_DATA_PATH, CALIBRATION_SAMPLES, CALIBRATION_METHOD
    CALIBRATION_DATA_PATH = args.calibration_data
    CALIBRATION_SAMPLES = args.calibration_samples
    CALIBRATION_METHOD = args.calibration_method

    if args.list:
        list_models()
        return

    if args.status:
        check_assets()
        return

    exports = []
    if args.export_all:
        exports = list(MODELS.keys())
    elif args.export:
        for key in args.export:
            if key in MODELS:
                exports.append(key)
            else:
                print(f"Unknown model: {key}")
                print(f"Available: {', '.join(MODELS.keys())}")
                return

    if not exports:
        parser.print_help()
        print(f"\nAvailable models: {', '.join(MODELS.keys())}")
        print("\nExamples:")
        print("  python export_to_onnx.py --export yolov8n")
        print("  python export_to_onnx.py --export yolov8n-quantized yolov8n-quantized-pc yolov8n-quantized-pc-noh")
        print("  python export_to_onnx.py --export-all")
        print("  python export_to_onnx.py --list")
        print("  python export_to_onnx.py --status")
        return

    exported = []
    failed = []

    for model_key in exports:
        results = export_model(model_key, args.output)
        if results:
            exported.extend(results)
        else:
            failed.append(model_key)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if exported:
        export_strs = []
        for model_key, method in exported:
            if method:
                export_strs.append(f"{model_key} ({method})")
            else:
                export_strs.append(model_key)
        print(f"Exported: {', '.join(export_strs)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Output directory: {args.output}")
    if any(m for _, m in exported if m):
        print("  [OK] Static (QDQ) quantization supported by QNN EP")

    if exported:
        print()
        print("Next steps:")
        print("  - Use analyze_ops.py to check NPU compatibility")
        print("  - Copy models to android/app/src/main/assets/")


if __name__ == "__main__":
    main()
