#!/usr/bin/env python3
"""
Download TFLite models from TensorFlow Hub and Ultralytics.

Usage:
    python convert_to_tflite.py --download-mobilenetv2-fp32
    python convert_to_tflite.py --download-mobilenetv2-int8
    python convert_to_tflite.py --download-yolov8n-fp32
    python convert_to_tflite.py --download-yolov8n-int8
    python convert_to_tflite.py --download-all
    python convert_to_tflite.py --list
"""

import argparse
import shutil
import urllib.request
from pathlib import Path


MODELS = {
    "mobilenetv2-fp32": {
        "url": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1?lite-format=tflite",
        "filename": "mobilenetv2_fp32.tflite",
        "description": "MobileNetV2 1.0 224x224 (FP32)",
        "input_shape": "1x224x224x3",
        "output": "1001 classes (ImageNet)",
        "dtype": "FP32",
    },
    "mobilenetv2-int8": {
        "url": "https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224_quantized/1/default/1?lite-format=tflite",
        "filename": "mobilenetv2_int8.tflite",
        "description": "MobileNetV2 1.0 224x224 (INT8 Quantized)",
        "input_shape": "1x224x224x3",
        "output": "1001 classes (ImageNet)",
        "dtype": "INT8",
    },
    "yolov8n-fp32": {
        "source": "ultralytics",
        "filename": "yolov8n_fp32.tflite",
        "description": "YOLOv8n Object Detection (FP32)",
        "input_shape": "1x640x640x3",
        "output": "Object detection boxes",
        "dtype": "FP32",
        "int8": False,
    },
    "yolov8n-int8": {
        "source": "ultralytics",
        "filename": "yolov8n_int8.tflite",
        "description": "YOLOv8n Object Detection (INT8 Quantized)",
        "input_shape": "1x640x640x3",
        "output": "Object detection boxes",
        "dtype": "INT8",
        "int8": True,
    },
}

ASSETS_DIR = Path(__file__).parent.parent.parent / "android" / "app" / "src" / "main" / "assets"


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {dest}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded: {dest.name} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_from_tfhub(model_key: str, output_dir: Path) -> bool:
    """Download TFLite model from TensorFlow Hub."""
    model = MODELS[model_key]
    dest = output_dir / model["filename"]

    print("=" * 60)
    print(f"Downloading {model['description']}")
    print("=" * 60)

    return download_file(model["url"], dest)


def download_yolov8n(model_key: str, output_dir: Path) -> bool:
    """Download YOLOv8n and export to TFLite using Ultralytics."""
    model = MODELS[model_key]
    dest = output_dir / model["filename"]
    is_int8 = model.get("int8", False)

    print("=" * 60)
    print(f"Downloading {model['description']}")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Install with: pip install ultralytics")
        return False

    try:
        print("Loading YOLOv8n model...")
        yolo = YOLO("yolov8n.pt")

        print(f"Exporting to TFLite ({'INT8' if is_int8 else 'FP32'})...")
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_int8:
            yolo.export(format="tflite", imgsz=640, int8=True)
            exported_file = Path("yolov8n_saved_model/yolov8n_full_integer_quant.tflite")
            if not exported_file.exists():
                exported_file = Path("yolov8n_full_integer_quant.tflite")
        else:
            yolo.export(format="tflite", imgsz=640)
            exported_file = Path("yolov8n_saved_model/yolov8n_float32.tflite")
            if not exported_file.exists():
                exported_file = Path("yolov8n_float32.tflite")

        if exported_file.exists():
            shutil.move(str(exported_file), str(dest))
            print(f"Exported: {dest.name} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")

            # Cleanup
            for cleanup_path in [Path("yolov8n.pt"), Path("yolov8n_saved_model")]:
                if cleanup_path.exists():
                    if cleanup_path.is_dir():
                        shutil.rmtree(cleanup_path)
                    else:
                        cleanup_path.unlink()

            return True
        else:
            print("Error: TFLite export file not found")
            return False

    except Exception as e:
        print(f"Export failed: {e}")
        return False


def download_model(model_key: str, output_dir: Path) -> bool:
    """Download a model by key."""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        return False

    model = MODELS[model_key]
    if model.get("source") == "ultralytics":
        return download_yolov8n(model_key, output_dir)
    else:
        return download_from_tfhub(model_key, output_dir)


def list_models():
    """List available models."""
    print("=" * 60)
    print("Available Models")
    print("=" * 60)
    print()
    for key, model in MODELS.items():
        print(f"  {key}:")
        print(f"    Description: {model['description']}")
        print(f"    Input: {model['input_shape']}")
        print(f"    Output: {model['output']}")
        print(f"    Data type: {model['dtype']}")
        print(f"    Filename: {model['filename']}")
        print()


def check_assets():
    """Check which models are already downloaded."""
    print("=" * 60)
    print(f"Model Status (in {ASSETS_DIR})")
    print("=" * 60)
    print()

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
        description="Download TFLite models for mobile inference benchmarking"
    )
    parser.add_argument(
        "--download-mobilenetv2-fp32",
        action="store_true",
        help="Download MobileNetV2 FP32 from TensorFlow Hub"
    )
    parser.add_argument(
        "--download-mobilenetv2-int8",
        action="store_true",
        help="Download MobileNetV2 INT8 quantized from TensorFlow Hub"
    )
    parser.add_argument(
        "--download-yolov8n-fp32",
        action="store_true",
        help="Download YOLOv8n FP32 (requires ultralytics)"
    )
    parser.add_argument(
        "--download-yolov8n-int8",
        action="store_true",
        help="Download YOLOv8n INT8 quantized (requires ultralytics)"
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check which models are already downloaded"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=ASSETS_DIR,
        help=f"Output directory (default: {ASSETS_DIR})"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.status:
        check_assets()
        return

    downloads = []
    if args.download_all:
        downloads = list(MODELS.keys())
    else:
        if args.download_mobilenetv2_fp32:
            downloads.append("mobilenetv2-fp32")
        if args.download_mobilenetv2_int8:
            downloads.append("mobilenetv2-int8")
        if args.download_yolov8n_fp32:
            downloads.append("yolov8n-fp32")
        if args.download_yolov8n_int8:
            downloads.append("yolov8n-int8")

    if not downloads:
        parser.print_help()
        print("\nExamples:")
        print("  python convert_to_tflite.py --download-mobilenetv2-fp32")
        print("  python convert_to_tflite.py --download-mobilenetv2-int8")
        print("  python convert_to_tflite.py --download-yolov8n-fp32")
        print("  python convert_to_tflite.py --download-yolov8n-int8")
        print("  python convert_to_tflite.py --download-all")
        print("  python convert_to_tflite.py --list")
        print("  python convert_to_tflite.py --status")
        return

    downloaded = []
    failed = []

    for model_key in downloads:
        if download_model(model_key, args.output):
            downloaded.append(model_key)
        else:
            failed.append(model_key)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if downloaded:
        print(f"Downloaded: {', '.join(downloaded)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
