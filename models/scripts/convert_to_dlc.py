#!/usr/bin/env python3
"""
Convert models to QNN DLC format.

Prerequisites:
1. QNN SDK installed and SNPE_ROOT/QNN_SDK_ROOT set
2. Model in ONNX format

Usage:
    python convert_to_dlc.py --input model.onnx --output model.dlc
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_qnn_sdk():
    """Check if QNN SDK is available."""
    qnn_root = os.environ.get('QNN_SDK_ROOT') or os.environ.get('SNPE_ROOT')

    if not qnn_root:
        print("Error: QNN_SDK_ROOT or SNPE_ROOT environment variable not set")
        print("Please install QNN SDK and set the environment variable")
        return None

    if not Path(qnn_root).exists():
        print(f"Error: QNN SDK path does not exist: {qnn_root}")
        return None

    return qnn_root


def find_converter(qnn_root: str):
    """Find the model converter tool."""
    # Try different possible paths
    possible_paths = [
        Path(qnn_root) / "bin" / "x86_64-linux-clang" / "qnn-onnx-converter",
        Path(qnn_root) / "bin" / "qnn-onnx-converter",
        Path(qnn_root) / "bin" / "snpe-onnx-to-dlc",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # Try finding in PATH
    try:
        result = subprocess.run(
            ["which", "qnn-onnx-converter"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return None


def convert_onnx_to_dlc(
    input_path: str,
    output_path: str,
    input_dims: str = None,
    quantize: bool = False
):
    """
    Convert ONNX model to DLC format.

    Args:
        input_path: Path to input ONNX file
        output_path: Path for output DLC file
        input_dims: Input dimensions override (e.g., "input:1,3,224,224")
        quantize: Whether to apply quantization
    """
    qnn_root = check_qnn_sdk()
    if not qnn_root:
        sys.exit(1)

    converter = find_converter(qnn_root)
    if not converter:
        print("Error: Could not find QNN converter tool")
        print("Expected: qnn-onnx-converter or snpe-onnx-to-dlc")
        sys.exit(1)

    print(f"Using converter: {converter}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Build command
    cmd = [converter, "-i", input_path, "-o", output_path]

    if input_dims:
        cmd.extend(["--input_dim", input_dims])

    if quantize:
        cmd.append("--quantize")

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"\nSuccess! Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion:")
        print(e.stderr)
        sys.exit(1)


def download_mobilenetv3():
    """Download MobileNetV3-Small ONNX model."""
    import urllib.request

    url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv3-small-100-224-opt.onnx"
    output_path = Path(__file__).parent.parent / "original" / "mobilenetv3_small.onnx"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Model already exists: {output_path}")
        return str(output_path)

    print(f"Downloading MobileNetV3-Small...")
    print(f"URL: {url}")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print("https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to QNN DLC format"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input ONNX model path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output DLC model path"
    )
    parser.add_argument(
        "--input-dims",
        help="Input dimensions (e.g., 'input:1,3,224,224')"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable quantization"
    )
    parser.add_argument(
        "--download-mobilenetv3",
        action="store_true",
        help="Download MobileNetV3-Small model"
    )

    args = parser.parse_args()

    if args.download_mobilenetv3:
        model_path = download_mobilenetv3()
        if model_path and not args.input:
            args.input = model_path
            args.output = str(
                Path(__file__).parent.parent / "converted" / "mobilenetv3_small.dlc"
            )

    if not args.input:
        parser.print_help()
        print("\nExample usage:")
        print("  python convert_to_dlc.py --download-mobilenetv3")
        print("  python convert_to_dlc.py -i model.onnx -o model.dlc")
        sys.exit(1)

    if not args.output:
        args.output = Path(args.input).stem + ".dlc"

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    convert_onnx_to_dlc(
        args.input,
        args.output,
        args.input_dims,
        args.quantize
    )


if __name__ == "__main__":
    main()
