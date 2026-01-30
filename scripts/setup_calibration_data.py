#!/usr/bin/env python3
"""
Download and setup calibration data for static quantization.

Supported datasets:
- ImageNet: Sample images for classification models (MobileNetV2)
- COCO: Sample images for detection models (YOLOv8)

Usage:
    python setup_calibration_data.py --download-imagenet
    python setup_calibration_data.py --download-coco
    python setup_calibration_data.py --download-all
    python setup_calibration_data.py --status
"""

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

# Calibration data directory
CALIBRATION_DIR = Path(__file__).parent / "calibration_data"

# Sample image sources
DATASETS = {
    "imagenet": {
        "description": "ImageNet validation samples for classification models",
        "url": "https://github.com/EliSchwartz/imagenet-sample-images/archive/refs/heads/master.zip",
        "subdir": "imagenet",
        "num_images": 1000,
        "input_size": 224,
        "models": ["MobileNetV2"],
    },
    "coco": {
        "description": "COCO validation samples for detection models",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "subdir": "coco",
        "num_images": 5000,
        "input_size": 640,
        "models": ["YOLOv8"],
        "large": True,  # ~1GB download
    },
}


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress indication."""
    print(f"{desc}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()  # New line after progress
        return True

    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip file."""
    print(f"  Extracting to: {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


def download_imagenet_samples() -> bool:
    """Download ImageNet sample images."""
    dataset = DATASETS["imagenet"]
    output_dir = CALIBRATION_DIR / dataset["subdir"]

    print("=" * 60)
    print(f"Downloading {dataset['description']}")
    print("=" * 60)

    if output_dir.exists() and any(output_dir.glob("*.JPEG")) or any(output_dir.glob("*.jpg")):
        print(f"  [OK] Already exists: {output_dir}")
        return True

    # Download zip
    zip_path = CALIBRATION_DIR / "imagenet_samples.zip"
    if not download_file(dataset["url"], zip_path, "Downloading ImageNet samples"):
        return False

    # Extract
    if not extract_zip(zip_path, CALIBRATION_DIR):
        return False

    # Move images to correct location
    extracted_dir = CALIBRATION_DIR / "imagenet-sample-images-master"
    if extracted_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy image files
        count = 0
        for img_file in extracted_dir.glob("*.JPEG"):
            shutil.copy(img_file, output_dir / img_file.name)
            count += 1
        for img_file in extracted_dir.glob("*.jpg"):
            shutil.copy(img_file, output_dir / img_file.name)
            count += 1

        # Cleanup
        shutil.rmtree(extracted_dir)
        zip_path.unlink()

        print(f"  [OK] Downloaded {count} images to {output_dir}")
        return True

    return False


def download_coco_samples(max_images: int = 100) -> bool:
    """Download COCO validation images (subset)."""
    dataset = DATASETS["coco"]
    output_dir = CALIBRATION_DIR / dataset["subdir"]

    print("=" * 60)
    print(f"Downloading {dataset['description']}")
    print("=" * 60)

    if output_dir.exists() and len(list(output_dir.glob("*.jpg"))) >= max_images:
        print(f"  [OK] Already exists: {output_dir}")
        return True

    print(f"  [!] COCO val2017 is ~1GB. Downloading first {max_images} images instead.")
    print("  Using COCO API to download individual images...")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download a subset of COCO images using direct URLs
        # These are from COCO val2017
        base_url = "http://images.cocodataset.org/val2017/"

        # Sample image IDs from COCO val2017
        sample_ids = [
            "000000000139", "000000000285", "000000000632", "000000000724",
            "000000000776", "000000000785", "000000000802", "000000000872",
            "000000000885", "000000001000", "000000001268", "000000001296",
            "000000001353", "000000001425", "000000001490", "000000001503",
            "000000001532", "000000001584", "000000001675", "000000001761",
            "000000001818", "000000001993", "000000002006", "000000002149",
            "000000002153", "000000002157", "000000002261", "000000002299",
            "000000002431", "000000002473", "000000002532", "000000002587",
            "000000002592", "000000002685", "000000002923", "000000003156",
            "000000003255", "000000003501", "000000003553", "000000003934",
            "000000004134", "000000004395", "000000004495", "000000004707",
            "000000004795", "000000004830", "000000005001", "000000005037",
            "000000005060", "000000005193", "000000005529", "000000005586",
            "000000005655", "000000005802", "000000006040", "000000006471",
            "000000006614", "000000006723", "000000006763", "000000006771",
            "000000006818", "000000007088", "000000007108", "000000007278",
            "000000007281", "000000007386", "000000007511", "000000007574",
            "000000007816", "000000007918", "000000008021", "000000008211",
            "000000008277", "000000008532", "000000008690", "000000008762",
            "000000008844", "000000008899", "000000009014", "000000009378",
            "000000009400", "000000009448", "000000009483", "000000009590",
            "000000009769", "000000009772", "000000009891", "000000009914",
            "000000010092", "000000010123", "000000010140", "000000010205",
            "000000010211", "000000010363", "000000010434", "000000010527",
            "000000010583", "000000010707", "000000010764", "000000010977",
        ]

        downloaded = 0
        for img_id in sample_ids[:max_images]:
            img_url = f"{base_url}{img_id}.jpg"
            img_path = output_dir / f"{img_id}.jpg"

            if img_path.exists():
                downloaded += 1
                continue

            try:
                urllib.request.urlretrieve(img_url, img_path)
                downloaded += 1
                print(f"\r  Downloaded: {downloaded}/{max_images}", end="", flush=True)
            except Exception:
                pass  # Skip failed downloads

        print()  # New line
        print(f"  [OK] Downloaded {downloaded} images to {output_dir}")
        return downloaded > 0

    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def check_status():
    """Check calibration data status."""
    print("=" * 60)
    print(f"Calibration Data Status")
    print(f"Directory: {CALIBRATION_DIR}")
    print("=" * 60)
    print()

    for name, dataset in DATASETS.items():
        data_dir = CALIBRATION_DIR / dataset["subdir"]

        if data_dir.exists():
            # Count images
            jpg_count = len(list(data_dir.glob("*.jpg")))
            jpeg_count = len(list(data_dir.glob("*.JPEG")))
            total = jpg_count + jpeg_count

            if total > 0:
                print(f"  [OK] {name}: {total} images")
                print(f"       Path: {data_dir}")
                print(f"       For: {', '.join(dataset['models'])}")
            else:
                print(f"  [--] {name}: directory exists but no images found")
        else:
            print(f"  [--] {name}: not downloaded")
            print(f"       Description: {dataset['description']}")

        print()


def create_calibration_reader_example():
    """Create example code for using calibration data."""
    example_code = '''
# Example: Using real calibration data for static quantization

from pathlib import Path
import numpy as np
from PIL import Image

class ImageNetCalibrationDataReader:
    """Calibration data reader using real ImageNet images."""

    def __init__(self, calibration_dir: str, input_name: str, num_samples: int = 100):
        self.input_name = input_name
        self.num_samples = num_samples
        self.current = 0

        # Load and preprocess images
        calibration_path = Path(calibration_dir)
        image_files = list(calibration_path.glob("*.JPEG")) + list(calibration_path.glob("*.jpg"))
        image_files = image_files[:num_samples]

        self.data = []
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            # NHWC to NCHW
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, axis=0)
            # Normalize (ImageNet mean/std)
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            img_array = (img_array - mean) / std
            self.data.append(img_array.astype(np.float32))

    def get_next(self):
        if self.current >= len(self.data):
            return None
        data = {self.input_name: self.data[self.current]}
        self.current += 1
        return data

    def rewind(self):
        self.current = 0


# Usage in export_to_onnx.py:
# calibration_reader = ImageNetCalibrationDataReader(
#     calibration_dir="models/calibration_data/imagenet",
#     input_name="input",
#     num_samples=100
# )
'''

    example_path = CALIBRATION_DIR / "calibration_reader_example.py"
    example_path.parent.mkdir(parents=True, exist_ok=True)
    example_path.write_text(example_code.strip())
    print(f"  Created example code: {example_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and setup calibration data for static quantization"
    )
    parser.add_argument(
        "--download-imagenet",
        action="store_true",
        help="Download ImageNet sample images (~50MB)"
    )
    parser.add_argument(
        "--download-coco",
        action="store_true",
        help="Download COCO sample images (100 images, ~20MB)"
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all calibration datasets"
    )
    parser.add_argument(
        "--coco-samples",
        type=int,
        default=100,
        help="Number of COCO samples to download (default: 100)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check calibration data status"
    )

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    downloads = []
    if args.download_all:
        downloads = ["imagenet", "coco"]
    else:
        if args.download_imagenet:
            downloads.append("imagenet")
        if args.download_coco:
            downloads.append("coco")

    if not downloads:
        parser.print_help()
        print("\nExamples:")
        print("  python setup_calibration_data.py --download-imagenet")
        print("  python setup_calibration_data.py --download-coco")
        print("  python setup_calibration_data.py --download-all")
        print("  python setup_calibration_data.py --status")
        print("\nCalibration data is used for static quantization to improve accuracy.")
        print("Without real data, random/synthetic data is used (less accurate).")
        return

    success = []
    failed = []

    for dataset in downloads:
        if dataset == "imagenet":
            if download_imagenet_samples():
                success.append(dataset)
            else:
                failed.append(dataset)
        elif dataset == "coco":
            if download_coco_samples(max_images=args.coco_samples):
                success.append(dataset)
            else:
                failed.append(dataset)

    # Create example calibration reader
    if success:
        create_calibration_reader_example()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if success:
        print(f"Downloaded: {', '.join(success)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Calibration data directory: {CALIBRATION_DIR}")

    if success:
        print()
        print("Next steps:")
        print("  1. Use --quant-method static with export_to_onnx.py")
        print("  2. Or modify export_to_onnx.py to use ImageNetCalibrationDataReader")
        print("     (see calibration_reader_example.py)")


if __name__ == "__main__":
    main()
