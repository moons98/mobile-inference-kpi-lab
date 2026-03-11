"""
Prepare deploy-ready model files in weights/deploy/.

All ONNX models are converted to external data format (.onnx + .onnx.data)
for consistent handling and efficient device storage.
Precompiled models (.onnx stub + .bin) are copied as-is.

Usage:
    python scripts/prepare_deploy_models.py [--force]
"""
import argparse
import shutil
from pathlib import Path

import onnx
from onnx.external_data_helper import convert_model_to_external_data

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEPLOY_DIR = PROJECT_DIR / "weights" / "deploy"

SD_ONNX_DIR = PROJECT_DIR / "weights" / "sd_v1.5_inpaint" / "onnx"
YOLO_ONNX_DIR = PROJECT_DIR / "weights" / "yolov8n_seg" / "onnx"

# Models to deploy: (source_path, deploy_filename)
ONNX_MODELS = {
    # SD Inpainting FP32
    "vae_encoder_fp32": SD_ONNX_DIR / "vae_encoder_fp32.onnx",
    "text_encoder_fp32": SD_ONNX_DIR / "text_encoder_fp32.onnx",
    "unet_fp32": SD_ONNX_DIR / "unet_fp32.onnx",
    "vae_decoder_fp32": SD_ONNX_DIR / "vae_decoder_fp32.onnx",
    # SD Inpainting INT8 QDQ (local quantization)
    "vae_encoder_int8_qdq": SD_ONNX_DIR / "vae_encoder_int8_qdq.onnx",
    "unet_int8_qdq": SD_ONNX_DIR / "unet_int8_qdq.onnx",
    "vae_decoder_int8_qdq": SD_ONNX_DIR / "vae_decoder_int8_qdq.onnx",
    # YOLO-seg FP32
    "yolov8n-seg_fp32": YOLO_ONNX_DIR / "yolov8n-seg_fp32.onnx",
    # YOLO-seg INT8 QDQ (local quantization)
    "yolov8n-seg_int8_qdq": YOLO_ONNX_DIR / "yolov8n-seg_int8_qdq.onnx",
}

# Precompiled models (stub .onnx + .bin) — copy as-is
PRECOMPILED_MODELS = {
    "yolov8n-seg_compiled": (
        YOLO_ONNX_DIR / "yolov8n-seg_compiled.onnx",
        YOLO_ONNX_DIR / "yolov8n-seg_compiled.bin",
    ),
}


def convert_to_external_data(src_onnx: Path, dst_onnx: Path, force: bool = False):
    """Load ONNX model and save with external data format."""
    dst_data = Path(str(dst_onnx) + ".data")

    if dst_onnx.exists() and dst_data.exists() and not force:
        src_size = src_onnx.stat().st_size
        data_path = Path(str(src_onnx) + ".data")
        if data_path.exists():
            src_size += data_path.stat().st_size
        dst_size = dst_onnx.stat().st_size + dst_data.stat().st_size
        # Allow 1% tolerance for size comparison
        if abs(src_size - dst_size) / max(src_size, 1) < 0.01:
            print(f"  [SKIP] {dst_onnx.name} (already exists)")
            return

    # If source already has external data, just copy both files
    src_data = Path(str(src_onnx) + ".data")
    if src_data.exists():
        print(f"  [COPY] {src_onnx.name} + .data (already external data)")
        shutil.copy2(src_onnx, dst_onnx)
        shutil.copy2(src_data, dst_data)
    else:
        print(f"  [CONV] {src_onnx.name} -> {dst_onnx.name} + .data")
        model = onnx.load(str(src_onnx), load_external_data=True)
        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=dst_data.name,
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save(model, str(dst_onnx))

    onnx_size = dst_onnx.stat().st_size / 1024
    data_size = dst_data.stat().st_size / 1024 / 1024
    print(f"         .onnx: {onnx_size:.1f} KB, .data: {data_size:.1f} MB")


def copy_precompiled(name: str, stub_src: Path, bin_src: Path, force: bool = False):
    """Copy precompiled stub .onnx + .bin as-is."""
    stub_dst = DEPLOY_DIR / stub_src.name
    bin_dst = DEPLOY_DIR / bin_src.name

    if stub_dst.exists() and bin_dst.exists() and not force:
        print(f"  [SKIP] {name} (already exists)")
        return

    print(f"  [COPY] {stub_src.name} ({stub_src.stat().st_size} B)")
    shutil.copy2(stub_src, stub_dst)

    print(f"  [COPY] {bin_src.name} ({bin_src.stat().st_size / 1024 / 1024:.1f} MB)")
    shutil.copy2(bin_src, bin_dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare deploy models")
    parser.add_argument("--force", action="store_true", help="Force re-conversion")
    args = parser.parse_args()

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Preparing deploy models in {DEPLOY_DIR}")
    print("=" * 60)

    # ONNX models → external data format
    print("\n--- ONNX Models (external data format) ---")
    for name, src_path in ONNX_MODELS.items():
        if not src_path.exists():
            print(f"  [MISS] {name}: {src_path}")
            continue
        dst_path = DEPLOY_DIR / src_path.name
        convert_to_external_data(src_path, dst_path, force=args.force)

    # Precompiled models → copy
    print("\n--- Precompiled Models (stub + bin) ---")
    for name, (stub, bin_file) in PRECOMPILED_MODELS.items():
        if not stub.exists() or not bin_file.exists():
            print(f"  [MISS] {name}: stub={stub.exists()}, bin={bin_file.exists()}")
            continue
        copy_precompiled(name, stub, bin_file, force=args.force)

    # Summary
    print("\n--- Deploy Directory ---")
    total_size = 0
    for f in sorted(DEPLOY_DIR.iterdir()):
        size = f.stat().st_size
        total_size += size
        if size > 1024 * 1024:
            print(f"  {f.name:45s} {size / 1024 / 1024:>8.1f} MB")
        else:
            print(f"  {f.name:45s} {size:>8,d} B")
    print(f"\n  Total: {total_size / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()
