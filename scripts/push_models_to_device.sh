#!/bin/bash
# Push ONNX models to Android device for AI Eraser benchmark.
#
# Usage:
#   ./scripts/push_models_to_device.sh           # push all available models
#   ./scripts/push_models_to_device.sh --fp32     # FP32 only
#   ./scripts/push_models_to_device.sh --int8     # INT8 only
#   ./scripts/push_models_to_device.sh --yolo     # YOLO-seg only
#   ./scripts/push_models_to_device.sh --status   # check what's on device

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Source directories
SD_ONNX_DIR="$PROJECT_DIR/weights/sd_v1.5_inpaint/onnx"
YOLO_ONNX_DIR="$PROJECT_DIR/weights/yolov8n_seg/onnx"

# Device target directory
DEVICE_DIR="/sdcard/sd_models"

echo "=== AI Eraser Model Push ==="
echo "Source (SD):   $SD_ONNX_DIR"
echo "Source (YOLO): $YOLO_ONNX_DIR"
echo "Target:        $DEVICE_DIR"
echo ""

# Check adb
if ! command -v adb &> /dev/null; then
    echo "ERROR: adb not found. Add Android SDK platform-tools to PATH."
    exit 1
fi

# Check device
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No Android device connected."
    exit 1
fi

DEVICE_MODEL=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')
echo "Device: $DEVICE_MODEL"
echo ""

# Create target directory
adb shell "mkdir -p $DEVICE_DIR" 2>/dev/null

push_file() {
    local src="$1"
    local filename="$(basename "$src")"
    local size_mb=$(( $(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null) / 1048576 ))

    # Check if already on device (compare size)
    local device_size=$(adb shell "stat -c%s $DEVICE_DIR/$filename 2>/dev/null" | tr -d '\r')
    local local_size=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null)

    if [ "$device_size" = "$local_size" ]; then
        echo "  [SKIP] $filename (${size_mb}MB, already on device)"
        return
    fi

    echo "  [PUSH] $filename (${size_mb}MB)..."
    adb push "$src" "$DEVICE_DIR/$filename"
}

if [ "$1" = "--status" ]; then
    echo "=== On-device models ($DEVICE_DIR) ==="
    adb shell "ls -la $DEVICE_DIR/ 2>/dev/null" || echo "  Directory not found"
    echo ""
    echo "=== Required for FP32/FP16 ==="
    for f in vae_encoder_fp32.onnx text_encoder_fp32.onnx unet_fp32.onnx unet_fp32.onnx.data vae_decoder_fp32.onnx; do
        if adb shell "test -f $DEVICE_DIR/$f" 2>/dev/null; then
            echo "  [OK] $f"
        else
            echo "  [--] $f"
        fi
    done
    echo ""
    echo "=== Required for W8A8 ==="
    for f in vae_encoder_int8_qdq.onnx text_encoder_int8_qdq.onnx unet_int8_qdq.onnx vae_decoder_int8_qdq.onnx; do
        if adb shell "test -f $DEVICE_DIR/$f" 2>/dev/null; then
            echo "  [OK] $f"
        else
            echo "  [--] $f"
        fi
    done
    echo ""
    echo "=== YOLO-seg ==="
    for f in yolov8n-seg.onnx yolov8n-seg_int8_qdq.onnx; do
        if adb shell "test -f $DEVICE_DIR/$f" 2>/dev/null; then
            echo "  [OK] $f"
        else
            echo "  [--] $f"
        fi
    done
    exit 0
fi

# Push SD models
push_sd_fp32() {
    echo "--- SD Inpainting FP32 (also used for FP16 runtime) ---"
    for f in vae_encoder_fp32.onnx text_encoder_fp32.onnx unet_fp32.onnx vae_decoder_fp32.onnx; do
        if [ -f "$SD_ONNX_DIR/$f" ]; then
            push_file "$SD_ONNX_DIR/$f"
        else
            echo "  [MISS] $f ??run: python scripts/sd/export_sd_to_onnx.py --export-all --precision fp32"
        fi
    done
    # UNet external data file
    if [ -f "$SD_ONNX_DIR/unet_fp32.onnx.data" ]; then
        push_file "$SD_ONNX_DIR/unet_fp32.onnx.data"
    fi
}

push_sd_int8() {
    echo "--- SD Inpainting INT8 QDQ (W8A8) ---"
    for f in vae_encoder_int8_qdq.onnx text_encoder_int8_qdq.onnx unet_int8_qdq.onnx vae_decoder_int8_qdq.onnx; do
        if [ -f "$SD_ONNX_DIR/$f" ]; then
            push_file "$SD_ONNX_DIR/$f"
        else
            echo "  [MISS] $f ??run: python scripts/sd/export_sd_to_onnx.py --export-all --precision int8"
        fi
    done
    # UNet INT8 external data file
    if [ -f "$SD_ONNX_DIR/unet_int8_qdq.onnx.data" ]; then
        push_file "$SD_ONNX_DIR/unet_int8_qdq.onnx.data"
    fi
}

push_yolo() {
    echo "--- YOLO-seg ---"
    for f in yolov8n-seg.onnx yolov8n-seg_int8_qdq.onnx; do
        if [ -f "$YOLO_ONNX_DIR/$f" ]; then
            push_file "$YOLO_ONNX_DIR/$f"
        else
            echo "  [MISS] $f"
        fi
    done
}

case "$1" in
    --fp32)
        push_sd_fp32
        ;;
    --int8)
        push_sd_int8
        ;;
    --yolo)
        push_yolo
        ;;
    *)
        push_sd_fp32
        echo ""
        push_sd_int8
        echo ""
        push_yolo
        ;;
esac

echo ""
echo "=== Done ==="
adb shell "ls -la $DEVICE_DIR/" 2>/dev/null | tail -20

