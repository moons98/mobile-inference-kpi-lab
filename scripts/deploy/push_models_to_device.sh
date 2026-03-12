#!/bin/bash
# Push ONNX models to Android device for SD v1.5 txt2img benchmark.
#
# Usage:
#   ./scripts/deploy/push_models_to_device.sh           # push all available models
#   ./scripts/deploy/push_models_to_device.sh --fp32     # FP32 only (text_enc + unet_base + unet_lcm + vae_dec)
#   ./scripts/deploy/push_models_to_device.sh --int8     # INT8 only
#   ./scripts/deploy/push_models_to_device.sh --deploy   # push all from weights/deploy/
#   ./scripts/deploy/push_models_to_device.sh --status   # check what's on device

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source directory
SD_ONNX_DIR="$PROJECT_DIR/weights/sd_v1.5/onnx"

# Device target directory
DEVICE_DIR="//sdcard/sd_models"

echo "=== SD v1.5 Txt2Img Model Push ==="
echo "Source: $SD_ONNX_DIR"
echo "Target: $DEVICE_DIR"
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
adb shell "mkdir -p /sdcard/sd_models" 2>/dev/null

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

push_with_data() {
    local src="$1"
    if [ -f "$src" ]; then
        push_file "$src"
        # Push external data file if exists
        if [ -f "${src}.data" ]; then
            push_file "${src}.data"
        fi
    else
        echo "  [MISS] $(basename "$src")"
    fi
}

if [ "$1" = "--status" ]; then
    echo "=== On-device models ($DEVICE_DIR) ==="
    adb shell "ls -la /sdcard/sd_models/ 2>/dev/null" || echo "  Directory not found"
    echo ""

    echo "=== Txt2Img Components ==="
    echo "--- Shared (Text Encoder / VAE Decoder) ---"
    for f in text_encoder_fp32.onnx vae_decoder_fp32.onnx vae_decoder_int8_qdq.onnx; do
        if adb shell "test -f /sdcard/sd_models/$f" 2>/dev/null; then
            echo "  [OK] $f"
        else
            echo "  [--] $f"
        fi
    done

    echo ""
    echo "--- UNet Base (SD v1.5) ---"
    for f in unet_base_fp32.onnx unet_base_fp32.onnx.data; do
        if adb shell "test -f /sdcard/sd_models/$f" 2>/dev/null; then
            echo "  [OK] $f"
        else
            echo "  [--] $f"
        fi
    done

    echo ""
    echo "--- UNet LCM ---"
    for f in unet_lcm_fp32.onnx unet_lcm_fp32.onnx.data; do
        if adb shell "test -f /sdcard/sd_models/$f" 2>/dev/null; then
            echo "  [OK] $f"
        else
            echo "  [--] $f"
        fi
    done

    echo ""
    echo "--- Cached Embeddings ---"
    for f in text_embeddings.npy uncond_embeddings.npy; do
        if adb shell "test -f /sdcard/sd_models/$f" 2>/dev/null; then
            echo "  [OK] $f"
        else
            echo "  [--] $f (generate: python scripts/deploy/generate_cached_embeddings.py)"
        fi
    done
    exit 0
fi

# Push cached text embeddings
push_embeddings() {
    echo "--- Cached Text Embeddings ---"
    for f in text_embeddings.npy uncond_embeddings.npy; do
        if [ -f "$SD_ONNX_DIR/$f" ]; then
            push_file "$SD_ONNX_DIR/$f"
        else
            echo "  [MISS] $f — run: python scripts/deploy/generate_cached_embeddings.py"
        fi
    done
}

# Push txt2img FP32 components
push_sd_fp32() {
    echo "--- Text Encoder FP32 ---"
    push_with_data "$SD_ONNX_DIR/text_encoder_fp32.onnx"

    echo ""
    echo "--- UNet Base (SD v1.5) FP32 ---"
    push_with_data "$SD_ONNX_DIR/unet_base_fp32.onnx"

    echo ""
    echo "--- UNet LCM FP32 ---"
    push_with_data "$SD_ONNX_DIR/unet_lcm_fp32.onnx"

    echo ""
    echo "--- VAE Decoder FP32 ---"
    push_with_data "$SD_ONNX_DIR/vae_decoder_fp32.onnx"
}

push_sd_int8() {
    echo "--- VAE Decoder INT8 ---"
    push_with_data "$SD_ONNX_DIR/vae_decoder_int8_qdq.onnx"

    echo ""
    echo "--- UNet Base INT8 ---"
    push_with_data "$SD_ONNX_DIR/unet_base_int8_qdq.onnx"

    echo ""
    echo "--- UNet LCM INT8 ---"
    push_with_data "$SD_ONNX_DIR/unet_lcm_int8_qdq.onnx"
}

push_deploy() {
    local DEPLOY_DIR="$PROJECT_DIR/weights/deploy"
    echo "--- Deploy Directory (precompiled models) ---"
    if [ ! -d "$DEPLOY_DIR" ]; then
        echo "  [MISS] Deploy dir not found. Run: python scripts/deploy/prepare_deploy_models.py"
        return
    fi
    for f in "$DEPLOY_DIR"/*; do
        [ -f "$f" ] || continue
        local filename="$(basename "$f")"
        case "$filename" in *.onnx.data) continue ;; esac
        push_file "$f"
        if [ -f "${f}.data" ]; then
            push_file "${f}.data"
        fi
    done
}

case "$1" in
    --fp32)
        push_sd_fp32
        echo ""
        push_embeddings
        ;;
    --int8)
        push_sd_int8
        ;;
    --deploy)
        push_deploy
        echo ""
        push_embeddings
        ;;
    *)
        push_sd_fp32
        echo ""
        push_sd_int8
        echo ""
        push_embeddings
        ;;
esac

echo ""
echo "=== Done ==="
adb shell "ls -la /sdcard/sd_models/" 2>/dev/null | tail -20
