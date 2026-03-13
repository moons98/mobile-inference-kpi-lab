#!/bin/bash
# Push ONNX models to Android device for SD v1.5 txt2img benchmark.
#
# Usage:
#   ./scripts/deploy/push_models_to_device.sh           # push all available models
#   ./scripts/deploy/push_models_to_device.sh --fp16    # FP16 only (text_enc + unet_base + unet_lcm + vae_dec)
#   ./scripts/deploy/push_models_to_device.sh --w8a8    # W8A8 only
#   ./scripts/deploy/push_models_to_device.sh --w8a16   # W8A16 only
#   ./scripts/deploy/push_models_to_device.sh --mixed   # Mixed PR only
#   ./scripts/deploy/push_models_to_device.sh --status  # check what's on device

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source directory
ONNX_DIR="$PROJECT_DIR/weights/sd_v1.5/onnx"

# Device target directory
DEVICE_DIR="//sdcard/sd_models"

echo "=== SD v1.5 Txt2Img Model Push ==="
echo "Source: $ONNX_DIR"
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

# push_file: push src with its own basename
push_file() {
    local src="$1"
    local dest_name="${2:-$(basename "$src")}"

    if [ ! -f "$src" ]; then
        echo "  [MISS] $dest_name"
        return
    fi

    local size_mb=$(( $(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null) / 1048576 ))

    # Check if already on device (compare size)
    local local_size=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null)
    local device_size=$(adb shell "stat -c%s $DEVICE_DIR/$dest_name 2>/dev/null" | tr -d '\r')

    if [ "$device_size" = "$local_size" ]; then
        echo "  [SKIP] $dest_name (${size_mb}MB, already on device)"
        return
    fi

    echo "  [PUSH] $dest_name (${size_mb}MB)..."
    adb push "$src" "$DEVICE_DIR/$dest_name"
}

# push model + optional external data
push_model() {
    local src_onnx="$1"
    local dest_onnx="${2:-$(basename "$src_onnx")}"

    push_file "$src_onnx" "$dest_onnx"

    # Try both .onnx.data and .data external data conventions
    local src_data_dot="${src_onnx}.data"
    local src_data_bare="${src_onnx%.onnx}.data"
    local dest_stem="${dest_onnx%.onnx}"

    if [ -f "$src_data_dot" ]; then
        push_file "$src_data_dot" "${dest_onnx}.data"
    elif [ -f "$src_data_bare" ]; then
        # bare .data convention — push keeping the original name reference
        push_file "$src_data_bare" "${dest_stem}.data"
    fi
}

# ──────────────────────────────────────────────
# Status check
# ──────────────────────────────────────────────
if [ "$1" = "--status" ]; then
    echo "=== On-device models ($DEVICE_DIR) ==="
    adb shell "ls -la /sdcard/sd_models/ 2>/dev/null" || echo "  Directory not found"
    echo ""

    check_file() {
        local f="$1"
        if adb shell "test -f /sdcard/sd_models/$f" 2>/dev/null; then
            local sz=$(adb shell "stat -c%s /sdcard/sd_models/$f 2>/dev/null" | tr -d '\r')
            local mb=$(( ${sz:-0} / 1048576 ))
            echo "  [OK] $f (${mb}MB)"
        else
            echo "  [--] $f"
        fi
    }

    echo "--- Text Encoder ---"
    check_file "text_encoder_fp32.onnx"
    check_file "text_encoder_w8a16.onnx"

    echo ""
    echo "--- UNet Base (SD v1.5) ---"
    check_file "unet_base_fp32.onnx"
    check_file "unet_base_fp32.onnx.data"
    check_file "unet_base_int8_qdq.onnx"
    check_file "unet_base_int8_qdq.onnx.data"
    check_file "unet_base_mixed_pr.onnx"
    check_file "unet_base_mixed_pr.onnx.data"
    check_file "unet_base_w8a16.onnx"
    check_file "unet_base_w8a16.onnx.data"

    echo ""
    echo "--- UNet LCM ---"
    check_file "unet_lcm_fp32.onnx"
    check_file "unet_lcm_fp32.onnx.data"
    check_file "unet_lcm_int8_qdq.onnx"
    check_file "unet_lcm_int8_qdq.onnx.data"
    check_file "unet_lcm_mixed_pr.onnx"
    check_file "unet_lcm_mixed_pr.onnx.data"
    check_file "unet_lcm_w8a16.onnx"
    check_file "unet_lcm_w8a16.data"

    echo ""
    echo "--- VAE Decoder ---"
    check_file "vae_decoder_fp32.onnx"
    check_file "vae_decoder_int8_qdq.onnx"
    check_file "vae_decoder_int8_qdq.onnx.data"
    check_file "vae_decoder_w8a16.onnx"

    echo ""
    echo "--- Cached Embeddings ---"
    check_file "text_embeddings.npy"
    check_file "uncond_embeddings.npy"
    exit 0
fi

# ──────────────────────────────────────────────
# Push functions by precision
# ──────────────────────────────────────────────
push_fp16() {
    echo "--- Text Encoder FP16 ---"
    push_model "$ONNX_DIR/text_encoder_fp32.onnx"

    echo ""
    echo "--- UNet Base FP16 ---"
    push_model "$ONNX_DIR/unet_base_fp32.onnx"

    echo ""
    echo "--- UNet LCM FP16 ---"
    push_model "$ONNX_DIR/unet_lcm_fp32.onnx"

    echo ""
    echo "--- VAE Decoder FP16 ---"
    push_model "$ONNX_DIR/vae_decoder_fp32.onnx"
}

push_w8a8() {
    echo "--- UNet Base W8A8 ---"
    push_model "$ONNX_DIR/unet_base_int8_qdq.onnx"

    echo ""
    echo "--- UNet LCM W8A8 ---"
    push_model "$ONNX_DIR/unet_lcm_int8_qdq.onnx"

    echo ""
    # vae_decoder_qai_int8.onnx → pushed as vae_decoder_int8_qdq.onnx (app-expected name)
    echo "--- VAE Decoder W8A8 ---"
    push_model "$ONNX_DIR/vae_decoder_qai_int8.onnx" "vae_decoder_int8_qdq.onnx"
    if [ -f "$ONNX_DIR/vae_decoder_qai_int8.onnx.data" ]; then
        push_file "$ONNX_DIR/vae_decoder_qai_int8.onnx.data" "vae_decoder_int8_qdq.onnx.data"
    fi
}

push_w8a16() {
    echo "--- Text Encoder W8A16 ---"
    push_model "$ONNX_DIR/text_encoder_w8a16.onnx"

    echo ""
    echo "--- UNet Base W8A16 ---"
    push_model "$ONNX_DIR/unet_base_w8a16.onnx"

    echo ""
    # unet_lcm_w8a16 uses bare .data convention (no .onnx prefix)
    echo "--- UNet LCM W8A16 ---"
    push_model "$ONNX_DIR/unet_lcm_w8a16.onnx"

    echo ""
    echo "--- VAE Decoder W8A16 ---"
    push_model "$ONNX_DIR/vae_decoder_w8a16.onnx"
}

push_mixed() {
    echo "--- UNet Base Mixed PR ---"
    push_model "$ONNX_DIR/unet_base_mixed_pr.onnx"

    echo ""
    echo "--- UNet LCM Mixed PR ---"
    push_model "$ONNX_DIR/unet_lcm_mixed_pr.onnx"
}

push_embeddings() {
    echo "--- Cached Text Embeddings ---"
    for f in text_embeddings.npy uncond_embeddings.npy; do
        if [ -f "$ONNX_DIR/$f" ]; then
            push_file "$ONNX_DIR/$f"
        else
            echo "  [MISS] $f — run: C:/Users/munsi/miniconda3/envs/mobile/python.exe scripts/deploy/extract_tokenizer_assets.py"
        fi
    done
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
case "$1" in
    --fp16)
        push_fp16
        echo ""
        push_embeddings
        ;;
    --w8a8)
        push_w8a8
        ;;
    --w8a16)
        push_w8a16
        ;;
    --mixed)
        push_mixed
        ;;
    *)
        push_fp16
        echo ""
        push_w8a8
        echo ""
        push_w8a16
        echo ""
        push_mixed
        echo ""
        push_embeddings
        ;;
esac

echo ""
echo "=== Done ==="
adb shell "ls -la /sdcard/sd_models/" 2>/dev/null | awk '{print $5, $9}' | column -t
