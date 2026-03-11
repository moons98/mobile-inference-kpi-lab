#!/usr/bin/env python3
"""
Export precomputed text embeddings for the fixed inpainting prompt.

Since the prompt is always "remove the object and fill the background naturally",
we can run the text encoder once and save the output as a .npy file.
The Android app can then load this file instead of running the text encoder model,
saving ~237MB of model storage and one ORT session.

Output: weights/deploy/text_embeddings.npy  (shape: [1, 77, 768], float32, ~236KB)

Usage:
    python scripts/sd/export_text_embeddings.py
"""

import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "weights" / "sd_v1.5_inpaint" / "onnx"
ASSETS_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"
DEPLOY_DIR = PROJECT_ROOT / "weights" / "deploy"

PROMPT = "remove the object and fill the background naturally"


def main():
    import onnxruntime as ort
    from transformers import CLIPTokenizer

    print("=" * 60)
    print("Export Text Embeddings (fixed prompt)")
    print(f"Prompt: \"{PROMPT}\"")
    print("=" * 60)

    # 1. Tokenize
    vocab_path = ASSETS_DIR / "vocab.json"
    merges_path = ASSETS_DIR / "merges.txt"
    if vocab_path.exists() and merges_path.exists():
        tokenizer = CLIPTokenizer(str(vocab_path), str(merges_path))
        print(f"Tokenizer: local assets (vocab.json + merges.txt)")
    else:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        print(f"Tokenizer: HuggingFace pretrained")

    tokens = tokenizer(
        PROMPT,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="np",
    )
    input_ids = tokens["input_ids"].astype(np.int64)  # [1, 77]
    print(f"Token IDs shape: {input_ids.shape}")
    print(f"Token IDs: {input_ids[0, :10].tolist()} ...")

    # 2. Run text encoder
    model_path = MODEL_DIR / "text_encoder_fp32.onnx"
    if not model_path.exists():
        print(f"[ERROR] Text encoder not found: {model_path}")
        return

    print(f"\nLoading text encoder: {model_path.name} ...", end="", flush=True)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    print(" OK")

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"Input: {input_name}, Outputs: {output_names}")

    print("Running text encoder ...", end="", flush=True)
    t0 = time.perf_counter()
    outputs = sess.run(None, {input_name: input_ids})
    dt = time.perf_counter() - t0
    print(f" ({dt:.2f}s)")

    # last_hidden_state is the text embedding used by UNet
    text_embeddings = outputs[0]  # [1, 77, 768]
    print(f"\nText embeddings shape: {text_embeddings.shape}")
    print(f"Dtype: {text_embeddings.dtype}")
    print(f"Range: [{text_embeddings.min():.4f}, {text_embeddings.max():.4f}]")
    print(f"Size: {text_embeddings.nbytes / 1024:.1f} KB")

    # 3. Save
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEPLOY_DIR / "text_embeddings.npy"
    np.save(str(out_path), text_embeddings)
    file_size = out_path.stat().st_size
    print(f"\nSaved: {out_path}  ({file_size / 1024:.1f} KB)")

    # 4. Verify round-trip
    loaded = np.load(str(out_path))
    assert np.array_equal(text_embeddings, loaded), "Round-trip verification failed!"
    print("Round-trip verification: OK")


if __name__ == "__main__":
    main()
