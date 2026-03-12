#!/usr/bin/env python3
"""
Extract CLIP tokenizer assets (vocab.json + merges.txt) for Android app.

These files are required by the on-device Tokenizer.kt for SD v1.5 inpainting.
Output goes to android/app/src/main/assets/ for APK bundling.

Usage:
    python scripts/deploy/extract_tokenizer_assets.py
    python scripts/deploy/extract_tokenizer_assets.py --output /path/to/output
"""

import argparse
import json
import shutil
from pathlib import Path

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "sd_v1.5_inpaint"
DEFAULT_OUTPUT = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"


def extract_tokenizer(output_dir: Path):
    from transformers import CLIPTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try local weights first
    if WEIGHTS_DIR.exists() and (WEIGHTS_DIR / "tokenizer").exists():
        source = str(WEIGHTS_DIR / "tokenizer")
        print(f"Loading tokenizer from local: {source}")
    else:
        source = SD_MODEL_ID
        print(f"Loading tokenizer from HuggingFace: {SD_MODEL_ID}")

    tokenizer = CLIPTokenizer.from_pretrained(
        source if source != SD_MODEL_ID else SD_MODEL_ID,
        subfolder="tokenizer" if source == SD_MODEL_ID else None,
    )

    # vocab.json
    vocab = tokenizer.get_vocab()
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  vocab.json: {len(vocab)} tokens ({vocab_path.stat().st_size / 1024:.0f} KB)")

    # merges.txt
    merges_path = output_dir / "merges.txt"

    # CLIPTokenizer stores merges in bpe_ranks
    tokenizer_dir = Path(source)
    local_merges = tokenizer_dir / "merges.txt"
    if local_merges.exists():
        shutil.copy2(local_merges, merges_path)
    else:
        # Save from tokenizer internals
        tokenizer.save_pretrained(str(output_dir / "_tmp_tokenizer"))
        tmp_merges = output_dir / "_tmp_tokenizer" / "merges.txt"
        if tmp_merges.exists():
            shutil.move(str(tmp_merges), str(merges_path))
        # Cleanup
        tmp_dir = output_dir / "_tmp_tokenizer"
        if tmp_dir.exists():
            shutil.rmtree(str(tmp_dir))

    merge_lines = merges_path.read_text().strip().split("\n")
    print(f"  merges.txt: {len(merge_lines) - 1} merge rules ({merges_path.stat().st_size / 1024:.0f} KB)")

    # Verify
    test_text = "remove the object and fill the background naturally"
    tokens = tokenizer.encode(test_text)
    print(f"\n  Verification: \"{test_text}\"")
    print(f"  Token IDs ({len(tokens)}): {tokens[:10]}...")
    print(f"\n  Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP tokenizer assets for Android")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    extract_tokenizer(args.output)


if __name__ == "__main__":
    main()


