"""
Prepare deploy-ready model files in weights/deploy/.

Reads deploy_config.json to find QAI Hub compile outputs (stub .onnx + .bin)
in qai-outputs/ and copies them with the correct deploy filenames.
Patches ep_cache_context in stub .onnx to point to the renamed .bin.

Usage:
    python scripts/deploy/prepare_deploy_models.py           # collect compiled models
    python scripts/deploy/prepare_deploy_models.py --force   # force overwrite
    python scripts/deploy/prepare_deploy_models.py --clean   # remove .data files from deploy/
"""
import argparse
import json
import shutil
from pathlib import Path

import onnx

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DEPLOY_DIR = PROJECT_DIR / "weights" / "deploy"
QAI_OUTPUTS_DIR = PROJECT_DIR / "qai-outputs"
DEPLOY_CONFIG = Path(__file__).resolve().parent / "deploy_config.json"


def deploy_compiled(force: bool = False):
    """Copy precompiled models (stub .onnx + .bin) from qai-outputs/ based on deploy_config.json."""
    if not DEPLOY_CONFIG.exists():
        print(f"  [ERROR] {DEPLOY_CONFIG} not found")
        return

    with open(DEPLOY_CONFIG, encoding="utf-8") as f:
        config = json.load(f)

    for entry in config.get("compiled", []):
        deploy_name = entry["deploy_name"]
        job_dir = QAI_OUTPUTS_DIR / entry["job_dir"]
        precision = entry.get("precision", "")

        modified_stub = job_dir / "model.modified.onnx"
        stub_src = modified_stub if modified_stub.exists() else job_dir / "model.onnx"
        bin_src = job_dir / "model.bin"

        dst_onnx = DEPLOY_DIR / f"{deploy_name}.onnx"
        dst_bin = DEPLOY_DIR / f"{deploy_name}.bin"

        if not job_dir.exists():
            print(f"  [MISS] {deploy_name}: job_dir not found: {job_dir}")
            continue
        stub_label = "model.modified.onnx" if modified_stub.exists() else "model.onnx"
        if not stub_src.exists():
            print(f"  [MISS] {deploy_name}: model.onnx not found in {job_dir}")
            continue
        if not bin_src.exists():
            print(f"  [MISS] {deploy_name}: model.bin not found in {job_dir}")
            continue

        if dst_onnx.exists() and dst_bin.exists() and not force:
            print(f"  [SKIP] {deploy_name} (already exists)")
            continue

        stub_size = stub_src.stat().st_size
        bin_size_mb = bin_src.stat().st_size / 1024 / 1024

        print(f"  [COPY] {deploy_name}.onnx ({stub_size:,d} B stub, src={stub_label}) + .bin ({bin_size_mb:.1f} MB)  [{precision}]")

        # Patch ep_cache_context in stub to point to renamed .bin
        model = onnx.load(str(stub_src))
        patched = False
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.name == "ep_cache_context" and attr.type == 3:  # STRING
                    old_val = attr.s.decode()
                    new_val = f"./{deploy_name}.bin"
                    attr.s = new_val.encode()
                    print(f"         ep_cache_context: {old_val} -> {new_val}")
                    patched = True
        if not patched:
            print(f"  [WARN] {deploy_name}: ep_cache_context not found in stub, copying as-is")
        onnx.save(model, str(dst_onnx))

        shutil.copy2(bin_src, dst_bin)


def clean_data_files():
    """List .onnx.data files in deploy/ for removal confirmation."""
    if not DEPLOY_DIR.exists():
        return
    data_files = sorted(DEPLOY_DIR.glob("*.data"))
    if not data_files:
        print("  No .data files found in deploy/")
        return

    print(f"  Found {len(data_files)} .data files:")
    total = 0
    for f in data_files:
        size = f.stat().st_size
        total += size
        print(f"    {f.name:45s} {size / 1024 / 1024:>8.1f} MB")
    print(f"    Total: {total / 1024 / 1024 / 1024:.2f} GB")

    answer = input("\n  Delete these files? [y/N] ").strip().lower()
    if answer == "y":
        for f in data_files:
            f.unlink()
            print(f"    [DEL] {f.name}")
        print("  Done.")
    else:
        print("  Skipped.")


def print_summary():
    """Print contents of deploy directory."""
    print("\n--- Deploy Directory ---")
    if not DEPLOY_DIR.exists():
        print("  (empty)")
        return

    total_size = 0
    for f in sorted(DEPLOY_DIR.iterdir()):
        size = f.stat().st_size
        total_size += size
        if size > 1024 * 1024:
            print(f"  {f.name:45s} {size / 1024 / 1024:>8.1f} MB")
        else:
            print(f"  {f.name:45s} {size:>8,d} B")
    print(f"\n  Total: {total_size / 1024 / 1024 / 1024:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Prepare deploy models (precompiled stub+bin)")
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    parser.add_argument("--clean", action="store_true", help="Remove leftover .data files from deploy/")
    args = parser.parse_args()

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean:
        print("--- Clean .data files from deploy/ ---")
        clean_data_files()
        print_summary()
        return

    print("=" * 60)
    print(f"Preparing deploy models in {DEPLOY_DIR}")
    print(f"Config: {DEPLOY_CONFIG}")
    print("=" * 60)

    print("\n--- Precompiled Models (stub .onnx + .bin from QAI Hub) ---")
    deploy_compiled(force=args.force)

    print_summary()


if __name__ == "__main__":
    main()
