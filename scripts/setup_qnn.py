#!/usr/bin/env python3
"""
QNN SDK Setup Script

Automatically finds and copies required QNN libraries to the Android project.
Also validates SDK structure and provides setup guidance.

Usage:
    python setup_qnn.py --sdk-path C:/Qualcomm/v2.42.0.251225
    python setup_qnn.py --sdk-path C:/Qualcomm/v2.42.0.251225 --target v73
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Required libraries for Android arm64
REQUIRED_LIBS = [
    "libQnnHtp.so",          # HTP (NPU) backend
    "libQnnGpu.so",          # GPU backend
    "libQnnHtpPrepare.so",   # HTP preparation
    "libQnnSystem.so",       # System utilities (if exists)
]

# HTP version-specific stub libraries
HTP_STUB_LIBS = {
    "v66": "libQnnHtpV66Stub.so",
    "v68": "libQnnHtpV68Stub.so",
    "v69": "libQnnHtpV69Stub.so",
    "v73": "libQnnHtpV73Stub.so",
    "v75": "libQnnHtpV75Stub.so",
    "v79": "libQnnHtpV79Stub.so",
    "v81": "libQnnHtpV81Stub.so",
}

# Chipset to HTP version mapping
CHIPSET_HTP_MAP = {
    "sd865": "v66",
    "sd888": "v68",
    "sd8gen1": "v69",
    "sd8gen2": "v73",
    "sd8gen3": "v75",
}


def find_sdk_root(sdk_path: str) -> Optional[Path]:
    """
    Find the actual SDK root containing lib/ and include/ directories.
    Handles nested directory structures like qairt/x.x.x/
    """
    sdk_path = Path(sdk_path)

    # Check if this is already the SDK root
    if (sdk_path / "lib").exists() and (sdk_path / "include").exists():
        return sdk_path

    # Try common nested structures
    possible_paths = [
        sdk_path,
        sdk_path / "qairt",
    ]

    # Add version subdirectories
    for p in list(possible_paths):
        if p.exists():
            for subdir in p.iterdir():
                if subdir.is_dir() and (subdir / "lib").exists():
                    possible_paths.append(subdir)

    # Check each path
    for p in possible_paths:
        if (p / "lib" / "aarch64-android").exists():
            return p

    return None


def get_android_lib_dir(sdk_root: Path) -> Path:
    """Get the aarch64-android library directory."""
    return sdk_root / "lib" / "aarch64-android"


def get_hexagon_lib_dir(sdk_root: Path, version: str) -> Optional[Path]:
    """Get the hexagon library directory for a specific version."""
    hexagon_dir = sdk_root / "lib" / f"hexagon-{version}"
    if hexagon_dir.exists():
        return hexagon_dir
    return None


def list_available_htp_versions(sdk_root: Path) -> List[str]:
    """List available HTP versions in the SDK."""
    lib_dir = sdk_root / "lib"
    versions = []

    for d in lib_dir.iterdir():
        if d.is_dir() and d.name.startswith("hexagon-v"):
            version = d.name.replace("hexagon-", "")
            versions.append(version)

    return sorted(versions)


def copy_libraries(
    sdk_root: Path,
    dest_dir: Path,
    target_versions: List[str],
    dry_run: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Copy required libraries to destination.

    Returns:
        Tuple of (copied_files, missing_files)
    """
    android_lib_dir = get_android_lib_dir(sdk_root)
    copied = []
    missing = []

    # Create destination directory
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy required base libraries
    for lib_name in REQUIRED_LIBS:
        src = android_lib_dir / lib_name
        if src.exists():
            if not dry_run:
                shutil.copy2(src, dest_dir / lib_name)
            copied.append(lib_name)
            print(f"  [OK] {lib_name}")
        else:
            # Some libraries may be optional
            if "System" not in lib_name:
                missing.append(lib_name)
                print(f"  [MISSING] {lib_name}")
            else:
                print(f"  [SKIP] {lib_name} (optional)")

    # Copy HTP stub libraries for target versions
    for version in target_versions:
        stub_name = HTP_STUB_LIBS.get(version)
        if stub_name:
            src = android_lib_dir / stub_name
            if src.exists():
                if not dry_run:
                    shutil.copy2(src, dest_dir / stub_name)
                copied.append(stub_name)
                print(f"  [OK] {stub_name} ({version})")
            else:
                missing.append(stub_name)
                print(f"  [MISSING] {stub_name}")

    return copied, missing


def setup_environment_hint(sdk_root: Path):
    """Print environment setup hints."""
    print("\n" + "=" * 60)
    print("Environment Setup")
    print("=" * 60)
    print(f"\nAdd to your shell profile or run before building:\n")
    print(f"  # Windows (PowerShell)")
    print(f'  $env:QNN_SDK_ROOT = "{sdk_root}"')
    print(f"\n  # Windows (CMD)")
    print(f'  set QNN_SDK_ROOT={sdk_root}')
    print(f"\n  # Linux/Mac")
    print(f'  export QNN_SDK_ROOT="{sdk_root}"')


def main():
    parser = argparse.ArgumentParser(
        description="Setup QNN SDK libraries for Android project"
    )
    parser.add_argument(
        "--sdk-path",
        required=True,
        help="Path to QNN SDK (e.g., C:/Qualcomm/v2.42.0.251225)"
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=["v69", "v73", "v75"],
        help="Target HTP versions (default: v69 v73 v75)"
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory (default: android/app/src/main/jniLibs/arm64-v8a)"
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Copy all available HTP version stubs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying"
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List available HTP versions and exit"
    )

    args = parser.parse_args()

    # Find SDK root
    print(f"Looking for QNN SDK at: {args.sdk_path}")
    sdk_root = find_sdk_root(args.sdk_path)

    if not sdk_root:
        print(f"\nError: Could not find QNN SDK at {args.sdk_path}")
        print("Expected structure: <sdk>/lib/aarch64-android/")
        sys.exit(1)

    print(f"Found SDK root: {sdk_root}")

    # List available versions
    available_versions = list_available_htp_versions(sdk_root)
    print(f"\nAvailable HTP versions: {', '.join(available_versions)}")

    if args.list_versions:
        print("\nChipset mapping:")
        for chipset, version in CHIPSET_HTP_MAP.items():
            print(f"  {chipset}: {version}")
        sys.exit(0)

    # Determine target versions
    if args.all_versions:
        target_versions = [v.replace("v", "") for v in available_versions]
        target_versions = [f"v{v}" for v in target_versions]
    else:
        target_versions = args.target

    # Normalize version format
    target_versions = [v if v.startswith("v") else f"v{v}" for v in target_versions]
    print(f"Target HTP versions: {', '.join(target_versions)}")

    # Determine destination
    if args.dest:
        dest_dir = Path(args.dest)
    else:
        # Find project root (look for android/ directory)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        dest_dir = project_root / "android" / "app" / "src" / "main" / "jniLibs" / "arm64-v8a"

    print(f"\nDestination: {dest_dir}")

    if args.dry_run:
        print("\n[DRY RUN - No files will be copied]")

    # Copy libraries
    print("\nCopying libraries:")
    copied, missing = copy_libraries(
        sdk_root,
        dest_dir,
        target_versions,
        dry_run=args.dry_run
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Copied: {len(copied)} files")
    print(f"  Missing: {len(missing)} files")

    if missing:
        print(f"\n  Warning: Missing files: {', '.join(missing)}")

    # Environment hints
    setup_environment_hint(sdk_root)

    # Next steps
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
1. For devices, push skeleton libraries:
   adb push <sdk>/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so /data/local/tmp/

2. Build the Android app:
   cd android && ./gradlew assembleDebug

3. Install and run:
   adb install app/build/outputs/apk/debug/app-debug.apk
""")


if __name__ == "__main__":
    main()
