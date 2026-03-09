#!/usr/bin/env python3
"""
Evaluate YOLOv8 ONNX model accuracy (mAP) on COCO val2017.

Uses ultralytics' built-in val() for accurate, GPU-accelerated evaluation.
Compares FP32 vs INT8 QDQ variants to measure quantization accuracy loss.

Requirements:
    pip install ultralytics

Usage:
    python eval_accuracy.py --model yolov8n                           # Single FP32 model
    python eval_accuracy.py --model yolov8n --all-variants            # FP32 + all QDQ variants
    python eval_accuracy.py --compare-all                             # All models x all variants
    python eval_accuracy.py --status                                  # Check model status

Note:
    - ultralytics auto-downloads COCO val2017 on first run (~1GB images + annotations)
    - GPU (CUDA) is used automatically if available, otherwise CPU
    - All QDQ variants use standard FP32 I/O and are evaluable by ultralytics val().
"""

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPTS_DIR.parent / "android" / "app" / "src" / "main" / "assets"

MODEL_VARIANTS = {
    "yolov8n": {
        "fp32": "yolov8n.onnx",
        "qdq": "yolov8n_int8_qdq.onnx",
        "qdq_pc": "yolov8n_int8_qdq_pc.onnx",
        "qdq_noh": "yolov8n_int8_qdq_noh.onnx",
        "qdq_pc_noh": "yolov8n_int8_qdq_pc_noh.onnx",
    },
    "yolov8s": {
        "fp32": "yolov8s.onnx",
        "qdq": "yolov8s_int8_qdq.onnx",
        "qdq_pc": "yolov8s_int8_qdq_pc.onnx",
        "qdq_noh": "yolov8s_int8_qdq_noh.onnx",
        "qdq_pc_noh": "yolov8s_int8_qdq_pc_noh.onnx",
    },
    "yolov8m": {
        "fp32": "yolov8m.onnx",
        "qdq": "yolov8m_int8_qdq.onnx",
        "qdq_pc": "yolov8m_int8_qdq_pc.onnx",
        "qdq_noh": "yolov8m_int8_qdq_noh.onnx",
        "qdq_pc_noh": "yolov8m_int8_qdq_pc_noh.onnx",
    },
}


# ============================================================
# ultralytics-based evaluation (FP32, QDQ)
# ============================================================

def eval_with_ultralytics(model_path: Path) -> dict:
    """Evaluate using ultralytics val(). Works for FP32 and QDQ ONNX models."""
    from ultralytics import YOLO

    print(f"  Loading model: {model_path.name}")
    model = YOLO(str(model_path), task="detect")

    print(f"  Running val() on COCO val2017...")
    metrics = model.val(
        data="coco.yaml",
        imgsz=640,
        batch=32,
        workers=8,
        verbose=False,
    )

    return {
        "mAP50_95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "mAP_small": float(metrics.box.maps[0]) if len(metrics.box.maps) > 0 else 0.0,
        "mAP_medium": float(metrics.box.maps[1]) if len(metrics.box.maps) > 1 else 0.0,
        "mAP_large": float(metrics.box.maps[2]) if len(metrics.box.maps) > 2 else 0.0,
        "speed_preprocess_ms": float(metrics.speed.get("preprocess", 0)),
        "speed_inference_ms": float(metrics.speed.get("inference", 0)),
        "speed_postprocess_ms": float(metrics.speed.get("postprocess", 0)),
    }


# ============================================================
# Comparison report
# ============================================================

OUTPUTS_DIR = SCRIPTS_DIR.parent / "outputs"


def evaluate_and_report(model_name: str, variants: list):
    """Evaluate model variants and print comparison table."""
    results = {}

    for variant in variants:
        if variant not in MODEL_VARIANTS.get(model_name, {}):
            continue
        filename = MODEL_VARIANTS[model_name][variant]
        model_path = ASSETS_DIR / filename
        if not model_path.exists():
            print(f"\n[SKIP] {filename} not found")
            continue

        label = f"{model_name} {variant.upper()}"
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label} ({filename})")
        print(f"{'=' * 60}")

        metrics = eval_with_ultralytics(model_path)

        if metrics:
            # Add model file size
            metrics["size_mb"] = model_path.stat().st_size / 1024 / 1024
            results[label] = metrics

    _print_comparison(model_name, results)
    _save_comparison(model_name, results)
    return results


def _format_comparison(model_name: str, results: dict) -> str:
    """Format comparison table as string."""
    if not results:
        return ""

    lines = []
    lines.append(f"{'=' * 100}")
    lines.append(f"Accuracy Comparison: {model_name}")
    lines.append(f"{'=' * 100}")
    lines.append(f"{'Model':<25} {'Size(MB)':>8} {'mAP@.5:.95':>10} {'Delta':>8} "
                 f"{'mAP@.5':>10} {'mAP@.75':>10} {'Small':>8} {'Medium':>8} {'Large':>8}")
    lines.append("-" * 100)

    fp32_map = None
    for label, m in results.items():
        if "FP32" in label:
            fp32_map = m.get("mAP50_95", 0)

        delta = ""
        if fp32_map is not None and "FP32" not in label:
            diff = m.get("mAP50_95", 0) - fp32_map
            pct = (diff / fp32_map * 100) if fp32_map > 0 else 0
            delta = f"{pct:+.1f}%"

        lines.append(f"{label:<25} {m.get('size_mb', 0):>8.2f} "
                     f"{m.get('mAP50_95', 0):>10.4f} {delta:>8} "
                     f"{m.get('mAP50', 0):>10.4f} {m.get('mAP75', 0):>10.4f} "
                     f"{m.get('mAP_small', 0):>8.4f} {m.get('mAP_medium', 0):>8.4f} "
                     f"{m.get('mAP_large', 0):>8.4f}")

    lines.append("-" * 100)
    if fp32_map:
        lines.append(f"  Delta = relative mAP@.5:.95 change from FP32 baseline")
    lines.append("")
    lines.append("Quantization Strategies:")
    lines.append("  QDQ          = per-tensor, full graph INT8")
    lines.append("  QDQ_PC       = per-channel Conv weights, full graph INT8")
    lines.append("  QDQ_NOH      = per-tensor, detection head (model.22) in FP32")
    lines.append("  QDQ_PC_NOH   = per-channel Conv weights, detection head in FP32")
    return "\n".join(lines)


def _print_comparison(model_name: str, results: dict):
    """Print formatted comparison table."""
    text = _format_comparison(model_name, results)
    if text:
        print(f"\n\n{text}")
    else:
        print("\nNo results to compare.")


def _save_comparison(model_name: str, results: dict):
    """Save comparison table to outputs/ directory."""
    if not results:
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"accuracy_{model_name}_{timestamp}.txt"
    filepath = OUTPUTS_DIR / filename

    text = _format_comparison(model_name, results)
    filepath.write_text(text, encoding="utf-8")
    print(f"\n  Results saved: {filepath}")


def check_status():
    """Print model status."""
    print("=" * 60)
    print("Model Status")
    print("=" * 60)
    for model_name, variants in MODEL_VARIANTS.items():
        print(f"\n  {model_name}:")
        for variant, filename in variants.items():
            path = ASSETS_DIR / filename
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"    [OK] {filename} ({size_mb:.1f} MB)")
            else:
                print(f"    [--] {filename}")

    print(f"\n  COCO val2017 will be auto-downloaded by ultralytics on first eval.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 ONNX model accuracy (mAP) on COCO val2017"
    )
    parser.add_argument("--status", action="store_true",
                        help="Check model status")
    parser.add_argument("--model", type=str, choices=["yolov8n", "yolov8s", "yolov8m"],
                        help="Model to evaluate")
    parser.add_argument("--variant", type=str, nargs="+",
                        help="Variant(s) to evaluate (fp32, qdq, qdq_pc, qdq_noh, qdq_pc_noh)")
    parser.add_argument("--all-variants", action="store_true",
                        help="Evaluate all available variants for the model")
    parser.add_argument("--compare-all", action="store_true",
                        help="All models x all variants")

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.compare_all:
        all_results = {}
        for model_name, variants in MODEL_VARIANTS.items():
            results = evaluate_and_report(model_name, list(variants.keys()))
            if results:
                all_results.update(results)

        if all_results:
            _print_comparison("all", all_results)
            _save_comparison("all", all_results)
        return

    if args.model:
        available = list(MODEL_VARIANTS.get(args.model, {}).keys())
        if args.all_variants:
            variants = available
        elif args.variant:
            variants = args.variant
        else:
            variants = ["fp32"]
        evaluate_and_report(args.model, variants)
        return

    parser.print_help()
    print(f"\nAvailable variants: fp32, qdq, qdq_pc, qdq_noh, qdq_pc_noh")
    print("\nExamples:")
    print("  python eval_accuracy.py --model yolov8n --all-variants")
    print("  python eval_accuracy.py --model yolov8n --variant fp32 qdq qdq_pc qdq_pc_noh")
    print("  python eval_accuracy.py --compare-all")
    print("  python eval_accuracy.py --model yolov8n --variant qdq_pc")


if __name__ == "__main__":
    main()
