#!/usr/bin/env python3
"""
Evaluate YOLOv8 ONNX model accuracy (mAP) on COCO val2017.

Uses ultralytics' built-in val() for accurate, GPU-accelerated evaluation.
Compares FP32 vs INT8 QDQ vs INT8 QIO to measure quantization accuracy loss.

Requirements:
    pip install ultralytics

Usage:
    python eval_accuracy.py --model yolov8n                           # Single FP32 model
    python eval_accuracy.py --model yolov8n --all-variants            # FP32 + QDQ + QIO
    python eval_accuracy.py --compare-all                             # All models x all variants
    python eval_accuracy.py --status                                  # Check model status

Note:
    - ultralytics auto-downloads COCO val2017 on first run (~1GB images + annotations)
    - GPU (CUDA) is used automatically if available, otherwise CPU
    - QIO models have UINT8 I/O and are not directly supported by ultralytics val();
      they are evaluated via ONNX Runtime with manual pre/postprocessing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPTS_DIR.parent / "android" / "app" / "src" / "main" / "assets"

MODEL_VARIANTS = {
    "yolov8n": {
        "fp32": "yolov8n.onnx",
        "qdq": "yolov8n_int8_qdq.onnx",
        "qio": "yolov8n_int8_qio.onnx",
    },
    "yolov8s": {
        "fp32": "yolov8s.onnx",
        "qdq": "yolov8s_int8_qdq.onnx",
        "qio": "yolov8s_int8_qio.onnx",
    },
    "yolov8m": {
        "fp32": "yolov8m.onnx",
        "qdq": "yolov8m_int8_qdq.onnx",
        "qio": "yolov8m_int8_qio.onnx",
    },
}

# COCO val2017 path (auto-downloaded by ultralytics)
# Used only for QIO fallback evaluation
COCO_DIR = SCRIPTS_DIR / "coco_val2017"
COCO_IMAGES_DIR = COCO_DIR / "val2017"
COCO_ANN_FILE = COCO_DIR / "annotations" / "instances_val2017.json"

# YOLOv8 80-class → COCO 91-class ID mapping
YOLO_TO_COCO_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


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
# QIO evaluation (ONNX Runtime fallback — UINT8 I/O)
# ============================================================

def eval_qio_model(model_path: Path) -> dict:
    """Evaluate QIO model via ONNX Runtime (UINT8 I/O not supported by ultralytics)."""
    import cv2
    import onnxruntime as ort

    print(f"  Loading QIO model: {model_path.name}")
    print(f"  QIO has UINT8 I/O — using ONNX Runtime fallback")

    # Prefer GPU if available
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        ep = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"  Using CUDA EP")
    else:
        ep = ["CPUExecutionProvider"]
        print(f"  Using CPU EP")

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_opts, providers=ep)
    input_name = session.get_inputs()[0].name

    # Get output dequant params from QDQ sibling
    qdq_path = Path(str(model_path).replace("_qio.", "_qdq."))
    out_scale, out_zp = _extract_output_quant_params(qdq_path)
    print(f"  Output dequant: scale={out_scale:.6f}, zp={out_zp}")

    # Need COCO val2017 dataset
    if not COCO_ANN_FILE.exists() or not COCO_IMAGES_DIR.exists():
        # Try ultralytics default location
        ul_coco = _find_ultralytics_coco()
        if ul_coco is None:
            print(f"  Error: COCO val2017 not found. Run --setup or evaluate FP32/QDQ first.")
            return {}
        ann_file, images_dir = ul_coco
    else:
        ann_file, images_dir = COCO_ANN_FILE, COCO_IMAGES_DIR

    with open(str(ann_file)) as f:
        coco_ann = json.load(f)

    available = []
    for img_info in coco_ann["images"]:
        if (images_dir / img_info["file_name"]).exists():
            available.append(img_info)

    print(f"  Evaluating on {len(available)} images...")

    all_detections = []
    total_time = 0
    for i, img_info in enumerate(available):
        img_bgr = cv2.imread(str(images_dir / img_info["file_name"]))
        if img_bgr is None:
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        blob, scale, pad_x, pad_y = _letterbox_uint8(img_bgr, 640)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: blob})
        total_time += time.perf_counter() - t0

        output = (outputs[0].astype(np.float32) - out_zp) * out_scale
        dets = _postprocess(output, scale, pad_x, pad_y, orig_w, orig_h)

        for det in dets:
            det["image_id"] = img_info["id"]
            all_detections.append(det)

        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(available)}")

    avg_ms = (total_time / max(len(available), 1)) * 1000
    print(f"  Done: {len(available)} images, {avg_ms:.1f} ms/img, {len(all_detections)} detections")

    # Compute mAP
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not all_detections:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0}

    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(all_detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list({d["image_id"] for d in all_detections})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP50_95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
        "speed_inference_ms": avg_ms,
    }


def _find_ultralytics_coco():
    """Find COCO val2017 in ultralytics default dataset location."""
    from ultralytics.utils import DATASETS_DIR
    ul_images = DATASETS_DIR / "coco" / "images" / "val2017"
    ul_ann = DATASETS_DIR / "coco" / "annotations" / "instances_val2017.json"
    if ul_images.exists() and ul_ann.exists():
        return ul_ann, ul_images
    return None


def _letterbox_uint8(img_bgr, size=640):
    """Letterbox preprocess for QIO (UINT8 CHW output)."""
    import cv2
    h, w = img_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    blob = np.expand_dims(np.transpose(canvas[:, :, ::-1].copy(), (2, 0, 1)), 0)
    return blob, scale, pad_x, pad_y


def _postprocess(output, scale, pad_x, pad_y, orig_w, orig_h,
                 conf_thresh=0.001, iou_thresh=0.7):
    """YOLOv8 postprocess: [1,84,8400] → COCO detections."""
    import cv2
    pred = output[0].T
    boxes = pred[:, :4]
    scores = pred[:, 4:]
    cls = np.argmax(scores, axis=1)
    conf = scores[np.arange(len(scores)), cls]
    mask = conf > conf_thresh
    boxes, conf, cls = boxes[mask], conf[mask], cls[mask]
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Per-class NMS via offset
    nms_boxes = xyxy.copy()
    offset = cls.astype(np.float32) * 7680
    nms_boxes[:, 0] += offset; nms_boxes[:, 1] += offset
    nms_boxes[:, 2] += offset; nms_boxes[:, 3] += offset
    indices = cv2.dnn.NMSBoxes(nms_boxes.tolist(), conf.tolist(), conf_thresh, iou_thresh)
    if len(indices) == 0:
        return []
    indices = np.array(indices).flatten()

    results = []
    for i in indices:
        bx1 = max(0, (xyxy[i, 0] - pad_x) / scale)
        by1 = max(0, (xyxy[i, 1] - pad_y) / scale)
        bx2 = min(orig_w, (xyxy[i, 2] - pad_x) / scale)
        by2 = min(orig_h, (xyxy[i, 3] - pad_y) / scale)
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 0 or bh <= 0:
            continue
        results.append({
            "bbox": [float(bx1), float(by1), float(bw), float(bh)],
            "score": float(conf[i]),
            "category_id": YOLO_TO_COCO_ID[cls[i]],
        })
    return results


def _extract_output_quant_params(qdq_model_path: Path) -> tuple:
    """Extract output DequantizeLinear scale/zp from QDQ model."""
    try:
        import onnx
        from onnx import numpy_helper
        model = onnx.load(str(qdq_model_path))
        graph_output_names = {out.name for out in model.graph.output}
        initializers = {init.name: init for init in model.graph.initializer}
        for node in model.graph.node:
            if node.op_type == "DequantizeLinear" and node.output[0] in graph_output_names:
                s = float(numpy_helper.to_array(initializers[node.input[1]]).flat[0])
                zp = 0
                if len(node.input) > 2 and node.input[2] in initializers:
                    zp = int(numpy_helper.to_array(initializers[node.input[2]]).flat[0])
                return s, zp
    except Exception as e:
        print(f"  Warning: Could not extract QDQ output params: {e}")
    return 1.0, 0


# ============================================================
# Comparison report
# ============================================================

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

        if variant == "qio":
            metrics = eval_qio_model(model_path)
        else:
            metrics = eval_with_ultralytics(model_path)

        if metrics:
            results[label] = metrics

    _print_comparison(model_name, results)
    return results


def _print_comparison(model_name: str, results: dict):
    """Print formatted comparison table."""
    if not results:
        print("\nNo results to compare.")
        return

    print(f"\n\n{'=' * 90}")
    print(f"Accuracy Comparison: {model_name}")
    print(f"{'=' * 90}")
    print(f"{'Model':<25} {'mAP@.5:.95':>10} {'Delta':>8} {'mAP@.5':>10} {'mAP@.75':>10} "
          f"{'Small':>8} {'Medium':>8} {'Large':>8}")
    print("-" * 90)

    fp32_map = None
    for label, m in results.items():
        if "FP32" in label:
            fp32_map = m.get("mAP50_95", 0)

        delta = ""
        if fp32_map is not None and "FP32" not in label:
            diff = m.get("mAP50_95", 0) - fp32_map
            pct = (diff / fp32_map * 100) if fp32_map > 0 else 0
            delta = f"{pct:+.1f}%"

        print(f"{label:<25} {m.get('mAP50_95', 0):>10.4f} {delta:>8} "
              f"{m.get('mAP50', 0):>10.4f} {m.get('mAP75', 0):>10.4f} "
              f"{m.get('mAP_small', 0):>8.4f} {m.get('mAP_medium', 0):>8.4f} "
              f"{m.get('mAP_large', 0):>8.4f}")

    print("-" * 90)
    if fp32_map:
        print(f"  Delta = relative mAP@.5:.95 change from FP32 baseline")


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
    parser.add_argument("--variant", type=str, choices=["fp32", "qdq", "qio"],
                        help="Specific variant (default: fp32)")
    parser.add_argument("--all-variants", action="store_true",
                        help="Evaluate FP32 + QDQ + QIO")
    parser.add_argument("--compare-all", action="store_true",
                        help="All models x all variants")

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.compare_all:
        all_results = {}
        for model_name in MODEL_VARIANTS:
            results = evaluate_and_report(model_name, ["fp32", "qdq", "qio"])
            if results:
                all_results.update(results)

        # Print grand summary
        if all_results:
            print(f"\n\n{'=' * 90}")
            print("Grand Summary: All Models")
            print(f"{'=' * 90}")
            _print_comparison("all", all_results)
        return

    if args.model:
        if args.all_variants:
            variants = ["fp32", "qdq", "qio"]
        elif args.variant:
            variants = [args.variant]
        else:
            variants = ["fp32"]
        evaluate_and_report(args.model, variants)
        return

    parser.print_help()
    print("\nExamples:")
    print("  python eval_accuracy.py --model yolov8n --all-variants   # n: FP32 vs QDQ vs QIO")
    print("  python eval_accuracy.py --compare-all                    # All models x all variants")
    print("  python eval_accuracy.py --model yolov8m --variant qdq    # Single variant")


if __name__ == "__main__":
    main()
