#!/usr/bin/env python3
"""
Evaluate YOLOv8n-seg INT8 quantization quality degradation vs FP32.

Compares detection and segmentation accuracy between precision variants:
- FP32: PyTorch-exported ONNX (ground truth)
- INT8 QDQ (noh): Static quantized with head excluded
- INT8 QDQ (full): All nodes quantized including head

Metrics:
- Detection: mAP@50, mAP@50:95, per-class AP
- Segmentation: mask mAP@50, mask mAP@50:95
- Per-image: bbox IoU, mask IoU between FP32 and INT8 predictions
- Inference speed: latency comparison on CPU

Methodology:
1. Run both models on COCO val2017 (or custom image set)
2. Compare predictions pairwise (same image, same NMS thresholds)
3. Optionally run ultralytics val() for official COCO mAP

Usage:
    python scripts/yolo/eval_yolo_seg_quality.py --compare --download-coco
    python scripts/yolo/eval_yolo_seg_quality.py --compare --data-dir path/to/coco/val2017/images/
    python scripts/yolo/eval_yolo_seg_quality.py --compare --num-images 50
    python scripts/yolo/eval_yolo_seg_quality.py --coco-val --data path/to/coco.yaml
    python scripts/yolo/eval_yolo_seg_quality.py --status
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "yolov8n_seg" / "onnx"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "yolo_seg_quality"

MODEL_NAME = "yolov8n-seg"
FP32_PATH = WEIGHTS_DIR / f"{MODEL_NAME}.onnx"
INT8_NOH_PATH = WEIGHTS_DIR / f"{MODEL_NAME}_int8_qdq_noh.onnx"
INT8_FULL_PATH = WEIGHTS_DIR / f"{MODEL_NAME}_int8_qdq.onnx"

DATASETS_DIR = PROJECT_ROOT / "datasets"
COCO_VAL_DIR = DATASETS_DIR / "coco" / "val2017"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
NUM_CLASSES = 80


# ============================================================
# COCO val2017 Download
# ============================================================

def download_coco_val2017(dest_dir: Path = COCO_VAL_DIR) -> Path:
    """Download COCO val2017 images (~1GB, 5000 images) if not present."""
    import zipfile
    import urllib.request
    import shutil

    if dest_dir.exists() and len(list(dest_dir.glob("*.jpg"))) > 4000:
        count = len(list(dest_dir.glob("*.jpg")))
        print(f"  COCO val2017 already exists: {dest_dir} ({count} images)")
        return dest_dir

    url = "http://images.cocodataset.org/zips/val2017.zip"
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir.parent / "val2017.zip"

    if not zip_path.exists():
        print(f"  Downloading COCO val2017 (~1GB)...")
        print(f"    URL: {url}")
        print(f"    Dest: {zip_path}")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                print(f"\r    {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(url, str(zip_path), reporthook=_progress)
        print()  # newline after progress
    else:
        print(f"  Reusing existing zip: {zip_path}")

    print(f"  Extracting to {dest_dir.parent}...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir.parent))

    count = len(list(dest_dir.glob("*.jpg")))
    print(f"  Done: {count} images in {dest_dir}")

    # Clean up zip
    zip_path.unlink()
    print(f"  Removed zip: {zip_path.name}")

    return dest_dir


# ============================================================
# ONNX Runtime Inference
# ============================================================

def create_session(model_path: Path):
    """Create an ORT inference session."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_opts, providers=["CPUExecutionProvider"]
    )


def preprocess_image(image_path: str, input_size: int = INPUT_SIZE):
    """Preprocess image for YOLOv8: resize + normalize + CHW."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Letterbox resize (maintain aspect ratio)
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to input_size x input_size
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    pad_x, pad_y = (input_size - new_w) // 2, (input_size - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))

    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    return arr, (orig_w, orig_h), (scale, pad_x, pad_y)


def run_inference(session, input_tensor):
    """Run YOLO-seg inference and return raw outputs."""
    input_name = session.get_inputs()[0].name
    t_start = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    latency_ms = (time.perf_counter() - t_start) * 1000
    return outputs, latency_ms


def decode_yolo_seg_outputs(outputs, conf_thresh=CONF_THRESHOLD,
                            iou_thresh=IOU_THRESHOLD, input_size=INPUT_SIZE):
    """Decode YOLOv8-seg raw outputs into boxes, scores, classes, masks.

    YOLOv8-seg outputs:
    - output0: [1, 116, 8400] ??4 bbox + 80 classes + 32 mask coefficients
    - output1: [1, 32, 160, 160] ??prototype masks

    Returns:
        boxes: (N, 4) in xyxy format (input_size scale)
        scores: (N,)
        classes: (N,)
        masks: (N, 160, 160) binary masks
    """
    det_output = outputs[0][0]  # (116, 8400)
    proto_masks = outputs[1][0]  # (32, 160, 160)

    # Transpose to (8400, 116)
    det_output = det_output.T

    # Split: bbox (4), class scores (80), mask coefficients (32)
    boxes_xywh = det_output[:, :4]
    class_scores = det_output[:, 4:4 + NUM_CLASSES]
    mask_coeffs = det_output[:, 4 + NUM_CLASSES:]  # (8400, 32)

    # Get best class per detection
    best_scores = class_scores.max(axis=1)
    best_classes = class_scores.argmax(axis=1)

    # Confidence filter
    mask = best_scores > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    best_scores = best_scores[mask]
    best_classes = best_classes[mask]
    mask_coeffs = mask_coeffs[mask]

    if len(best_scores) == 0:
        return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int),
                np.zeros((0, 160, 160)))

    # xywh -> xyxy
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # NMS per class
    keep = _nms_per_class(boxes_xyxy, best_scores, best_classes, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]
    best_scores = best_scores[keep]
    best_classes = best_classes[keep]
    mask_coeffs = mask_coeffs[keep]

    # Decode masks: coeffs @ proto -> sigmoid
    # mask_coeffs: (N, 32), proto_masks: (32, 160, 160)
    mask_maps = mask_coeffs @ proto_masks.reshape(32, -1)  # (N, 25600)
    mask_maps = mask_maps.reshape(-1, 160, 160)
    mask_maps = _sigmoid(mask_maps)

    # Crop masks to bbox region (scaled to 160x160)
    scale = 160 / input_size
    for i in range(len(boxes_xyxy)):
        x1 = max(0, int(boxes_xyxy[i, 0] * scale))
        y1 = max(0, int(boxes_xyxy[i, 1] * scale))
        x2 = min(160, int(boxes_xyxy[i, 2] * scale + 0.5))
        y2 = min(160, int(boxes_xyxy[i, 3] * scale + 0.5))
        crop_mask = np.zeros((160, 160), dtype=np.float32)
        crop_mask[y1:y2, x1:x2] = mask_maps[i, y1:y2, x1:x2]
        mask_maps[i] = crop_mask

    # Binarize masks
    binary_masks = (mask_maps > 0.5).astype(np.float32)

    return boxes_xyxy, best_scores, best_classes, binary_masks


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _nms_per_class(boxes, scores, classes, iou_thresh):
    """Simple per-class NMS."""
    keep = []
    for cls in np.unique(classes):
        cls_mask = classes == cls
        cls_indices = np.where(cls_mask)[0].tolist()
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        order = cls_scores.argsort()[::-1].tolist()
        cls_indices = [cls_indices[o] for o in order]
        cls_boxes = cls_boxes[order]

        selected = []
        suppressed = set()
        for i_idx, i in enumerate(cls_indices):
            if i in suppressed:
                continue
            selected.append(i)
            # Suppress overlapping boxes
            for j_idx in range(i_idx + 1, len(cls_indices)):
                j = cls_indices[j_idx]
                if j in suppressed:
                    continue
                iou = _compute_iou_single(cls_boxes[i_idx], cls_boxes[j_idx])
                if iou >= iou_thresh:
                    suppressed.add(j)

        keep.extend(selected)

    return sorted(keep)


def _compute_iou_single(box_a, box_b):
    """Compute IoU between two single boxes (xyxy format)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / (union + 1e-7)


def _compute_iou(box, boxes):
    """Compute IoU of one box against many boxes."""
    x1 = np.maximum(box[:, 0], boxes[:, 0])
    y1 = np.maximum(box[:, 1], boxes[:, 1])
    x2 = np.minimum(box[:, 2], boxes[:, 2])
    y2 = np.minimum(box[:, 3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter

    return inter / (union + 1e-7)


# ============================================================
# Pairwise Comparison Metrics
# ============================================================

def match_detections(boxes_a, scores_a, classes_a, masks_a,
                     boxes_b, scores_b, classes_b, masks_b,
                     iou_thresh=0.5):
    """Match detections between two sets (FP32 vs INT8) by bbox IoU.

    Returns per-match metrics: bbox IoU, mask IoU, score diff, class match.
    """
    if len(boxes_a) == 0 and len(boxes_b) == 0:
        return {
            "matched": 0, "fp32_only": 0, "int8_only": 0,
            "bbox_iou_mean": 1.0, "mask_iou_mean": 1.0,
            "score_diff_mean": 0.0, "class_match_rate": 1.0,
            "matches": [],
        }

    if len(boxes_a) == 0:
        return {
            "matched": 0, "fp32_only": 0, "int8_only": len(boxes_b),
            "bbox_iou_mean": 0.0, "mask_iou_mean": 0.0,
            "score_diff_mean": 0.0, "class_match_rate": 0.0,
            "matches": [],
        }

    if len(boxes_b) == 0:
        return {
            "matched": 0, "fp32_only": len(boxes_a), "int8_only": 0,
            "bbox_iou_mean": 0.0, "mask_iou_mean": 0.0,
            "score_diff_mean": 0.0, "class_match_rate": 0.0,
            "matches": [],
        }

    # Compute pairwise bbox IoU matrix
    iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))
    for i in range(len(boxes_a)):
        for j in range(len(boxes_b)):
            iou_matrix[i, j] = _compute_iou_single(boxes_a[i], boxes_b[j])

    # Greedy matching: highest IoU first
    matched_a = set()
    matched_b = set()
    matches = []

    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for idx in flat_indices:
        i, j = divmod(int(idx), len(boxes_b))
        if i in matched_a or j in matched_b:
            continue
        if iou_matrix[i, j] < iou_thresh:
            break

        # Compute mask IoU
        mask_inter = (masks_a[i] * masks_b[j]).sum()
        mask_union = ((masks_a[i] + masks_b[j]) > 0).sum()
        mask_iou = float(mask_inter / (mask_union + 1e-7))

        matches.append({
            "bbox_iou": float(iou_matrix[i, j]),
            "mask_iou": mask_iou,
            "score_diff": float(abs(scores_a[i] - scores_b[j])),
            "class_match": int(classes_a[i]) == int(classes_b[j]),
            "class_a": int(classes_a[i]),
            "class_b": int(classes_b[j]),
            "score_a": float(scores_a[i]),
            "score_b": float(scores_b[j]),
        })
        matched_a.add(i)
        matched_b.add(j)

    n_matched = len(matches)
    fp32_only = len(boxes_a) - n_matched
    int8_only = len(boxes_b) - n_matched

    return {
        "matched": n_matched,
        "fp32_only": fp32_only,
        "int8_only": int8_only,
        "bbox_iou_mean": np.mean([m["bbox_iou"] for m in matches]) if matches else 0.0,
        "mask_iou_mean": np.mean([m["mask_iou"] for m in matches]) if matches else 0.0,
        "score_diff_mean": np.mean([m["score_diff"] for m in matches]) if matches else 0.0,
        "class_match_rate": np.mean([m["class_match"] for m in matches]) if matches else 0.0,
        "matches": matches,
    }


# ============================================================
# Pairwise Comparison Pipeline
# ============================================================

def run_pairwise_comparison(image_dir: Path, num_images: int = 50,
                            variants: list = None):
    """Run FP32 and INT8 variants on the same images and compare pairwise.

    Args:
        variants: list of (label, model_path) tuples to compare against FP32.
                  Defaults to all available INT8 models.
    """
    import glob as glob_mod

    if variants is None:
        variants = []
        if INT8_NOH_PATH.exists():
            variants.append(("INT8_noh", INT8_NOH_PATH))
        if INT8_FULL_PATH.exists():
            variants.append(("INT8_full", INT8_FULL_PATH))

    if not FP32_PATH.exists():
        print(f"  Error: FP32 model not found: {FP32_PATH}")
        print(f"  Run: python scripts/yolo/export_yolo_seg.py --export-all")
        return {}

    if not variants:
        print(f"  Error: No INT8 models found.")
        print(f"  Run: python scripts/yolo/export_yolo_seg.py --export-all")
        return {}

    print("\n" + "=" * 70)
    print("YOLOv8n-seg Pairwise Quality Comparison: FP32 vs INT8 variants")
    print("=" * 70)

    size_mb = FP32_PATH.stat().st_size / 1024 / 1024
    print(f"  FP32: {FP32_PATH.name} ({size_mb:.1f} MB)")
    for label, path in variants:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {label}: {path.name} ({size_mb:.1f} MB)")

    # Gather images
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob_mod.glob(str(image_dir / ext)))
    image_files = sorted(image_files)[:num_images]

    if not image_files:
        print(f"\n  Error: No images found in {image_dir}")
        print("  Provide images with --data-dir or use --download-coco")
        return {}

    print(f"\n  Images: {len(image_files)} from {image_dir}")

    # Load sessions
    print("\n  Loading FP32 session...")
    sess_fp32 = create_session(FP32_PATH)
    variant_sessions = {}
    for label, path in variants:
        print(f"  Loading {label} session...")
        variant_sessions[label] = create_session(path)

    # Warmup
    print("  Warmup (3 runs each)...")
    dummy = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    for _ in range(3):
        run_inference(sess_fp32, dummy)
        for sess in variant_sessions.values():
            run_inference(sess, dummy)

    # Run comparison per variant
    all_results = {}

    for label, sess_int8 in variant_sessions.items():
        print(f"\n  --- Comparing FP32 vs {label} ---")
        per_image_results = []
        fp32_latencies = []
        int8_latencies = []

        for idx, img_path in enumerate(image_files):
            input_tensor, orig_size, transform = preprocess_image(img_path)

            # FP32
            out_fp32, lat_fp32 = run_inference(sess_fp32, input_tensor)
            boxes_fp32, scores_fp32, cls_fp32, masks_fp32 = decode_yolo_seg_outputs(out_fp32)

            # INT8 variant
            out_int8, lat_int8 = run_inference(sess_int8, input_tensor)
            boxes_int8, scores_int8, cls_int8, masks_int8 = decode_yolo_seg_outputs(out_int8)

            fp32_latencies.append(lat_fp32)
            int8_latencies.append(lat_int8)

            result = match_detections(
                boxes_fp32, scores_fp32, cls_fp32, masks_fp32,
                boxes_int8, scores_int8, cls_int8, masks_int8,
            )
            result["image"] = Path(img_path).name
            result["fp32_dets"] = len(boxes_fp32)
            result["int8_dets"] = len(boxes_int8)
            result["fp32_latency_ms"] = lat_fp32
            result["int8_latency_ms"] = lat_int8
            per_image_results.append(result)

            if (idx + 1) % 10 == 0 or idx == len(image_files) - 1:
                print(f"  [{idx+1}/{len(image_files)}] "
                      f"FP32: {len(boxes_fp32)} dets, {label}: {len(boxes_int8)} dets, "
                      f"matched: {result['matched']}, "
                      f"bbox_IoU: {result['bbox_iou_mean']:.3f}, "
                      f"mask_IoU: {result['mask_iou_mean']:.3f}")

        agg = _aggregate_results(per_image_results, fp32_latencies, int8_latencies)
        agg["per_image"] = per_image_results
        agg["label"] = label
        all_results[label] = agg

    return all_results


def _aggregate_results(per_image, fp32_lats, int8_lats):
    """Compute aggregate metrics from per-image results."""
    total_fp32_dets = sum(r["fp32_dets"] for r in per_image)
    total_int8_dets = sum(r["int8_dets"] for r in per_image)
    total_matched = sum(r["matched"] for r in per_image)
    total_fp32_only = sum(r["fp32_only"] for r in per_image)
    total_int8_only = sum(r["int8_only"] for r in per_image)

    # Weighted mean IoU (by number of matches)
    all_bbox_ious = []
    all_mask_ious = []
    all_score_diffs = []
    all_class_matches = []
    for r in per_image:
        for m in r["matches"]:
            all_bbox_ious.append(m["bbox_iou"])
            all_mask_ious.append(m["mask_iou"])
            all_score_diffs.append(m["score_diff"])
            all_class_matches.append(m["class_match"])

    return {
        "num_images": len(per_image),
        "total_fp32_dets": total_fp32_dets,
        "total_int8_dets": total_int8_dets,
        "det_count_diff_pct": (total_int8_dets - total_fp32_dets) / max(total_fp32_dets, 1) * 100,
        "total_matched": total_matched,
        "total_fp32_only": total_fp32_only,
        "total_int8_only": total_int8_only,
        "match_rate": total_matched / max(total_fp32_dets, 1) * 100,
        "bbox_iou_mean": float(np.mean(all_bbox_ious)) if all_bbox_ious else 0.0,
        "bbox_iou_std": float(np.std(all_bbox_ious)) if all_bbox_ious else 0.0,
        "bbox_iou_min": float(np.min(all_bbox_ious)) if all_bbox_ious else 0.0,
        "mask_iou_mean": float(np.mean(all_mask_ious)) if all_mask_ious else 0.0,
        "mask_iou_std": float(np.std(all_mask_ious)) if all_mask_ious else 0.0,
        "mask_iou_min": float(np.min(all_mask_ious)) if all_mask_ious else 0.0,
        "score_diff_mean": float(np.mean(all_score_diffs)) if all_score_diffs else 0.0,
        "score_diff_max": float(np.max(all_score_diffs)) if all_score_diffs else 0.0,
        "class_match_rate": float(np.mean(all_class_matches)) if all_class_matches else 0.0,
        "fp32_latency_mean_ms": float(np.mean(fp32_lats)),
        "fp32_latency_std_ms": float(np.std(fp32_lats)),
        "int8_latency_mean_ms": float(np.mean(int8_lats)),
        "int8_latency_std_ms": float(np.std(int8_lats)),
        "speedup": float(np.mean(fp32_lats) / np.mean(int8_lats)) if np.mean(int8_lats) > 0 else 0,
    }


# ============================================================
# COCO Validation (ultralytics)
# ============================================================

def run_coco_val(data_yaml: str):
    """Run official COCO mAP evaluation using ultralytics for both models.

    Requires: ultralytics, COCO val2017 dataset, coco.yaml
    """
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("YOLOv8n-seg COCO Validation: FP32 vs INT8 variants")
    print("=" * 70)

    results = {}

    model_list = [("FP32", FP32_PATH), ("INT8_noh", INT8_NOH_PATH), ("INT8_full", INT8_FULL_PATH)]
    for label, model_path in model_list:
        if not model_path.exists():
            print(f"\n  {label}: Model not found ??{model_path}")
            continue

        print(f"\n  Evaluating {label}: {model_path.name}")
        model = YOLO(str(model_path), task="segment")

        metrics = model.val(
            data=data_yaml,
            imgsz=INPUT_SIZE,
            batch=1,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        results[label] = {
            "box_map50": float(metrics.box.map50),
            "box_map50_95": float(metrics.box.map),
            "seg_map50": float(metrics.seg.map50),
            "seg_map50_95": float(metrics.seg.map),
        }

        print(f"    Box  mAP@50:    {results[label]['box_map50']:.4f}")
        print(f"    Box  mAP@50:95: {results[label]['box_map50_95']:.4f}")
        print(f"    Mask mAP@50:    {results[label]['seg_map50']:.4f}")
        print(f"    Mask mAP@50:95: {results[label]['seg_map50_95']:.4f}")

    int8_labels = [l for l in results if l != "FP32"]
    if "FP32" in results and int8_labels:
        # Build header
        header = f"  {'Metric':<20} {'FP32':>10}"
        for il in int8_labels:
            header += f" {il:>12} {'Drop%':>8}"
        print(f"\n{header}")
        print("  " + "-" * (20 + 10 + len(int8_labels) * 22))
        for key, name in [
            ("box_map50", "Box mAP@50"),
            ("box_map50_95", "Box mAP@50:95"),
            ("seg_map50", "Mask mAP@50"),
            ("seg_map50_95", "Mask mAP@50:95"),
        ]:
            fp32_val = results["FP32"][key]
            row = f"  {name:<20} {fp32_val:>10.4f}"
            for il in int8_labels:
                int8_val = results[il][key]
                delta_pct = (int8_val - fp32_val) / fp32_val * 100 if fp32_val > 0 else 0
                row += f" {int8_val:>12.4f} {delta_pct:>+7.2f}%"
            print(row)

    return results


# ============================================================
# Output
# ============================================================

def print_summary(all_agg: dict):
    """Print formatted summary of pairwise comparison for all variants."""
    print(f"\n\n{'=' * 80}")
    print("YOLOv8n-seg Quantization Quality Summary")
    print(f"{'=' * 80}")

    print(f"\n  Model:  {MODEL_NAME}")
    print(f"  FP32:   {FP32_PATH.name} ({FP32_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    for label, agg in all_agg.items():
        print(f"\n  {'?' * 70}")
        print(f"  Variant: {label}")
        print(f"  Images:  {agg['num_images']}")

        print(f"\n    Detection Count:")
        print(f"      FP32 total:    {agg['total_fp32_dets']}")
        print(f"      {label} total: {agg['total_int8_dets']} ({agg['det_count_diff_pct']:+.1f}%)")
        print(f"      Matched:       {agg['total_matched']} ({agg['match_rate']:.1f}% of FP32)")
        print(f"      FP32-only:     {agg['total_fp32_only']}")
        print(f"      {label}-only:  {agg['total_int8_only']}")

        print(f"\n    Matched Detection Quality:")
        print(f"      Bbox IoU:    mean={agg['bbox_iou_mean']:.4f}  std={agg['bbox_iou_std']:.4f}  min={agg['bbox_iou_min']:.4f}")
        print(f"      Mask IoU:    mean={agg['mask_iou_mean']:.4f}  std={agg['mask_iou_std']:.4f}  min={agg['mask_iou_min']:.4f}")
        print(f"      Score diff:  mean={agg['score_diff_mean']:.4f}  max={agg['score_diff_max']:.4f}")
        print(f"      Class match: {agg['class_match_rate'] * 100:.1f}%")

        print(f"\n    Latency (CPU):")
        print(f"      FP32:    {agg['fp32_latency_mean_ms']:.1f} 짹 {agg['fp32_latency_std_ms']:.1f} ms")
        print(f"      {label}: {agg['int8_latency_mean_ms']:.1f} 짹 {agg['int8_latency_std_ms']:.1f} ms")
        print(f"      Speedup: {agg['speedup']:.2f}x")

    # Comparison table if multiple variants
    if len(all_agg) > 1:
        print(f"\n  {'=' * 70}")
        print(f"  Variant Comparison")
        print(f"  {'?' * 70}")
        header = f"  {'Metric':<22}"
        for label in all_agg:
            header += f" {label:>14}"
        print(header)
        print(f"  {'?' * 70}")
        for metric, fmt in [
            ("mask_iou_mean", ".4f"), ("bbox_iou_mean", ".4f"),
            ("match_rate", ".1f"), ("class_match_rate", ".1%"),
            ("score_diff_mean", ".4f"), ("speedup", ".2f"),
        ]:
            row = f"  {metric:<22}"
            for agg in all_agg.values():
                val = agg[metric]
                if fmt == ".1%":
                    row += f" {val * 100:>13.1f}%"
                else:
                    row += f" {val:>14{fmt}}"
            print(row)

    # Interpretation
    print(f"\n  Interpretation:")
    for label, agg in all_agg.items():
        miou = agg["mask_iou_mean"]
        mrate = agg["match_rate"]
        if miou >= 0.95:
            qual = "Excellent ??negligible impact"
        elif miou >= 0.90:
            qual = "Good ??minor differences"
        elif miou >= 0.80:
            qual = "Moderate ??visible differences"
        else:
            qual = "Significant degradation"
        print(f"    {label}: Mask IoU={miou:.4f}, Match={mrate:.1f}% ??{qual}")

    print()


def save_results(all_agg: dict, coco_results: dict = None):
    """Save results to JSON."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUTS_DIR / f"yolo_seg_quality_{timestamp}.json"

    pairwise_data = {}
    for label, agg in all_agg.items():
        pairwise_data[label] = {k: v for k, v in agg.items() if k != "per_image"}

    data = {
        "model": MODEL_NAME,
        "fp32_model": FP32_PATH.name,
        "int8_models": {
            "INT8_noh": INT8_NOH_PATH.name,
            "INT8_full": INT8_FULL_PATH.name,
        },
        "timestamp": timestamp,
        "settings": {
            "input_size": INPUT_SIZE,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
        "pairwise": pairwise_data if pairwise_data else None,
        "coco_val": coco_results,
    }

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    json_path.write_text(json.dumps(data, indent=2, default=convert))
    print(f"  Results saved: {json_path}")
    return json_path


# ============================================================
# Status Check
# ============================================================

def check_status():
    """Check model availability and previous results."""
    print("=" * 60)
    print(f"YOLOv8n-seg Quality Evaluation Status")
    print("=" * 60)

    print(f"\n  Model directory: {WEIGHTS_DIR}")
    for label, path in [("FP32", FP32_PATH), ("INT8 QDQ (noh)", INT8_NOH_PATH),
                         ("INT8 QDQ (full)", INT8_FULL_PATH)]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"    [OK] {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"    [--] {path.name}")

    print(f"\n  COCO val2017: {COCO_VAL_DIR}")
    if COCO_VAL_DIR.exists():
        count = len(list(COCO_VAL_DIR.glob("*.jpg")))
        print(f"    [OK] {count} images")
    else:
        print(f"    [--] Not downloaded (use --download-coco)")

    if OUTPUTS_DIR.exists():
        results = list(OUTPUTS_DIR.glob("yolo_seg_quality_*.json"))
        if results:
            print(f"\n  Previous evaluations ({len(results)}):")
            for r in sorted(results)[-3:]:
                print(f"    {r.name}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8n-seg INT8 quantization quality vs FP32"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run pairwise comparison on images (FP32 vs INT8 predictions)",
    )
    parser.add_argument(
        "--coco-val",
        action="store_true",
        help="Run official COCO mAP evaluation using ultralytics",
    )
    parser.add_argument(
        "--download-coco",
        action="store_true",
        help="Download COCO val2017 images (~1GB) if not present",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with images for pairwise comparison (e.g., COCO val2017/images/)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco.yaml",
        help="COCO dataset YAML for --coco-val (default: coco.yaml)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of images for pairwise comparison (default: 50)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check model availability and previous results",
    )

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.download_coco:
        download_coco_val2017()
        if not args.compare and not args.coco_val:
            return

    if not args.compare and not args.coco_val:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/yolo/eval_yolo_seg_quality.py --compare --download-coco")
        print("  python scripts/yolo/eval_yolo_seg_quality.py --compare --data-dir path/to/images/ --num-images 100")
        print("  python scripts/yolo/eval_yolo_seg_quality.py --coco-val --data coco.yaml")
        print("  python scripts/yolo/eval_yolo_seg_quality.py --status")
        return

    all_agg = {}
    coco_results = None

    if args.compare:
        if args.data_dir is None:
            # Try default COCO path
            default_dirs = [
                COCO_VAL_DIR,
                Path("datasets/coco/val2017"),
                Path("../datasets/coco/val2017"),
                Path.home() / "datasets" / "coco" / "val2017",
            ]
            for d in default_dirs:
                if d.exists():
                    args.data_dir = d
                    break

            if args.data_dir is None:
                print("Error: No --data-dir specified and no COCO path found.")
                print("Use --download-coco to download COCO val2017.")
                return

        all_agg = run_pairwise_comparison(args.data_dir, args.num_images)
        if all_agg:
            print_summary(all_agg)

    if args.coco_val:
        coco_results = run_coco_val(args.data)

    if all_agg or coco_results:
        save_results(all_agg or {}, coco_results)


if __name__ == "__main__":
    main()


