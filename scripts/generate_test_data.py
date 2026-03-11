#!/usr/bin/env python3
"""
Generate test_images + test_masks for AI Eraser benchmark.

Runs YOLOv8n-seg FP32 on sample_image.jpg, picks 3 objects by ROI size
(Small ~128², Medium ~256², Large ~400²), and saves binary masks.

Output:
  android/app/src/main/assets/test_images/scene_{small,medium,large}.jpg
  android/app/src/main/assets/test_masks/mask_{small,medium,large}.png

Usage:
    python scripts/generate_test_data.py
    python scripts/generate_test_data.py --image path/to/image.jpg
    python scripts/generate_test_data.py --show  # visualize detections
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPTS_DIR.parent
ASSETS_DIR = PROJECT_DIR / "android" / "app" / "src" / "main" / "assets"
YOLO_MODEL = PROJECT_DIR / "weights" / "yolov8n_seg" / "onnx" / "yolov8n-seg_fp32.onnx"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.7
NUM_CLASSES = 80
MASK_PROTO_DIM = 32

# Target ROI sizes (bbox diagonal in pixels, for original image coordinates)
ROI_TARGETS = {
    "small":  128,   # ~128x128 bbox
    "medium": 256,   # ~256x256 bbox
    "large":  400,   # ~400x400 bbox
}


def letterbox(img, new_shape=640):
    """Resize with letterbox padding (same as ultralytics)."""
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_h, pad_w = (new_shape - new_h) / 2, (new_shape - new_w) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, (left, top)


def xywh2xyxy(x):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes, scores, iou_threshold=0.7):
    """Simple NMS."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


def run_yolo_seg(image_bgr, model_path):
    """Run YOLOv8n-seg and return list of (bbox_xyxy_orig, mask_orig, score, class_id)."""
    h_orig, w_orig = image_bgr.shape[:2]
    padded, scale, (pad_left, pad_top) = letterbox(image_bgr, INPUT_SIZE)

    # Preprocess
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalize
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW

    # Inference
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: blob})

    # Parse outputs: output0 = [1, 116, 8400], output1 = [1, 32, 160, 160]
    pred = outputs[0][0]  # [116, 8400]
    proto = outputs[1][0]  # [32, 160, 160]

    pred = pred.T  # [8400, 116]
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:4 + NUM_CLASSES]
    mask_coeffs = pred[:, 4 + NUM_CLASSES:]  # [8400, 32]

    # Confidence filter
    max_scores = class_scores.max(axis=1)
    mask = max_scores > CONF_THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    class_scores = class_scores[mask]
    mask_coeffs = mask_coeffs[mask]
    max_scores = max_scores[mask]
    class_ids = class_scores.argmax(axis=1)

    if len(boxes_xywh) == 0:
        return []

    # NMS
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, max_scores, IOU_THRESHOLD)
    boxes_xyxy = boxes_xyxy[keep]
    max_scores = max_scores[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    # Decode masks
    results = []
    proto_h, proto_w = proto.shape[1], proto.shape[2]

    for i in range(len(boxes_xyxy)):
        # Mask from prototypes: sigmoid(coeffs @ proto)
        mask_pred = (mask_coeffs[i] @ proto.reshape(32, -1)).reshape(proto_h, proto_w)
        mask_pred = 1 / (1 + np.exp(-mask_pred))  # sigmoid

        # Resize mask to input size
        mask_resized = cv2.resize(mask_pred, (INPUT_SIZE, INPUT_SIZE))

        # Crop mask to bbox (in letterbox coords)
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(INPUT_SIZE, x2), min(INPUT_SIZE, y2)

        # Zero out outside bbox
        cropped_mask = np.zeros_like(mask_resized)
        cropped_mask[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]

        # Remove letterbox padding and rescale to original image
        unpadded = cropped_mask[pad_top:pad_top + int(h_orig * scale),
                                pad_left:pad_left + int(w_orig * scale)]
        mask_orig = cv2.resize(unpadded, (w_orig, h_orig))
        mask_binary = (mask_orig > 0.5).astype(np.uint8) * 255

        # Convert bbox to original coordinates
        bbox_orig = np.array([
            (x1 - pad_left) / scale,
            (y1 - pad_top) / scale,
            (x2 - pad_left) / scale,
            (y2 - pad_top) / scale
        ]).clip(0)
        bbox_orig[2] = min(bbox_orig[2], w_orig)
        bbox_orig[3] = min(bbox_orig[3], h_orig)

        results.append((bbox_orig, mask_binary, float(max_scores[i]), int(class_ids[i])))

    return results


def bbox_size(bbox):
    """Bbox diagonal size."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return math.sqrt(w * w + h * h)


def select_by_roi_size(detections, targets):
    """Select best detection for each ROI size target."""
    selected = {}
    used = set()

    for size_name, target_diag in sorted(targets.items(), key=lambda x: x[1]):
        best_idx = None
        best_diff = float("inf")
        for i, (bbox, mask, score, cls_id) in enumerate(detections):
            if i in used:
                continue
            diag = bbox_size(bbox)
            diff = abs(diag - target_diag)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_idx is not None:
            used.add(best_idx)
            bbox, mask, score, cls_id = detections[best_idx]
            diag = bbox_size(bbox)
            selected[size_name] = {
                "bbox": bbox,
                "mask": mask,
                "score": score,
                "class_id": cls_id,
                "diag": diag,
                "index": best_idx,
            }
            print(f"  {size_name}: detection #{best_idx}, class={cls_id}, "
                  f"score={score:.2f}, diag={diag:.0f}px (target={target_diag})")

    return selected


def save_test_data(image_bgr, selected, assets_dir):
    """Save test images and masks to assets."""
    img_dir = assets_dir / "test_images"
    mask_dir = assets_dir / "test_masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for size_name, info in selected.items():
        # Save scene image (same image for all sizes)
        img_path = img_dir / f"scene_{size_name}.jpg"
        cv2.imwrite(str(img_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved: {img_path.relative_to(assets_dir.parent.parent.parent.parent)}")

        # Save binary mask
        mask_path = mask_dir / f"mask_{size_name}.png"
        cv2.imwrite(str(mask_path), info["mask"])
        print(f"  Saved: {mask_path.relative_to(assets_dir.parent.parent.parent.parent)}")


def visualize(image_bgr, detections, selected):
    """Show detections with selected ones highlighted."""
    vis = image_bgr.copy()
    COCO_NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]

    selected_indices = {v["index"] for v in selected.values()}
    selected_names = {v["index"]: k for k, v in selected.items()}

    for i, (bbox, mask, score, cls_id) in enumerate(detections):
        x1, y1, x2, y2 = bbox.astype(int)
        is_selected = i in selected_indices

        color = (0, 255, 0) if is_selected else (128, 128, 128)
        thickness = 2 if is_selected else 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        cls_name = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else str(cls_id)
        label = f"#{i} {cls_name} {score:.2f} d={bbox_size(bbox):.0f}"
        if is_selected:
            label = f"[{selected_names[i].upper()}] {label}"

        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

        # Overlay mask
        if is_selected:
            overlay = vis.copy()
            overlay[mask > 128] = color
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    cv2.imshow("YOLO-seg Detections", vis)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Generate test data for AI Eraser benchmark")
    parser.add_argument("--image", type=str, default=str(ASSETS_DIR / "sample_image.jpg"))
    parser.add_argument("--model", type=str, default=str(YOLO_MODEL))
    parser.add_argument("--show", action="store_true", help="Visualize detections")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    print("=== Generate Test Data for AI Eraser ===")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    print("\nRunning YOLO-seg...")
    detections = run_yolo_seg(image, args.model)
    print(f"Found {len(detections)} objects")

    if len(detections) == 0:
        print("ERROR: No objects detected. Try a different image.")
        sys.exit(1)

    # Print all detections
    print("\nAll detections:")
    for i, (bbox, mask, score, cls_id) in enumerate(detections):
        diag = bbox_size(bbox)
        area_pct = (mask > 128).sum() / (image.shape[0] * image.shape[1]) * 100
        print(f"  #{i}: class={cls_id}, score={score:.2f}, "
              f"diag={diag:.0f}px, mask_area={area_pct:.1f}%")

    # Select by ROI size
    print("\nSelecting by ROI size:")
    selected = select_by_roi_size(detections, ROI_TARGETS)

    if len(selected) < 3:
        print(f"\nWARNING: Only {len(selected)} objects selected (need 3). "
              f"Using available detections.")
        # Fill missing sizes with closest available
        for size_name in ROI_TARGETS:
            if size_name not in selected and detections:
                # Use any remaining detection
                for i, (bbox, mask, score, cls_id) in enumerate(detections):
                    if i not in {v["index"] for v in selected.values()}:
                        selected[size_name] = {
                            "bbox": bbox, "mask": mask, "score": score,
                            "class_id": cls_id, "diag": bbox_size(bbox), "index": i,
                        }
                        print(f"  {size_name} (fallback): detection #{i}")
                        break

    # Save
    print("\nSaving test data:")
    save_test_data(image, selected, ASSETS_DIR)

    if args.show:
        visualize(image, detections, selected)

    print("\nDone!")


if __name__ == "__main__":
    main()
