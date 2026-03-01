#!/usr/bin/env python3
"""
Add pre/post-processing to YOLOv8n ONNX model.

Creates an end-to-end model where pre/post processing ops can be
offloaded to NPU/GPU via QNN Execution Provider.

Preprocessing (added before model):
  Input:  uint8 [1, 640, 640, 3]  (HWC, letterbox-resized on CPU)
  -> Cast to float32
  -> Divide by 255.0 (normalize to [0, 1])
  -> Transpose HWC -> CHW  [1, 3, 640, 640]

Postprocessing (added after model):
  Model output: float32 [1, 84, 8400]
  -> Transpose to [1, 8400, 84]
  -> Split: boxes [1, 8400, 4] (cx,cy,w,h) + scores [1, 8400, 80]
  -> Convert boxes: center -> corner (x1,y1,x2,y2)
  -> NonMaxSuppression (optional, per-class, IoU=0.45, conf=0.25)

Modes:
  --mode pre       Preprocessing only
  --mode post      Postprocessing only (box convert + optional NMS)
  --mode full      Both pre and post (default)

Usage:
  python add_pre_post.py
  python add_pre_post.py --input yolov8n.onnx --output yolov8n_e2e.onnx
  python add_pre_post.py --mode pre
  python add_pre_post.py --mode full --no-nms
  python add_pre_post.py --verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference


ASSETS_DIR = Path(__file__).parent.parent / "android" / "app" / "src" / "main" / "assets"

# YOLOv8 postprocessing defaults
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS_PER_CLASS = 300


def get_model_io(model):
    """Get original model input/output names and shapes."""
    graph = model.graph
    input_info = graph.input[0]
    output_info = graph.output[0]

    def extract_shape(tensor_type):
        return [
            d.dim_value if d.dim_value > 0 else 1
            for d in tensor_type.shape.dim
        ]

    return {
        "input_name": input_info.name,
        "input_shape": extract_shape(input_info.type.tensor_type),
        "input_type": input_info.type.tensor_type.elem_type,
        "output_name": output_info.name,
        "output_shape": extract_shape(output_info.type.tensor_type),
    }


def add_preprocessing(model):
    """Add preprocessing: uint8 HWC [1,H,W,3] -> float32 NCHW [1,3,H,W] normalized.

    Letterbox resize is NOT included (stays on CPU via Bitmap API).
    This handles the compute-heavy normalize + transpose step.
    """
    graph = model.graph
    io = get_model_io(model)

    orig_input_name = io["input_name"]
    _, C, H, W = io["input_shape"]

    print(f"  Original input: '{orig_input_name}' float32 {io['input_shape']}")
    print(f"  New input:      'input_image' uint8 [1, {H}, {W}, {C}]")

    new_input_name = "input_image"

    # New uint8 HWC input
    new_input = helper.make_tensor_value_info(
        new_input_name, TensorProto.UINT8, [1, H, W, C]
    )

    # Constant: 255.0
    const_255 = numpy_helper.from_array(
        np.array(255.0, dtype=np.float32), name="pre/const_255"
    )

    # Preprocessing nodes
    pre_nodes = [
        # Cast uint8 -> float32
        helper.make_node(
            "Cast", [new_input_name], ["pre/cast_out"],
            to=TensorProto.FLOAT, name="pre/Cast"
        ),
        # Divide by 255.0 -> [0.0, 1.0]
        helper.make_node(
            "Div", ["pre/cast_out", "pre/const_255"], ["pre/normalized"],
            name="pre/Normalize"
        ),
        # Transpose HWC -> CHW: [1,H,W,3] -> [1,3,H,W]
        helper.make_node(
            "Transpose", ["pre/normalized"], [orig_input_name],
            perm=[0, 3, 1, 2], name="pre/Transpose"
        ),
    ]

    # Replace graph input
    inputs = list(graph.input)
    inputs[0] = new_input
    del graph.input[:]
    graph.input.extend(inputs)

    # Add constant
    graph.initializer.append(const_255)

    # Prepend preprocessing nodes before existing model nodes
    existing_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(pre_nodes + existing_nodes)

    print(f"  Added {len(pre_nodes)} preprocessing nodes")
    return model


def add_postprocessing(model, include_nms=True,
                       conf_threshold=CONFIDENCE_THRESHOLD,
                       iou_threshold=NMS_IOU_THRESHOLD):
    """Add postprocessing: [1,84,8400] -> boxes_xyxy + scores [+ NMS indices].

    Box conversion: center (cx,cy,w,h) -> corner (x1,y1,x2,y2)
    NMS: ONNX NonMaxSuppression op (per-class, center_point_box=1)
    """
    graph = model.graph
    io = get_model_io(model)

    orig_output_name = io["output_name"]
    _, features, num_det = io["output_shape"]  # [1, 84, 8400]
    num_classes = features - 4  # 80

    print(f"  Original output: '{orig_output_name}' float32 {io['output_shape']}")
    print(f"  Detections: {num_det}, Classes: {num_classes}")

    # Constants
    initializers = [
        numpy_helper.from_array(
            np.array([4, num_classes], dtype=np.int64), "post/split_sizes_feat"
        ),
        numpy_helper.from_array(
            np.array([2, 2], dtype=np.int64), "post/split_sizes_box"
        ),
        numpy_helper.from_array(
            np.array(2.0, dtype=np.float32), "post/const_2"
        ),
    ]

    post_nodes = [
        # Transpose [1, 84, 8400] -> [1, 8400, 84]
        helper.make_node(
            "Transpose", [orig_output_name], ["post/transposed"],
            perm=[0, 2, 1], name="post/Transpose"
        ),

        # Split features -> boxes [1,8400,4] + scores [1,8400,80]
        helper.make_node(
            "Split", ["post/transposed", "post/split_sizes_feat"],
            ["post/boxes_cxcywh", "post/scores"],
            axis=2, name="post/SplitFeat"
        ),

        # Split boxes -> center_xy [1,8400,2] + wh [1,8400,2]
        helper.make_node(
            "Split", ["post/boxes_cxcywh", "post/split_sizes_box"],
            ["post/center_xy", "post/wh"],
            axis=2, name="post/SplitBox"
        ),

        # half_wh = wh / 2.0
        helper.make_node(
            "Div", ["post/wh", "post/const_2"], ["post/half_wh"],
            name="post/HalfWH"
        ),

        # xy1 = center_xy - half_wh (top-left corner)
        helper.make_node(
            "Sub", ["post/center_xy", "post/half_wh"], ["post/xy1"],
            name="post/XY1"
        ),

        # xy2 = center_xy + half_wh (bottom-right corner)
        helper.make_node(
            "Add", ["post/center_xy", "post/half_wh"], ["post/xy2"],
            name="post/XY2"
        ),

        # boxes_xyxy = concat(xy1, xy2) -> [1, 8400, 4]
        helper.make_node(
            "Concat", ["post/xy1", "post/xy2"], ["boxes"],
            axis=2, name="post/ConcatBoxes"
        ),

        # Identity for clean output naming
        helper.make_node(
            "Identity", ["post/scores"], ["scores"],
            name="post/ScoresOut"
        ),
    ]

    # Output definitions
    new_outputs = [
        helper.make_tensor_value_info(
            "boxes", TensorProto.FLOAT, [1, num_det, 4]
        ),
        helper.make_tensor_value_info(
            "scores", TensorProto.FLOAT, [1, num_det, num_classes]
        ),
    ]

    if include_nms:
        nms_initializers = [
            numpy_helper.from_array(
                np.array([MAX_DETECTIONS_PER_CLASS], dtype=np.int64),
                "post/max_per_class"
            ),
            numpy_helper.from_array(
                np.array([iou_threshold], dtype=np.float32),
                "post/iou_threshold"
            ),
            numpy_helper.from_array(
                np.array([conf_threshold], dtype=np.float32),
                "post/score_threshold"
            ),
        ]
        initializers.extend(nms_initializers)

        nms_nodes = [
            # Transpose scores: [1, 8400, 80] -> [1, 80, 8400] (NMS expects this)
            helper.make_node(
                "Transpose", ["post/scores"], ["post/scores_nms"],
                perm=[0, 2, 1], name="post/TransposeScores"
            ),

            # NonMaxSuppression (per-class, center_point_box=1)
            # Input boxes in center format (cx,cy,w,h), scores [1, classes, detections]
            # Output: selected_indices [N, 3] = (batch_idx, class_idx, box_idx)
            helper.make_node(
                "NonMaxSuppression",
                ["post/boxes_cxcywh", "post/scores_nms",
                 "post/max_per_class", "post/iou_threshold", "post/score_threshold"],
                ["nms_indices"],
                center_point_box=1, name="post/NMS"
            ),
        ]
        post_nodes.extend(nms_nodes)

        new_outputs.append(
            helper.make_tensor_value_info("nms_indices", TensorProto.INT64, [None, 3])
        )
        print(f"  NMS: max_per_class={MAX_DETECTIONS_PER_CLASS}, "
              f"iou={iou_threshold}, conf={conf_threshold}")

    # Add initializers
    graph.initializer.extend(initializers)

    # Append postprocessing nodes
    graph.node.extend(post_nodes)

    # Replace outputs
    del graph.output[:]
    graph.output.extend(new_outputs)

    print(f"  Added {len(post_nodes)} postprocessing nodes")
    return model


def finalize_model(model, opset_version=17):
    """Run shape inference and validate."""
    # Ensure opset version is set
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            if opset.version < opset_version:
                opset.version = opset_version

    # Shape inference
    try:
        model = shape_inference.infer_shapes(model)
        print("  Shape inference: OK")
    except Exception as e:
        print(f"  Shape inference warning: {e}")

    # Validate
    try:
        onnx.checker.check_model(model)
        print("  Model validation: OK")
    except Exception as e:
        print(f"  Model validation warning: {e}")

    return model


def verify(original_path, modified_path, mode):
    """Verify that the modified model produces correct results vs original."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Skipping verification (onnxruntime not installed)")
        return

    print("\nVerification:")

    # Load original model info
    orig_model = onnx.load(str(original_path))
    orig_io = get_model_io(orig_model)
    _, C, H, W = orig_io["input_shape"]

    # Create deterministic test input
    np.random.seed(42)
    test_image_hwc = np.random.randint(0, 256, (1, H, W, C), dtype=np.uint8)

    # Manual preprocessing (reference)
    preprocessed = test_image_hwc.astype(np.float32) / 255.0
    preprocessed = np.transpose(preprocessed, (0, 3, 1, 2))  # HWC -> CHW

    # Run original model
    orig_session = ort.InferenceSession(str(original_path))
    orig_input_name = orig_session.get_inputs()[0].name
    orig_results = orig_session.run(None, {orig_input_name: preprocessed})
    raw_output = orig_results[0]  # [1, 84, 8400]

    # Run modified model
    mod_session = ort.InferenceSession(str(modified_path))
    mod_input_name = mod_session.get_inputs()[0].name

    if mode in ("pre", "full"):
        mod_input = {mod_input_name: test_image_hwc}
    else:  # post only
        mod_input = {mod_input_name: preprocessed}

    mod_results = mod_session.run(None, mod_input)

    if mode == "pre":
        # Output should match original raw output
        diff = np.abs(raw_output - mod_results[0]).max()
        print(f"  Raw output max diff: {diff:.8f}")
        print(f"  {'PASS' if diff < 1e-4 else 'FAIL'}")

    elif mode in ("post", "full"):
        # Compute reference boxes and scores
        transposed = np.transpose(raw_output, (0, 2, 1))  # [1, 8400, 84]
        ref_scores = transposed[:, :, 4:]
        boxes_cxcywh = transposed[:, :, :4]

        cx = boxes_cxcywh[:, :, 0:1]
        cy = boxes_cxcywh[:, :, 1:2]
        w = boxes_cxcywh[:, :, 2:3]
        h = boxes_cxcywh[:, :, 3:4]
        ref_boxes = np.concatenate([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=-1)

        boxes_diff = np.abs(ref_boxes - mod_results[0]).max()
        scores_diff = np.abs(ref_scores - mod_results[1]).max()

        print(f"  Boxes  max diff: {boxes_diff:.8f}")
        print(f"  Scores max diff: {scores_diff:.8f}")
        ok = boxes_diff < 1e-4 and scores_diff < 1e-4
        print(f"  {'PASS' if ok else 'FAIL'}")

        if len(mod_results) > 2:
            nms_indices = mod_results[2]
            print(f"  NMS selected: {nms_indices.shape[0]} detections")


def print_summary(mode, include_nms, output_path):
    """Print summary of changes needed on Android side."""
    print()
    print("=" * 60)
    print("Android-side changes needed")
    print("=" * 60)

    if mode in ("pre", "full"):
        print("""
[Preprocessing - simplified]
  Before: Bitmap -> getPixels -> normalize (÷255) -> HWC->CHW loop -> FloatArray
  After:  Bitmap -> getPixels -> pack as uint8 ByteArray -> OnnxTensor(UINT8)

  The normalize + transpose now runs on NPU/GPU inside the ONNX graph.
  Letterbox resize stays on CPU (Bitmap.createScaledBitmap).

  // Kotlin pseudo-code:
  val pixels = IntArray(640 * 640)
  letterboxed.getPixels(pixels, 0, 640, 0, 0, 640, 640)
  val hwc = ByteArray(640 * 640 * 3)
  for (i in pixels.indices) {
      val p = pixels[i]
      hwc[i * 3]     = ((p shr 16) and 0xFF).toByte()  // R
      hwc[i * 3 + 1] = ((p shr 8) and 0xFF).toByte()   // G
      hwc[i * 3 + 2] = (p and 0xFF).toByte()            // B
  }
  val tensor = OnnxTensor.createTensor(env, ByteBuffer.wrap(hwc),
      longArrayOf(1, 640, 640, 3), OnnxJavaType.UINT8)""")

    if mode in ("post", "full"):
        if include_nms:
            print("""
[Postprocessing - simplified with NMS]
  Before: Parse [1,84,8400] -> confidence filter -> box convert -> per-class NMS
  After:  Read 3 outputs from model:
    - boxes  [1, 8400, 4]  (x1, y1, x2, y2) in 640x640 letterbox space
    - scores [1, 8400, 80] per-class scores
    - nms_indices [N, 3]   (batch_idx, class_idx, box_idx)

  // Kotlin pseudo-code:
  val boxes = outputs[0].value as Array<Array<FloatArray>>     // [1][8400][4]
  val scores = outputs[1].value as Array<Array<FloatArray>>    // [1][8400][80]
  val indices = outputs[2].value as Array<LongArray>           // [N][3]

  val detections = indices.map { idx ->
      val classId = idx[1].toInt()
      val boxIdx = idx[2].toInt()
      val box = boxes[0][boxIdx]
      val score = scores[0][boxIdx][classId]
      // Reverse letterbox transform
      Detection(
          x1 = (box[0] - padLeft) / scale,
          y1 = (box[1] - padTop) / scale,
          x2 = (box[2] - padLeft) / scale,
          y2 = (box[3] - padTop) / scale,
          confidence = score, classId = classId
      )
  }""")
        else:
            print("""
[Postprocessing - box conversion offloaded, NMS on CPU]
  Before: Parse [1,84,8400] -> confidence filter -> box convert -> per-class NMS
  After:  Read 2 outputs:
    - boxes  [1, 8400, 4]  (x1, y1, x2, y2) in 640x640 letterbox space
    - scores [1, 8400, 80] per-class scores
  Then apply confidence filter + NMS on CPU (same as before, but simpler parsing).""")

    print(f"\nModel saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add pre/post-processing to YOLOv8n ONNX model"
    )
    parser.add_argument(
        "--input", type=Path,
        default=ASSETS_DIR / "yolov8n.onnx",
        help="Input ONNX model path (default: assets/yolov8n.onnx)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output ONNX model path (default: auto-generated based on mode)"
    )
    parser.add_argument(
        "--mode", choices=["pre", "post", "full"], default="full",
        help="What to add: pre (preprocessing only), post (postprocessing only), "
             "full (both, default)"
    )
    parser.add_argument(
        "--no-nms", action="store_true",
        help="Skip NMS in postprocessing (box conversion only)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify output matches original model (requires onnxruntime)"
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"NMS confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=NMS_IOU_THRESHOLD,
        help=f"NMS IoU threshold (default: {NMS_IOU_THRESHOLD})"
    )

    args = parser.parse_args()

    # Auto-generate output name
    if args.output is None:
        stem = args.input.stem
        suffixes = {
            "pre": "_pre",
            "post": "_post" + ("" if args.no_nms else "_nms"),
            "full": "_e2e" + ("" if args.no_nms else "_nms"),
        }
        args.output = args.input.parent / f"{stem}{suffixes[args.mode]}.onnx"

    # Check input exists
    if not args.input.exists():
        print(f"Error: Input model not found: {args.input}")
        sys.exit(1)

    # Load model
    print(f"Loading: {args.input}")
    model = onnx.load(str(args.input))
    io = get_model_io(model)
    print(f"  Input:  {io['input_name']} {io['input_shape']}")
    print(f"  Output: {io['output_name']} {io['output_shape']}")

    # Add preprocessing
    if args.mode in ("pre", "full"):
        print(f"\nAdding preprocessing:")
        model = add_preprocessing(model)

    # Add postprocessing
    include_nms = not args.no_nms
    if args.mode in ("post", "full"):
        print(f"\nAdding postprocessing (NMS={'yes' if include_nms else 'no'}):")
        model = add_postprocessing(
            model, include_nms=include_nms,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
        )

    # Finalize
    print(f"\nFinalizing:")
    model = finalize_model(model)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nSaved: {args.output.name} ({size_mb:.2f} MB)")

    # Verify
    if args.verify:
        verify(args.input, args.output, args.mode)

    # Summary
    print_summary(args.mode, include_nms, args.output)


if __name__ == "__main__":
    main()
