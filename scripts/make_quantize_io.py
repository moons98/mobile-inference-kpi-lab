#!/usr/bin/env python3
"""
Convert a QDQ (INT8) ONNX model to QuantizeIO format by removing boundary Q/DQ nodes.

QDQ model: FP32 I/O → QuantizeLinear → [INT8 compute] → DequantizeLinear → FP32 I/O
QIO model: UINT8 I/O → [INT8 compute] → UINT8 I/O  (boundary Q/DQ removed)

This eliminates the CPU fallback QuantizeLinear/DequantizeLinear ops at the graph boundary,
enabling 100% NPU offload on QNN EP. The app must handle quantize/dequantize in pre/post processing.

Usage:
    python make_quantize_io.py                              # Default: QDQ → QIO
    python make_quantize_io.py --input path/to/qdq.onnx     # Custom input
    python make_quantize_io.py --input qdq.onnx --output qio.onnx
"""

import argparse
import sys
from pathlib import Path

try:
    import onnx
    from onnx import numpy_helper, TensorProto, helper
except ImportError:
    print("Error: onnx not installed. Install with: pip install onnx")
    sys.exit(1)

ASSETS_DIR = Path(__file__).parent.parent / "android" / "app" / "src" / "main" / "assets"


def get_tensor_value(tensor):
    """Extract scalar value from an ONNX TensorProto."""
    arr = numpy_helper.to_array(tensor)
    return arr.flat[0]


def remove_boundary_qdq(model_path: Path, output_path: Path) -> dict:
    """Remove boundary QuantizeLinear/DequantizeLinear nodes from QDQ model.

    Returns dict with extracted I/O quantization parameters.
    """
    model = onnx.load(str(model_path))
    initializers = {init.name: init for init in model.graph.initializer}
    graph = model.graph

    # Build node output → node map for quick lookup
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Build node input consumer map
    input_to_consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp not in input_to_consumers:
                input_to_consumers[inp] = []
            input_to_consumers[inp].append(node)

    graph_input_names = {inp.name for inp in graph.input}
    graph_output_names = {out.name for out in graph.output}

    params = {"inputs": [], "outputs": []}
    nodes_to_remove = set()
    rewire_map = {}  # old_tensor_name → new_tensor_name

    # --- Input boundary: find QuantizeLinear consuming graph inputs ---
    for node in list(graph.node):
        if node.op_type != "QuantizeLinear":
            continue
        if node.input[0] not in graph_input_names:
            continue

        input_name = node.input[0]
        scale_tensor = initializers.get(node.input[1])
        zp_tensor = initializers.get(node.input[2]) if len(node.input) > 2 else None

        scale = float(get_tensor_value(scale_tensor))
        zp = int(get_tensor_value(zp_tensor)) if zp_tensor is not None else 0

        params["inputs"].append({
            "name": input_name,
            "scale": scale,
            "zero_point": zp,
        })

        # The QuantizeLinear output feeds into downstream nodes
        ql_output = node.output[0]

        # Mark this QuantizeLinear for removal
        nodes_to_remove.add(id(node))

        # Rewire: downstream consumers of ql_output should use graph input directly
        # But graph input type must change to UINT8
        rewire_map[ql_output] = input_name

        print(f"  Input '{input_name}': removing QuantizeLinear (scale={scale:.6f}, zp={zp})")

    # --- Output boundary: find DequantizeLinear producing graph outputs ---
    for node in list(graph.node):
        if node.op_type != "DequantizeLinear":
            continue
        if node.output[0] not in graph_output_names:
            continue

        output_name = node.output[0]
        scale_tensor = initializers.get(node.input[1])
        zp_tensor = initializers.get(node.input[2]) if len(node.input) > 2 else None

        scale = float(get_tensor_value(scale_tensor))
        zp = int(get_tensor_value(zp_tensor)) if zp_tensor is not None else 0

        params["outputs"].append({
            "name": output_name,
            "scale": scale,
            "zero_point": zp,
        })

        # The DequantizeLinear input comes from upstream
        dql_input = node.input[0]

        # Mark this DequantizeLinear for removal
        nodes_to_remove.add(id(node))

        # Rewire: graph output should come from dql_input instead
        rewire_map[output_name] = dql_input

        print(f"  Output '{output_name}': removing DequantizeLinear (scale={scale:.6f}, zp={zp})")

    if not params["inputs"] and not params["outputs"]:
        print("WARNING: No boundary Q/DQ nodes found. Is this already a QIO model?")
        return params

    # --- Apply rewiring to all remaining nodes ---
    new_nodes = []
    for node in graph.node:
        if id(node) in nodes_to_remove:
            continue
        # Rewire inputs
        new_inputs = []
        for inp in node.input:
            new_inputs.append(rewire_map.get(inp, inp))
        # Rewire outputs
        new_outputs = []
        for out in node.output:
            new_outputs.append(rewire_map.get(out, out))

        new_node = helper.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=new_outputs,
            name=node.name,
            domain=node.domain,
        )
        new_node.attribute.extend(node.attribute)
        new_nodes.append(new_node)

    # --- Update graph input types to UINT8 ---
    new_inputs = []
    for inp in graph.input:
        if any(p["name"] == inp.name for p in params["inputs"]):
            # Change type from FLOAT to UINT8
            shape_dims = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape_dims.append(dim.dim_value)
                else:
                    shape_dims.append(dim.dim_param or 1)
            new_inp = helper.make_tensor_value_info(
                inp.name, TensorProto.UINT8, shape_dims
            )
            new_inputs.append(new_inp)
            print(f"  Graph input '{inp.name}': FLOAT → UINT8")
        else:
            new_inputs.append(inp)

    # --- Update graph output types to UINT8 ---
    new_outputs = []
    for out in graph.output:
        if any(p["name"] == out.name for p in params["outputs"]):
            # Output now comes from a different tensor (the DQL input)
            new_out_name = rewire_map.get(out.name, out.name)
            shape_dims = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape_dims.append(dim.dim_value)
                else:
                    shape_dims.append(dim.dim_param or 1)
            new_out = helper.make_tensor_value_info(
                new_out_name, TensorProto.UINT8, shape_dims
            )
            new_outputs.append(new_out)
            print(f"  Graph output '{out.name}' → '{new_out_name}': FLOAT → UINT8")
        else:
            new_outputs.append(out)

    # --- Build new graph ---
    new_graph = helper.make_graph(
        new_nodes,
        graph.name,
        new_inputs,
        new_outputs,
        initializer=list(graph.initializer),
    )

    new_model = helper.make_model(new_graph, opset_imports=list(model.opset_import))
    new_model.ir_version = model.ir_version

    # Validate
    try:
        onnx.checker.check_model(new_model)
        print("  [OK] Model validation passed")
    except Exception as e:
        print(f"  [WARN] Model validation: {e}")

    # Save
    onnx.save(new_model, str(output_path))
    orig_size = model_path.stat().st_size / 1024 / 1024
    new_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_path.name} ({new_size:.2f} MB, was {orig_size:.2f} MB)")

    # Count nodes
    orig_ql = sum(1 for n in graph.node if n.op_type == "QuantizeLinear")
    orig_dql = sum(1 for n in graph.node if n.op_type == "DequantizeLinear")
    new_ql = sum(1 for n in new_nodes if n.op_type == "QuantizeLinear")
    new_dql = sum(1 for n in new_nodes if n.op_type == "DequantizeLinear")
    print(f"  Nodes: {len(graph.node)} → {len(new_nodes)} "
          f"(QL: {orig_ql}→{new_ql}, DQL: {orig_dql}→{new_dql})")

    return params


def print_quant_params(params: dict):
    """Print I/O quantization parameters to console."""
    for inp in params["inputs"]:
        print(f"    Input '{inp['name']}': scale={inp['scale']:.6f}, zp={inp['zero_point']}")
    for out in params["outputs"]:
        print(f"    Output '{out['name']}': scale={out['scale']:.6f}, zp={out['zero_point']}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert QDQ ONNX model to QuantizeIO (remove boundary Q/DQ nodes)"
    )
    parser.add_argument(
        "--input", type=Path,
        default=ASSETS_DIR / "yolov8n_int8_qdq.onnx",
        help="Input QDQ model path (default: assets/yolov8n_int8_qdq.onnx)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output QIO model path (default: <input_dir>/<name>_qio.onnx)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input model not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        stem = args.input.stem.replace("_qdq", "").replace("_int8", "_int8")
        args.output = args.input.parent / f"{stem}_qio.onnx"

    print(f"{'=' * 60}")
    print(f"Converting QDQ → QIO (QuantizeIO)")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}")

    params = remove_boundary_qdq(args.input, args.output)

    if params["inputs"] or params["outputs"]:
        print_quant_params(params)

        print(f"\n{'=' * 60}")
        print("Next steps:")
        print("  1. Update OrtRunner.kt YOLOV8N_INT8_QIO with the scale/zp values above")
        print("  2. Build & deploy the app")
        print("  3. Run experiment 5 (QDQ vs QIO)")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
