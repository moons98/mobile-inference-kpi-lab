#!/usr/bin/env python3
"""
Analyze model operations and identify potential fallback ops.

This script examines an ONNX model to identify operations that may not be
fully supported on the NPU and could cause fallback to CPU/GPU.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Error: onnx package not installed")
    print("Install with: pip install onnx")
    sys.exit(1)


# Operations known to have potential NPU support issues
# This list is based on common QNN/SNPE limitations
POTENTIALLY_UNSUPPORTED_OPS = {
    # Dynamic operations
    "Shape", "Gather", "Squeeze", "Unsqueeze", "Reshape",
    "Expand", "Tile", "NonZero", "Where",

    # Complex operations
    "LSTM", "GRU", "RNN",
    "Attention", "MultiHeadAttention",

    # Reduction operations (may have limitations)
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
    "ReduceProd", "ReduceL1", "ReduceL2",

    # Activation functions (some variants)
    "HardSwish", "HardSigmoid", "Mish", "Selu", "Celu",

    # Other potentially problematic ops
    "Resize", "Upsample",  # Mode-dependent
    "InstanceNormalization",
    "GroupNormalization",
    "LayerNormalization",
}

# Operations typically well-supported on NPU
WELL_SUPPORTED_OPS = {
    "Conv", "ConvTranspose",
    "MaxPool", "AveragePool", "GlobalAveragePool",
    "Relu", "Sigmoid", "Tanh", "Softmax",
    "BatchNormalization",
    "Add", "Mul", "Sub", "Div",
    "MatMul", "Gemm",
    "Flatten", "Concat",
    "Clip",  # ReLU6
}


def load_model(model_path: str):
    """Load ONNX model."""
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return model


def analyze_ops(model) -> dict:
    """Analyze operations in the model."""
    graph = model.graph
    op_counts = Counter()
    op_details = []

    for node in graph.node:
        op_type = node.op_type
        op_counts[op_type] += 1

        # Collect details for potentially problematic ops
        if op_type in POTENTIALLY_UNSUPPORTED_OPS:
            op_details.append({
                "name": node.name,
                "op_type": op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            })

    return {
        "total_ops": sum(op_counts.values()),
        "unique_ops": len(op_counts),
        "op_counts": dict(op_counts),
        "potentially_unsupported": op_details,
    }


def check_dynamic_shapes(model) -> list:
    """Check for dynamic shapes in inputs/outputs."""
    dynamic_shapes = []

    for input_tensor in model.graph.input:
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_param:  # Named dimension (dynamic)
                shape.append(f"?({dim.dim_param})")
            elif dim.dim_value == 0:  # Unknown dimension
                shape.append("?")
            else:
                shape.append(dim.dim_value)

        if any(isinstance(s, str) for s in shape):
            dynamic_shapes.append({
                "name": input_tensor.name,
                "shape": shape,
                "type": "input"
            })

    return dynamic_shapes


def generate_report(model_path: str):
    """Generate analysis report."""
    model = load_model(model_path)

    print("\n" + "=" * 60)
    print("MODEL ANALYSIS REPORT")
    print("=" * 60)

    # Basic info
    print(f"\nModel: {Path(model_path).name}")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")

    # Op analysis
    analysis = analyze_ops(model)

    print(f"\n--- Operation Summary ---")
    print(f"Total operations: {analysis['total_ops']}")
    print(f"Unique op types: {analysis['unique_ops']}")

    print(f"\n--- Operation Counts ---")
    sorted_ops = sorted(analysis['op_counts'].items(), key=lambda x: -x[1])
    for op, count in sorted_ops:
        status = ""
        if op in POTENTIALLY_UNSUPPORTED_OPS:
            status = " [!] Potential fallback"
        elif op in WELL_SUPPORTED_OPS:
            status = " [✓] Well supported"
        print(f"  {op}: {count}{status}")

    # Potentially unsupported ops
    unsupported = analysis['potentially_unsupported']
    if unsupported:
        print(f"\n--- Potentially Unsupported Operations ({len(unsupported)}) ---")
        for op in unsupported[:10]:  # Show first 10
            print(f"  [{op['op_type']}] {op['name']}")
        if len(unsupported) > 10:
            print(f"  ... and {len(unsupported) - 10} more")
    else:
        print(f"\n--- No potentially unsupported operations found ---")

    # Dynamic shapes
    dynamic = check_dynamic_shapes(model)
    if dynamic:
        print(f"\n--- Dynamic Shapes Found ---")
        for d in dynamic:
            print(f"  {d['type']}: {d['name']} = {d['shape']}")
        print("\n  [!] Dynamic shapes may cause issues on NPU")
        print("  Consider using fixed batch size of 1")
    else:
        print(f"\n--- All shapes are static ---")

    # Recommendations
    print(f"\n--- Recommendations ---")

    if unsupported:
        print("  1. Test with NPU_FALLBACK mode first")
        print("  2. Profile to identify which ops actually fall back")
        print("  3. Consider graph transformations for critical ops:")
        for op in set(o['op_type'] for o in unsupported[:5]):
            if op == "HardSwish":
                print(f"     - {op}: Can approximate with ReLU6")
            elif op in ("Reshape", "Squeeze", "Unsqueeze"):
                print(f"     - {op}: Ensure shapes are static")
            elif op == "LayerNormalization":
                print(f"     - {op}: Check if can use BatchNorm instead")

    if dynamic:
        print("  - Fix dynamic dimensions to static values")
        print("  - Use input_dims flag during conversion")

    print("\n" + "=" * 60)

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ONNX model for NPU compatibility"
    )
    parser.add_argument(
        "model",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--json",
        help="Output analysis as JSON to file"
    )

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    analysis = generate_report(args.model)

    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {args.json}")


if __name__ == "__main__":
    main()
