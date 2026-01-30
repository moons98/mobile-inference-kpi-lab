#!/usr/bin/env python3
"""
Analyze model operations and identify potential fallback ops.

This script examines an ONNX model to identify operations that may not be
fully supported on the NPU and could cause fallback to CPU/GPU.

Based on ONNX Runtime QNN Execution Provider supported operators:
https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

Target: Snapdragon 8 Gen 2 (SM8550) HTP backend
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


# QNN Execution Provider supported operators (ONNX Runtime 1.19+, QNN 2.26+)
# Reference: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
QNN_SUPPORTED_OPS = {
    # Mathematical & Logical Operations
    "Abs", "Add", "And", "Div", "Equal", "Exp", "Floor",
    "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
    "Log", "Mul", "Neg", "Not", "Or", "Pow", "Round", "Sign", "Sqrt", "Sub",

    # Activation Functions
    "Relu", "Sigmoid", "Tanh", "Elu", "Gelu", "LeakyRelu",
    "HardSwish", "HardSigmoid", "Softmax", "LogSoftmax",

    # Neural Network Layers
    "Conv", "ConvTranspose", "Gemm", "MatMul",
    "BatchNormalization", "InstanceNormalization", "LayerNormalization",

    # Pooling Operations
    "AveragePool", "MaxPool", "GlobalAveragePool",

    # Shape Operations
    "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Reshape",
    "Expand", "Pad", "Slice", "Concat", "Split", "Tile",

    # Specialized Operations
    "Gather", "Resize", "GridSample", "TopK", "Where",
    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum", "ReduceProd",
    "ArgMax", "ArgMin",

    # Quantization Operations
    "DequantizeLinear", "QuantizeLinear",

    # Other supported ops
    "Clip", "Cast", "Constant", "Identity",
}

# Operations NOT supported by QNN EP (will fallback to CPU)
QNN_UNSUPPORTED_OPS = {
    # Control flow (explicitly not supported)
    "Loop", "If", "Scan",

    # RNN variants (limited or no support)
    "LSTM", "GRU", "RNN",

    # Attention (custom ops, not standard ONNX)
    "Attention", "MultiHeadAttention",

    # Other unsupported ops
    "NonZero", "NonMaxSuppression",
    "RoiAlign", "RoiPool",
    "Mish", "Selu", "Celu",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
    "Einsum", "MatMulInteger", "ConvInteger",
    "DynamicQuantizeLinear",  # Dynamic quantization not supported on HTP
}

# Operators with known limitations on Snapdragon 8 Gen 2 HTP
# These work but may have constraints (data types, shapes, etc.)
QNN_LIMITED_SUPPORT_OPS = {
    "Gather": "Only supports positive indices",
    "MatMul": "Limited input data types on HTP backend",
    "Resize": "Mode-dependent (nearest, linear supported)",
    "Cast": "Limited type conversions on HTP",
    "DequantizeLinear": "uint8/uint16 primarily supported",
    "QuantizeLinear": "uint8/uint16 primarily supported",
}

# Legacy aliases for backward compatibility
POTENTIALLY_UNSUPPORTED_OPS = QNN_UNSUPPORTED_OPS
WELL_SUPPORTED_OPS = QNN_SUPPORTED_OPS


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
    unsupported_details = []
    limited_details = []
    unknown_details = []

    for node in graph.node:
        op_type = node.op_type
        op_counts[op_type] += 1

        detail = {
            "name": node.name,
            "op_type": op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }

        if op_type in QNN_UNSUPPORTED_OPS:
            unsupported_details.append(detail)
        elif op_type in QNN_LIMITED_SUPPORT_OPS:
            detail["limitation"] = QNN_LIMITED_SUPPORT_OPS[op_type]
            limited_details.append(detail)
        elif op_type not in QNN_SUPPORTED_OPS:
            unknown_details.append(detail)

    return {
        "total_ops": sum(op_counts.values()),
        "unique_ops": len(op_counts),
        "op_counts": dict(op_counts),
        "unsupported": unsupported_details,
        "limited_support": limited_details,
        "unknown": unknown_details,
        # Legacy field
        "potentially_unsupported": unsupported_details + unknown_details,
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

    print(f"\n--- Operation Counts (QNN EP compatibility) ---")
    sorted_ops = sorted(analysis['op_counts'].items(), key=lambda x: -x[1])
    for op, count in sorted_ops:
        if op in QNN_UNSUPPORTED_OPS:
            status = " [X] Not supported - CPU fallback"
        elif op in QNN_LIMITED_SUPPORT_OPS:
            status = " [~] Limited support"
        elif op in QNN_SUPPORTED_OPS:
            status = " [OK]"
        else:
            status = " [?] Unknown"
        print(f"  {op}: {count}{status}")

    # Unsupported ops (will fallback to CPU)
    unsupported = analysis['unsupported']
    if unsupported:
        print(f"\n--- Unsupported Operations ({len(unsupported)}) - Will fallback to CPU ---")
        for op in unsupported[:10]:
            print(f"  [X] {op['op_type']}: {op['name']}")
        if len(unsupported) > 10:
            print(f"  ... and {len(unsupported) - 10} more")

    # Limited support ops
    limited = analysis['limited_support']
    if limited:
        print(f"\n--- Limited Support Operations ({len(limited)}) ---")
        seen_types = set()
        for op in limited:
            if op['op_type'] not in seen_types:
                print(f"  [~] {op['op_type']}: {op['limitation']}")
                seen_types.add(op['op_type'])

    # Unknown ops
    unknown = analysis['unknown']
    if unknown:
        print(f"\n--- Unknown Operations ({len(unknown)}) ---")
        unknown_types = set(op['op_type'] for op in unknown)
        for op_type in unknown_types:
            count = sum(1 for op in unknown if op['op_type'] == op_type)
            print(f"  [?] {op_type}: {count} occurrences")

    if not unsupported and not unknown:
        print(f"\n--- All operations are QNN EP compatible ---")

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
    print(f"\n--- Recommendations for Snapdragon 8 Gen 2 ---")

    if unsupported:
        print("  [!] Unsupported ops will cause CPU fallback:")
        unsupported_types = set(o['op_type'] for o in unsupported)
        for op in unsupported_types:
            if op == "DynamicQuantizeLinear":
                print(f"     - {op}: Use static quantization (QuantizeLinear/DequantizeLinear)")
            elif op in ("MatMulInteger", "ConvInteger"):
                print(f"     - {op}: Use QDQ format quantization instead")
            elif op in ("LSTM", "GRU", "RNN"):
                print(f"     - {op}: Consider using Transformer or splitting model")
            else:
                print(f"     - {op}: Will run on CPU")

    if unknown:
        print("  [?] Unknown ops - verify with actual device profiling")

    if limited:
        print("  [~] Limited support ops - check data types and shapes")

    if dynamic:
        print("  [!] Dynamic shapes detected:")
        print("     - Fix to static dimensions for HTP backend")
        print("     - Use batch_size=1 for mobile inference")

    if not unsupported and not unknown and not dynamic:
        print("  Model appears fully compatible with QNN EP on Snapdragon 8 Gen 2")

    print("\n--- References ---")
    print("  QNN EP Docs: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html")
    print("  Target: Snapdragon 8 Gen 2 (SM8550) HTP backend")

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
