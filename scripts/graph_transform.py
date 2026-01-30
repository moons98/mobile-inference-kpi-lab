#!/usr/bin/env python3
"""
Low-risk graph transformations to improve NPU compatibility.

These transformations do not require retraining and aim to:
1. Remove/replace unsupported operations
2. Fix dynamic shapes
3. Fuse operations for better performance
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import numpy as np
except ImportError:
    print("Error: Required packages not installed")
    print("Install with: pip install onnx numpy")
    sys.exit(1)


def load_model(model_path: str):
    """Load and validate ONNX model."""
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return model


def save_model(model, output_path: str):
    """Save ONNX model."""
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved model to: {output_path}")


def fix_dynamic_batch(model, batch_size: int = 1):
    """
    Replace dynamic batch dimension with fixed size.

    Args:
        model: ONNX model
        batch_size: Fixed batch size to use

    Returns:
        Modified model
    """
    print(f"Fixing dynamic batch dimension to {batch_size}")

    for input_tensor in model.graph.input:
        shape = input_tensor.type.tensor_type.shape
        if len(shape.dim) > 0:
            dim = shape.dim[0]
            if dim.dim_param or dim.dim_value == 0:
                print(f"  Fixed input '{input_tensor.name}' batch dim")
                dim.dim_value = batch_size
                dim.dim_param = ""

    for output_tensor in model.graph.output:
        shape = output_tensor.type.tensor_type.shape
        if len(shape.dim) > 0:
            dim = shape.dim[0]
            if dim.dim_param or dim.dim_value == 0:
                print(f"  Fixed output '{output_tensor.name}' batch dim")
                dim.dim_value = batch_size
                dim.dim_param = ""

    return model


def replace_hardswish_with_relu6(model):
    """
    Replace HardSwish with ReLU6 approximation.

    HardSwish(x) = x * ReLU6(x + 3) / 6
    Approximation: ReLU6(x) (simpler, slightly different but NPU-friendly)

    This is a lossy transformation - verify accuracy impact!
    """
    print("Replacing HardSwish with ReLU6 approximation")

    nodes_to_remove = []
    nodes_to_add = []

    for i, node in enumerate(model.graph.node):
        if node.op_type == "HardSwish":
            print(f"  Replacing: {node.name}")

            # Create Clip node (ReLU6 = Clip(0, 6))
            clip_node = helper.make_node(
                "Clip",
                inputs=[node.input[0]],
                outputs=[node.output[0]],
                name=f"{node.name}_relu6",
                min=0.0,
                max=6.0
            )

            nodes_to_remove.append(node)
            nodes_to_add.append((i, clip_node))

    # Apply changes
    for node in nodes_to_remove:
        model.graph.node.remove(node)

    for i, node in nodes_to_add:
        model.graph.node.insert(i, node)

    if nodes_to_remove:
        print(f"  Replaced {len(nodes_to_remove)} HardSwish operations")
    else:
        print("  No HardSwish operations found")

    return model


def remove_identity_nodes(model):
    """Remove unnecessary Identity nodes."""
    print("Removing Identity nodes")

    nodes_to_remove = []
    replacements = {}

    for node in model.graph.node:
        if node.op_type == "Identity":
            # Map output to input
            replacements[node.output[0]] = node.input[0]
            nodes_to_remove.append(node)

    # Update references
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in replacements:
                node.input[i] = replacements[inp]

    # Remove Identity nodes
    for node in nodes_to_remove:
        model.graph.node.remove(node)

    print(f"  Removed {len(nodes_to_remove)} Identity nodes")
    return model


def fold_constants(model):
    """
    Fold constant operations.

    Uses ONNX optimizer to fold constant expressions.
    """
    print("Folding constants")

    try:
        from onnx import optimizer
        passes = ["eliminate_deadend", "eliminate_identity", "fuse_consecutive_transposes"]
        model = optimizer.optimize(model, passes)
        print("  Applied ONNX optimizations")
    except ImportError:
        print("  ONNX optimizer not available, skipping")

    return model


def validate_transformation(original_path: str, transformed_path: str, tolerance: float = 1e-5):
    """
    Validate that transformation doesn't significantly change outputs.

    Args:
        original_path: Path to original model
        transformed_path: Path to transformed model
        tolerance: Maximum allowed difference

    Returns:
        Tuple of (is_valid, max_diff)
    """
    print("\nValidating transformation...")

    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping validation")
        print("  Install with: pip install onnxruntime")
        return True, 0.0

    # Load models
    original_session = ort.InferenceSession(original_path)
    transformed_session = ort.InferenceSession(transformed_path)

    # Get input info
    input_info = original_session.get_inputs()[0]
    input_shape = [d if isinstance(d, int) else 1 for d in input_info.shape]

    # Create dummy input
    np.random.seed(42)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Run both models
    original_output = original_session.run(None, {input_info.name: dummy_input})[0]
    transformed_output = transformed_session.run(None, {input_info.name: dummy_input})[0]

    # Compare
    max_diff = np.max(np.abs(original_output - transformed_output))
    mean_diff = np.mean(np.abs(original_output - transformed_output))

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    is_valid = max_diff < tolerance
    if is_valid:
        print(f"  ✓ Validation passed (tolerance: {tolerance})")
    else:
        print(f"  ✗ Validation failed (exceeds tolerance: {tolerance})")

    return is_valid, max_diff


def main():
    parser = argparse.ArgumentParser(
        description="Apply low-risk graph transformations for NPU compatibility"
    )
    parser.add_argument(
        "input",
        help="Input ONNX model path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output ONNX model path"
    )
    parser.add_argument(
        "--fix-batch",
        type=int,
        metavar="SIZE",
        help="Fix dynamic batch dimension to specified size"
    )
    parser.add_argument(
        "--replace-hardswish",
        action="store_true",
        help="Replace HardSwish with ReLU6"
    )
    parser.add_argument(
        "--remove-identity",
        action="store_true",
        help="Remove Identity nodes"
    )
    parser.add_argument(
        "--fold-constants",
        action="store_true",
        help="Fold constant operations"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Apply all transformations"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output matches original"
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not args.output:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_optimized.onnx")

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()

    # Load model
    model = load_model(args.input)

    # Apply transformations
    if args.all or args.fix_batch:
        batch_size = args.fix_batch if args.fix_batch else 1
        model = fix_dynamic_batch(model, batch_size)

    if args.all or args.remove_identity:
        model = remove_identity_nodes(model)

    if args.all or args.replace_hardswish:
        model = replace_hardswish_with_relu6(model)

    if args.all or args.fold_constants:
        model = fold_constants(model)

    # Save model
    save_model(model, args.output)

    # Validate if requested
    if args.validate:
        is_valid, max_diff = validate_transformation(args.input, args.output)
        if not is_valid:
            print("\nWarning: Transformation changed model outputs significantly!")
            print("Review the changes before using in production.")


if __name__ == "__main__":
    main()
