"""Analyze ONNX model op types and identify quantization-sensitive nodes."""
import sys
from collections import Counter
from pathlib import Path

import onnx

def analyze(model_path):
    model = onnx.load(str(model_path))
    graph = model.graph

    op_counts = Counter(n.op_type for n in graph.node)
    print(f"=== {Path(model_path).name} ===")
    print(f"Total nodes: {len(graph.node)}")
    print(f"\nOp Type Distribution:")
    for op, cnt in op_counts.most_common():
        print(f"  {op}: {cnt}")

    # Identify potentially problematic ops for INT8
    sensitive_ops = {"Softmax", "LayerNormalization", "GroupNorm",
                     "InstanceNormalization", "Sigmoid", "Tanh",
                     "Attention", "MultiHeadAttention", "Resize",
                     "ReduceMean", "BatchNormalization"}
    found = {op: cnt for op, cnt in op_counts.items() if op in sensitive_ops}
    if found:
        print(f"\nQuantization-sensitive ops:")
        for op, cnt in sorted(found.items(), key=lambda x: -x[1]):
            print(f"  {op}: {cnt}")

    # Show node names for sensitive ops
    print(f"\nSensitive node names (for nodes_to_exclude):")
    for node in graph.node:
        if node.op_type in sensitive_ops:
            print(f"  [{node.op_type}] {node.name}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "weights/sd_v1.5_inpaint/onnx/vae_decoder_fp32.onnx"
    analyze(path)
