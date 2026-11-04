"""
Utility script to generate an initial YOLO nano ONNX model for pipeline verification.
Creates a valid ONNX graph with input 'images' [1, 3, 640, 640] and output 'output0' [1, 5, 8400].
This allows test.py and evaluate.py to be executed and tested immediately.
"""

import os
import sys
from pathlib import Path
import numpy as np


def generate_pipeline_onnx(output_path: str = "models/model.onnx"):
    """Generates a standard YOLOv8/v11-compatible single-class ONNX model."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import onnx
        from onnx import helper, TensorProto

        # Create input: 'images' [1, 3, 640, 640]
        input_tensor = helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, [1, 3, 640, 640]
        )

        # Create output: 'output0' [1, 5, 8400]
        output_tensor = helper.make_tensor_value_info(
            "output0", TensorProto.FLOAT, [1, 5, 8400]
        )

        # Construct a lightweight convolutional pipeline node
        # Weights for a conv layer mapping 3 -> 5 channels with adaptive pooling / resize to 8400
        # For pipeline verification, we create a constant or simple graph
        # Node: Constant tensor for output or simple operations
        # Create dummy weights tensor: [1, 5, 8400]
        dummy_weights = np.zeros((1, 5, 8400), dtype=np.float32)
        # Assign default normalized center coordinates across 8400 anchor grid
        grid_x = np.linspace(50, 590, 8400, dtype=np.float32)
        grid_y = np.linspace(50, 590, 8400, dtype=np.float32)
        dummy_weights[0, 0, :] = grid_x
        dummy_weights[0, 1, :] = grid_y
        dummy_weights[0, 2, :] = 200.0  # width
        dummy_weights[0, 3, :] = 100.0  # height
        dummy_weights[0, 4, :] = 0.05   # baseline low confidence (rejected unless note is placed)

        const_tensor = helper.make_tensor(
            name="const_pred",
            data_type=TensorProto.FLOAT,
            dims=[1, 5, 8400],
            vals=dummy_weights.flatten().tolist(),
        )

        # Create node: ReduceSum of input * 0.0 + const_pred to maintain input dependency
        reduce_node = helper.make_node(
            "ReduceMean",
            inputs=["images"],
            outputs=["img_mean"],
            axes=[1, 2, 3],
            keepdims=1,
        )

        zero_node = helper.make_node(
            "Mul",
            inputs=["img_mean", "zero_factor"],
            outputs=["zero_out"],
        )

        add_node = helper.make_node(
            "Add",
            inputs=["const_pred", "zero_out"],
            outputs=["output0"],
        )

        zero_const = helper.make_tensor(
            name="zero_factor",
            data_type=TensorProto.FLOAT,
            dims=[1, 1, 1, 1],
            vals=[0.0],
        )

        graph = helper.make_graph(
            nodes=[reduce_node, zero_node, add_node],
            name="yolo_zed_50_detector",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[const_tensor, zero_const],
        )

        onnx_model = helper.make_model(graph, producer_name="ZED-Digital-Pipeline", opset_imports=[helper.make_opsetid("", 12)])
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(out_file))
        print(f"[+] Successfully generated YOLO-compatible pipeline ONNX model at: {out_file.resolve()}")
        return True

    except Exception as e:
        print(f"[!] Note on ONNX creation: {e}")
        return False


if __name__ == "__main__":
    out_p = sys.argv[1] if len(sys.argv) > 1 else "models/model.onnx"
    generate_pipeline_onnx(out_p)
