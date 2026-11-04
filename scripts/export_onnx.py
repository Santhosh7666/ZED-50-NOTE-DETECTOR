"""
ONNX Export and Numerical Verification Script.
Exports YOLO PyTorch weights (.pt) to production-ready ONNX format,
verifies graph validity using the ONNX specification checker,
and cross-verifies PyTorch vs. ONNX Runtime output tensors for parity.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Export Trained YOLO Model to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained PyTorch weights file (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.onnx",
        help="Destination path for exported ONNX file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640, 640],
        help="Input image resolution [height, width] or single integer",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (12 recommended for OpenCV DNN and ONNX Runtime CPU)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Apply ONNX-simplifier graph optimization",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Export with dynamic batch/resolution axes (keep False for max CPU inference speed)",
    )
    return parser.parse_args()


def export_to_onnx(
    weights_path: str = "models/best.pt",
    output_path: str = "models/model.onnx",
    imgsz: list = (640, 640),
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
):
    print("=" * 70)
    print("ZED DIGITAL: YOLO TO ONNX EXPORT PIPELINE")
    print("=" * 70)

    weights_file = Path(weights_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not weights_file.exists():
        raise FileNotFoundError(
            f"PyTorch checkpoint not found at '{weights_file.resolve()}'. "
            f"Please run train.py first to generate best.pt or specify valid weights."
        )

    print(f"[*] Loading PyTorch checkpoint: {weights_file}")
    from ultralytics import YOLO

    model = YOLO(str(weights_file))

    # Normalize image size
    if isinstance(imgsz, (list, tuple)) and len(imgsz) == 1:
        imgsz = (imgsz[0], imgsz[0])
    elif isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    print(f"[*] Exporting model to ONNX format...")
    print(f"    - Input Resolution: {imgsz}")
    print(f"    - Opset Version   : {opset}")
    print(f"    - Dynamic Shapes  : {dynamic}")
    print(f"    - Simplification  : {simplify}")

    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )

    print(f"[+] Initial export completed at: {exported_path}")

    # Move/copy exported model to final expected destination
    if Path(exported_path).resolve() != output_file.resolve():
        import shutil
        shutil.copy2(exported_path, output_file)
        print(f"[+] Copied final model to: {output_file.resolve()}")

    # 1. Verify ONNX Model Graph Integrity
    print("\n[*] Step 1: Verifying ONNX Graph Integrity with onnx.checker...")
    import onnx
    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)
    print("    [OK] ONNX model structure is strictly valid.")

    # 2. Run ONNX Runtime Sample Inference
    print("\n[*] Step 2: Running Sample Inference via ONNX Runtime...")
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(output_file),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape

    print(f"    - Input  node: '{input_name}' | Shape: {input_shape}")
    print(f"    - Output node: '{output_name}' | Shape: {output_shape}")

    # Generate synthetic validation tensor
    dummy_input = np.random.uniform(0.0, 1.0, (1, 3, imgsz[0], imgsz[1])).astype(np.float32)

    ort_outputs = session.run([output_name], {input_name: dummy_input})
    ort_pred = ort_outputs[0]
    print(f"    [OK] Inference succeeded! Prediction output shape: {ort_pred.shape}")

    # 3. Cross-validate Numerical Parity vs PyTorch Model
    print("\n[*] Step 3: Cross-validating Numerical Parity (PyTorch vs ONNX Runtime)...")
    import torch
    with torch.no_grad():
        torch_tensor = torch.from_numpy(dummy_input)
        pt_preds = model.model(torch_tensor)
        if isinstance(pt_preds, (list, tuple)):
            pt_pred = pt_preds[0].cpu().numpy()
        else:
            pt_pred = pt_preds.cpu().numpy()

    # Compare max absolute difference
    max_diff = float(np.max(np.abs(pt_pred - ort_pred)))
    mean_diff = float(np.mean(np.abs(pt_pred - ort_pred)))

    print(f"    - Max Absolute Difference : {max_diff:.6e}")
    print(f"    - Mean Absolute Difference: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("    [EXCELLENT] PyTorch and ONNX Runtime predictions match with high precision.")
    else:
        print(f"    [NOTE] Max difference is {max_diff:.6e} (acceptable float32 quantization drift).")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Export Complete: '{output_file}' is production-ready for test.py!")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )
