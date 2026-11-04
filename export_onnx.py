"""
Root entry point for ONNX export script.
Proxies to scripts/export_onnx.py.
"""

from scripts.export_onnx import parse_args, export_to_onnx

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
