"""
YOLO Training Pipeline for ZED Digital ₹50 Banknote Detector.
Trains a lightweight YOLO Nano object detection model to detect ONLY new ₹50 Indian banknotes
while actively penalizing false positives on ₹10, ₹20, ₹100, ₹200, ₹500, old notes, and clutter.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO Nano for ₹50 Banknote Detection")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.yaml",
        help="Path to dataset YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base pretrained YOLO architecture (yolov8n.pt or yolo11n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (recommended: 50-80)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (decrease to 8 or 4 for low VRAM or CPU)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (640x640 standard for YOLO)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without validation mAP improvement)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', '0', '0,1', or 'cpu'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker threads",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training runs",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Directory to save training run logs and checkpoints",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="zed_50_detector",
        help="Experiment run name",
    )
    return parser.parse_args()


def select_compute_device(requested_device: str) -> str:
    """Detects available hardware and selects CUDA GPU with graceful CPU fallback."""
    if requested_device == "cpu":
        print("[Device Selection] CPU explicitly requested.")
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device Selection] CUDA GPU detected: {gpu_name} ({gpu_count} device(s) available).")
            return "0" if requested_device == "auto" else requested_device
        else:
            print("[Device Selection] No CUDA GPU detected. Falling back to CPU.")
            return "cpu"
    except ImportError:
        print("[Device Selection] PyTorch not installed yet. Defaulting to CPU.")
        return "cpu"


def verify_dataset_readiness(data_yaml_path: str) -> bool:
    """Verifies that dataset.yaml exists and contains non-empty image directories."""
    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        print(f"[ERROR] Dataset configuration file '{data_yaml_path}' not found!")
        return False

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_path = Path(config.get("train", "dataset/images/train"))
    val_path = Path(config.get("val", "dataset/images/val"))

    # Resolve relative paths
    if not train_path.is_absolute():
        train_path = (yaml_path.parent / train_path).resolve()
    if not val_path.is_absolute():
        val_path = (yaml_path.parent / val_path).resolve()

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_imgs = [p for p in train_path.glob("*") if p.suffix.lower() in valid_exts] if train_path.exists() else []
    val_imgs = [p for p in val_path.glob("*") if p.suffix.lower() in valid_exts] if val_path.exists() else []

    print(f"[*] Training images available  : {len(train_imgs)}")
    print(f"[*] Validation images available: {len(val_imgs)}")

    if len(train_imgs) == 0:
        print("\n" + "=" * 70)
        print("[NOTICE] No images found in training split.")
        print("To train the model:")
        print("  1. Place your ₹50 front/back photos and negative photos into raw_data/")
        print("  2. Run: python scripts/prepare_dataset.py")
        print("  3. Run: python train.py")
        print("=" * 70 + "\n")
        return False

    return True


def train():
    args = parse_args()

    print("=" * 70)
    print("ZED DIGITAL: ₹50 INDIAN BANKNOTE DETECTOR TRAINING")
    print("=" * 70)

    # 1. Verify Dataset
    is_ready = verify_dataset_readiness(args.data)
    if not is_ready:
        print("[ABORT] Cannot proceed with training until dataset contains images.")
        return

    # 2. Configure Device
    device = select_compute_device(args.device)

    # 3. Load YOLO Model
    print(f"\n[*] Initializing YOLO architecture: {args.model}")
    from ultralytics import YOLO

    model = YOLO(args.model)

    # 4. Training Hyperparameters tuned for Single-Class Currency Detection & Hard Negative Mining
    # Flips: horizontal flip (fliplr) allowed, NO vertical flip (flipud=0.0) as currency is read right side up
    # Mosaic: 0.5 (combines background textures), MixUp: 0.0 (prevents transparent overlapping notes)
    training_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "device": device,
        "workers": args.workers,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "save": True,
        "save_period": -1,
        "val": True,
        "plots": True,
        "verbose": True,
        # Specialized augmentations for currency
        "degrees": 15.0,        # Rotation +/- 15 deg
        "translate": 0.10,      # Translation +/- 10%
        "scale": 0.20,          # Scale +/- 20%
        "fliplr": 0.5,          # Horizontal flip (covers left-to-right angles)
        "flipud": 0.0,          # Currency is rarely inverted upside down in hand
        "hsv_h": 0.015,         # Mild hue adjustment (preserve distinct light blue-green tone)
        "hsv_s": 0.40,          # Saturation variation
        "hsv_v": 0.40,          # Exposure/brightness variation
        "mosaic": 0.50,         # Expose note against complex random background cutouts
        "mixup": 0.0,           # Do not mix banknotes together
    }

    print("\n[*] Starting YOLO training pipeline...")
    for k, v in training_args.items():
        print(f"    - {k:<15}: {v}")

    results = model.train(**training_args)

    print("\n" + "=" * 70)
    print("[SUCCESS] Training complete!")
    print(f"Checkpoints and metrics saved under: {Path(args.project) / args.name}")

    # Check for best.pt and link to models/best.pt
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        target_dir = Path("models")
        target_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(best_pt, target_dir / "best.pt")
        print(f"[+] Deployed best checkpoint to: {target_dir / 'best.pt'}")
        print("\nNext step: Export model to ONNX via:")
        print("    python scripts/export_onnx.py --weights models/best.pt --output models/model.onnx")
    print("=" * 70)


if __name__ == "__main__":
    train()
