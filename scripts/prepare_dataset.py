"""
Dataset Preparation and Ingestion Pipeline for ZED Digital ₹50 Detector.
Processes raw mobile phone photographs (positive ₹50 and negative denominations/objects),
ensures clean annotation pairing, applies leak-free stratified splitting, and validates integrity.
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import our validator module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset_validator import DatasetValidator


def parse_args():
    parser = argparse.ArgumentParser(description="ZED Digital ₹50 Dataset Preparation Pipeline")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="raw_data",
        help="Path to folder containing raw mobile phone captures (subfolders: positive, negative)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Target YOLO dataset root directory",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of data allocated to training split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of data allocated to validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified splitting",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation checks on existing dataset without modifying files",
    )
    return parser.parse_args()


def create_initial_folder_structure(raw_dir: Path):
    """Creates convenient raw collection template directories for mobile phone uploads."""
    subfolders = [
        raw_dir / "positive" / "front",
        raw_dir / "positive" / "back",
        raw_dir / "negative" / "rs_10",
        raw_dir / "negative" / "rs_20",
        raw_dir / "negative" / "rs_100",
        raw_dir / "negative" / "rs_200",
        raw_dir / "negative" / "rs_500",
        raw_dir / "negative" / "old_rs_50",
        raw_dir / "negative" / "backgrounds_and_objects",
    ]
    for folder in subfolders:
        folder.mkdir(parents=True, exist_ok=True)


def verify_image(img_path: Path) -> bool:
    """Checks whether an image file is readable and uncorrupted."""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_dataset(
    raw_dir: str = "raw_data",
    output_dir: str = "dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    # Initialize raw template directory if missing
    create_initial_folder_structure(raw_path)

    # Prepare destination directories
    for split in ["train", "val", "test"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect valid images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pos_images = []
    neg_images = []

    pos_dir = raw_path / "positive"
    if pos_dir.exists():
        for f in pos_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in valid_exts and verify_image(f):
                pos_images.append(f)

    neg_dir = raw_path / "negative"
    if neg_dir.exists():
        for f in neg_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in valid_exts and verify_image(f):
                neg_images.append(f)

    print(f"\n[Dataset Preparation Summary]")
    print(f"  Positive ₹50 mobile captures found : {len(pos_images)}")
    print(f"  Negative (other currency/objects)  : {len(neg_images)}")

    if len(pos_images) == 0 and len(neg_images) == 0:
        print("\n[INFO] No raw images found in raw_data/. Created sample folders under:")
        print(f"       -> {raw_path.resolve()}/positive (front & back ₹50 notes)")
        print(f"       -> {raw_path.resolve()}/negative (₹10, ₹20, ₹100, ₹200, ₹500, old ₹50, objects)")
        print("Please place your captured photos in these folders and re-run this script.")
        return

    # Shuffle datasets independently for stratified distribution
    random.shuffle(pos_images)
    random.shuffle(neg_images)

    def split_list(items: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]

    pos_train, pos_val, pos_test = split_list(pos_images)
    neg_train, neg_val, neg_test = split_list(neg_images)

    splits = {
        "train": (pos_train, neg_train),
        "val": (pos_val, neg_val),
        "test": (pos_test, neg_test),
    }

    print("\n[*] Splitting and copying files into destination splits...")
    for split_name, (pos_list, neg_list) in splits.items():
        dst_img_dir = out_path / "images" / split_name
        dst_lbl_dir = out_path / "labels" / split_name

        # Copy positive samples and look for matching annotation or create prompt
        for p_img in pos_list:
            dst_img = dst_img_dir / p_img.name
            shutil.copy2(p_img, dst_img)

            # Check if matching label file exists alongside raw image
            src_lbl = p_img.with_suffix(".txt")
            dst_lbl = dst_lbl_dir / f"{p_img.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # If no manual annotation yet, create template note
                if not dst_lbl.exists():
                    # Placeholder or alert
                    pass

        # Copy negative samples and write guaranteed EMPTY label file
        for n_img in neg_list:
            dst_img = dst_img_dir / n_img.name
            shutil.copy2(n_img, dst_img)
            dst_lbl = dst_lbl_dir / f"{n_img.stem}.txt"
            # Negatives MUST have empty label file
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("")

        print(
            f"  Split '{split_name}': {len(pos_list)} positive + {len(neg_list)} negative "
            f"= {len(pos_list) + len(neg_list)} total"
        )

    print("\n[*] Running post-split validation...")
    validator = DatasetValidator(dataset_root=output_dir)
    validator.run_full_validation()


if __name__ == "__main__":
    args = parse_args()
    if args.validate_only:
        validator = DatasetValidator(dataset_root=args.output_dir)
        validator.run_full_validation()
    else:
        prepare_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
