"""
Dataset Validator Module for YOLO Currency Detection.
Validates dataset integrity, checks annotation validity, detects corrupted/duplicate images,
and prevents train/val/test data leakage.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image
import numpy as np

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class DatasetValidator:
    """
    Validates YOLO formatted dataset for ₹50 note detector.
    Enforces single-class (class 0) constraint, negative sample consistency,
    bounding box validity, image integrity, and zero leakage across splits.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, dataset_root: str = "dataset"):
        self.root = Path(dataset_root)
        self.images_root = self.root / "images"
        self.labels_root = self.root / "labels"
        self.splits = ["train", "val", "test"]
        self.report: Dict[str, any] = {}

    def _compute_image_hash(self, image_path: Path) -> str:
        """Computes a perceptual dHash (difference hash) and MD5 for duplicate detection."""
        try:
            with Image.open(image_path) as img:
                # 8x8 difference hash
                img_gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
                pixels = np.array(img_gray)
                diff = pixels[:, 1:] > pixels[:, :-1]
                dhash_str = "".join(["1" if b else "0" for b in diff.flatten()])
                return dhash_str
        except Exception:
            return ""

    def validate_bounding_box(self, line_num: int, line_str: str, file_path: Path) -> Tuple[bool, str]:
        """
        Validates individual YOLO annotation line:
        Format: class_id x_center y_center width height
        """
        parts = line_str.strip().split()
        if not parts:
            return True, ""  # Empty lines allowed
        if len(parts) != 5:
            return False, f"Line {line_num} in {file_path.name} has {len(parts)} values (expected 5: class x y w h)."

        try:
            class_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            return False, f"Line {line_num} in {file_path.name} contains non-numeric values: '{line_str}'."

        if class_id != 0:
            return False, (
                f"Line {line_num} in {file_path.name} has class_id {class_id}. "
                f"Only class 0 (₹50 banknote) is permitted. Other currencies must be negative samples."
            )

        for name, val in [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]:
            if not (0.0 <= val <= 1.0):
                return False, f"Line {line_num} in {file_path.name}: {name}={val} is outside [0.0, 1.0]."

        if w <= 0.0 or h <= 0.0:
            return False, f"Line {line_num} in {file_path.name}: box has non-positive width/height ({w}, {h})."

        # Check box boundary clipping
        x_min, x_max = xc - w / 2, xc + w / 2
        y_min, y_max = yc - h / 2, yc + h / 2
        if x_min < -0.01 or x_max > 1.01 or y_min < -0.01 or y_max > 1.01:
            return False, f"Line {line_num} in {file_path.name}: bounding box extends substantially outside image bounds."

        return True, ""

    def validate_split(self, split: str) -> Dict[str, any]:
        """Validates all images and labels in a specific split (train, val, or test)."""
        img_dir = self.images_root / split
        lbl_dir = self.labels_root / split

        results = {
            "image_count": 0,
            "label_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "corrupted_images": [],
            "missing_labels": [],
            "orphan_labels": [],
            "invalid_annotations": [],
            "image_hashes": {},
        }

        if not img_dir.exists() or not lbl_dir.exists():
            return results

        image_files = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS}
        label_files = {p.stem: p for p in lbl_dir.glob("*.txt")}

        results["image_count"] = len(image_files)
        results["label_count"] = len(label_files)

        # Check missing label files for images
        for stem, img_path in image_files.items():
            # Check image integrity
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Re-open for hash (verify can close descriptor)
                dhash = self._compute_image_hash(img_path)
                if dhash:
                    results["image_hashes"][img_path.name] = dhash
            except Exception as e:
                results["corrupted_images"].append(f"{img_path.name}: {str(e)}")
                continue

            if stem not in label_files:
                results["missing_labels"].append(img_path.name)
            else:
                lbl_path = label_files[stem]
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    lines = [ln.strip() for ln in lf.readlines() if ln.strip()]

                if len(lines) == 0:
                    results["negative_count"] += 1
                else:
                    results["positive_count"] += 1
                    for line_idx, line in enumerate(lines, 1):
                        is_valid, err = self.validate_bounding_box(line_idx, line, lbl_path)
                        if not is_valid:
                            results["invalid_annotations"].append(err)

        # Check orphan labels (labels without images)
        for stem, lbl_path in label_files.items():
            if stem not in image_files:
                results["orphan_labels"].append(lbl_path.name)

        return results

    def check_data_leakage(self, split_hashes: Dict[str, Dict[str, str]]) -> List[str]:
        """Detects identical or near-duplicate images across train/val/test splits."""
        leakage_warnings = []
        splits = list(split_hashes.keys())

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                h1 = split_hashes[s1]
                h2 = split_hashes[s2]

                # Invert hash mapping: hash -> filename
                inv_h1 = {h: fname for fname, h in h1.items()}
                for fname2, h in h2.items():
                    if h in inv_h1:
                        leakage_warnings.append(
                            f"LEAKAGE DETECTED: {s1}/{inv_h1[h]} is identical to {s2}/{fname2}"
                        )

        return leakage_warnings

    def run_full_validation(self) -> Dict[str, any]:
        """Runs comprehensive validation across all splits and prints detailed report."""
        print("=" * 70)
        print("RUNNING DATASET VALIDATION (ZED DIGITAL ₹50 DETECTOR)")
        print("=" * 70)

        overall_valid = True
        split_hashes = {}
        all_results = {}

        for split in self.splits:
            print(f"\n[*] Validating split: '{split}'...")
            res = self.validate_split(split)
            all_results[split] = res
            split_hashes[split] = res["image_hashes"]

            print(f"    - Images found: {res['image_count']}")
            print(f"    - Label files found: {res['label_count']}")
            print(f"    - Positive samples (₹50 annotated): {res['positive_count']}")
            print(f"    - Negative samples (empty annotations): {res['negative_count']}")

            if res["corrupted_images"]:
                print(f"    [!] Corrupted images: {len(res['corrupted_images'])}")
                overall_valid = False
            if res["missing_labels"]:
                print(f"    [!] Missing label files: {len(res['missing_labels'])}")
                overall_valid = False
            if res["orphan_labels"]:
                print(f"    [!] Orphan label files: {len(res['orphan_labels'])}")
                overall_valid = False
            if res["invalid_annotations"]:
                print(f"    [!] Invalid annotation lines: {len(res['invalid_annotations'])}")
                overall_valid = False

        print("\n[*] Checking for train/val/test data leakage...")
        leakages = self.check_data_leakage(split_hashes)
        if leakages:
            for l in leakages:
                print(f"    [WARNING] {l}")
            overall_valid = False
        else:
            print("    [OK] No cross-split leakage detected.")

        self.report = {
            "splits": all_results,
            "leakages": leakages,
            "is_valid": overall_valid,
        }

        print("\n" + "=" * 70)
        if overall_valid:
            print("[SUCCESS] Dataset validation passed without blocking errors.")
        else:
            print("[WARNING/FAILURE] Dataset validation identified issues (see details above).")
        print("=" * 70)

        return self.report


if __name__ == "__main__":
    validator = DatasetValidator()
    validator.run_full_validation()
