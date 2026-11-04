"""
Compilation and Master Execution Pipeline for ZED Digital ₹50 Detector.
Executes end-to-end:
  1. Python bytecode compilation & syntax verification across all modules.
  2. Dataset validation and class constraint checks.
  3. Real-time banknote detection pipeline execution.
"""

import sys
import py_compile
import subprocess
from pathlib import Path

# Ensure UTF-8 and unbuffered console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


def compile_project() -> bool:
    print("=" * 70, flush=True)
    print("STAGE 1: COMPILING PROJECT SOURCE FILES", flush=True)
    print("=" * 70, flush=True)

    source_files = [
        "train.py",
        "test.py",
        "evaluate.py",
        "export_onnx.py",
        "scripts/prepare_dataset.py",
        "scripts/export_onnx.py",
        "scripts/create_initial_model.py",
        "src/__init__.py",
        "src/dataset_validator.py",
        "src/preprocessing.py",
        "src/augmentation.py",
        "src/inference.py",
        "src/postprocessing.py",
        "src/visualization.py",
        "src/metrics.py",
    ]

    all_passed = True
    for sf in source_files:
        path = Path(sf)
        if not path.exists():
            print(f"  [MISSING] {sf}", flush=True)
            all_passed = False
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"  [COMPILED OK] {sf:<32} -> __pycache__/", flush=True)
        except py_compile.PyCompileError as e:
            print(f"  [SYNTAX ERROR] {sf}: {e}", flush=True)
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All 15 modules compiled with zero syntax errors.", flush=True)
    else:
        print("\n[FAILURE] Compilation issues detected.", flush=True)
    return all_passed


def validate_dataset():
    print("\n" + "=" * 70, flush=True)
    print("STAGE 2: DATASET STRUCTURE & INTEGRITY VALIDATION", flush=True)
    print("=" * 70, flush=True)
    from src.dataset_validator import DatasetValidator
    validator = DatasetValidator(dataset_root="dataset")
    validator.run_full_validation()


def run_pipeline(max_frames: int = 15):
    print("\n" + "=" * 70, flush=True)
    print("STAGE 3: EXECUTING REAL-TIME ₹50 BANKNOTE DETECTION SYSTEM", flush=True)
    print("=" * 70, flush=True)
    cmd = [sys.executable, "test.py", "--model", "model.onnx"]
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    print(f"[*] Running command: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd)


if __name__ == "__main__":
    max_f = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 15
    if compile_project():
        validate_dataset()
        run_pipeline(max_frames=max_f)
