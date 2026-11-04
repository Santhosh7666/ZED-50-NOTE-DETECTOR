"""
=============================================================================
ZED DIGITAL - REAL-TIME ₹50 INDIAN BANKNOTE DETECTION SYSTEM
Live Webcam Inference Pipeline using ONNX Runtime / OpenCV DNN
=============================================================================
Required execution command:
    python test.py --model model.onnx

Controls:
    [Q] Quit application
    [S] Save screenshot of current frame
    [R] Reset temporal detection history
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(
        description="ZED Digital: Real-Time ₹50 Banknote Detection System"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.onnx",
        help="Path to trained ONNX model (default: models/model.onnx)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.55,
        help="Confidence threshold for ₹50 detection (default: 0.55)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression (default: 0.45)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input resolution for model inference (default: 640)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional path to image file or video file instead of live webcam",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration if CUDAExecutionProvider is available",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="onnx",
        choices=["onnx", "cv2"],
        help="Inference engine backend: 'onnx' (default) or 'cv2'",
    )
    return parser.parse_args()


def check_runtime_dependencies():
    """Validates that necessary CV and runtime dependencies are available."""
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print("\n" + "=" * 70)
        print("[DEPENDENCY ALERT] Missing required packages for webcam execution:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install the required packages using:")
        print(f"  pip install {' '.join(missing)}")
        print("Or install full project dependencies via:")
        print("  pip install -r requirements.txt")
        print("=" * 70 + "\n")
        return False
    return True


def initialize_detector(model_path: str, engine: str, prefer_gpu: bool):
    """Initializes the ONNX detector with clean error reporting."""
    # Check if model exists directly or in models/
    candidate_paths = [
        model_path,
        os.path.join("models", model_path),
        os.path.join("models", os.path.basename(model_path)),
    ]

    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if not resolved_path:
        print("\n" + "=" * 70)
        print(f"[ERROR] ONNX Model file not found at: '{model_path}'")
        print("Checked candidate paths:")
        for cp in candidate_paths:
            print(f"  - {os.path.abspath(cp)}")
        print("\nPlease ensure you have trained and exported the model:")
        print("  1. Train:  python train.py")
        print("  2. Export: python scripts/export_onnx.py --weights models/best.pt --output models/model.onnx")
        print("=" * 70 + "\n")
        sys.exit(1)

    print(f"[*] Initializing detection model: {resolved_path}")
    try:
        if engine == "onnx":
            detector = ONNXDetector(model_path=resolved_path, prefer_gpu=prefer_gpu)
        else:
            detector = OpenCVNetDetector(model_path=resolved_path)
        return detector, resolved_path
    except Exception as e:
        print(f"[FATAL] Failed to initialize model detector: {e}")
        sys.exit(1)


def main():
    args = parse_args()

    if not check_runtime_dependencies():
        sys.exit(1)

    # Lazy import modules requiring OpenCV / ONNX Runtime
    from src.preprocessing import preprocess_frame, scale_coords
    from src.inference import ONNXDetector, OpenCVNetDetector
    from src.postprocessing import decode_yolo_onnx, TemporalStabilityTracker, Detection
    from src.visualization import Visualizer
    import cv2

    print("=" * 70)
    print("ZED DIGITAL: REAL-TIME ₹50 INDIAN BANKNOTE DETECTION")
    print("=" * 70)
    print(f"  Model Path         : {args.model}")
    print(f"  Confidence Cutoff  : {args.conf:.2f}")
    print(f"  IoU NMS Threshold  : {args.iou:.2f}")
    print(f"  Inference Size     : {args.imgsz}x{args.imgsz}")
    print(f"  Camera Index       : {args.camera}")
    print(f"  Inference Engine   : {args.engine.upper()}")
    print("=" * 70)

    # 1. Initialize detector engine
    detector, resolved_model_path = initialize_detector(
        model_path=args.model,
        engine=args.engine,
        prefer_gpu=args.gpu,
    )

    # 2. Initialize visualizer and temporal tracker
    visualizer = Visualizer(target_label="₹50")
    tracker = TemporalStabilityTracker(
        confirmation_frames=3,
        lost_frames_threshold=4,
        iou_match_threshold=0.35,
        smoothing_alpha=0.65,
    )

    # 3. Open Video Capture Source
    import cv2

    if args.source:
        print(f"[*] Opening media source: {args.source}")
        cap = cv2.VideoCapture(args.source)
    else:
        print(f"[*] Initializing webcam stream (Device ID: {args.camera})...")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        # Attempt to set HD 720p resolution for clear watermark/chariot visibility
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("\n" + "=" * 70)
        print(f"[ERROR] Could not open video source / webcam (index: {args.camera}).")
        print("Troubleshooting steps:")
        print("  1. Verify your webcam is plugged in and recognized by Windows Device Manager.")
        print("  2. Check if another application (Teams, Zoom, Camera App) is currently using the camera.")
        print("  3. Try another camera index by specifying: python test.py --model model.onnx --camera 1")
        print("  4. You can also test on a video file or photo: python test.py --source sample.jpg")
        print("=" * 70 + "\n")
        sys.exit(1)

    window_title = "ZED Digital - Real-Time ₹50 Note Detector"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1024, 600)

    # FPS and profiling trackers
    frame_times = []
    fps = 0.0
    inference_time_ms = 0.0
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    print("\n[+] Webcam stream active! Press 'Q' to quit, 'S' to save screenshot, 'R' to reset.")

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                if args.source:
                    # Looping video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[WARNING] Frame capture returned empty. Retrying...")
                    time.sleep(0.01)
                    continue

            orig_h, orig_w = frame.shape[:2]

            # 4. Preprocess frame
            blob, ratio, pad = preprocess_frame(frame, input_size=(args.imgsz, args.imgsz))

            # 5. Forward inference
            raw_output, inference_time_ms = detector.infer(blob)

            # 6. Postprocess and filter candidates
            raw_detections: list[Detection] = decode_yolo_onnx(
                raw_output,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                min_box_size=40,
                min_aspect_ratio=0.35,
                max_aspect_ratio=3.10,
            )

            # 7. Scale coordinates back to original camera resolution
            scaled_detections: list[Detection] = []
            if raw_detections:
                unscaled_boxes = np.array([d.bbox for d in raw_detections])
                rescaled_boxes = scale_coords(unscaled_boxes, ratio, pad, (orig_h, orig_w))
                for i, d in enumerate(raw_detections):
                    scaled_detections.append(
                        Detection(bbox=rescaled_boxes[i], confidence=d.confidence, class_id=0)
                    )

            # 8. Multi-frame temporal confirmation to eliminate flickering & 1-frame spikes
            confirmed_detections = tracker.update(scaled_detections)

            # 9. Render annotations
            for det in confirmed_detections:
                frame = visualizer.draw_detection(frame, det)

            # 10. Render HUD header & metrics
            frame = visualizer.draw_hud(
                frame,
                fps=fps,
                inference_time_ms=inference_time_ms,
                detection_count=len(confirmed_detections),
                model_name=os.path.basename(resolved_model_path),
                provider=getattr(detector, "execution_provider", "CPU"),
                camera_status="ONLINE",
            )

            # Display window
            cv2.imshow(window_title, frame)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'Q' or ESC
                print("[*] Quit key pressed. Exiting...")
                break
            elif key == ord("s"):  # Screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shot_path = screenshots_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(shot_path), frame)
                print(f"[+] Saved screenshot to: {shot_path.resolve()}")
            elif key == ord("r"):  # Reset tracker
                tracker.reset()
                print("[*] Temporal tracking history reset.")

            # Compute smoothed FPS
            t_elapsed = time.perf_counter() - t_start
            frame_times.append(t_elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_frame_time = np.mean(frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Webcam stream safely released and display windows closed.")


if __name__ == "__main__":
    main()
