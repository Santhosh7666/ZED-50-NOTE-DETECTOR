"""
Postprocessing and Advanced False Positive Filtering Module.
Decodes YOLO ONNX tensor outputs, applies IoU Non-Maximum Suppression (NMS),
filters geometric inconsistencies (aspect ratio, minimum area), and performs
temporal multi-frame confirmation to prevent false triggers and display flicker.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


class Detection:
    """Represents a single confirmed bounding box detection."""

    def __init__(self, bbox: List[float], confidence: float, class_id: int = 0):
        self.bbox = [float(v) for v in bbox]  # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.class_id = int(class_id)

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def aspect_ratio(self) -> float:
        """Width / Height aspect ratio."""
        return self.width / (self.height + 1e-6)

    @property
    def area(self) -> float:
        return self.width * self.height


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection over Union (IoU) of two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    interArea = inter_w * inter_h

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


def decode_yolo_onnx(
    output: np.ndarray,
    conf_threshold: float = 0.50,
    iou_threshold: float = 0.45,
    min_box_size: int = 40,
    min_aspect_ratio: float = 0.30,
    max_aspect_ratio: float = 3.20,
) -> List[Detection]:
    """
    Decodes YOLOv8 / YOLO11 raw ONNX output tensor into filtered Detections.

    Expected tensor shape:
        (1, 5, 8400) or (1, 8400, 5) where 5 values are:
        [center_x, center_y, width, height, class_0_confidence]

    Args:
        output: Raw numpy output from ONNX runtime
        conf_threshold: Minimum confidence score to retain candidate
        iou_threshold: IoU threshold for Non-Maximum Suppression
        min_box_size: Minimum pixel dimension for width and height
        min_aspect_ratio: Minimum acceptable aspect ratio (width/height)
        max_aspect_ratio: Maximum acceptable aspect ratio (width/height)

    Returns:
        List of Detection objects
    """
    if isinstance(output, list):
        output = output[0]

    if output.ndim == 3:
        if output.shape[1] < output.shape[2]:
            # Shape is (1, 5, 8400) -> transpose to (8400, 5)
            preds = output[0].transpose(1, 0)
        else:
            # Shape is (1, 8400, 5)
            preds = output[0]
    elif output.ndim == 2:
        preds = output
    else:
        raise ValueError(f"Unexpected YOLO output shape: {output.shape}")

    # Extract boxes and confidence scores
    # Columns: [cx, cy, w, h, score]
    cx = preds[:, 0]
    cy = preds[:, 1]
    w = preds[:, 2]
    h = preds[:, 3]
    scores = preds[:, 4]

    # Confidence threshold mask
    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    cx = cx[mask]
    cy = cy[mask]
    w = w[mask]
    h = h[mask]
    scores = scores[mask]

    # Convert cx, cy, w, h -> x1, y1, x2, y2
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    boxes_xywh = []
    boxes_xyxy = []
    confs = []

    for i in range(len(scores)):
        curr_w = float(w[i])
        curr_h = float(h[i])

        # Dimension and aspect ratio validity check (Banknote aspect ratio ~2.045:1 or ~1:2.045 vertically)
        if curr_w < min_box_size or curr_h < min_box_size:
            continue
        ar = curr_w / (curr_h + 1e-6)
        if ar < min_aspect_ratio or ar > max_aspect_ratio:
            continue

        boxes_xywh.append([int(x1[i]), int(y1[i]), int(curr_w), int(curr_h)])
        boxes_xyxy.append([float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])])
        confs.append(float(scores[i]))

    if not boxes_xywh:
        return []

    # Non-Maximum Suppression (OpenCV DNN accelerated, with pure Python fallback)
    try:
        import cv2
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh,
            scores=confs,
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold,
        )
        if len(indices) > 0:
            indices = np.array(indices).flatten().tolist()
        else:
            indices = []
    except ImportError:
        # Vectorized pure Python IoU NMS
        order = np.argsort(-np.array(confs))
        indices = []
        while len(order) > 0:
            i = order[0]
            indices.append(i)
            if len(order) == 1:
                break
            ious = np.array([calculate_iou(boxes_xyxy[i], boxes_xyxy[j]) for j in order[1:]])
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]

    detections = []
    for idx in indices:
        detections.append(Detection(bbox=boxes_xyxy[idx], confidence=confs[idx], class_id=0))

    return detections


class TemporalStabilityTracker:
    """
    Prevents false triggers and flickering by requiring multi-frame confirmation
    and smoothing bounding box coordinates across consecutive video frames.
    """

    def __init__(
        self,
        confirmation_frames: int = 3,
        lost_frames_threshold: int = 4,
        iou_match_threshold: float = 0.40,
        smoothing_alpha: float = 0.65,
    ):
        """
        Args:
            confirmation_frames: Number of consecutive detections required before displaying note.
            lost_frames_threshold: Frames without detection before confirming note has left camera.
            iou_match_threshold: IoU overlap to match tracking across frames.
            smoothing_alpha: EMA weight for smoothing bounding box (1.0 = no smoothing, 0.5 = heavy smoothing).
        """
        self.confirmation_frames = confirmation_frames
        self.lost_frames_threshold = lost_frames_threshold
        self.iou_match_threshold = iou_match_threshold
        self.smoothing_alpha = smoothing_alpha

        # Active track state:
        # dict of: {"bbox": [x1, y1, x2, y2], "confidence": float, "hits": int, "misses": int}
        self.tracks: List[Dict[str, any]] = []

    def reset(self):
        """Resets all tracking history."""
        self.tracks = []

    def update(self, current_detections: List[Detection]) -> List[Detection]:
        """
        Updates temporal state with current frame detections.
        Returns only CONFIRMED, smoothed detections.
        """
        matched_det_indices = set()
        updated_tracks = []

        # Match current detections with existing tracks
        for track in self.tracks:
            best_iou = 0.0
            best_idx = -1

            for idx, det in enumerate(current_detections):
                if idx in matched_det_indices:
                    continue
                iou = calculate_iou(track["bbox"], det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= self.iou_match_threshold and best_idx != -1:
                # Match found: update EMA bounding box and increase hits
                matched_det_indices.add(best_idx)
                det = current_detections[best_idx]

                alpha = self.smoothing_alpha
                smoothed_bbox = [
                    alpha * det.bbox[i] + (1.0 - alpha) * track["bbox"][i]
                    for i in range(4)
                ]
                updated_tracks.append({
                    "bbox": smoothed_bbox,
                    "confidence": alpha * det.confidence + (1.0 - alpha) * track["confidence"],
                    "hits": min(track["hits"] + 1, self.confirmation_frames + 10),
                    "misses": 0,
                })
            else:
                # Missed in this frame
                if track["misses"] + 1 <= self.lost_frames_threshold:
                    updated_tracks.append({
                        "bbox": track["bbox"],
                        "confidence": track["confidence"] * 0.9,
                        "hits": track["hits"],
                        "misses": track["misses"] + 1,
                    })

        # Add unmatched new detections as tentative tracks
        for idx, det in enumerate(current_detections):
            if idx not in matched_det_indices:
                updated_tracks.append({
                    "bbox": list(det.bbox),
                    "confidence": det.confidence,
                    "hits": 1,
                    "misses": 0,
                })

        self.tracks = updated_tracks

        # Return only confirmed tracks that meet hit threshold
        confirmed_detections = []
        for track in self.tracks:
            if track["hits"] >= self.confirmation_frames and track["misses"] == 0:
                confirmed_detections.append(
                    Detection(
                        bbox=track["bbox"],
                        confidence=track["confidence"],
                        class_id=0,
                    )
                )

        return confirmed_detections

