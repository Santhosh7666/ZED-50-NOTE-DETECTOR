"""
Metrics Evaluation Module for YOLO Currency Detection.
Computes Precision, Recall, F1-Score, mAP@50, mAP@50-95, False Positive Rate (FPR),
and inference latency benchmarks across test subsets.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two arrays of boxes [x1, y1, x2, y2].
    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
    Returns:
        iou_matrix: (N, M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    union = np.maximum(union, 1e-6)

    return inter / union


class DetectionMetrics:
    """
    Evaluates object detection performance specifically accounting for
    strict negative sample rejection (₹10, ₹20, ₹100, ₹200, ₹500, backgrounds).
    """

    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        if iou_thresholds is None:
            # Standard COCO evaluation thresholds [0.50 : 0.05 : 0.95]
            self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        else:
            self.iou_thresholds = np.array(iou_thresholds)

        self.reset()

    def reset(self):
        self.stats = []  # List of dicts for each sample
        self.latencies_ms = []
        self.total_images = 0
        self.positive_images = 0
        self.negative_images = 0
        self.false_positive_images = 0
        self.true_negative_images = 0

    def add_latency(self, latency_ms: float):
        self.latencies_ms.append(latency_ms)

    def update(
        self,
        pred_boxes: np.ndarray,      # (N, 4) [x1, y1, x2, y2]
        pred_scores: np.ndarray,     # (N,)
        gt_boxes: np.ndarray,        # (M, 4) [x1, y1, x2, y2]
    ):
        """Updates evaluation state with single test image predictions and ground truth."""
        self.total_images += 1
        is_negative_image = (len(gt_boxes) == 0)

        if is_negative_image:
            self.negative_images += 1
            if len(pred_boxes) > 0:
                self.false_positive_images += 1
            else:
                self.true_negative_images += 1
        else:
            self.positive_images += 1

        # Match predictions to GT at various IoU thresholds
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        if n_pred == 0:
            return

        if n_gt == 0:
            # All predictions are false positives
            for i in range(n_pred):
                self.stats.append({
                    "score": float(pred_scores[i]),
                    "correct": np.zeros(len(self.iou_thresholds), dtype=bool),
                })
            return

        # Sort predictions by confidence score descending
        sort_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]

        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)  # [M, N]

        correct = np.zeros((n_pred, len(self.iou_thresholds)), dtype=bool)

        for t_idx, iou_thresh in enumerate(self.iou_thresholds):
            detected_gt = set()
            for p_idx in range(n_pred):
                # Find best matching GT
                ious = iou_mat[:, p_idx]
                best_gt = np.argmax(ious)
                if ious[best_gt] >= iou_thresh and best_gt not in detected_gt:
                    correct[p_idx, t_idx] = True
                    detected_gt.add(best_gt)

        for i in range(n_pred):
            self.stats.append({
                "score": float(pred_scores[i]),
                "correct": correct[i],
            })

    def compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """Computes Average Precision (AP) using standard 101-point COCO interpolation."""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # Ensure precision is non-increasing
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 101-point interpolation
        recall_levels = np.linspace(0, 1, 101)
        interpolated_prec = np.interp(recall_levels, mrec, mpre)
        return float(np.mean(interpolated_prec))

    def evaluate(self, total_gt_count: int) -> Dict[str, any]:
        """Calculates final comprehensive metrics."""
        if not self.stats or total_gt_count == 0:
            prec = 1.0 if not self.stats and self.false_positive_images == 0 else 0.0
            rec = 0.0
            f1 = 0.0
            map50 = 0.0
            map50_95 = 0.0
        else:
            # Sort stats by score descending
            scores = np.array([s["score"] for s in self.stats])
            correct_mat = np.array([s["correct"] for s in self.stats])  # [N_pred, num_thresholds]

            sort_idx = np.argsort(-scores)
            correct_mat = correct_mat[sort_idx]

            # At IoU=0.50 (index 0)
            tp_cum = np.cumsum(correct_mat[:, 0])
            fp_cum = np.cumsum(~correct_mat[:, 0])

            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
            recalls = tp_cum / max(1, total_gt_count)

            prec = float(precisions[-1]) if len(precisions) > 0 else 0.0
            rec = float(recalls[-1]) if len(recalls) > 0 else 0.0
            f1 = float(2 * (prec * rec) / (prec + rec + 1e-6))

            map50 = self.compute_ap(recalls, precisions)

            # Compute across all thresholds for mAP@50-95
            aps = []
            for t_idx in range(len(self.iou_thresholds)):
                tp_c = np.cumsum(correct_mat[:, t_idx])
                fp_c = np.cumsum(~correct_mat[:, t_idx])
                p_c = tp_c / np.maximum(tp_c + fp_c, 1e-6)
                r_c = tp_c / max(1, total_gt_count)
                aps.append(self.compute_ap(r_c, p_c))

            map50_95 = float(np.mean(aps))

        # False positive rate on negative test set
        fpr = (
            float(self.false_positive_images / self.negative_images)
            if self.negative_images > 0
            else 0.0
        )

        # Latency statistics
        if self.latencies_ms:
            lat = np.array(self.latencies_ms)
            mean_lat = float(np.mean(lat))
            median_lat = float(np.median(lat))
            p95_lat = float(np.percentile(lat, 95))
            fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
        else:
            mean_lat = median_lat = p95_lat = fps = 0.0

        return {
            "total_images": self.total_images,
            "positive_images": self.positive_images,
            "negative_images": self.negative_images,
            "total_gt_notes": total_gt_count,
            "false_positive_images": self.false_positive_images,
            "true_negative_images": self.true_negative_images,
            "false_positive_rate": fpr,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "map_50": map50,
            "map_50_95": map50_95,
            "latency_mean_ms": mean_lat,
            "latency_median_ms": median_lat,
            "latency_p95_ms": p95_lat,
            "fps": fps,
        }

