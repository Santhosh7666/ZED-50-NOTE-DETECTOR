"""
Visualization and UI Module for Live Webcam Banknote Detection.
Renders high-visibility bounding boxes, confidence badges, HUD status metrics,
temporal stabilization indicators, and user shortcut hints.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from src.postprocessing import Detection


class Visualizer:
    """
    Renders user interface and bounding annotations compliant with
    ZED Digital FEDS specifications.
    """

    # Modern color palette (BGR format for OpenCV)
    COLOR_CONFIRMED_GREEN = (46, 204, 113)    # Bright Emerald Green
    COLOR_BOX_SHADOW = (20, 100, 50)          # Shadow accent
    COLOR_PANEL_BG = (28, 28, 30)             # Dark slate background
    COLOR_PANEL_BORDER = (60, 60, 67)         # Border gray
    COLOR_TEXT_PRIMARY = (255, 255, 255)      # White
    COLOR_TEXT_MUTED = (160, 160, 160)        # Muted gray
    COLOR_ACCENT_BLUE = (255, 170, 0)         # Soft Cyan/Blue

    def __init__(self, target_label: str = "₹50"):
        self.target_label = target_label

    def draw_rounded_rectangle(
        self,
        image: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        radius: int = 8,
    ):
        """Draws a smooth rounded rectangle on the given OpenCV image."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

        # Draw lines
        cv2.line(image, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(image, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(image, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(image, (x2, y1 + r), (x2, y2 - r), color, thickness)

        # Draw arcs for corners
        cv2.ellipse(image, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        show_corners: bool = True,
    ) -> np.ndarray:
        """
        Draws green bounding box and high-visibility badge for confirmed ₹50 banknote.
        Format: ₹50 | Confidence: XX.X%
        """
        x1, y1, x2, y2 = map(int, detection.bbox)
        conf_pct = detection.confidence * 100.0
        label_text = f"{self.target_label} | Confidence: {conf_pct:.1f}%"

        # Draw outer box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_CONFIRMED_GREEN, 3, cv2.LINE_AA)

        # Draw corner highlight brackets for futuristic target look
        if show_corners:
            corner_len = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
            corner_thick = 5
            c_color = (255, 255, 255)

            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), c_color, corner_thick)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), c_color, corner_thick)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), c_color, corner_thick)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), c_color, corner_thick)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), c_color, corner_thick)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), c_color, corner_thick)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), c_color, corner_thick)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), c_color, corner_thick)

        # Draw filled label tag above box
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        badge_x1 = x1
        badge_y1 = max(0, y1 - text_h - 14)
        badge_x2 = badge_x1 + text_w + 18
        badge_y2 = y1

        cv2.rectangle(
            frame,
            (badge_x1, badge_y1),
            (badge_x2, badge_y2),
            self.COLOR_CONFIRMED_GREEN,
            -1,
        )

        # Text inside tag (Dark text on green badge for high contrast readability)
        cv2.putText(
            frame,
            label_text,
            (badge_x1 + 8, badge_y2 - 6),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        inference_time_ms: float,
        detection_count: int,
        model_name: str = "model.onnx",
        provider: str = "CPUExecutionProvider",
        camera_status: str = "ONLINE",
    ) -> np.ndarray:
        """
        Renders header HUD bar and footer status compliant with assignment specs.
        """
        h, w = frame.shape[:2]

        # Top translucent HUD banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), self.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(
            frame,
            "ZED DIGITAL - ₹50 BANKNOTE DETECTOR",
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.COLOR_TEXT_PRIMARY,
            2,
            cv2.LINE_AA,
        )

        # Camera & Model Info Subtitle
        info_sub = f"Model: {model_name} ({provider}) | Cam: {camera_status}"
        cv2.putText(
            frame,
            info_sub,
            (16, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Real-time Metrics (Right side of header)
        fps_text = f"FPS: {fps:.1f}"
        lat_text = f"Lat: {inference_time_ms:.1f}ms"
        cv2.putText(
            frame,
            fps_text,
            (w - 190, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.COLOR_CONFIRMED_GREEN if fps >= 15 else (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            lat_text,
            (w - 190, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Status badge when NO ₹50 note is detected
        if detection_count == 0:
            status_text = "No ₹50 note detected."
            font = cv2.FONT_HERSHEY_SIMPLEX
            (st_w, st_h), _ = cv2.getTextSize(status_text, font, 0.65, 1)

            st_box_x = 16
            st_box_y = 70
            cv2.rectangle(
                frame,
                (st_box_x, st_box_y),
                (st_box_x + st_w + 24, st_box_y + st_h + 16),
                (40, 40, 45),
                -1,
            )
            cv2.rectangle(
                frame,
                (st_box_x, st_box_y),
                (st_box_x + st_w + 24, st_box_y + st_h + 16),
                (80, 80, 85),
                1,
            )
            cv2.putText(
                frame,
                status_text,
                (st_box_x + 12, st_box_y + st_h + 8),
                font,
                0.60,
                self.COLOR_TEXT_MUTED,
                1,
                cv2.LINE_AA,
            )

        # Footer shortcut keys helper
        footer_y = h - 12
        shortcuts_text = "[Q] Quit  |  [S] Save Screenshot  |  [R] Reset History"
        cv2.putText(
            frame,
            shortcuts_text,
            (16, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return frame

