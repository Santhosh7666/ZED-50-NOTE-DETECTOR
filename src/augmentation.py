"""
Data Augmentation Pipeline for YOLO Currency Detection.
Applies realistic photographic augmentations (lighting, rotation, scale, shadows,
perspective, blur, noise) while preserving banknotes' authentic security patterns.
Updates YOLO bounding box coordinates accordingly.
"""

import random
from typing import Tuple, List, Optional
import numpy as np
import cv2


class BanknoteAugmentor:
    """
    Augmentation engine specifically tuned for currency detection.
    Prevents unrealistic color inversions or heavy distortion while simulating
    real-world mobile camera capture conditions.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15.0, 15.0),
        scale_range: Tuple[float, float] = (0.85, 1.15),
        brightness_range: Tuple[float, float] = (0.75, 1.25),
        contrast_range: Tuple[float, float] = (0.80, 1.20),
        p_blur: float = 0.25,
        p_noise: float = 0.25,
        p_shadow: float = 0.35,
    ):
        self.rot_min, self.rot_max = rotation_range
        self.scale_min, self.scale_max = scale_range
        self.b_min, self.b_max = brightness_range
        self.c_min, self.c_max = contrast_range
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.p_shadow = p_shadow

    def adjust_photometry(self, image: np.ndarray) -> np.ndarray:
        """Adjusts brightness, contrast, and exposure realistically."""
        img = image.astype(np.float32)

        # Contrast and brightness: img * alpha + beta
        alpha = random.uniform(self.c_min, self.c_max)
        beta = random.uniform((self.b_min - 1.0) * 128, (self.b_max - 1.0) * 128)

        img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)
        return img

    def apply_shadow(self, image: np.ndarray) -> np.ndarray:
        """Simulates natural ambient shadow / hand occluding part of the light."""
        if random.random() > self.p_shadow:
            return image

        h, w = image.shape[:2]
        # Generate random polygon shadow mask
        mask = np.zeros((h, w), dtype=np.float32)
        top_x = random.randint(0, w)
        bot_x = random.randint(0, w)
        pts = np.array([
            [top_x, 0],
            [w, 0],
            [w, h],
            [bot_x, h],
        ], dtype=np.int32)

        cv2.fillPoly(mask, [pts], 1.0)
        # Blur the shadow boundary for soft shadow effect
        mask = cv2.GaussianBlur(mask, (51, 51), 0)

        shadow_intensity = random.uniform(0.40, 0.75)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Darken Value channel
        hsv[:, :, 2] = hsv[:, :, 2] * (1.0 - mask * (1.0 - shadow_intensity))
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def apply_blur_and_noise(self, image: np.ndarray) -> np.ndarray:
        """Simulates camera lens defocus or sensor noise."""
        out = image.copy()
        if random.random() < self.p_blur:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)

        if random.random() < self.p_noise:
            noise = np.random.normal(0, random.uniform(3, 10), out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return out

    def transform_geometry(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Applies rotation, scaling, and translation, transforming bounding boxes accordingly.
        boxes format: [[class_id, x_c, y_c, w, h], ...] (normalized)
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        angle = random.uniform(self.rot_min, self.rot_max)
        scale = random.uniform(self.scale_min, self.scale_max)
        dx = random.uniform(-0.05, 0.05) * w
        dy = random.uniform(-0.05, 0.05) * h

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += dx
        M[1, 2] += dy

        warped_img = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        if not boxes:
            return warped_img, []

        new_boxes = []
        for box in boxes:
            cls_id, xc_n, yc_n, bw_n, bh_n = box
            xc, yc = xc_n * w, yc_n * h
            bw, bh = bw_n * w, bh_n * h

            # 4 corners in pixel coords
            pts = np.array([
                [xc - bw / 2, yc - bh / 2, 1.0],
                [xc + bw / 2, yc - bh / 2, 1.0],
                [xc + bw / 2, yc + bh / 2, 1.0],
                [xc - bw / 2, yc + bh / 2, 1.0],
            ]).T  # 3x4

            transformed_pts = (M @ pts).T  # 4x2
            x_min = np.clip(np.min(transformed_pts[:, 0]), 0, w)
            x_max = np.clip(np.max(transformed_pts[:, 0]), 0, w)
            y_min = np.clip(np.min(transformed_pts[:, 1]), 0, h)
            y_max = np.clip(np.max(transformed_pts[:, 1]), 0, h)

            nw = x_max - x_min
            nh = y_max - y_min

            # Ensure transformed box is still valid and substantial
            if nw > 20 and nh > 20:
                n_xc = (x_min + nw / 2) / w
                n_yc = (y_min + nh / 2) / h
                n_w = nw / w
                n_h = nh / h
                new_boxes.append([cls_id, n_xc, n_yc, n_w, n_h])

        return warped_img, new_boxes

    def augment(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Full augmentation pipeline returning augmented image and normalized boxes."""
        # 1. Photometric adjustments
        img = self.adjust_photometry(image)

        # 2. Ambient shadow
        img = self.apply_shadow(img)

        # 3. Blur & sensor noise
        img = self.apply_blur_and_noise(img)

        # 4. Geometric rotation & scaling
        img, aug_boxes = self.transform_geometry(img, boxes)

        return img, aug_boxes
