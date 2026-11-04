"""
Preprocessing Module for YOLO ONNX Real-Time Banknote Detector.
Provides aspect-ratio preserving letterbox resizing, BGR->RGB conversion,
normalization, batch tensor formation, and reverse coordinate mapping.
"""

from typing import Tuple, List
import numpy as np
import cv2


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints and preserving aspect ratio.

    Args:
        image: Original BGR image (H, W, C)
        new_shape: Target dimensions (height, width) or int
        color: Padding fill color
        auto: Minimum rectangle pad
        scaleFill: Stretch to new_shape
        scaleup: Allow scaling up
        stride: Stride multiple (default 32 for YOLO)

    Returns:
        padded_image: Image ready for model input (H, W, C)
        ratio: Resize ratio (new / old)
        (dw, dh): Left/right and top/bottom padding amounts
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, r, (dw, dh)


def preprocess_frame(
    frame: np.ndarray,
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Standardizes OpenCV webcam frame for YOLO ONNX input.
    - Letterboxes frame
    - Converts BGR to RGB
    - Transposes (H, W, C) -> (1, C, H, W)
    - Normalizes uint8 [0, 255] to float32 [0.0, 1.0]

    Returns:
        blob: Transposed, normalized numpy array [1, 3, H, W]
        ratio: Scaling ratio
        pad: Padding offsets (dw, dh)
    """
    # 1. Letterbox resize
    padded_img, ratio, pad = letterbox(frame, new_shape=input_size, auto=False, scaleup=True)

    # 2. BGR to RGB
    rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

    # 3. Transpose (H, W, C) -> (C, H, W)
    blob = rgb_img.transpose((2, 0, 1))

    # 4. Add batch dimension & make contiguous
    blob = np.ascontiguousarray(blob, dtype=np.float32)

    # 5. Normalize [0, 255] -> [0.0, 1.0]
    blob /= 255.0
    blob = np.expand_dims(blob, axis=0)

    return blob, ratio, pad


def scale_coords(
    boxes: np.ndarray,
    ratio: float,
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Maps bounding boxes from letterboxed model input coordinates back to original frame coordinates.

    Args:
        boxes: Array of [x1, y1, x2, y2]
        ratio: Letterbox scale ratio
        pad: (dw, dh) letterbox padding
        orig_shape: (height, width) of original frame

    Returns:
        scaled_boxes: Bounding boxes clamped within original image dimensions
    """
    if len(boxes) == 0:
        return boxes

    dw, dh = pad
    scaled = boxes.copy()
    scaled[:, [0, 2]] -= dw
    scaled[:, [1, 3]] -= dh
    scaled[:, :4] /= ratio

    # Clip to image boundary
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, orig_shape[1])
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, orig_shape[0])

    return scaled
