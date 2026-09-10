"""
Pure bounding-box geometry utilities — deterministic, side-effect-free.

All functions operate on plain Python/NumPy primitives.
No OpenCV or Ultralytics imports here so that unit tests stay fast and isolated.

Mathematical reference (from thesis spec):
  C_m(b) = (x1 - m*w,  y1 - m*h,  x2 + m*w,  y2 + m*h)
  where m = margin_fraction, w = x2-x1, h = y2-y1
"""
from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Box expansion
# ──────────────────────────────────────────────────────────────────────────────

def expand_box(
    box: tuple[float, float, float, float],
    margin: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """
    Expand a bounding box by *margin* fraction and clamp to frame boundaries.

    Args:
        box:     (x1, y1, x2, y2) in pixel coordinates.
        margin:  Fractional expansion, e.g. 0.10 = 10 % of width/height.
        frame_w: Frame width  (px) used for clamping.
        frame_h: Frame height (px) used for clamping.

    Returns:
        Clamped integer (x1, y1, x2, y2).

    Raises:
        ValueError: if the input box is degenerate (zero area).
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError(f"Degenerate bounding box: {box}")

    dx = margin * w
    dy = margin * h
    nx1 = max(0, int(x1 - dx))
    ny1 = max(0, int(y1 - dy))
    nx2 = min(frame_w, int(x2 + dx))
    ny2 = min(frame_h, int(y2 + dy))
    return nx1, ny1, nx2, ny2


# ──────────────────────────────────────────────────────────────────────────────
# Resize modes
# ──────────────────────────────────────────────────────────────────────────────

def letterbox_crop(
    crop: np.ndarray,
    target_size: int = 224,
    padding_value: int = 114,
) -> np.ndarray:
    """
    Resize *crop* to (target_size × target_size) preserving aspect ratio.
    Remaining area is filled with *padding_value* (grey).

    This is the spec-required 'letterbox' mode that avoids squashing
    extreme-aspect-ratio person crops.

    Args:
        crop:          H×W×3 uint8 image array.
        target_size:   Square output side length.
        padding_value: Scalar fill value for the padded regions.

    Returns:
        target_size × target_size × 3 uint8 array.
    """
    import cv2  # deferred — keeps unit tests import-free

    h, w = crop.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), padding_value, dtype=np.uint8)
    pad_top  = (target_size - nh) // 2
    pad_left = (target_size - nw) // 2
    canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized
    return canvas


def direct_resize_crop(
    crop: np.ndarray,
    target_size: int = 224,
) -> np.ndarray:
    """
    Resize *crop* directly to (target_size × target_size) — no aspect preservation.
    Use only when comparing against training code that used the same transform.

    Args:
        crop:        H×W×3 uint8 image array.
        target_size: Square output side length.

    Returns:
        target_size × target_size × 3 uint8 array.
    """
    import cv2  # deferred

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────────────────────
# Frame ↔ time conversions
# ──────────────────────────────────────────────────────────────────────────────

def seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert a duration in *seconds* to frame count given *fps*."""
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return max(1, int(round(seconds * fps)))


def frames_to_seconds(frames: int, fps: float) -> float:
    """Convert a *frame* count to seconds given *fps*."""
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return frames / fps


# ──────────────────────────────────────────────────────────────────────────────
# IoU
# ──────────────────────────────────────────────────────────────────────────────

def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """
    Intersection over Union for two xyxy boxes.

    Returns a value in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0
