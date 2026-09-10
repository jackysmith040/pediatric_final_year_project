"""
SAHI Plugin (Slicing Aided Hyper Inference).

What it is:
Slicing Aided Hyper Inference (SAHI) is an inference framework specialized for detecting
small or distant objects in high-resolution, wide-angle imagery. Standard object detectors
(like YOLO) resize high-resolution camera feeds (1080p, 2K, 4K) down to a fixed input size
(such as 640x640). In a wide-angle CCTV camera mounted on a high ceiling, a small toddler or
infant far across the room might measure only 24x24 pixels. When downscaled to 640x640, that
child shrinks to 6x6 pixels—losing critical spatial features and becoming undetectable.

How SAHI Works:
1. Slicing: Cuts the full-resolution frame into overlapping rectangular tiles (e.g. 640x640 with 20% overlap).
2. Native Inference: Runs the detection model on each slice at native resolution without extreme downscaling.
3. Coordinate Shift: Maps the detected bounding boxes from slice coordinates back to full-frame coordinates.
4. Non-Maximum Suppression (NMS): Merges duplicate overlapping detections across slice boundaries.

Usage:
This plugin is OPTIONAL and disabled by default. Because running multiple slices per frame
multiplies inference passes, it is recommended only for ultra-high-resolution feeds where small
children are too distant for standard full-frame detectors to resolve.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple
import numpy as np

from pediatric_counter.plugins.base import ControlSignal, FrameResult, PipelinePlugin


class SAHIPlugin(PipelinePlugin):
    """
    Optional plugin providing Slicing Aided Hyper Inference for small-object CCTV detection.
    Kept disabled by default; can be enabled for 4K / wide-angle camera feeds.
    """

    name = "sahi_sliced_inference"
    priority = 25

    def __init__(
        self,
        enabled: bool = False,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        iou_threshold: float = 0.50,
    ) -> None:
        self.enabled = enabled
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.iou_threshold = iou_threshold

    def get_slice_boxes(self, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Compute top-left and bottom-right crop coordinates for all sliding windows."""
        h, w = image_shape[:2]
        step_y = int(self.slice_height * (1.0 - self.overlap_height_ratio))
        step_x = int(self.slice_width * (1.0 - self.overlap_width_ratio))

        y_coords = list(range(0, max(1, h - self.slice_height + 1), step_y))
        if len(y_coords) == 0 or y_coords[-1] + self.slice_height < h:
            y_coords.append(max(0, h - self.slice_height))

        x_coords = list(range(0, max(1, w - self.slice_width + 1), step_x))
        if len(x_coords) == 0 or x_coords[-1] + self.slice_width < w:
            x_coords.append(max(0, w - self.slice_width))

        slices = []
        for y1 in y_coords:
            for x1 in x_coords:
                y2 = min(h, y1 + self.slice_height)
                x2 = min(w, x1 + self.slice_width)
                slices.append((x1, y1, x2, y2))
        return slices

    def on_frame(self, result: FrameResult) -> ControlSignal:
        """Passive monitor when registered as a standard pipeline plugin."""
        return ControlSignal.CONTINUE
