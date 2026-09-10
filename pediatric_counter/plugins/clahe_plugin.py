"""
CLAHE Plugin (Contrast Limited Adaptive Histogram Equalization).

What it is:
Contrast Limited Adaptive Histogram Equalization (CLAHE) is an advanced computer vision
enhancement technique designed to improve local contrast and reveal hidden details in
harshly lit CCTV scenes (e.g. strong backlighting, heavy shadows under desks, low-light hallways).
Unlike standard global histogram equalization, which stretches the contrast across the entire
image uniformly and can blow out highlights or amplify background sensor noise, CLAHE divides
the frame into small contextual tiles (e.g. 8x8 pixels), computes local histograms, and clips
the contrast enhancement at a defined limit (clipLimit) to prevent noise amplification.

Usage:
This plugin is OPTIONAL and disabled by default. It can be toggled on/off in the configuration
or used programmatically to pre-process frames before detection or crop classification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple
import cv2
import numpy as np

from pediatric_counter.plugins.base import ControlSignal, FrameResult, PipelinePlugin, PipelineSummary


class CLAHEPlugin(PipelinePlugin):
    """
    Optional plugin providing Contrast Limited Adaptive Histogram Equalization.
    Kept disabled by default; can be enabled for low-contrast/backlit feeds.
    """

    name = "clahe_enhancer"
    priority = 20  # Runs early in pipeline processing

    def __init__(
        self,
        enabled: bool = False,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self.enabled = enabled
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe: cv2.CLAHE | None = None
        if self.enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )

    def enhance(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement on the Luminance (L) channel in LAB color space.
        Preserves original color balance while enhancing shadow/highlight details.
        """
        if not self.enabled:
            return bgr_image

        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )

        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._clahe.apply(l_channel)
        merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def on_frame(self, result: FrameResult) -> ControlSignal:
        """Passive monitor when registered as a standard pipeline plugin."""
        return ControlSignal.CONTINUE
