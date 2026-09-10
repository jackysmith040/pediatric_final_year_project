"""
Kids Sieve Plugin for Dedicated Pediatric Demographic Filtering.

Observes person proposals from the Base Model and verifies child characteristics
using the specialized pediatric-kids-only model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from pediatric_counter.plugins.base import ControlSignal, FrameResult, PipelinePlugin, PipelineSummary

logger = logging.getLogger("pediatric_counter.plugins.kids_sieve")


class KidsSievePlugin(PipelinePlugin):
    """
    Dedicated kids-only sieve plugin.
    Runs pediatric-kids-only.onnx (or .pt) on candidate person crops.
    """

    name: str = "kids_sieve"
    priority: int = 40  # Runs before visualizers and exporters

    def __init__(
        self,
        model_path: str = "computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-kids-only.onnx",
        confidence_threshold: float = 0.25,
        enabled: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled
        self._model = None
        self.sieve_history: Dict[int, list[float]] = {}

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        if not self.enabled:
            logger.info("KidsSievePlugin is disabled.")
            return

        if not self.model_path.exists():
            logger.warning(f"KidsSievePlugin model not found at {self.model_path}")
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path), task="detect")
            if hasattr(self._model, "overrides"):
                self._model.overrides["verbose"] = False
            logger.info(f"KidsSievePlugin loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"KidsSievePlugin failed to load model: {e}")
            self._model = None

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if not self.enabled or self._model is None:
            return ControlSignal.CONTINUE

        raw_frame = result.raw_frame
        h_frame, w_frame = raw_frame.shape[:2]
        sieve_results: Dict[int, float] = {}

        for i, tid in enumerate(result.tracked_ids):
            if i >= len(result.bounding_boxes):
                continue
            box = result.bounding_boxes[i]
            x1, y1, x2, y2 = box
            x1 = max(0, min(w_frame - 1, x1))
            y1 = max(0, min(h_frame - 1, y1))
            x2 = max(x1 + 1, min(w_frame, x2))
            y2 = max(y1 + 1, min(h_frame, y2))

            crop = raw_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                preds = self._model(crop, conf=self.confidence_threshold, verbose=False)[0]
                child_conf = 0.0
                for b in preds.boxes:
                    c_name = str(self._model.names.get(int(b.cls[0]))).lower()
                    if "child" in c_name:
                        child_conf = max(child_conf, float(b.conf[0]))
                
                sieve_results[tid] = child_conf
                if tid not in self.sieve_history:
                    self.sieve_history[tid] = []
                self.sieve_history[tid].append(child_conf)
            except Exception as e:
                logger.debug(f"KidsSieve crop inference error on track #{tid}: {e}")

        result.extra["kids_sieve"] = sieve_results
        return ControlSignal.CONTINUE

    def on_finish(self, summary: PipelineSummary) -> None:
        if not self.enabled or not self.sieve_history:
            return

        summary.extra["kids_sieve_tracks_evaluated"] = len(self.sieve_history)
        logger.info(f"KidsSievePlugin finalized. Evaluated {len(self.sieve_history)} distinct tracks.")
