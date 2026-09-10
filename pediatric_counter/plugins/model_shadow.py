"""
Model Shadow Evaluator Plugin.

Evaluates an alternative model (e.g. pediatric-smaller-dataset-trained.onnx)
in parallel with the baseline pipeline, logging comparative agreement metrics
to model_comparison.json without altering the core counting state machine.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pediatric_counter.plugins.base import ControlSignal, FrameResult, PipelinePlugin, PipelineSummary

logger = logging.getLogger("pediatric_counter.plugins.model_shadow")


class ModelShadowPlugin(PipelinePlugin):
    """
    Shadow benchmark plugin for testing fine-tuned models side-by-side.
    """

    name: str = "model_shadow"
    priority: int = 45

    def __init__(
        self,
        model_path: str = "computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-smaller-dataset-trained.onnx",
        confidence_threshold: float = 0.20,
        enabled: bool = True,
        output_filename: str = "model_comparison.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled
        self.output_filename = output_filename
        self._model = None
        self._output_dir: Optional[Path] = None
        self.comparison_records: List[Dict[str, Any]] = []
        self.total_inference_time_ms: float = 0.0
        self.total_crops_evaluated: int = 0

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        self._output_dir = output_dir
        if not self.enabled:
            logger.info("ModelShadowPlugin is disabled.")
            return

        if not self.model_path.exists():
            logger.warning(f"ModelShadowPlugin model not found at {self.model_path}")
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path), task="detect")
            if hasattr(self._model, "overrides"):
                self._model.overrides["verbose"] = False
            logger.info(f"ModelShadowPlugin loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"ModelShadowPlugin failed to load model: {e}")
            self._model = None

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if not self.enabled or self._model is None:
            return ControlSignal.CONTINUE

        raw_frame = result.raw_frame
        h_frame, w_frame = raw_frame.shape[:2]

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

            t0 = time.perf_counter()
            try:
                preds = self._model(crop, conf=self.confidence_threshold, verbose=False)[0]
                dt_ms = (time.perf_counter() - t0) * 1000.0
                self.total_inference_time_ms += dt_ms
                self.total_crops_evaluated += 1

                top_cls = "unknown"
                top_conf = 0.0
                for b in preds.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    c_name = str(self._model.names.get(cls_id, "unknown")).lower()
                    if conf > top_conf:
                        top_conf = conf
                        top_cls = c_name

                baseline_label = result.track_labels.get(tid, "detecting")
                self.comparison_records.append({
                    "frame_idx": result.frame_idx,
                    "track_id": tid,
                    "baseline_label": baseline_label,
                    "shadow_prediction": top_cls,
                    "shadow_confidence": round(top_conf, 3),
                    "agrees": (baseline_label == top_cls) if baseline_label in ("child", "adult") else None,
                    "latency_ms": round(dt_ms, 2),
                })
            except Exception as e:
                logger.debug(f"ModelShadow error on track #{tid}: {e}")

        return ControlSignal.CONTINUE

    def on_finish(self, summary: PipelineSummary) -> None:
        if not self.enabled or self._output_dir is None:
            return

        evaluated_pairs = [r for r in self.comparison_records if r["agrees"] is not None]
        agreements = sum(1 for r in evaluated_pairs if r["agrees"] is True)
        total_eval = len(evaluated_pairs)
        agreement_rate = (agreements / total_eval) if total_eval > 0 else 0.0
        avg_latency = (self.total_inference_time_ms / max(1, self.total_crops_evaluated))

        report = {
            "model_tested": str(self.model_path.name),
            "model_full_path": str(self.model_path),
            "total_crops_evaluated": self.total_crops_evaluated,
            "definitive_pairs_compared": total_eval,
            "agreement_count": agreements,
            "agreement_rate": round(agreement_rate, 4),
            "average_latency_ms": round(avg_latency, 2),
            "summary": {
                "baseline_children": summary.distinct_child_count,
                "baseline_adults": summary.distinct_adult_count,
            },
            "recent_sample_comparisons": self.comparison_records[-30:] if self.comparison_records else [],
        }

        out_path = self._output_dir / self.output_filename
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"ModelShadowPlugin saved comparative report to {out_path} (Agreement: {agreement_rate*100:.1f}%)")
        except Exception as e:
            logger.error(f"Failed to write model comparison report: {e}")
