"""
CSV Exporter Plugin — records per-frame bounding boxes, demographic probabilities, and tracking states.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, TextIO, Tuple

from pediatric_counter.plugins.base import (
    ControlSignal,
    FrameResult,
    PipelinePlugin,
    PipelineSummary,
)


class PerFrameCSVPlugin(PipelinePlugin):
    """Logs individual detection bounding boxes and states per frame to a CSV file."""

    name: str = "csv_exporter"
    priority: int = 10

    def __init__(self, filename: str = "per_frame.csv") -> None:
        self.filename = filename
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        csv_path = output_dir / self.filename
        self._file = csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "frame",
                "track_id",
                "state",
                "label",
                "child_prob",
                "counted",
                "uncertain",
                "box_x1",
                "box_y1",
                "box_x2",
                "box_y2",
            ],
        )
        self._writer.writeheader()

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if not self._writer:
            return ControlSignal.CONTINUE

        for i, tid in enumerate(result.tracked_ids):
            box = result.bounding_boxes[i] if i < len(result.bounding_boxes) else (0, 0, 0, 0)
            state = result.track_states.get(tid, "tentative")
            label = result.track_labels.get(tid, "unknown")
            prob = result.child_probabilities.get(tid, 0.5)
            is_counted = 1 if state in ("confirmed", "retired") else 0
            is_uncertain = 1 if prob >= 0.4 and prob <= 0.6 else 0

            self._writer.writerow(
                {
                    "frame": result.frame_idx,
                    "track_id": tid,
                    "state": state,
                    "label": label,
                    "child_prob": round(prob, 4),
                    "counted": is_counted,
                    "uncertain": is_uncertain,
                    "box_x1": int(box[0]),
                    "box_y1": int(box[1]),
                    "box_x2": int(box[2]),
                    "box_y2": int(box[3]),
                }
            )
        return ControlSignal.CONTINUE

    def on_finish(self, summary: PipelineSummary) -> None:
        self.close()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
