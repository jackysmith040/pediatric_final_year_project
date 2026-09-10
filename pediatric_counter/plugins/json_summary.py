"""
JSON Summary Plugin — records final distinct demographic totals and run metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

from pediatric_counter.plugins.base import (
    ControlSignal,
    FrameResult,
    PipelinePlugin,
    PipelineSummary,
)


class JSONSummaryPlugin(PipelinePlugin):
    """Outputs structured run summary JSON at EOF."""

    name: str = "json_summary"
    priority: int = 15

    def __init__(self, filename: str = "summary.json") -> None:
        self.filename = filename
        self._output_dir: Path | None = None

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        self._output_dir = output_dir

    def on_frame(self, result: FrameResult) -> ControlSignal:
        return ControlSignal.CONTINUE

    def on_finish(self, summary: PipelineSummary) -> None:
        if not self._output_dir:
            self._output_dir = summary.output_dir

        summary_data = {
            "run_id": summary.run_id,
            "distinct_child_count": summary.distinct_child_count,
            "distinct_adult_count": summary.distinct_adult_count,
            "total_distinct_count": summary.total_distinct_count,
            "counted_child_ids": summary.counted_child_ids,
            "counted_adult_ids": summary.counted_adult_ids,
            "uncertain_ids": summary.uncertain_ids,
            "total_frames_processed": summary.total_frames_processed,
            "detection_events_count": len(summary.events),
            "config_hash": summary.config_hash,
        }

        target_file = self._output_dir / self.filename
        with target_file.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
