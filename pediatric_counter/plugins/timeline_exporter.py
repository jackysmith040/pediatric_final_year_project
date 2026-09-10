"""
Timeline Exporter Plugin — clusters per-frame detections into discrete video timestamps.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from pediatric_counter.counting.events import DetectionEvent, EventIndexer
from pediatric_counter.plugins.base import (
    ControlSignal,
    FrameResult,
    PipelinePlugin,
    PipelineSummary,
)


class TimelineExporterPlugin(PipelinePlugin):
    """Indexes continuous person sightings and exports timestamped markdown, JSON, and CSV."""

    name: str = "timeline_exporter"
    priority: int = 20

    def __init__(
        self,
        cctv_start_time: Optional[datetime] = None,
        min_duration_seconds: float = 0.5,
    ) -> None:
        self.cctv_start_time = cctv_start_time
        self.min_duration_seconds = min_duration_seconds
        self._indexer: Optional[EventIndexer] = None
        self._output_dir: Optional[Path] = None

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        self._output_dir = output_dir
        self._indexer = EventIndexer(
            fps=fps,
            cctv_start_time=self.cctv_start_time,
        )

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if not self._indexer:
            return ControlSignal.CONTINUE

        for i, tid in enumerate(result.tracked_ids):
            state = result.track_states.get(tid, "tentative")
            if state in ("confirmed", "tentative"):
                lbl = result.track_labels.get(tid, "unknown")
                prob = result.child_probabilities.get(tid, 0.5)
                box = list(result.bounding_boxes[i]) if i < len(result.bounding_boxes) else [0, 0, 0, 0]
                self._indexer.update(
                    frame_idx=result.frame_idx,
                    track_id=tid,
                    demographic=lbl,
                    confidence=prob,
                    box=box,
                )
        return ControlSignal.CONTINUE

    def on_finish(self, summary: PipelineSummary) -> None:
        if not self._indexer or not self._output_dir:
            return

        events: List[DetectionEvent] = self._indexer.finalize_all()
        summary.events = events

        self._indexer.save_json(self._output_dir / "detection_events.json")
        self._indexer.save_csv(self._output_dir / "detection_events.csv")
        self._indexer.save_markdown(self._output_dir / "detection_timeline.md")
