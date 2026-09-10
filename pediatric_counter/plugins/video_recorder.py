"""
Video Recorder Plugin — encodes annotated frames into an MP4 video file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2

from pediatric_counter.plugins.base import (
    ControlSignal,
    FrameResult,
    PipelinePlugin,
    PipelineSummary,
)


class VideoRecorderPlugin(PipelinePlugin):
    """Encodes annotated video frames to an MP4 video file."""

    name: str = "video_recorder"
    priority: int = 80

    def __init__(self, filename: str = "annotated.mp4") -> None:
        self.filename = filename
        self._writer: Optional[cv2.VideoWriter] = None

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        height, width = frame_shape[:2]
        out_path = str(output_dir / self.filename)
        self._writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if self._writer is not None:
            self._writer.write(result.annotated_frame)
        return ControlSignal.CONTINUE

    def on_finish(self, summary: PipelineSummary) -> None:
        self.close()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
