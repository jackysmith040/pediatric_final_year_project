"""
Base interfaces and contracts for the Pediatric & Adult CCTV Counter Plugin Architecture.

Design Invariants:
1. Immutable Core: Plugins observe and augment execution; they do not alter the distinct counting state machine.
2. Isolated Memory: Visualizer plugins receive an annotated copy; raw video buffers remain pristine.
3. Two-Way Control: Plugins can send non-blocking ControlSignals (CONTINUE, PAUSE, STEP, QUIT, SPEED_CHANGE).
4. Exception Barrier: Failures in plugins are captured without interrupting the counting core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ControlSignal(str, Enum):
    """Signals sent from a plugin (e.g. Live HUD keyboard handler) back to the core pipeline."""
    CONTINUE = "continue"
    PAUSE = "pause"
    STEP = "step"
    QUIT = "quit"
    SPEED_CHANGE = "speed_change"


@dataclass
class FrameResult:
    """Per-frame analytical payload passed to all registered plugins."""
    frame_idx: int
    raw_frame: np.ndarray
    annotated_frame: np.ndarray
    tracked_ids: List[int]
    bounding_boxes: List[Tuple[int, int, int, int]]
    track_labels: Dict[int, str]
    child_probabilities: Dict[int, float]
    track_states: Dict[int, str]
    counted_child_count: int
    counted_adult_count: int
    total_count: int
    fps: float
    cctv_timestamp_str: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSummary:
    """Final summary payload passed to plugins at EOF or upon session termination."""
    run_id: str
    output_dir: Path
    distinct_child_count: int
    distinct_adult_count: int
    total_distinct_count: int
    counted_child_ids: List[int]
    counted_adult_ids: List[int]
    uncertain_ids: List[int]
    total_frames_processed: int
    fps: float
    events: List[Any] = field(default_factory=list)
    config_hash: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class PipelinePlugin:
    """
    Abstract base class for all pipeline extensions.
    Any external reporter, exporter, or UI overlay subclasses this interface.
    """

    name: str = "base_plugin"
    priority: int = 50  # Lower numbers execute earlier (e.g., Exporters: 10, Visualizers: 90)

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        """Called once before the frame processing loop begins."""
        pass

    def on_frame(self, result: FrameResult) -> ControlSignal:
        """
        Called after each frame is tracked and counted.
        Returns a ControlSignal to guide playback or flow.
        """
        return ControlSignal.CONTINUE

    def on_finish(self, summary: PipelineSummary) -> None:
        """Called at EOF or when processing completes successfully."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called if an unhandled error occurs during pipeline execution."""
        pass

    def close(self) -> None:
        """Release any held handles, open files, or OpenCV windows."""
        pass
