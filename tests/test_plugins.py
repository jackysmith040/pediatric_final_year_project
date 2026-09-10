"""
Tests for the Pediatric Counter Plugin Architecture and Fault Isolation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from pediatric_counter.plugins.base import (
    ControlSignal,
    FrameResult,
    PipelinePlugin,
    PipelineSummary,
)
from pediatric_counter.plugins.csv_exporter import PerFrameCSVPlugin
from pediatric_counter.plugins.json_summary import JSONSummaryPlugin
from pediatric_counter.plugins.live_hud import LiveHUDPlugin
from pediatric_counter.plugins.manager import PluginManager


class PriorityTrackerPlugin(PipelinePlugin):
    def __init__(self, name: str, priority: int, record_list: list[str]) -> None:
        self.name = name
        self.priority = priority
        self.record_list = record_list

    def on_frame(self, result: FrameResult) -> ControlSignal:
        self.record_list.append(self.name)
        return ControlSignal.CONTINUE


class FaultyPlugin(PipelinePlugin):
    name = "faulty_plugin"
    priority = 5

    def __init__(self) -> None:
        self.call_count = 0

    def on_frame(self, result: FrameResult) -> ControlSignal:
        self.call_count += 1
        raise RuntimeError("Simulated plugin crash: memory allocation failure or bug")


class QuitSignalPlugin(PipelinePlugin):
    name = "quit_plugin"
    priority = 99

    def on_frame(self, result: FrameResult) -> ControlSignal:
        return ControlSignal.QUIT


def test_plugin_priority_ordering(tmp_path: Path) -> None:
    """Plugins must be invoked strictly in ascending order of priority."""
    manager = PluginManager()
    call_log: list[str] = []

    p_high = PriorityTrackerPlugin("p_high", 90, call_log)
    p_low = PriorityTrackerPlugin("p_low", 10, call_log)
    p_mid = PriorityTrackerPlugin("p_mid", 50, call_log)

    # Register in arbitrary order
    manager.register(p_high)
    manager.register(p_low)
    manager.register(p_mid)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = FrameResult(
        frame_idx=0,
        raw_frame=dummy_frame,
        annotated_frame=dummy_frame.copy(),
        tracked_ids=[],
        bounding_boxes=[],
        track_labels={},
        child_probabilities={},
        track_states={},
        counted_child_count=0,
        counted_adult_count=0,
        total_count=0,
        fps=25.0,
    )

    manager.broadcast_frame(res)
    assert call_log == ["p_low", "p_mid", "p_high"]


def test_plugin_fault_isolation(tmp_path: Path) -> None:
    """A crashed plugin MUST NOT raise out of broadcast_frame or stop other plugins."""
    manager = PluginManager()
    call_log: list[str] = []

    faulty = FaultyPlugin()
    good = PriorityTrackerPlugin("good_plugin", 50, call_log)

    manager.register(faulty)
    manager.register(good)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = FrameResult(
        frame_idx=0,
        raw_frame=dummy_frame,
        annotated_frame=dummy_frame.copy(),
        tracked_ids=[1],
        bounding_boxes=[(10, 10, 50, 50)],
        track_labels={1: "child"},
        child_probabilities={1: 0.95},
        track_states={1: "confirmed"},
        counted_child_count=1,
        counted_adult_count=0,
        total_count=1,
        fps=25.0,
    )

    # First broadcast: FaultyPlugin crashes, but good_plugin STILL runs!
    signal = manager.broadcast_frame(res)
    assert signal == ControlSignal.CONTINUE
    assert call_log == ["good_plugin"]
    assert faulty.call_count == 1

    # Second broadcast: FaultyPlugin is now disabled and never called again
    manager.broadcast_frame(res)
    assert call_log == ["good_plugin", "good_plugin"]
    assert faulty.call_count == 1  # Not called again


def test_plugin_control_signal_aggregation() -> None:
    """Plugin returning QUIT causes aggregated broadcast_frame to return QUIT."""
    manager = PluginManager()
    manager.register(QuitSignalPlugin())

    dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    res = FrameResult(
        frame_idx=0,
        raw_frame=dummy_frame,
        annotated_frame=dummy_frame,
        tracked_ids=[],
        bounding_boxes=[],
        track_labels={},
        child_probabilities={},
        track_states={},
        counted_child_count=0,
        counted_adult_count=0,
        total_count=0,
        fps=25.0,
    )

    sig = manager.broadcast_frame(res)
    assert sig == ControlSignal.QUIT


def test_per_frame_csv_plugin(tmp_path: Path) -> None:
    """PerFrameCSVPlugin accurately writes detection bounding boxes and states."""
    plugin = PerFrameCSVPlugin(filename="test_frames.csv")
    plugin.on_start(output_dir=tmp_path, fps=25.0, total_frames=10, frame_shape=(100, 100, 3))

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = FrameResult(
        frame_idx=42,
        raw_frame=dummy_frame,
        annotated_frame=dummy_frame,
        tracked_ids=[7],
        bounding_boxes=[(12, 24, 60, 80)],
        track_labels={7: "child"},
        child_probabilities={7: 0.88},
        track_states={7: "confirmed"},
        counted_child_count=1,
        counted_adult_count=0,
        total_count=1,
        fps=25.0,
    )

    plugin.on_frame(res)
    plugin.close()

    csv_path = tmp_path / "test_frames.csv"
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "frame,track_id,state,label,child_prob,counted,uncertain,box_x1,box_y1,box_x2,box_y2" in content
    assert "42,7,confirmed,child,0.88,1,0,12,24,60,80" in content


def test_clahe_plugin_enhancement() -> None:
    from pediatric_counter.plugins.clahe_plugin import CLAHEPlugin
    
    # Test disabled
    plugin_disabled = CLAHEPlugin(enabled=False)
    dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    enhanced = plugin_disabled.enhance(dummy_img)
    assert np.array_equal(dummy_img, enhanced)

    # Test enabled
    plugin_enabled = CLAHEPlugin(enabled=True, clip_limit=2.0)
    enhanced = plugin_enabled.enhance(dummy_img)
    assert enhanced.shape == dummy_img.shape


def test_sahi_plugin_slices() -> None:
    from pediatric_counter.plugins.sahi_plugin import SAHIPlugin

    plugin = SAHIPlugin(enabled=False, slice_height=640, slice_width=640)
    # High-resolution 1080p frame
    slices = plugin.get_slice_boxes((1080, 1920, 3))
    assert len(slices) > 1
    # Check that all slice coordinates are within bounds
    for x1, y1, x2, y2 in slices:
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080


def test_json_summary_plugin(tmp_path: Path) -> None:
    """JSONSummaryPlugin outputs structured metrics at completion."""
    plugin = JSONSummaryPlugin(filename="test_summary.json")
    plugin.on_start(output_dir=tmp_path, fps=25.0, total_frames=100, frame_shape=(100, 100, 3))

    summary = PipelineSummary(
        run_id="run_test_123",
        output_dir=tmp_path,
        distinct_child_count=3,
        distinct_adult_count=5,
        total_distinct_count=8,
        counted_child_ids=[1, 2, 3],
        counted_adult_ids=[4, 5, 6, 7, 8],
        uncertain_ids=[],
        total_frames_processed=100,
        fps=25.0,
        config_hash="abc12345",
    )

    plugin.on_finish(summary)

    json_path = tmp_path / "test_summary.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_test_123"
    assert data["distinct_child_count"] == 3
    assert data["distinct_adult_count"] == 5
    assert data["total_distinct_count"] == 8
