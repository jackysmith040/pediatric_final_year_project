"""
Tests for the developer Agent Kit.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from pediatric_counter.agent_kit.diagnostics import probe_video, probe_frame_clarity
from pediatric_counter.agent_kit.crop_inspector import inspect_frame_crops


def test_probe_video_nonexistent(tmp_path: Path) -> None:
    res = probe_video(tmp_path / "nonexistent.mp4")
    assert res["exists"] is False
    assert "error" in res


def test_probe_frame_clarity_error_nonexistent(tmp_path: Path) -> None:
    res = probe_frame_clarity(tmp_path / "nonexistent.mp4", 0)
    assert "error" in res or res.get("artifact_detected") is True
