"""
Agent Kit Fast Runner.
Provides a streamlined one-line runner for inspecting counting results and benchmark stats
without spinning up full CLI overhead or external processes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pediatric_counter.app.config import load_config
from pediatric_counter.app.pipeline import run_pipeline


def quick_run(
    video_path: str | Path,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    config_path: str | Path = "pediatric_counter/configs/room_default.yaml",
    live_view: bool = False,
    save_annotated_video: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    """Execute a quick run of the pediatric counter and return a structured dictionary."""
    cfg = load_config(config_path)
    cfg.room.video_path = Path(video_path)
    cfg.room.start_frame = start_frame
    if max_frames is not None:
        cfg.room.max_frames = max_frames
    cfg.artifacts.live_view = live_view
    cfg.artifacts.save_annotated_video = save_annotated_video

    for k, v in overrides.items():
        if hasattr(cfg.tracking, k):
            setattr(cfg.tracking, k, v)
        elif hasattr(cfg.lifecycle, k):
            setattr(cfg.lifecycle, k, v)
        elif hasattr(cfg.counting, k):
            setattr(cfg.counting, k, v)
        elif hasattr(cfg.room, k):
            setattr(cfg.room, k, v)

    summary = run_pipeline(cfg)
    return {
        "video": Path(video_path).name,
        "children": summary.distinct_child_count,
        "adults": summary.distinct_adult_count,
        "total": summary.total_distinct_count,
        "child_ids": summary.counted_child_ids,
        "adult_ids": summary.counted_adult_ids,
        "uncertain_ids": summary.uncertain_ids,
        "frames_processed": summary.total_frames_processed,
        "fps": round(summary.fps, 2),
    }
