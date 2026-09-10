"""
Agent Kit Diagnostics & Probing Utilities.
Provides reusable helpers for inspecting video clarity, HEVC keyframes,
demographic classification confidence, and Stage 1/Stage 2 pipeline behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import cv2
import numpy as np


def probe_video(video_path: str | Path) -> dict[str, Any]:
    """Inspect video file properties, resolution, frame count, and playability."""
    p = Path(video_path)
    if not p.exists():
        return {"exists": False, "error": f"File not found: {p}"}

    cap = cv2.VideoCapture(str(p))
    opened = cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    ret, frame = cap.read() if opened else (False, None)
    std_val = float(np.std(frame)) if ret and frame is not None else 0.0
    cap.release()

    return {
        "exists": True,
        "filename": p.name,
        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "is_opened": opened,
        "can_read_first_frame": ret,
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "first_frame_std": round(std_val, 2),
        "is_vertical": h > w,
    }


def probe_frame_clarity(video_path: str | Path, frame_idx: int) -> dict[str, float]:
    """Assess whether a decoded frame is clear or suffering from flat-gray HEVC macroblocking."""
    from pediatric_counter.io.video_reader import VideoReader

    p = Path(video_path)
    if not p.exists():
        return {"frame_idx": frame_idx, "error": 1.0, "artifact_detected": True}

    with VideoReader(p, start_frame=frame_idx) as reader:
        for idx, frame in reader.frames(max_frames=1):
            std_dev = float(np.std(frame))
            mean_val = float(np.mean(frame))
            is_clear = std_dev >= 30.0
            return {
                "frame_idx": idx,
                "mean": round(mean_val, 2),
                "std": round(std_dev, 2),
                "is_clear": is_clear,
                "artifact_detected": not is_clear,
            }
    return {"frame_idx": frame_idx, "error": 1.0}
