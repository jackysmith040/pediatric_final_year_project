"""
Video reader — thin OpenCV wrapper returning frames as numpy arrays.

Responsibilities:
- Detect or validate FPS (avoids hardcoded assumptions).
- Yield (frame_index, frame_ndarray) pairs.
- Fail gracefully on missing file or unreadable codec.
"""
from __future__ import annotations

import os
import sys

# Silence C-level FFmpeg and OpenCV decoder notices globally
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from pathlib import Path
from typing import Iterator

import cv2
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)

import numpy as np


class VideoReader:
    """Context-manager wrapper around cv2.VideoCapture."""

    def __init__(self, path: str | Path, start_frame: int = 0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._start_frame = start_frame
        self._cap: cv2.VideoCapture | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open video '{self.path}'. The video container may be corrupt, "
                f"truncated, or missing its 'moov' atom. Please check video file integrity."
            )
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Attempt to read one test frame
            ret, _ = self._cap.read()
            if not ret:
                raise RuntimeError(
                    f"Video '{self.path}' opened but contains 0 readable frames "
                    f"(corrupt container or incomplete CCTV export)."
                )
        if self._start_frame > 0:
            self._seek_start_frame(self._start_frame)
        return self

    def _seek_start_frame(self, target_frame: int) -> None:
        """Seek to target_frame using decoder-friendly pre-roll so HEVC/H.264 reference frames are primed.

        Directly seeking to an arbitrary frame in HEVC can land on a P/B-frame without
        its preceding keyframe (I-frame) in the Decoded Picture Buffer (DPB), yielding
        flat-gray (128, 128, 128) unreferenced frames.
        For start frames <= 2500, sequentially grabbing from frame 0 maintains 100% DPB
        reference integrity without FFmpeg reference frame dropping.
        For larger offsets, pre-rolls up to 250 frames (typical CCTV GOP length).
        """
        import os
        try:
            null_fd = os.open(os.devnull, os.O_WRONLY)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(null_fd, 2)
                if target_frame <= 2500:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    for _ in range(target_frame):
                        if not self._cap.grab():
                            break
                else:
                    pre_roll_count = min(250, target_frame)
                    pre_roll_start = max(0, target_frame - pre_roll_count)
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, pre_roll_start)
                    for _ in range(pre_roll_count):
                        if not self._cap.grab():
                            break
            finally:
                os.dup2(old_stderr_fd, 2)
                os.close(null_fd)
                os.close(old_stderr_fd)
        except Exception:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def __exit__(self, *_) -> None:
        if self._cap:
            import os
            try:
                null_fd = os.open(os.devnull, os.O_WRONLY)
                old_stderr_fd = os.dup(2)
                try:
                    os.dup2(null_fd, 2)
                    self._cap.release()
                finally:
                    os.dup2(old_stderr_fd, 2)
                    os.close(null_fd)
                    os.close(old_stderr_fd)
            except Exception:
                self._cap.release()
            self._cap = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        assert self._cap, "Call inside a `with` block"
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 25.0

    @property
    def total_frames(self) -> int:
        assert self._cap, "Call inside a `with` block"
        return max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    @property
    def width(self) -> int:
        assert self._cap, "Call inside a `with` block"
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        assert self._cap, "Call inside a `with` block"
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Frame iteration ───────────────────────────────────────────────────────

    def frames(self, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, bgr_frame) pairs up to max_frames or EOF."""
        assert self._cap, "Call inside a `with` block"
        self._current_frame_idx = self._start_frame
        self._yielded_frames = 0
        while True:
            if max_frames is not None and self._yielded_frames >= max_frames:
                break
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            yield self._current_frame_idx, frame
            self._current_frame_idx += 1
            self._yielded_frames += 1

    def skip(self, count: int) -> int:
        """Fast-forward by skipping `count` frames using cap.grab() without decoding RGB."""
        if not self._cap or count <= 0:
            return 0
        skipped = 0
        for _ in range(count):
            if not self._cap.grab():
                break
            skipped += 1
        if hasattr(self, "_current_frame_idx"):
            self._current_frame_idx += skipped
        if hasattr(self, "_yielded_frames"):
            self._yielded_frames += skipped
        return skipped
