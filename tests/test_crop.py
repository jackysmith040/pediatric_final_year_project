"""
Unit tests for bounding-box geometry utilities.

These tests import only pediatric_counter.vision.crop — no OpenCV,
no torch, no heavy dependencies.  They should run in milliseconds.
"""
import pytest
from pediatric_counter.vision.crop import (
    expand_box,
    frames_to_seconds,
    iou,
    seconds_to_frames,
)


# ──────────────────────────────────────────────────────────────────────────────
# expand_box
# ──────────────────────────────────────────────────────────────────────────────

class TestExpandBox:
    def test_zero_margin_returns_same_box(self):
        box = (10, 20, 110, 70)
        result = expand_box(box, margin=0.0, frame_w=640, frame_h=480)
        assert result == (10, 20, 110, 70)

    def test_10pct_margin_expands_correctly(self):
        # w=100, h=50 → dx=10, dy=5
        box = (50, 100, 150, 150)
        x1, y1, x2, y2 = expand_box(box, margin=0.10, frame_w=640, frame_h=480)
        assert x1 == 40
        assert y1 == 95
        assert x2 == 160
        assert y2 == 155

    def test_clamped_to_frame_left_top(self):
        box = (2, 3, 50, 50)
        x1, y1, x2, y2 = expand_box(box, margin=0.20, frame_w=640, frame_h=480)
        assert x1 == 0   # clamped at 0
        assert y1 == 0   # clamped at 0

    def test_clamped_to_frame_right_bottom(self):
        box = (580, 440, 638, 478)
        x1, y1, x2, y2 = expand_box(box, margin=0.30, frame_w=640, frame_h=480)
        assert x2 == 640  # clamped at frame_w
        assert y2 == 480  # clamped at frame_h

    def test_degenerate_box_raises(self):
        with pytest.raises(ValueError):
            expand_box((10, 10, 10, 10), margin=0.10, frame_w=640, frame_h=480)

    def test_degenerate_inverted_box_raises(self):
        with pytest.raises(ValueError):
            expand_box((100, 100, 50, 50), margin=0.10, frame_w=640, frame_h=480)


# ──────────────────────────────────────────────────────────────────────────────
# seconds_to_frames / frames_to_seconds
# ──────────────────────────────────────────────────────────────────────────────

class TestFpsConversions:
    def test_seconds_to_frames_25fps(self):
        assert seconds_to_frames(1.0, fps=25.0) == 25

    def test_seconds_to_frames_30fps(self):
        assert seconds_to_frames(30.0, fps=30.0) == 900

    def test_seconds_to_frames_zero_returns_at_least_one(self):
        assert seconds_to_frames(0.0, fps=25.0) == 1

    def test_seconds_to_frames_invalid_fps_raises(self):
        with pytest.raises(ValueError):
            seconds_to_frames(1.0, fps=0.0)

    def test_frames_to_seconds_round_trip(self):
        fps = 29.97
        seconds = 10.0
        frames = seconds_to_frames(seconds, fps)
        recovered = frames_to_seconds(frames, fps)
        assert abs(recovered - seconds) < 0.1   # within 100 ms tolerance


# ──────────────────────────────────────────────────────────────────────────────
# IoU
# ──────────────────────────────────────────────────────────────────────────────

class TestIoU:
    def test_identical_boxes_return_one(self):
        box = (10, 10, 50, 50)
        assert iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_return_zero(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        assert iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        # a=(0,0,10,10), b=(5,0,15,10): overlap=(5,0,10,10)=50, union=150
        a = (0, 0, 10, 10)
        b = (5, 0, 15, 10)
        assert iou(a, b) == pytest.approx(50 / 150)

    def test_contained_box(self):
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        result = iou(outer, inner)
        # inner area = 2500, outer = 10000, union = 10000, iou = 2500/10000
        assert result == pytest.approx(0.25)
