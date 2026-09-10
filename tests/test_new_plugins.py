"""
Unit tests for KidsSievePlugin and ModelShadowPlugin.
"""
from pathlib import Path
import numpy as np
import pytest

from pediatric_counter.plugins.base import ControlSignal, FrameResult, PipelineSummary
from pediatric_counter.plugins.kids_sieve import KidsSievePlugin
from pediatric_counter.plugins.model_shadow import ModelShadowPlugin
from pediatric_counter.plugins.pose import PosePlugin


def test_kids_sieve_disabled():
    plugin = KidsSievePlugin(enabled=False)
    plugin.on_start(Path("runs/test"), fps=25.0, total_frames=10, frame_shape=(270, 480, 3))
    assert plugin._model is None

    dummy_frame = np.zeros((270, 480, 3), dtype=np.uint8)
    res = FrameResult(
        frame_idx=0,
        raw_frame=dummy_frame,
        annotated_frame=dummy_frame,
        tracked_ids=[1],
        bounding_boxes=[(10, 10, 50, 50)],
        track_labels={1: "child"},
        child_probabilities={1: 0.8},
        track_states={1: "confirmed"},
        counted_child_count=1,
        counted_adult_count=0,
        total_count=1,
        fps=25.0,
    )
    sig = plugin.on_frame(res)
    assert sig == ControlSignal.CONTINUE
    assert "kids_sieve" not in res.extra


def test_model_shadow_disabled():
    plugin = ModelShadowPlugin(enabled=False)
    plugin.on_start(Path("runs/test"), fps=25.0, total_frames=10, frame_shape=(270, 480, 3))
    assert plugin._model is None

    dummy_frame = np.zeros((270, 480, 3), dtype=np.uint8)
    res = FrameResult(
        frame_idx=0,
        raw_frame=dummy_frame,
        annotated_frame=dummy_frame,
        tracked_ids=[1],
        bounding_boxes=[(10, 10, 50, 50)],
        track_labels={1: "child"},
        child_probabilities={1: 0.8},
        track_states={1: "confirmed"},
        counted_child_count=1,
        counted_adult_count=0,
        total_count=1,
        fps=25.0,
    )
    sig = plugin.on_frame(res)
    assert sig == ControlSignal.CONTINUE


def test_pose_plugin_disabled():
    plugin = PosePlugin(enabled=False)
    plugin.on_start(Path("runs/test"), fps=25.0, total_frames=10, frame_shape=(270, 480, 3))
    assert plugin._model is None

    dummy_frame = np.zeros((270, 480, 3), dtype=np.uint8)
    res = FrameResult(
        frame_idx=0,
        raw_frame=dummy_frame,
        annotated_frame=dummy_frame,
        tracked_ids=[1],
        bounding_boxes=[(10, 10, 50, 50)],
        track_labels={1: "child"},
        child_probabilities={1: 0.8},
        track_states={1: "confirmed"},
        counted_child_count=1,
        counted_adult_count=0,
        total_count=1,
        fps=25.0,
    )
    sig = plugin.on_frame(res)
    assert sig == ControlSignal.CONTINUE


def test_pose_plugin_anatomical_analysis():
    plugin = PosePlugin(enabled=True)
    # Synthetic 17-point keypoints array
    kpts = np.zeros((17, 2), dtype=np.float32)
    kconf = np.ones((17,), dtype=np.float32)

    # Shoulders (5, 6) at y=20
    kpts[5] = [40, 20]
    kpts[6] = [60, 20]

    # Hips (11, 12) at y=60 (torso length = 40)
    kpts[11] = [45, 60]
    kpts[12] = [55, 60]

    # Ankles (15, 16) at y=100 (leg length = 40)
    kpts[15] = [45, 100]
    kpts[16] = [55, 100]

    # Torso/Leg ratio = 40 / 40 = 1.0 (Toddler morphology >= 0.80)
    analysis = plugin._analyze_skeleton(kpts, kconf, crop_height=110)
    assert analysis is not None
    assert analysis["torso_length_px"] == 40.0
    assert analysis["leg_length_px"] == 40.0
    assert analysis["torso_to_leg_ratio"] == 1.0
    assert analysis["pose_predicted_demographic"] == "child"

    # Adult morphology: longer legs (ankles at y=140, leg length = 80)
    kpts[15] = [45, 140]
    kpts[16] = [55, 140]
    # Torso/Leg ratio = 40 / 80 = 0.50 (Adult morphology < 0.80)
    adult_analysis = plugin._analyze_skeleton(kpts, kconf, crop_height=150)
    assert adult_analysis is not None
    assert adult_analysis["torso_to_leg_ratio"] == 0.50
    assert adult_analysis["pose_predicted_demographic"] == "adult"

