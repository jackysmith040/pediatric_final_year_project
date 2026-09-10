import json
import numpy as np
import pytest
from pathlib import Path

from pediatric_counter.plugins.base import FrameResult, PipelineSummary
from pediatric_counter.plugins.reid import ReIDPlugin


def test_reid_plugin_initialization(tmp_path: Path):
    gallery = tmp_path / "gallery.json"
    plugin = ReIDPlugin(
        enabled=True,
        mode="advisory",
        similarity_threshold=0.80,
        gallery_path=str(gallery),
        output_filename="reid_test.json",
    )
    assert plugin.enabled is True
    assert plugin.mode == "advisory"
    assert plugin.similarity_threshold == 0.80
    assert plugin.name == "reid"


def test_reid_descriptor_extraction():
    plugin = ReIDPlugin(enabled=True)

    # Blank / invalid crop returns None
    assert plugin.extract_descriptor(None) is None
    assert plugin.extract_descriptor(np.zeros((10, 10, 3), dtype=np.uint8)) is None

    # Red crop vs Blue crop
    red_crop = np.zeros((100, 50, 3), dtype=np.uint8)
    red_crop[:, :] = [0, 0, 255]  # Red in BGR

    blue_crop = np.zeros((100, 50, 3), dtype=np.uint8)
    blue_crop[:, :] = [255, 0, 0]  # Blue in BGR

    red_sig = plugin.extract_descriptor(red_crop)
    blue_sig = plugin.extract_descriptor(blue_crop)

    assert red_sig is not None
    assert blue_sig is not None
    assert red_sig.shape == (256,)
    assert blue_sig.shape == (256,)

    # Norm must be ~1.0
    assert np.isclose(np.linalg.norm(red_sig), 1.0, atol=1e-3)
    assert np.isclose(np.linalg.norm(blue_sig), 1.0, atol=1e-3)

    # Identical crop should have similarity ~ 1.0
    sim_self = plugin.compute_similarity(red_sig, red_sig)
    assert sim_self > 0.99

    # Distinct colors should have low similarity
    sim_diff = plugin.compute_similarity(red_sig, blue_sig)
    assert sim_diff < 0.20


def test_reid_plugin_lifecycle(tmp_path: Path):
    out_dir = tmp_path / "run_reid"
    gallery_file = tmp_path / "global_gallery.json"

    plugin = ReIDPlugin(
        enabled=True,
        gallery_path=str(gallery_file),
        similarity_threshold=0.75,
        output_filename="reid_analysis.json",
    )

    # Start
    plugin.on_start(output_dir=out_dir, fps=25.0)

    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:200, 50:120] = [0, 128, 255]  # Person 1 (Orange)
    frame[50:200, 200:270] = [255, 128, 0]  # Person 2 (Cyan)

    frame_res = FrameResult(
        frame_idx=0,
        raw_frame=frame,
        annotated_frame=frame.copy(),
        tracked_ids=[1, 2],
        bounding_boxes=[(50, 50, 120, 200), (200, 50, 270, 200)],
        track_labels={1: "adult", 2: "child"},
        child_probabilities={1: 0.05, 2: 0.92},
        track_states={1: "CONFIRMED", 2: "CONFIRMED"},
        counted_child_count=1,
        counted_adult_count=1,
        total_count=2,
        fps=25.0,
        cctv_timestamp_str="08:00:00",
    )

    plugin.on_frame(frame_res)
    assert "reid_links" in frame_res.extra

    # Summary
    summary = PipelineSummary(
        run_id="cam_01",
        output_dir=out_dir,
        distinct_child_count=1,
        distinct_adult_count=1,
        total_distinct_count=2,
        counted_child_ids=[2],
        counted_adult_ids=[1],
        uncertain_ids=[],
        total_frames_processed=1,
        fps=25.0,
    )

    plugin.on_end(summary)

    # Verify output files
    assert (out_dir / "reid_analysis.json").exists()
    assert gallery_file.exists()

    with open(out_dir / "reid_analysis.json") as f:
        data = json.load(f)
        assert data["run_id"] == "cam_01"
        assert data["total_tracks_profiled"] == 2
