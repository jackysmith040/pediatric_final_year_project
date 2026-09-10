"""
Capture Real, Authentic Full-Color Inference Screenshots Directly from Production Pipeline.
Eliminates:
1. Gray-screen codec artifact (uses production VideoReader with DPB keyframe pre-roll).
2. Child spam / misclassification (uses production LifecycleManager, Bayesian consensus, and spatial priors).
3. Inaccurate Live HUD counts (renders LiveHUD with true DistinctChildCounter outputs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pediatric_counter.app.config import PipelineConfig
from pediatric_counter.app.pipeline import run_pipeline
from pediatric_counter.plugins.base import ControlSignal, FrameResult, PipelinePlugin
from pediatric_counter.plugins.live_hud import LiveHUDPlugin

FIGURES_DIR = REPO_ROOT / "thesis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class SnapshotPlugin(PipelinePlugin):
    """Intercepts the real pipeline at a target frame and saves a broadcast-quality annotated HUD frame."""

    name = "snapshot_capture"
    priority = 100

    def __init__(self, target_frame: int, output_path: Path):
        self.target_frame = target_frame
        self.output_path = output_path
        self.captured = False
        self._hud = LiveHUDPlugin()

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if result.frame_idx >= self.target_frame and not self.captured:
            raw_h, raw_w = result.annotated_frame.shape[:2]
            
            # Upscale low-res feeds (e.g. 270p) for publication clarity, or scale 2.7K UHD down slightly for sharp readability
            if raw_w < 1280:
                target_w = 1280
                target_h = int(raw_h * (1280.0 / raw_w))
                display_frame = cv2.resize(result.annotated_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            elif raw_w > 1920:
                target_w = 1920
                target_h = int(raw_h * (1920.0 / raw_w))
                display_frame = cv2.resize(result.annotated_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                display_frame = result.annotated_frame.copy()

            # Render the authentic glassmorphic HUD status cards directly on top
            self._hud._render_hud(display_frame, result)

            cv2.imwrite(str(self.output_path), display_frame)
            self.captured = True
            print(f"--> [CAPTURED AUTHENTIC SNAPSHOT] Frame {result.frame_idx} saved to: {self.output_path}")
            print(f"    Counts on HUD: Children={result.counted_child_count}, Adults={result.counted_adult_count}, Total={result.total_count}")
        return ControlSignal.CONTINUE


def capture_feed(
    video_path: Path,
    start_frame: int,
    max_frames: int,
    target_frame: int,
    out_path: Path,
    room_id: str = "general",
    demographic_prior: str = "adult_dominant",
):
    print(f"\n=======================================================")
    print(f"Running Authentic Pipeline on: {video_path.name}")
    print(f"Start Frame: {start_frame} | Max: {max_frames} | Target: {target_frame}")
    print(f"=======================================================")

    cfg = PipelineConfig.from_yaml(REPO_ROOT / "pediatric_counter" / "configs" / "room_default.yaml")
    cfg.room.video_path = video_path
    cfg.room.id = room_id
    cfg.room.demographic_prior = demographic_prior
    cfg.room.start_frame = start_frame
    cfg.room.max_frames = max_frames
    cfg.artifacts.output_dir = REPO_ROOT / "runs" / "thesis_snapshots"
    cfg.artifacts.live_view = False
    cfg.artifacts.save_annotated_video = False
    cfg.artifacts.save_per_frame_csv = False
    cfg.artifacts.save_summary_json = False

    snapshot = SnapshotPlugin(target_frame=target_frame, output_path=out_path)
    run_pipeline(cfg, plugins=[snapshot])


def main():
    # 1. Kindergarten Classroom (Child-dominant: 5 kids, 1 teacher)
    cctv_kindergarten = REPO_ROOT / "assets" / "cctv_videos" / "cctv_kindergarten_classroom.mp4"
    if cctv_kindergarten.exists():
        capture_feed(
            video_path=cctv_kindergarten,
            start_frame=260,
            max_frames=90,
            target_frame=340,
            out_path=FIGURES_DIR / "fig2_kindergarten_inference.png",
            room_id="kindergarten",
            demographic_prior="child_dominant",
        )

    # 2. Child Room Drawers (Child-dominant: 2 kids, 1 mother)
    cctv_drawers = REPO_ROOT / "assets" / "cctv_videos" / "cctv_child_room_drawers.mp4"
    if cctv_drawers.exists():
        capture_feed(
            video_path=cctv_drawers,
            start_frame=650,
            max_frames=110,
            target_frame=740,
            out_path=FIGURES_DIR / "fig3_drawers_inference.png",
            room_id="drawers",
            demographic_prior="child_dominant",
        )

    # 3. Hospital Outpatient Queue (Midday Shift: Adult-dominant: 12 adults, 0 kids)
    cctv_hospital_opd = REPO_ROOT / "assets" / "cctv_videos" / "Hospital OTMC GF OPD_Hospital_Hospital_20260616095600_20260616113920_117658515.mp4"
    if cctv_hospital_opd.exists():
        capture_feed(
            video_path=cctv_hospital_opd,
            start_frame=430,
            max_frames=30,
            target_frame=450,
            out_path=FIGURES_DIR / "fig4_hospital_opd_inference.png",
            room_id="hospital_opd",
            demographic_prior="adult_dominant",
        )

    # 4. Hospital Pharmacy Lobby (Adult-dominant: 6 adults, 0 kids)
    cctv_hospital_pharmacy = REPO_ROOT / "assets" / "cctv_videos" / "Hospital_Old_GF_Pharmacy_Hospital_20260708075415_20260708080722 (1).mp4"
    if cctv_hospital_pharmacy.exists():
        capture_feed(
            video_path=cctv_hospital_pharmacy,
            start_frame=0,
            max_frames=35,
            target_frame=25,
            out_path=FIGURES_DIR / "fig5_hospital_pharmacy_inference.png",
            room_id="hospital_pharmacy",
            demographic_prior="adult_dominant",
        )

    print("\n[SUCCESS] All authentic pipeline snapshots captured and saved to thesis/figures/!")


if __name__ == "__main__":
    main()
