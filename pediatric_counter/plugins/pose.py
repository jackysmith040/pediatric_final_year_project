"""
Pose Estimation & Anatomical Analysis Plugin.

Evaluates human skeletal keypoints (e.g. yolo26s-pose.pt / yolo26s-pose.onnx)
to compute scale-invariant anatomical ratios (torso length, leg length,
torso-to-leg ratio) without altering the core distinct counting state machine.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pediatric_counter.plugins.base import ControlSignal, FrameResult, PipelinePlugin, PipelineSummary

logger = logging.getLogger("pediatric_counter.plugins.pose")

# Standard COCO 17 Keypoint Indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Facial features
    (5, 6),                                   # Shoulder girdle
    (5, 7), (7, 9),                           # Left arm
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12),                         # Torso sides
    (11, 12),                                 # Pelvis
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]


class PosePlugin(PipelinePlugin):
    """
    Decoupled plugin for evaluating pose estimation models and anatomical ratios.
    """

    name: str = "pose"
    priority: int = 42

    def __init__(
        self,
        model_path: str = "computer_vision_models/yolo26/onnx/pose_models/yolo26s-pose.onnx",
        confidence_threshold: float = 0.25,
        enabled: bool = False,
        draw_on_frame: bool = True,
        output_filename: str = "pose_analysis.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled
        self.draw_on_frame = draw_on_frame
        self.output_filename = output_filename
        self._model = None
        self._output_dir: Optional[Path] = None
        self.pose_records: List[Dict[str, Any]] = []
        self.total_inference_time_ms: float = 0.0
        self.total_evaluations: int = 0

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        self._output_dir = output_dir
        if not self.enabled:
            logger.info("PosePlugin is disabled.")
            return

        # Auto-fallback between ONNX and PyTorch weights
        if not self.model_path.exists():
            for candidate in [
                Path("computer_vision_models/yolo26/onnx/pose_models/yolo26s-pose.onnx"),
                Path("computer_vision_models/yolo26/base_model/pose_models/yolo26s-pose.pt"),
                Path("assets/export_from_ultralytics_platform/yolo26s-pose.onnx"),
            ]:
                if candidate.exists():
                    self.model_path = candidate
                    break

        if not self.model_path.exists():
            logger.warning(
                f"PosePlugin model not found at {self.model_path}. "
                f"Place yolo26-pose model at this path to activate pose estimation."
            )
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path), task="pose")
            if hasattr(self._model, "overrides"):
                self._model.overrides["verbose"] = False
            logger.info(f"PosePlugin successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"PosePlugin failed to load model: {e}")
            self._model = None

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if not self.enabled or self._model is None:
            return ControlSignal.CONTINUE

        raw_frame = result.raw_frame
        h_frame, w_frame = raw_frame.shape[:2]

        for i, tid in enumerate(result.tracked_ids):
            if i >= len(result.bounding_boxes):
                continue
            box = result.bounding_boxes[i]
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(w_frame - 1, x1))
            y1 = max(0, min(h_frame - 1, y1))
            x2 = max(x1 + 1, min(w_frame, x2))
            y2 = max(y1 + 1, min(h_frame, y2))

            crop = raw_frame[y1:y2, x1:x2]
            if crop.size == 0 or (x2 - x1) < 15 or (y2 - y1) < 15:
                continue

            t0 = time.perf_counter()
            try:
                preds = self._model(crop, conf=self.confidence_threshold, verbose=False)[0]
                dt_ms = (time.perf_counter() - t0) * 1000.0
                self.total_inference_time_ms += dt_ms
                self.total_evaluations += 1

                if preds.keypoints is not None and len(preds.keypoints) > 0:
                    # Extract 17 keypoint coordinates [17, 2] and confidences [17]
                    kpts = preds.keypoints.xy[0].cpu().numpy()
                    kconf = preds.keypoints.conf[0].cpu().numpy() if preds.keypoints.conf is not None else None

                    analysis = self._analyze_skeleton(kpts, kconf, crop_height=(y2 - y1))
                    if analysis:
                        analysis.update({
                            "frame_idx": result.frame_idx,
                            "track_id": tid,
                            "baseline_label": result.track_labels.get(tid, "detecting"),
                            "latency_ms": round(dt_ms, 2),
                        })
                        self.pose_records.append(analysis)

                    # Draw skeleton directly onto the display canvas if requested
                    if self.draw_on_frame and result.annotated_frame is not None:
                        is_child = (analysis and analysis.get("pose_predicted_demographic") == "child") or (result.track_labels.get(tid) == "child")
                        bone_color = (118, 230, 0) if is_child else (255, 176, 0)
                        joint_color = (0, 255, 255) if is_child else (255, 77, 124)

                        # Draw skeletal bones
                        for p1_idx, p2_idx in COCO_SKELETON:
                            if p1_idx < len(kpts) and p2_idx < len(kpts):
                                pt1 = kpts[p1_idx]
                                pt2 = kpts[p2_idx]
                                conf1 = kconf[p1_idx] if kconf is not None else 1.0
                                conf2 = kconf[p2_idx] if kconf is not None else 1.0
                                if conf1 >= 0.25 and conf2 >= 0.25 and (pt1[0] > 0 or pt1[1] > 0) and (pt2[0] > 0 or pt2[1] > 0):
                                    g_pt1 = (int(x1 + pt1[0]), int(y1 + pt1[1]))
                                    g_pt2 = (int(x2 - (x2 - x1) + pt2[0]), int(y1 + pt2[1]))
                                    cv2.line(result.annotated_frame, g_pt1, g_pt2, bone_color, 2, cv2.LINE_AA)

                        # Draw skeletal joints
                        for pt_idx, pt in enumerate(kpts):
                            pt_conf = kconf[pt_idx] if kconf is not None else 1.0
                            if pt_conf >= 0.25 and (pt[0] > 0 or pt[1] > 0):
                                g_pt = (int(x1 + pt[0]), int(y1 + pt[1]))
                                cv2.circle(result.annotated_frame, g_pt, 3, joint_color, -1, cv2.LINE_AA)

                        # Draw anatomical ratio badge above tracklet
                        if analysis and analysis.get("torso_to_leg_ratio") is not None:
                            ratio_val = analysis["torso_to_leg_ratio"]
                            badge_text = f"Torso/Leg: {ratio_val:.2f}"
                            cv2.putText(
                                result.annotated_frame,
                                badge_text,
                                (x1, max(14, y1 - 8)),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.38,
                                bone_color,
                                1,
                                cv2.LINE_AA,
                            )
            except Exception as e:
                logger.debug(f"PosePlugin error on track #{tid}: {e}")

        return ControlSignal.CONTINUE

    def _analyze_skeleton(
        self,
        kpts: np.ndarray,
        kconf: Optional[np.ndarray],
        crop_height: int,
    ) -> Optional[Dict[str, Any]]:
        """Calculate anatomical proportions from COCO 17 skeletal keypoints."""
        if len(kpts) < 17:
            return None

        # Helper to get valid point if confidence threshold passes
        def get_point(idx: int) -> Optional[np.ndarray]:
            if kconf is not None and kconf[idx] < 0.25:
                return None
            pt = kpts[idx]
            if pt[0] == 0 and pt[1] == 0:
                return None
            return pt

        ls, rs = get_point(L_SHOULDER), get_point(R_SHOULDER)
        lh, rh = get_point(L_HIP), get_point(R_HIP)
        lk, rk = get_point(L_KNEE), get_point(R_KNEE)
        la, ra = get_point(L_ANKLE), get_point(R_ANKLE)

        # Mid-shoulder and mid-hip
        shoulders = [p for p in (ls, rs) if p is not None]
        hips = [p for p in (lh, rh) if p is not None]

        if not shoulders or not hips:
            return None

        mid_shoulder = np.mean(shoulders, axis=0)
        mid_hip = np.mean(hips, axis=0)

        # Torso length (shoulder to hip)
        torso_len = float(np.linalg.norm(mid_shoulder - mid_hip))

        # Leg length (hip to ankle)
        leg_lengths: List[float] = []
        if lh is not None and la is not None:
            leg_lengths.append(float(np.linalg.norm(lh - la)))
        if rh is not None and ra is not None:
            leg_lengths.append(float(np.linalg.norm(rh - ra)))

        leg_len = float(np.mean(leg_lengths)) if leg_lengths else None

        # Compute anatomical ratios
        # Toddlers exhibit cephalocaudal development: higher torso-to-leg ratio (typically >= 0.85)
        # Adults exhibit elongated lower limbs: lower torso-to-leg ratio (typically <= 0.70)
        torso_leg_ratio = (torso_len / leg_len) if (leg_len and leg_len > 0) else None
        torso_crop_ratio = torso_len / max(1.0, crop_height)

        predicted_demographic = "unknown"
        if torso_leg_ratio is not None:
            predicted_demographic = "child" if torso_leg_ratio >= 0.80 else "adult"
        elif torso_crop_ratio is not None:
            predicted_demographic = "child" if torso_crop_ratio >= 0.45 else "adult"

        return {
            "torso_length_px": round(torso_len, 1),
            "leg_length_px": round(leg_len, 1) if leg_len else None,
            "torso_to_leg_ratio": round(torso_leg_ratio, 3) if torso_leg_ratio else None,
            "torso_to_crop_ratio": round(torso_crop_ratio, 3),
            "pose_predicted_demographic": predicted_demographic,
        }

    def on_finish(self, summary: PipelineSummary) -> None:
        if not self.enabled or self._output_dir is None:
            return

        evaluated = len(self.pose_records)
        agreements = sum(
            1 for r in self.pose_records
            if r.get("pose_predicted_demographic") == r.get("baseline_label")
            and r.get("pose_predicted_demographic") in ("child", "adult")
        )
        agreement_rate = (agreements / evaluated) if evaluated > 0 else 0.0
        avg_latency = (self.total_inference_time_ms / max(1, self.total_evaluations))

        report = {
            "model_tested": str(self.model_path.name),
            "model_path": str(self.model_path),
            "total_pose_evaluations": self.total_evaluations,
            "total_valid_skeletons": evaluated,
            "baseline_agreement_rate": round(agreement_rate, 4),
            "average_latency_ms": round(avg_latency, 2),
            "sample_skeletal_analyses": self.pose_records[-30:] if self.pose_records else [],
        }

        out_path = self._output_dir / self.output_filename
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"PosePlugin saved report to {out_path} (Skeletons analyzed: {evaluated})")
        except Exception as e:
            logger.error(f"Failed to write pose analysis report: {e}")
