"""
Person Re-Identification (Re-ID) & Multi-Camera Tracking Plugin.

Extracts visual appearance signatures (color distributions, upper/lower torso
descriptors, and optional deep embeddings) to maintain persistent identity
across long-term occlusions and disjoint camera views without altering the
distinct counting core state machine.

Architecture & Future-Proofing:
- Default Fast CPU Mode: Dual-zone Upper/Lower HSV color descriptor + aspect ratio (< 0.8ms).
- GPU / Deep Mode: Pluggable ONNX / PyTorch embedding backbone (OSNet, FastReID, Torbreck)
  with CUDAExecutionProvider support.
- Multi-Camera Global Gallery: Persists occupant appearance profiles to a shared
  gallery, enabling cross-camera verification.
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

logger = logging.getLogger("pediatric_counter.plugins.reid")


class ReIDPlugin(PipelinePlugin):
    """
    Decoupled plugin for appearance-based Re-Identification and cross-camera matching.
    """

    name: str = "reid"
    priority: int = 43

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "advisory",
        similarity_threshold: float = 0.78,
        gallery_path: str = "runs/reid_gallery/global_gallery.json",
        output_filename: str = "reid_analysis.json",
        deep_model_path: str = "",
        use_gpu: bool = False,
    ) -> None:
        self.enabled = enabled
        self.mode = mode
        self.similarity_threshold = float(similarity_threshold)
        self.gallery_path = Path(gallery_path)
        self.output_filename = output_filename
        self.deep_model_path = deep_model_path
        self.use_gpu = use_gpu

        self._deep_session = None
        self._output_dir: Optional[Path] = None
        self._track_signatures: Dict[int, List[np.ndarray]] = {}
        self._track_metadata: Dict[int, Dict[str, Any]] = {}
        self.reid_matches: List[Dict[str, Any]] = []
        self.global_gallery: Dict[str, Any] = {}
        self.total_extractions: int = 0
        self.total_extraction_time_ms: float = 0.0

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int, int]] = None,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not self.enabled:
            return

        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing global multi-camera gallery if available
        if self.gallery_path.exists():
            try:
                with open(self.gallery_path, "r", encoding="utf-8") as f:
                    self.global_gallery = json.load(f)
                logger.info(f"Loaded global Re-ID gallery with {len(self.global_gallery.get('profiles', {}))} profiles.")
            except Exception as e:
                logger.warning(f"Could not load Re-ID gallery at {self.gallery_path}: {e}")
                self.global_gallery = {"profiles": {}, "version": "1.0"}
        else:
            self.global_gallery = {"profiles": {}, "version": "1.0"}

        # Initialize deep embedding engine if path provided (Future GPU/ONNX path)
        if self.deep_model_path and Path(self.deep_model_path).exists():
            try:
                import onnxruntime as ort
                providers = ["CPUExecutionProvider"]
                if self.use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._deep_session = ort.InferenceSession(str(self.deep_model_path), providers=providers)
                logger.info(f"Loaded deep Re-ID model from {self.deep_model_path} with {providers[0]}")
            except Exception as e:
                logger.warning(f"Failed to load deep Re-ID model: {e}. Falling back to Fast HSV descriptor.")
                self._deep_session = None

    def extract_descriptor(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract an appearance signature from a person crop.
        - Deep Model (if loaded): 512-D L2-normalized embedding.
        - Default Fast CPU Mode: Dual-zone Upper/Lower HSV color histogram (256-D L2-norm).
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 15:
            return None

        t0 = time.perf_counter()

        # 1. Deep Feature Path (GPU or CPU ONNX)
        if self._deep_session is not None:
            try:
                # Resize to standard Re-ID input size 256x128
                resized = cv2.resize(crop, (128, 256))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                normed = (rgb - mean) / std
                blob = np.transpose(normed, (2, 0, 1))[np.newaxis, ...]
                input_name = self._deep_session.get_inputs()[0].name
                emb = self._deep_session.run(None, {input_name: blob})[0].flatten()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                self.total_extraction_time_ms += (time.perf_counter() - t0) * 1000
                self.total_extractions += 1
                return emb
            except Exception as e:
                logger.debug(f"Deep Re-ID inference failed: {e}, falling back to HSV")

        # 2. Fast CPU Dual-Zone HSV Path
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape

        # Dual-zone: Upper 60% (shirt/torso/head) vs Lower 40% (pants/shoes)
        split_y = max(1, int(h * 0.60))
        upper_zone = hsv[:split_y, :]
        lower_zone = hsv[split_y:, :]

        # 16 Hue bins, 8 Saturation bins = 128 bins per zone
        hist_upper = cv2.calcHist([upper_zone], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist_lower = cv2.calcHist([lower_zone], [0, 1], None, [16, 8], [0, 180, 0, 256])

        # Normalize each zone
        u_norm = np.linalg.norm(hist_upper)
        l_norm = np.linalg.norm(hist_lower)
        v_upper = (hist_upper / (u_norm + 1e-6)).flatten()
        v_lower = (hist_lower / (l_norm + 1e-6)).flatten()

        # Combine into single 256-D descriptor
        descriptor = np.concatenate([v_upper, v_lower])
        d_norm = np.linalg.norm(descriptor)
        if d_norm > 0:
            descriptor = descriptor / d_norm

        self.total_extraction_time_ms += (time.perf_counter() - t0) * 1000
        self.total_extractions += 1
        return descriptor

    @staticmethod
    def compute_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Compute cosine similarity between two unit-norm descriptors (range: -1.0 to 1.0)."""
        if sig_a is None or sig_b is None:
            return 0.0
        return float(np.dot(sig_a, sig_b))

    def on_frame(self, frame_result: FrameResult) -> Optional[ControlSignal]:
        if not self.enabled:
            return None

        raw_frame = frame_result.raw_frame
        reid_links: Dict[int, Dict[str, Any]] = {}

        for tid, box in zip(frame_result.tracked_ids, frame_result.bounding_boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = raw_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = raw_frame[y1:y2, x1:x2]
            descriptor = self.extract_descriptor(crop)
            if descriptor is None:
                continue

            if tid not in self._track_signatures:
                self._track_signatures[tid] = []
                # First time seeing this track: query against previous tracks in session
                best_match_id = None
                best_sim = 0.0
                for prev_id, sigs in self._track_signatures.items():
                    if prev_id == tid or not sigs:
                        continue
                    mean_sig = np.mean(sigs, axis=0)
                    mean_sig = mean_sig / (np.linalg.norm(mean_sig) + 1e-6)
                    sim = self.compute_similarity(descriptor, mean_sig)
                    if sim > best_sim and sim >= self.similarity_threshold:
                        best_sim = sim
                        best_match_id = prev_id

                if best_match_id is not None:
                    match_record = {
                        "frame_idx": frame_result.frame_idx,
                        "cctv_time": frame_result.cctv_timestamp_str or "",
                        "new_track_id": tid,
                        "matched_track_id": best_match_id,
                        "similarity": round(best_sim, 3),
                        "demographic": frame_result.track_labels.get(tid, "UNKNOWN"),
                    }
                    self.reid_matches.append(match_record)
                    reid_links[tid] = match_record

            # Append to rolling window (keep up to 10 latest snapshots)
            self._track_signatures[tid].append(descriptor)
            if len(self._track_signatures[tid]) > 10:
                self._track_signatures[tid].pop(0)

            # Store metadata
            self._track_metadata[tid] = {
                "demographic": frame_result.track_labels.get(tid, "UNKNOWN"),
                "last_seen_frame": frame_result.frame_idx,
                "last_seen_time": frame_result.cctv_timestamp_str or "",
            }

        # Inject into frame_result extra payload for LiveHUD and downstream export
        frame_result.extra["reid_links"] = reid_links
        return None

    def on_finish(self, summary: PipelineSummary) -> None:
        if not self.enabled or not self._output_dir:
            return

        # Compute stable profile for each confirmed track
        session_profiles: Dict[str, Any] = {}
        for tid, sigs in self._track_signatures.items():
            if not sigs:
                continue
            mean_sig = np.mean(sigs, axis=0)
            mean_sig = (mean_sig / (np.linalg.norm(mean_sig) + 1e-6)).tolist()
            meta = self._track_metadata.get(tid, {})
            session_profiles[str(tid)] = {
                "track_id": tid,
                "demographic": meta.get("demographic", "UNKNOWN"),
                "last_seen_time": meta.get("last_seen_time", ""),
                "signature": mean_sig,
            }

        # Cross-camera matching against global gallery
        cross_camera_matches = []
        global_profiles = self.global_gallery.get("profiles", {})
        for tid_str, s_prof in session_profiles.items():
            s_vec = np.array(s_prof["signature"])
            for g_id, g_prof in global_profiles.items():
                g_vec = np.array(g_prof["signature"])
                sim = self.compute_similarity(s_vec, g_vec)
                if sim >= self.similarity_threshold:
                    cross_camera_matches.append({
                        "session_track_id": int(tid_str),
                        "global_profile_id": g_id,
                        "global_camera_id": g_prof.get("camera_source", "unknown"),
                        "similarity": round(sim, 3),
                        "demographic": s_prof["demographic"],
                    })

        # Register new profiles into global gallery for future runs
        camera_id = summary.run_id
        for tid_str, s_prof in session_profiles.items():
            global_id = f"{camera_id}_track_{tid_str}"
            global_profiles[global_id] = {
                "camera_source": camera_id,
                "track_id": int(tid_str),
                "demographic": s_prof["demographic"],
                "last_seen_time": s_prof["last_seen_time"],
                "signature": s_prof["signature"],
            }
        self.global_gallery["profiles"] = global_profiles
        self.global_gallery["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save global gallery
        try:
            with open(self.gallery_path, "w", encoding="utf-8") as f:
                json.dump(self.global_gallery, f, indent=2)
            logger.info(f"Saved updated Re-ID gallery to {self.gallery_path}")
        except Exception as e:
            logger.error(f"Failed to save Re-ID gallery: {e}")

        # Save session reid_analysis.json
        avg_extract_ms = (
            self.total_extraction_time_ms / self.total_extractions
            if self.total_extractions > 0
            else 0.0
        )
        report_data = {
            "run_id": summary.run_id,
            "total_tracks_profiled": len(session_profiles),
            "intra_session_reid_matches": self.reid_matches,
            "cross_camera_matches": cross_camera_matches,
            "metrics": {
                "total_extractions": self.total_extractions,
                "avg_extraction_time_ms": round(avg_extract_ms, 3),
                "deep_model_active": self._deep_session is not None,
                "gpu_accelerated": self.use_gpu and self._deep_session is not None,
            },
        }

        out_path = self._output_dir / self.output_filename
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Re-ID analysis saved to: {out_path}")
        except Exception as e:
            logger.error(f"Failed to write Re-ID analysis report: {e}")

        # Store in summary.extra
        summary.extra["reid_matches_count"] = len(self.reid_matches)
        summary.extra["cross_camera_matches_count"] = len(cross_camera_matches)

    # Alias for backward compatibility
    on_end = on_finish
