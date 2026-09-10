"""
Baseline pipeline — Stable v1 Core Orchestrator with Decoupled Plugin Architecture.

Processing loop per frame:
  1. Read frame (VideoReader or LiveCameraReader)
  2. Detect persons (Ultralytics YOLO -> Supervision Detections)
  3. Track with ByteTrack (Supervision ByteTrack)
  4. For each tracked person: extract crop, classify demographic
  5. Advance lifecycle state machine
  6. Update distinct child and adult counters
  7. Dispatch per-frame analytical payload (FrameResult) to PluginManager
  8. Handle two-way ControlSignals (e.g. QUIT, PAUSE, STEP from Live HUD)
  9. At EOF: broadcast PipelineSummary to plugins and release all resources
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

import cv2
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from pediatric_counter.app.config import PipelineConfig
from pediatric_counter.counting.distinct_counter import DistinctChildCounter
from pediatric_counter.counting.events import parse_cctv_start_time
from pediatric_counter.counting.lifecycle import LifecycleManager
from pediatric_counter.evaluation.run_manifest import create_manifest, save_manifest
from pediatric_counter.io.video_reader import VideoReader
from pediatric_counter.plugins import (
    ControlSignal,
    FrameResult,
    JSONSummaryPlugin,
    LiveHUDPlugin,
    PerFrameCSVPlugin,
    PipelinePlugin,
    PipelineSummary,
    PluginManager,
    TimelineExporterPlugin,
    VideoRecorderPlugin,
    KidsSievePlugin,
    ModelShadowPlugin,
    PosePlugin,
    ReIDPlugin,
)
from pediatric_counter.vision.crop import (
    direct_resize_crop,
    expand_box,
    letterbox_crop,
    seconds_to_frames,
)

logger = logging.getLogger("pediatric_counter.pipeline")


def _load_detector(cfg: PipelineConfig):
    from ultralytics import YOLO
    p = Path(cfg.models.detector_path)
    if not p.exists():
        for candidate in [
            Path("computer_vision_models/yolo26/onnx/yolo26s.onnx"),
            Path("computer_vision_models/yolo26/base_model/yolo26s.pt"),
            Path(str(p).replace(".pt", ".onnx")),
            Path(str(p).replace(".onnx", ".pt")),
        ]:
            if candidate.exists():
                p = candidate
                break

    if p.suffix.lower() == ".onnx":
        m = YOLO(str(p), task="detect")
    else:
        m = YOLO(str(p))
    if hasattr(m, "overrides"):
        m.overrides["verbose"] = False
    return m


def _load_tracker(cfg: PipelineConfig, fps: float = 25.0):
    backend = str(cfg.tracking.backend).lower()
    import supervision as sv
    import yaml

    track_act = cfg.tracking.new_track_threshold
    match_th = cfg.tracking.match_threshold
    lost_buf = int(cfg.tracking.max_lost_time_seconds * fps)

    cfg_file = getattr(cfg.tracking, "tracker_config", None)
    if cfg_file and Path(cfg_file).exists():
        try:
            with open(cfg_file, "r") as f:
                y = yaml.safe_load(f)
                if isinstance(y, dict):
                    if "new_track_thresh" in y:
                        track_act = float(y["new_track_thresh"])
                    if "match_thresh" in y:
                        match_th = float(y["match_thresh"])
                    if "track_buffer" in y:
                        lost_buf = max(lost_buf, int(y["track_buffer"]))
        except Exception:
            pass

    tracker_cls = getattr(sv.ByteTrack, "wrapped", sv.ByteTrack)
    logger.info(f"Initialized tracking backend '{backend}' with activation={track_act}, match={match_th}, buffer={lost_buf}")
    return tracker_cls(
        track_activation_threshold=track_act,
        lost_track_buffer=lost_buf,
        minimum_matching_threshold=match_th,
        frame_rate=int(fps),
    )


def _load_classifier(cfg: PipelineConfig):
    if not cfg.models.classifier_path:
        return None
    path = Path(cfg.models.classifier_path)
    if not path.exists():
        for candidate in [
            Path("computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-model.onnx"),
            Path("computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt"),
            Path(str(path).replace(".pt", ".onnx")),
            Path(str(path).replace(".onnx", ".pt")),
        ]:
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        return None
    if path.suffix in (".pt", ".onnx"):
        from ultralytics import YOLO
        m = YOLO(str(path), task="detect")
        m.overrides["verbose"] = False
        return m
    import torch
    model = torch.load(str(path), map_location="cpu")
    model.eval()
    return model


def _extract_crop(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    cfg: PipelineConfig,
) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(box, cfg.crop.margin_fraction, w, h)
    raw = frame[y1:y2, x1:x2]
    if raw.size == 0:
        return np.zeros((cfg.crop.input_size, cfg.crop.input_size, 3), dtype=np.uint8)
    if cfg.crop.resize_mode == "letterbox":
        return letterbox_crop(raw, cfg.crop.input_size, cfg.crop.padding_value)
    return direct_resize_crop(raw, cfg.crop.input_size)


def _classify_crop(
    crop: np.ndarray,
    classifier,
    cfg: PipelineConfig,
    detector_cls_id: int | None = None,
    detector_conf: float = 1.0,
    detector_names: dict[int, str] | None = None,
    box_coords: tuple[float, float, float, float] | None = None,
    frame_shape: tuple[int, ...] | None = None,
    track_max_h_ratio: float | None = None,
) -> float:
    """Return posterior P(child) in [0.0, 1.0] calibrated against anthropometric scale and room prior."""
    prior = getattr(cfg.room, "demographic_prior", "adult_dominant")

    # 1. Anthropometric scale analysis from full-frame CCTV bounding box
    is_tall_adult = False
    is_sitting_adult = False
    is_pediatric_scale = False

    if box_coords is not None and frame_shape is not None:
        x1, y1, x2, y2 = box_coords
        w_box = max(1.0, x2 - x1)
        h_box = max(1.0, y2 - y1)
        h_frame = float(frame_shape[0])
        w_frame = float(frame_shape[1])
        h_ratio = h_box / max(1.0, h_frame)
        peak_h_ratio = max(h_ratio, track_max_h_ratio or 0.0)

        # Scale-invariant Anthropometric Analysis:
        if prior == "child_dominant":
            # In Kindergarten and Playroom Drawers, children are <= 28% of vertical frame height
            # Adults standing are >= 32% of frame height (Teacher: 33-42%, Drawers adult: 45-64%)
            is_tall_adult = (peak_h_ratio >= 0.32)
            is_pediatric_scale = (h_ratio <= 0.28)
            is_sitting_adult = False
        else:
            is_tall_adult = (peak_h_ratio >= 0.22)
            is_pediatric_scale = (peak_h_ratio <= 0.18)
            is_sitting_adult = (peak_h_ratio >= 0.20 and (w_box / max(1.0, w_frame) >= 0.08))


    # 2. Extract valid detections from crop classifier
    valid_child_conf = 0.0
    valid_adult_conf = 0.0

    if classifier is not None:

        try:
            from ultralytics import YOLO
            if isinstance(classifier, YOLO):
                res = classifier(crop, conf=0.10, verbose=False)[0]
                crop_area = max(1.0, float(crop.shape[0] * crop.shape[1]))

                for b in res.boxes:
                    c_name = str(classifier.names.get(int(b.cls[0]))).lower()
                    c_conf = float(b.conf[0])
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    b_cov = ((bx2 - bx1) * (by2 - by1)) / crop_area

                    if c_name == "child":
                        # Require at least 35% crop coverage to reject partial-torso child hits on tall/sitting adults
                        if b_cov >= 0.35 and not is_tall_adult and not is_sitting_adult:
                            valid_child_conf = max(valid_child_conf, c_conf)
                    elif c_name == "adult":
                        valid_adult_conf = max(valid_adult_conf, c_conf)
        except Exception:
            pass

        try:
            import torch
            import torchvision.transforms.functional as TF
            tensor = TF.to_tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
            with torch.no_grad():
                logits = classifier(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            child_idx = cfg.models.classifier_classes.index("child")
            return float(probs[child_idx])
        except Exception:
            pass

    # 3. Probabilistic synthesis conditioned on room demographic prior
    if is_tall_adult or is_sitting_adult:
        # Strong physical constraint: Adult
        prob = 0.10
        if valid_adult_conf > 0.0:
            prob = min(prob, 1.0 - valid_adult_conf)
    elif prior == "child_dominant":
        # In kindergarten/playroom, pediatric scale defaults to Child even if crop detector missed low-res box
        if is_pediatric_scale:
            prob = 0.85 if valid_child_conf > 0.0 else 0.75
        elif valid_child_conf > 0.0:
            prob = 0.60 + 0.30 * valid_child_conf
        else:
            prob = 0.20  # Large individual in classroom is teacher
    elif prior == "adult_dominant":
        # In hospital OPD, adult is null hypothesis ($H_0 = Adult$).
        # Requires pediatric scale and high classifier confidence to promote to child
        if is_pediatric_scale and valid_child_conf >= 0.80:
            prob = 0.75
        else:
            prob = 0.15
    else:
        # Neutral baseline
        if is_pediatric_scale and valid_child_conf > 0.0:
            prob = 0.70
        else:
            prob = 0.30

    # Direct resolution fallback if detector is demographic-aware
    if detector_cls_id is not None:
        cls_name = ""
        if detector_names and detector_cls_id in detector_names:
            cls_name = str(detector_names[detector_cls_id]).lower()
        elif cfg.models.classifier_classes and 0 <= detector_cls_id < len(cfg.models.classifier_classes):
            cls_name = str(cfg.models.classifier_classes[detector_cls_id]).lower()

        if cls_name == "child":
            return max(0.55, detector_conf)
        elif cls_name == "adult":
            return min(0.45, max(0.01, 1.0 - detector_conf))

    return float(np.clip(prob, 0.05, 0.95))


def run_pipeline(
    cfg: PipelineConfig,
    plugins: Optional[List[PipelinePlugin]] = None,
) -> PipelineSummary:
    """
    Execute the core counting pipeline with decoupled plugin orchestration.

    Parameters:
        cfg: Validated PipelineConfig object.
        plugins: Optional list of additional custom or third-party PipelinePlugin instances.
    """
    import supervision as sv

    output_dir = Path(cfg.artifacts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v_str = str(cfg.room.video_path)
    is_live_camera = (
        v_str.isdigit()
        or v_str.startswith("camera:")
        or v_str.startswith("rtsp://")
        or v_str.startswith("http://")
    )
    if is_live_camera:
        from pediatric_counter.io.camera import LiveCameraReader
        cam_src = int(v_str.replace("camera:", "")) if v_str.replace("camera:", "").isdigit() else v_str
        reader_context = LiveCameraReader(source=cam_src)
    else:
        reader_context = VideoReader(cfg.room.video_path, start_frame=cfg.room.start_frame)

    # ── Initialize Plugin Manager ─────────────────────────────────────────────
    plugin_manager = PluginManager()

    # Core system plugins
    plugin_manager.register(PerFrameCSVPlugin())
    plugin_manager.register(JSONSummaryPlugin())

    cctv_time = None
    if is_live_camera:
        cctv_time = datetime.now()
    else:
        cctv_time = parse_cctv_start_time(cfg.room.video_path)
    plugin_manager.register(TimelineExporterPlugin(cctv_start_time=cctv_time))

    if cfg.artifacts.save_annotated_video:
        plugin_manager.register(VideoRecorderPlugin())

    if cfg.artifacts.live_view:
        plugin_manager.register(
            LiveHUDPlugin(
                scale=cfg.artifacts.live_view_scale,
                initial_speed=getattr(cfg.artifacts, "playback_speed", 1.0),
            )
        )

    # Configurable optional plugins
    kids_sieve_cfg = getattr(cfg, "kids_sieve", None)
    if kids_sieve_cfg and getattr(kids_sieve_cfg, "enabled", False):
        plugin_manager.register(
            KidsSievePlugin(
                model_path=kids_sieve_cfg.model_path,
                confidence_threshold=kids_sieve_cfg.confidence_threshold,
                enabled=True,
            )
        )

    model_shadow_cfg = getattr(cfg, "model_shadow", None)
    if model_shadow_cfg and getattr(model_shadow_cfg, "enabled", False):
        plugin_manager.register(
            ModelShadowPlugin(
                model_path=model_shadow_cfg.model_path,
                confidence_threshold=model_shadow_cfg.confidence_threshold,
                enabled=True,
                output_filename=model_shadow_cfg.output_filename,
            )
        )

    pose_cfg = getattr(cfg, "pose", None)
    if pose_cfg and getattr(pose_cfg, "enabled", False):
        plugin_manager.register(
            PosePlugin(
                model_path=pose_cfg.model_path,
                confidence_threshold=pose_cfg.confidence_threshold,
                enabled=True,
                output_filename=pose_cfg.output_filename,
            )
        )

    reid_cfg = getattr(cfg, "reid", None)
    if reid_cfg and getattr(reid_cfg, "enabled", False):
        plugin_manager.register(
            ReIDPlugin(
                enabled=True,
                mode=reid_cfg.mode,
                similarity_threshold=reid_cfg.similarity_threshold,
                gallery_path=reid_cfg.gallery_path,
                output_filename=reid_cfg.output_filename,
                deep_model_path=reid_cfg.deep_model_path,
                use_gpu=reid_cfg.use_gpu,
            )
        )

    # Register any custom third-party plugins passed by caller
    if plugins:
        for p in plugins:
            plugin_manager.register(p)

    with reader_context as reader:
        fps = reader.fps if cfg.room.fps == "auto" else float(cfg.room.fps)
        max_lost_frames = seconds_to_frames(cfg.tracking.max_lost_time_seconds, fps)
        max_frames = cfg.room.max_frames

        # Run manifest
        manifest = create_manifest(cfg, fps, reader.total_frames)
        save_manifest(manifest, output_dir)

        # Notify plugins of session start
        frame_shape = (reader.height, reader.width, 3)
        plugin_manager.broadcast_start(
            output_dir=output_dir,
            fps=fps,
            total_frames=reader.total_frames,
            frame_shape=frame_shape,
            config=cfg,
        )

        model = _load_detector(cfg)
        tracker = _load_tracker(cfg, fps=fps)
        classifier = _load_classifier(cfg)

        lifecycle = LifecycleManager(
            min_confirmed_observations=cfg.lifecycle.min_confirmed_observations,
            confirmation_window_frames=cfg.lifecycle.confirmation_window_frames,
            min_child_probability=cfg.lifecycle.min_child_probability,
            label_window_size=cfg.lifecycle.label_window_size,
            uncertain_margin=cfg.lifecycle.uncertain_margin,
            max_lost_frames=max_lost_frames,
            enable_stitching=getattr(cfg.lifecycle, "enable_stitching", True),
        )
        counter = DistinctChildCounter(

            count_adults=cfg.counting.count_adults,
            count_children=cfg.counting.count_children,
        )

        total_to_process = min(reader.total_frames, max_frames) if max_frames else reader.total_frames
        
        # Resolution-adaptive visual annotations: prevents oversized labels on low-res (e.g. 270p) feeds
        text_scale = max(0.32, min(0.60, float(reader.height) / 1080.0 * 0.55))
        text_thickness = 1 if reader.height < 600 else 2
        box_annotator = sv.BoxAnnotator(thickness=max(1, int(reader.height / 540)))
        label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=text_thickness, text_padding=3)

        # Detect active LiveHUDPlugin to coordinate hardware playback speed with reader frame skipping
        hud_plugin = None
        for p in getattr(plugin_manager, "_plugins", []):
            if type(p).__name__ == "LiveHUDPlugin":
                hud_plugin = p
                break

        frame_idx = cfg.room.start_frame
        wall_clock_start_time: Optional[float] = None
        wall_clock_start_frame: Optional[int] = None
        summary: Optional[PipelineSummary] = None

        try:
            with Progress(SpinnerColumn(), *Progress.get_default_columns(),
                          TimeElapsedColumn()) as progress:
                task = progress.add_task("Processing frames…", total=total_to_process)

                for frame_idx, frame in reader.frames(max_frames=max_frames):
                    # ── 1. Detection ──────────────────────────────────────────
                    # Feed detections above track_low_threshold to ByteTrack so partially occluded subjects are tracked
                    det_conf = min(0.20, cfg.tracking.track_low_threshold) if cfg.tracking.track_low_threshold else 0.20
                    results = model(
                        frame,
                        classes=cfg.models.detector_class_ids,
                        conf=det_conf,
                        verbose=False,
                    )[0]
                    sv_dets = sv.Detections.from_ultralytics(results)

                    # ── 2. Tracking ───────────────────────────────────────────
                    tracked = tracker.update_with_detections(sv_dets)

                    # ── 3. Classification ─────────────────────────────────────
                    active_ids: set[int] = set()
                    child_probs: dict[int, float] = {}
                    tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
                    boxes_list: list[tuple[int, int, int, int]] = []
                    boxes_dict: dict[int, tuple[float, float, float, float]] = {}

                    for i, tid in enumerate(tracker_ids):
                        if tid is None:
                            continue
                        tid_int = int(tid)
                        active_ids.add(tid_int)
                        box_coords = tracked.xyxy[i].tolist()
                        box_tuple = (float(box_coords[0]), float(box_coords[1]), float(box_coords[2]), float(box_coords[3]))
                        boxes_list.append((int(box_coords[0]), int(box_coords[1]), int(box_coords[2]), int(box_coords[3])))
                        boxes_dict[tid_int] = box_tuple

                        canon_id = lifecycle.get_canonical_id(tid_int)
                        rec = lifecycle._tracks.get(canon_id)
                        track_peak_h = rec.max_height_ratio if rec else None

                        # Amortized inference: reuse cached probability for confirmed tracks, re-evaluating every 5 frames
                        if rec and len(rec.child_probs) >= 5 and (frame_idx % 5 != 0):
                            prob = rec.child_probs[-1]
                        else:
                            crop = _extract_crop(frame, box_tuple, cfg)
                            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else None
                            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                            prob = _classify_crop(
                                crop,
                                classifier,
                                cfg,
                                detector_cls_id=cls_id,
                                detector_conf=conf,
                                detector_names=getattr(model, "names", None),
                                box_coords=box_tuple,
                                frame_shape=frame.shape,
                                track_max_h_ratio=track_peak_h,
                            )
                        child_probs[tid_int] = prob
                        child_probs[canon_id] = prob

                    # ── 4. Lifecycle & Distinct Counting ──────────────────────
                    records = lifecycle.update(
                        active_ids,
                        child_probs,
                        frame_idx,
                        bounding_boxes=boxes_dict,
                        frame_shape=frame.shape,
                    )
                    counter.update(records)

                    # ── 5. Prepare FrameResult ────────────────────────────────
                    labels_dict: dict[int, str] = {}
                    states_dict: dict[int, str] = {}
                    overlay_labels: list[str] = []
                    canonical_tracked_ids: list[int] = []

                    for tid in tracker_ids:
                        if tid is None:
                            continue
                        raw_tid_int = int(tid)
                        canon_tid = lifecycle.get_canonical_id(raw_tid_int)
                        canonical_tracked_ids.append(canon_tid)

                        rec = records.get(canon_tid)
                        lbl = rec.track_label if rec and rec.track_label else "detecting"
                        st = rec.state.value if rec else "tentative"
                        prob = child_probs.get(raw_tid_int, child_probs.get(canon_tid, 0.5))
                        labels_dict[canon_tid] = lbl
                        states_dict[canon_tid] = st

                        # Bounding box tag: e.g. "#4 child 94%"
                        prob_pct = int(prob * 100) if lbl == "child" else int((1.0 - prob) * 100)
                        overlay_labels.append(f"#{canon_tid} {lbl} ({prob_pct}%)")

                    # Draw base supervision annotations onto isolated frame copy
                    annotated_frame = frame.copy()
                    if len(tracker_ids) > 0:
                        annotated_frame = box_annotator.annotate(annotated_frame, tracked)
                        if overlay_labels:
                            annotated_frame = label_annotator.annotate(
                                annotated_frame, tracked, labels=overlay_labels
                            )

                    # Compute CCTV timestamp string if available
                    cctv_time_str = None
                    if cctv_time:
                        from datetime import timedelta
                        current_cctv_time = cctv_time + timedelta(seconds=(frame_idx / fps))
                        cctv_time_str = current_cctv_time.strftime("%Y-%m-%d %H:%M:%S")

                    frame_result = FrameResult(
                        frame_idx=frame_idx,
                        raw_frame=frame,
                        annotated_frame=annotated_frame,
                        tracked_ids=canonical_tracked_ids,
                        bounding_boxes=boxes_list,
                        track_labels=labels_dict,
                        child_probabilities=child_probs,
                        track_states=states_dict,
                        counted_child_count=counter.distinct_child_count,
                        counted_adult_count=counter.distinct_adult_count,
                        total_count=counter.total_count,
                        fps=fps,
                        cctv_timestamp_str=cctv_time_str,
                    )


                    # ── 6. Broadcast to Plugins ───────────────────────────────
                    signal = plugin_manager.broadcast_frame(frame_result)
                    if signal == ControlSignal.QUIT:
                        print("\n[INFO] Pipeline execution terminated early by user.")
                        break

                    # Fast-forward frames or adaptively synchronize with real-time camera clock
                    if hud_plugin and getattr(hud_plugin, "enabled", True):
                        current_speed = getattr(hud_plugin, "playback_speed", 1.0)
                        if current_speed > 1.25:
                            skip_n = max(0, int(round(current_speed)) - 1)
                            if skip_n > 0:
                                skipped = reader.skip(skip_n)
                                progress.advance(task, skipped)
                        elif getattr(cfg.artifacts, "realtime_sync", False) and current_speed >= 0.9:
                            if wall_clock_start_time is None:
                                wall_clock_start_time = time.perf_counter()
                                wall_clock_start_frame = frame_idx
                            else:
                                elapsed_real_sec = time.perf_counter() - wall_clock_start_time
                                elapsed_video_sec = (frame_idx - wall_clock_start_frame) / fps
                                lag_sec = elapsed_real_sec - elapsed_video_sec
                                is_live_input = str(cfg.room.video_path).startswith("camera:") or getattr(reader, "is_live", False)
                                if is_live_input and lag_sec > 0.08:  # live camera: drop stale frames to keep up with sensor buffer
                                    skip_frames = min(int(lag_sec * fps), 4)
                                    if skip_frames > 0:
                                        skipped = reader.skip(skip_frames)
                                        progress.advance(task, skipped)
                                elif not is_live_input and lag_sec < -0.01:  # recorded video: throttle if running faster than 1.0x
                                    time.sleep(min(0.2, abs(lag_sec)))

                    progress.advance(task)

        except Exception as e:
            plugin_manager.broadcast_error(e)
            raise e
        finally:
            lifecycle.retire_all()

            # Compile summary
            summary = PipelineSummary(
                run_id=manifest["run_id"],
                output_dir=output_dir,
                distinct_child_count=counter.distinct_child_count,
                distinct_adult_count=counter.distinct_adult_count,
                total_distinct_count=counter.total_count,
                counted_child_ids=counter.state.counted_child_ids,
                counted_adult_ids=counter.state.counted_adult_ids,
                uncertain_ids=counter.state.uncertain_ids,
                total_frames_processed=max(0, frame_idx + 1 - cfg.room.start_frame),
                fps=fps,
                config_hash=manifest["config_hash"],
            )

            # Broadcast session finish and close plugins
            plugin_manager.broadcast_finish(summary)
            plugin_manager.close_all()

        # Terminal reporting
        print(f"\n[SUCCESS] Distinct individuals counted:")
        print(f"  - Children: {counter.distinct_child_count}")
        print(f"  - Adults:   {counter.distinct_adult_count}")
        print(f"  - Total:    {counter.total_count}")

        if summary and summary.events:
            print(f"\n[TIMESTAMPS] Found {len(summary.events)} discrete sighting event(s):")
            for ev in summary.events:
                wc_info = f" (CCTV Time: {ev.wall_clock_start} - {ev.wall_clock_end})" if ev.wall_clock_start else ""
                print(f"  * Track #{ev.track_id} [{ev.demographic.upper()}]: {ev.start_time} - {ev.end_time}{wc_info} | frames {ev.start_frame}-{ev.end_frame} ({ev.duration_seconds}s)")
            print(f"  --> Saved timeline to: {output_dir / 'detection_timeline.md'}")

        print(f"[ARTIFACTS] Saved to: {output_dir}")
        return summary
