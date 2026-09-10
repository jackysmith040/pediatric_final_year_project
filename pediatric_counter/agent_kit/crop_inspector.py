"""
Agent Kit Crop Inspector.
Runs Stage 1 detection and Stage 2 crop classification on a single frame or video segment,
returning structured bounding boxes, confidence, class, and child probability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import cv2
import numpy as np

from pediatric_counter.app.config import load_config
from pediatric_counter.app.pipeline import _load_detector, _load_classifier, _classify_crop, _extract_crop
from pediatric_counter.io.video_reader import VideoReader


def inspect_frame_crops(
    video_path: str | Path,
    frame_idx: int = 0,
    config_path: str | Path = "pediatric_counter/configs/room_default.yaml",
) -> List[dict[str, Any]]:
    """Inspect person detections and Stage 2 demographic probabilities on a single frame."""
    cfg = load_config(config_path)
    cfg.room.video_path = Path(video_path)
    
    detector = _load_detector(cfg)
    classifier = _load_classifier(cfg)

    with VideoReader(video_path, start_frame=frame_idx) as reader:
        for idx, frame in reader.frames(max_frames=1):
            res = detector(frame, classes=cfg.models.detector_class_ids, conf=0.15, verbose=False)[0]
            boxes = res.boxes
            results: List[dict[str, Any]] = []

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                
                crop = _extract_crop(frame, tuple(xyxy), cfg)
                child_prob = _classify_crop(
                    crop,
                    classifier,
                    cfg,
                    detector_cls_id=cls_id,
                    detector_conf=conf,
                    detector_names=getattr(detector, "names", None),
                    box_coords=tuple(xyxy),
                    frame_shape=frame.shape,
                )
                
                label = "child" if child_prob >= cfg.lifecycle.min_child_probability else "adult"
                results.append({
                    "crop_index": i,
                    "bbox": [round(c, 1) for c in xyxy],
                    "detector_conf": round(conf, 3),
                    "child_prob": round(child_prob, 3),
                    "assigned_label": label,
                    "crop_shape": crop.shape,
                })
            return results
    return []
