from ultralytics import YOLO
import numpy as np
from pediatric_counter.io.video_reader import VideoReader
from pediatric_counter.app.pipeline import _extract_crop
from pediatric_counter.app.config import load_config

cfg = load_config('pediatric_counter/configs/room_default.yaml')
base = YOLO('computer_vision_models/yolo26/base_model/yolo26s.pt')
clf = YOLO('computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt')
clf.overrides['verbose'] = False

def classify_person_candidate(crop, box_coords, frame_shape, classifier, prior="child_dominant"):
    x1, y1, x2, y2 = box_coords
    w_box = max(1.0, x2 - x1)
    h_box = max(1.0, y2 - y1)
    h_frame = float(frame_shape[0])
    w_frame = float(frame_shape[1])
    h_ratio = h_box / max(1.0, h_frame)
    box_area = w_box * h_box
    frame_area = h_frame * w_frame
    area_ratio = box_area / max(1.0, frame_area)
    
    # 1. Scale-invariant Anthropometric Analysis
    # A standing adult occupies at least 22% of the vertical frame in indoor surveillance
    # (Hospital: 0.23-0.28, Kindergarten teacher: 0.35-0.53, Drawers room adult: 0.45)
    is_tall_adult = (h_ratio >= 0.22)
    # Sitting adult: wide profile and significant area ratio
    is_sitting_adult = (area_ratio >= 0.015 and (w_box / max(1.0, w_frame) >= 0.07 or h_ratio >= 0.19))
    # Pediatric scale: small relative vertical height (<= 18% of frame)
    is_pediatric_scale = (h_ratio <= 0.18)

    valid_child_conf = 0.0
    valid_adult_conf = 0.0
    
    if classifier is not None:
        res = classifier(crop, conf=0.10, verbose=False)[0]
        crop_area = max(1.0, float(crop.shape[0] * crop.shape[1]))
        
        for b in res.boxes:
            c_name = classifier.names.get(int(b.cls[0]))
            c_conf = float(b.conf[0])
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            b_cov = ((bx2 - bx1) * (by2 - by1)) / crop_area
            
            if c_name == "child":
                if b_cov >= 0.35 and not is_tall_adult and not is_sitting_adult:
                    valid_child_conf = max(valid_child_conf, c_conf)
            elif c_name == "adult":
                valid_adult_conf = max(valid_adult_conf, c_conf)

    if is_tall_adult or is_sitting_adult:
        prob = 0.10
    elif prior == "child_dominant":
        if is_pediatric_scale:
            prob = 0.85 if valid_child_conf > 0.0 else 0.75
        elif valid_child_conf > 0.0:
            prob = 0.60 + 0.30 * valid_child_conf
        else:
            prob = 0.20
    elif prior == "adult_dominant":
        if is_pediatric_scale and valid_child_conf >= 0.80:
            prob = 0.75
        else:
            prob = 0.15
    else:
        prob = 0.70 if (is_pediatric_scale and valid_child_conf > 0.0) else 0.30

    return float(np.clip(prob, 0.05, 0.95))

# Test on Frame 692 of drawers feed
vpath = 'assets/cctv_videos/cctv_child_room_drawers.mp4'
print("\nTesting scale-invariant classifier on Drawers Frame 692:")
with VideoReader(vpath, start_frame=692) as r:
    for idx, frame in r.frames(max_frames=1):
        res = base(frame, classes=[0], conf=0.30, verbose=False)[0]
        for i, b in enumerate(res.boxes):
            box = tuple(b.xyxy[0].tolist())
            crop = _extract_crop(frame, box, cfg)
            prob = classify_person_candidate(crop, box, frame.shape, clf, prior="child_dominant")
            label = "CHILD" if prob >= 0.60 else "ADULT"
            h_ratio = (box[3] - box[1]) / frame.shape[0]
            print(f"  Person {i} at ({box[0]:.0f}, {box[1]:.0f}) h_ratio={h_ratio:.2f}: P(child)={prob:.2f} -> {label}")
