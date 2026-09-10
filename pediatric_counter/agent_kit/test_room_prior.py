from ultralytics import YOLO
import numpy as np
from pediatric_counter.io.video_reader import VideoReader
from pediatric_counter.app.pipeline import _extract_crop
from pediatric_counter.app.config import load_config

cfg = load_config('pediatric_counter/configs/room_default.yaml')
base = YOLO('computer_vision_models/yolo26/base_model/yolo26s.pt')
clf = YOLO('computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt')

def classify_person_candidate(crop, box_coords, frame_shape, classifier, prior="adult_dominant"):
    x1, y1, x2, y2 = box_coords
    w_box = x2 - x1
    h_box = y2 - y1
    h_frame, w_frame = frame_shape[:2]
    h_ratio = h_box / max(1.0, h_frame)
    box_area = w_box * h_box
    
    # 1. Anthropometric scale analysis
    is_tall_adult = h_box >= 250 and ((h_ratio >= 0.19 and h_box >= 280) or (h_ratio >= 0.25))
    is_sitting_adult = (box_area >= 50000 and (w_box >= 180 or h_box >= 260))
    is_pediatric_scale = (h_box <= 220 and box_area <= 45000)

    # 2. Crop classifier evidence
    valid_child_conf = 0.0
    valid_adult_conf = 0.0
    
    if classifier is not None:
        res = classifier(crop, conf=0.10, verbose=False)[0]
        crop_area = crop.shape[0] * crop.shape[1]
        
        for b in res.boxes:
            c_name = classifier.names.get(int(b.cls[0]))
            c_conf = float(b.conf[0])
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            b_cov = ((bx2 - bx1) * (by2 - by1)) / max(1.0, crop_area)
            
            if c_name == "child":
                # Discard partial-torso child hits on tall or sitting adults
                if b_cov >= 0.35 and not is_tall_adult and not is_sitting_adult:
                    valid_child_conf = max(valid_child_conf, c_conf)
            elif c_name == "adult":
                valid_adult_conf = max(valid_adult_conf, c_conf)

    # 3. Probabilistic synthesis conditioned on room demographic prior
    if is_tall_adult or is_sitting_adult:
        prob = 0.10
    elif prior == "child_dominant":
        if is_pediatric_scale:
            prob = 0.85 if valid_child_conf > 0.0 else 0.75
        elif valid_child_conf > 0.0:
            prob = 0.60 + 0.30 * valid_child_conf
        else:
            prob = 0.20  # large person in classroom is teacher
    elif prior == "adult_dominant":
        # In hospital OPD, adult is null hypothesis. Only confident pediatric-scale triggers child
        if is_pediatric_scale and valid_child_conf >= 0.80:
            prob = 0.75
        else:
            prob = 0.15
    else:
        # Neutral
        if is_pediatric_scale and valid_child_conf > 0.0:
            prob = 0.70
        else:
            prob = 0.30

    return float(np.clip(prob, 0.05, 0.95))

videos = [
    ("Hospital OTMC Frame 14 (Ground truth: 0 Children, 12 Adults)", "assets/cctv_videos/Hospital OTMC GF OPD_Hospital_Hospital_20260616150730_20260616163715_117801518.mp4", 14, "adult_dominant"),
    ("Kindergarten Frame 0 (Ground truth: 5 Children, 1 Adult)", "assets/cctv_videos/cctv_kindergarten_classroom.mp4", 0, "child_dominant"),
    ("Child Room Frame 0 (Ground truth: 1 Child, 0 Adults)", "assets/cctv_videos/cctv_child_room_drawers.mp4", 0, "child_dominant"),
]

for label, vpath, start, prior in videos:
    print(f"\n=================== {label} (prior={prior}) ===================")
    with VideoReader(vpath, start_frame=start) as r:
        for idx, frame in r.frames(max_frames=1):
            res = base(frame, classes=[0], conf=0.30, verbose=False)[0]
            children_count = 0
            adult_count = 0
            for i, b in enumerate(res.boxes):
                box = tuple(b.xyxy[0].tolist())
                crop = _extract_crop(frame, box, cfg)
                prob = classify_person_candidate(crop, box, frame.shape, clf, prior=prior)
                label_pred = "CHILD" if prob >= 0.60 else "ADULT"
                if label_pred == "CHILD":
                    children_count += 1
                else:
                    adult_count += 1
                w_box = box[2] - box[0]
                h_box = box[3] - box[1]
                print(f"  Person {i} ({w_box:.0f}x{h_box:.0f}px): P(child)={prob:.2f} -> {label_pred}")
            print(f">> Total Detected: {children_count} Children, {adult_count} Adults")
