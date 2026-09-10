from ultralytics import YOLO
import numpy as np
from pediatric_counter.io.video_reader import VideoReader
from pediatric_counter.app.pipeline import _extract_crop
from pediatric_counter.app.config import load_config

cfg = load_config('pediatric_counter/configs/room_default.yaml')
base = YOLO('computer_vision_models/yolo26/base_model/yolo26s.pt')
clf = YOLO('computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt')

def classify_person_candidate(crop, box_coords, frame_shape, classifier):
    x1, y1, x2, y2 = box_coords
    w_box = x2 - x1
    h_box = y2 - y1
    h_frame, w_frame = frame_shape[:2]
    h_ratio = h_box / max(1.0, h_frame)
    box_area = w_box * h_box
    
    # 1. Anthropometric scale analysis
    # Standing adult check: tall vertical ratio
    is_tall_adult = (h_ratio >= 0.22 and h_box >= 320) or (h_ratio >= 0.28)
    # Sitting adult check: massive area / wide profile in high-res CCTV
    is_sitting_adult = (box_area >= 65000 and (w_box >= 220 or h_box >= 280))
    # Pediatric scale check: small relative height and small area
    is_pediatric_scale = (h_ratio <= 0.18 and h_box <= 230 and box_area <= 45000)

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
                # Only accept child detection if it covers at least 35% of crop AND is not a known large adult
                if b_cov >= 0.35 and not is_tall_adult and not is_sitting_adult:
                    valid_child_conf = max(valid_child_conf, c_conf)
            elif c_name == "adult":
                valid_adult_conf = max(valid_adult_conf, c_conf)

    # 3. Probabilistic synthesis
    if is_tall_adult or is_sitting_adult:
        # Strong physical constraint: Adult
        prob = 0.15
        if valid_adult_conf > 0.0:
            prob = min(prob, 1.0 - valid_adult_conf)
    elif is_pediatric_scale:
        # Strong physical scale: Child
        if valid_child_conf > 0.0:
            prob = 0.75 + 0.20 * valid_child_conf
        else:
            # Model didn't fire due to low-res / blur in crop, but scale is unmistakably child
            prob = 0.70
    elif valid_child_conf > 0.0 or valid_adult_conf > 0.0:
        prob = 0.50 + 0.40 * (valid_child_conf - valid_adult_conf)
    else:
        # Ambiguous / mid-scale
        prob = 0.35  # Leans adult in surveillance

    return float(np.clip(prob, 0.05, 0.95))

videos = [
    ("Hospital OTMC Frame 14 (Ground truth: 0 Children, 12 Adults)", "assets/cctv_videos/Hospital OTMC GF OPD_Hospital_Hospital_20260616150730_20260616163715_117801518.mp4", 14),
    ("Kindergarten Frame 0 (Ground truth: 5 Children, 1 Adult)", "assets/cctv_videos/cctv_kindergarten_classroom.mp4", 0),
    ("Child Room Frame 0 (Ground truth: 1 Child, 0 Adults)", "assets/cctv_videos/cctv_child_room_drawers.mp4", 0),
]

for label, vpath, start in videos:
    print(f"\n=================== {label} ===================")
    with VideoReader(vpath, start_frame=start) as r:
        for idx, frame in r.frames(max_frames=1):
            res = base(frame, classes=[0], conf=0.30, verbose=False)[0]
            children_count = 0
            adult_count = 0
            for i, b in enumerate(res.boxes):
                box = tuple(b.xyxy[0].tolist())
                crop = _extract_crop(frame, box, cfg)
                prob = classify_person_candidate(crop, box, frame.shape, clf)
                label_pred = "CHILD" if prob >= 0.60 else "ADULT"
                if label_pred == "CHILD":
                    children_count += 1
                else:
                    adult_count += 1
                w_box = box[2] - box[0]
                h_box = box[3] - box[1]
                print(f"  Person {i} ({w_box:.0f}x{h_box:.0f}px): P(child)={prob:.2f} -> {label_pred}")
            print(f">> Total Detected: {children_count} Children, {adult_count} Adults")
