from ultralytics import YOLO
from pediatric_counter.io.video_reader import VideoReader
from pediatric_counter.app.pipeline import _extract_crop
from pediatric_counter.app.config import load_config

cfg = load_config('pediatric_counter/configs/room_default.yaml')
base = YOLO('computer_vision_models/yolo26/base_model/yolo26s.pt')
m_fine = YOLO('computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt')

videos = [
    ("Hospital OTMC Frame 14 (Adults Only)", "assets/cctv_videos/Hospital OTMC GF OPD_Hospital_Hospital_20260616150730_20260616163715_117801518.mp4", 14),
    ("Kindergarten Frame 0 (5 Kids, 1 Adult Teacher)", "assets/cctv_videos/cctv_kindergarten_classroom.mp4", 0),
    ("Child Room Frame 0 (Toddler)", "assets/cctv_videos/cctv_child_room_drawers.mp4", 0),
]

for label, vpath, start in videos:
    print(f"\n=================== {label} ===================")
    with VideoReader(vpath, start_frame=start) as r:
        for idx, frame in r.frames(max_frames=1):
            H, W = frame.shape[:2]
            res = base(frame, classes=[0], conf=0.30, verbose=False)[0]
            for i, b in enumerate(res.boxes):
                box = tuple(b.xyxy[0].tolist())
                h_box = box[3] - box[1]
                w_box = box[2] - box[0]
                crop = _extract_crop(frame, box, cfg)
                
                rf = m_fine(crop, conf=0.10, verbose=False)[0]
                child_boxes = [cb for cb in rf.boxes if m_fine.names.get(int(cb.cls[0])) == 'child']
                
                det_info = []
                for cb in child_boxes:
                    c_conf = float(cb.conf[0])
                    cx1, cy1, cx2, cy2 = cb.xyxy[0].tolist()
                    c_area = (cx2 - cx1) * (cy2 - cy1)
                    crop_area = crop.shape[0] * crop.shape[1]
                    cov = c_area / max(1.0, crop_area)
                    det_info.append(f"conf={c_conf:.2f}, cov={cov:.1%}")
                
                print(f"Person {i}: frame_box={w_box:.0f}x{h_box:.0f} (h/H={h_box/H:.2f}) -> {det_info}")
