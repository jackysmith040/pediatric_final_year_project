from ultralytics import YOLO
from pediatric_counter.io.video_reader import VideoReader
from pediatric_counter.app.pipeline import _extract_crop
from pediatric_counter.app.config import load_config

cfg = load_config('pediatric_counter/configs/room_default.yaml')
base = YOLO('computer_vision_models/yolo26/base_model/yolo26s.pt')
m_fine = YOLO('computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt')
m_dist = YOLO('computer_vision_models/yolo26/distilled_model/best.pt')

videos = [
    ("Hospital OTMC Frame 14", "assets/cctv_videos/Hospital OTMC GF OPD_Hospital_Hospital_20260616150730_20260616163715_117801518.mp4", 14),
    ("Kindergarten Frame 0", "assets/cctv_videos/cctv_kindergarten_classroom.mp4", 0),
    ("Child Room Frame 0", "assets/cctv_videos/cctv_child_room_drawers.mp4", 0),
]

for label, vpath, start in videos:
    print(f"\n=================== {label} ===================")
    with VideoReader(vpath, start_frame=start) as r:
        for idx, frame in r.frames(max_frames=1):
            res = base(frame, classes=[0], conf=0.30, verbose=False)[0]
            print(f"Base detector found {len(res.boxes)} people in frame:")
            for i, b in enumerate(res.boxes):
                box = tuple(b.xyxy[0].tolist())
                crop = _extract_crop(frame, box, cfg)
                
                rf = m_fine(crop, conf=0.10, verbose=False)[0]
                fine_dets = [(m_fine.names[int(x.cls[0])], round(float(x.conf[0]), 2)) for x in rf.boxes]
                
                rd = m_dist(crop, conf=0.10, verbose=False)[0]
                dist_dets = [(m_dist.names[int(x.cls[0])], round(float(x.conf[0]), 2)) for x in rd.boxes]
                
                h_box = int(box[3] - box[1])
                w_box = int(box[2] - box[0])
                print(f"  Person {i} ({w_box}x{h_box}px at y={int(box[1])}): fine_tune={fine_dets} | distilled={dist_dets}")
