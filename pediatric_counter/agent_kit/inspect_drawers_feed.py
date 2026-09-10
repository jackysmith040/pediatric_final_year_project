from ultralytics import YOLO
from pediatric_counter.io.video_reader import VideoReader
from pediatric_counter.app.pipeline import _extract_crop, _classify_crop
from pediatric_counter.app.config import load_config

cfg = load_config('pediatric_counter/configs/room_default.yaml')
cfg.room.demographic_prior = 'child_dominant'
base = YOLO('computer_vision_models/yolo26/base_model/yolo26s.pt')
clf = YOLO('computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt')
clf.overrides['verbose'] = False

vpath = 'assets/cctv_videos/cctv_child_room_drawers.mp4'

for target_frame in [625, 692]:
    print(f"\n================ Frame {target_frame} ================")
    with VideoReader(vpath, start_frame=target_frame) as r:
        for idx, frame in r.frames(max_frames=1):
            H, W = frame.shape[:2]
            print(f"Video native dimensions: {W}x{H}")
            res = base(frame, classes=[0], conf=0.30, verbose=False)[0]
            print(f"Found {len(res.boxes)} people:")
            for i, b in enumerate(res.boxes):
                box = tuple(b.xyxy[0].tolist())
                w_box = box[2] - box[0]
                h_box = box[3] - box[1]
                h_ratio = h_box / H
                box_area = w_box * h_box
                crop = _extract_crop(frame, box, cfg)
                prob = _classify_crop(crop, clf, cfg, box_coords=box, frame_shape=frame.shape)
                label = "CHILD" if prob >= 0.60 else "ADULT"
                print(f"  Person {i} at ({box[0]:.0f}, {box[1]:.0f}) size {w_box:.0f}x{h_box:.0f} (h/H={h_ratio:.2f}, area={box_area:.0f}): P(child)={prob:.2f} -> {label}")
