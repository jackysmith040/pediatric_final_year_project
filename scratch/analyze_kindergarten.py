import cv2
import supervision as sv
from ultralytics import YOLO
from pediatric_counter.io.video_reader import VideoReader

m_base = YOLO('computer_vision_models/yolo26/base_model/yolo26s.pt')
tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=60, minimum_matching_threshold=0.50)

track_data = {}

with VideoReader('assets/cctv_videos/cctv_kindergarten_classroom.mp4') as r:
    for idx, frame in r.frames():
        res = m_base(frame, classes=[0], conf=0.25, verbose=False)[0]
        sv_dets = sv.Detections.from_ultralytics(res)
        tracked = tracker.update_with_detections(sv_dets)
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                box = tracked.xyxy[i].tolist()
                h = box[3] - box[1]
                w = box[2] - box[0]
                if tid not in track_data:
                    track_data[tid] = {'start': idx, 'end': idx, 'count': 0, 'max_h': h, 'max_w': w, 'h_ratios': []}
                track_data[tid]['end'] = idx
                track_data[tid]['count'] += 1
                track_data[tid]['max_h'] = max(track_data[tid]['max_h'], h)
                track_data[tid]['h_ratios'].append(h / 1280.0)

print('Track summaries for kindergarten classroom:')
for tid, d in sorted(track_data.items()):
    avg_hr = sum(d['h_ratios']) / len(d['h_ratios'])
    max_hr = max(d['h_ratios'])
    start = d['start']
    end = d['end']
    cnt = d['count']
    max_h = d['max_h']
    print(f"Track #{tid}: frames {start}-{end} (hits={cnt}), max_h={max_h:.1f}, max_ratio={max_hr:.2f}, avg_ratio={avg_hr:.2f}")
