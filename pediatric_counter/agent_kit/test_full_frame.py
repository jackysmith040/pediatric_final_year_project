from ultralytics import YOLO
from pediatric_counter.io.video_reader import VideoReader

def test_hospital():
    m = YOLO('computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt')
    vpath = 'assets/cctv_videos/Hospital OTMC GF OPD_Hospital_Hospital_20260616150730_20260616163715_117801518.mp4'
    with VideoReader(vpath, start_frame=14) as r:
        for idx, f in r.frames(max_frames=1):
            res = m(f, conf=0.25, verbose=False)[0]
            print(f'Full-frame Hospital Frame 14 detections with pediatric-model.pt ({len(res.boxes)} found):')
            for b in res.boxes:
                print(f'  cls={m.names[int(b.cls[0])]}, conf={float(b.conf[0]):.2f}, box={[int(x) for x in b.xyxy[0].tolist()]}')

def test_kindergarten():
    m = YOLO('computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt')
    vpath = 'assets/cctv_videos/cctv_kindergarten_classroom.mp4'
    with VideoReader(vpath, start_frame=0) as r:
        for idx, f in r.frames(max_frames=1):
            res = m(f, conf=0.25, verbose=False)[0]
            print(f'Full-frame Kindergarten Frame 0 detections with pediatric-model.pt ({len(res.boxes)} found):')
            for b in res.boxes:
                print(f'  cls={m.names[int(b.cls[0])]}, conf={float(b.conf[0]):.2f}, box={[int(x) for x in b.xyxy[0].tolist()]}')

if __name__ == '__main__':
    test_hospital()
    print('-'*50)
    test_kindergarten()
