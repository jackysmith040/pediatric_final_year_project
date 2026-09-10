import cv2
from ultralytics import YOLO
from pediatric_counter.io.video_reader import VideoReader

shadow_model = YOLO("computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-smaller-dataset-trained.onnx", task="detect")
main_model = YOLO("computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt", task="detect")
base = YOLO("computer_vision_models/yolo26/base_model/yolo26s.pt")

print("=" * 60)
print("COMPARING SHADOW MODEL VS MAIN MODEL ON CROPS")
print("=" * 60)

print("\n--- Kindergarten Classroom (Frame 10) ---")
with VideoReader("assets/cctv_videos/cctv_kindergarten_classroom.mp4", start_frame=10) as r:
    for idx, f in r.frames(max_frames=1):
        dets = base(f, classes=[0], conf=0.20, verbose=False)[0]
        for i, b in enumerate(dets.boxes):
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            crop = f[y1:y2, x1:x2]
            
            res_shadow = shadow_model(crop, conf=0.10, verbose=False)[0]
            preds_shadow = [(shadow_model.names[int(box.cls[0])], round(float(box.conf[0]), 3)) for box in res_shadow.boxes]

            res_main = main_model(crop, conf=0.10, verbose=False)[0]
            preds_main = [(main_model.names[int(box.cls[0])], round(float(box.conf[0]), 3)) for box in res_main.boxes]

            print(f"Crop #{i} ({x2-x1}x{y2-y1}px):")
            print(f"  - Shadow Model (smaller-dataset-trained.onnx): {preds_shadow if preds_shadow else 'NO DETECTIONS'}")
            print(f"  - Main Model   (pediatric-model.pt):          {preds_main if preds_main else 'NO DETECTIONS'}")

print("\n--- Playroom Drawers (Frame 700 - Child + Adult present) ---")
with VideoReader("assets/cctv_videos/cctv_child_room_drawers.mp4", start_frame=700) as r:
    for idx, f in r.frames(max_frames=1):
        dets = base(f, classes=[0], conf=0.20, verbose=False)[0]
        for i, b in enumerate(dets.boxes):
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            crop = f[y1:y2, x1:x2]
            
            res_shadow = shadow_model(crop, conf=0.10, verbose=False)[0]
            preds_shadow = [(shadow_model.names[int(box.cls[0])], round(float(box.conf[0]), 3)) for box in res_shadow.boxes]

            res_main = main_model(crop, conf=0.10, verbose=False)[0]
            preds_main = [(main_model.names[int(box.cls[0])], round(float(box.conf[0]), 3)) for box in res_main.boxes]

            print(f"Crop #{i} ({x2-x1}x{y2-y1}px):")
            print(f"  - Shadow Model (smaller-dataset-trained.onnx): {preds_shadow if preds_shadow else 'NO DETECTIONS'}")
            print(f"  - Main Model   (pediatric-model.pt):          {preds_main if preds_main else 'NO DETECTIONS'}")
