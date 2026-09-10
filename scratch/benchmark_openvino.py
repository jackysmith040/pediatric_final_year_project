import time
import cv2
import numpy as np
from ultralytics import YOLO
from pediatric_counter.io.video_reader import VideoReader

def run_benchmark():
    print("=" * 70)
    print("🚀 OPENVINO VS PYTORCH BASE DETECTOR BENCHMARK ON CPU")
    print("=" * 70)

    models_to_test = [
        ("yolo26s.pt (PyTorch)", "computer_vision_models/yolo26/base_model/yolo26s.pt"),
        ("yolo26s (OpenVINO FP16)", "computer_vision_models/yolo26/openvino/base_model/yolo26s_openvino_model"),
        ("yolo26m (OpenVINO FP16)", "computer_vision_models/yolo26/openvino/base_model/yolo26m_openvino_model"),
    ]

    # Load test frames: 1080p Hospital OPD and 1080p Kindergarten
    frames = {}
    with VideoReader("assets/cctv_videos/Hospital OTMC GF OPD_Hospital_Hospital_20260616095600_20260616113920_117658515.mp4", start_frame=430) as r:
        for idx, f in r.frames(max_frames=1):
            frames["Hospital OPD (1080p)"] = f

    with VideoReader("assets/cctv_videos/cctv_kindergarten_classroom.mp4", start_frame=10) as r:
        for idx, f in r.frames(max_frames=1):
            frames["Kindergarten (1080p)"] = f

    for name, path in models_to_test:
        print(f"\nEvaluating: {name}...")
        try:
            m = YOLO(path, task="detect")
            if hasattr(m, "overrides"):
                m.overrides["verbose"] = False
            
            for scene_name, frame in frames.items():
                # Warmup
                _ = m(frame, classes=[0], conf=0.20, verbose=False)[0]
                
                # Benchmark 20 iterations
                times = []
                last_res = None
                for _ in range(20):
                    t0 = time.perf_counter()
                    res = m(frame, classes=[0], conf=0.20, verbose=False)[0]
                    dt = (time.perf_counter() - t0) * 1000.0
                    times.append(dt)
                    last_res = res

                mean_ms = np.mean(times)
                std_ms = np.std(times)
                fps = 1000.0 / mean_ms
                det_count = len(last_res.boxes) if last_res else 0

                print(f"  [{scene_name}]:")
                print(f"    - Latency: {mean_ms:.2f} ms ± {std_ms:.2f} ms")
                print(f"    - Throughput: {fps:.1f} FPS")
                print(f"    - Detections (Person class): {det_count}")
        except Exception as e:
            print(f"  FAILED to evaluate {name}: {e}")

    # Test Pose Models
    print("\n" + "=" * 70)
    print("🤸 POSE ESTIMATION OPENVINO MODELS TEST")
    print("=" * 70)
    pose_models = [
        ("yolo26s-pose (OpenVINO FP16)", "computer_vision_models/yolo26/openvino/pose/yolo26s-pose_openvino_model"),
        ("yolo26m-pose (OpenVINO FP16)", "computer_vision_models/yolo26/openvino/pose/yolo26m-pose_openvino_model"),
    ]

    k_frame = frames.get("Kindergarten (1080p)")
    for name, path in pose_models:
        print(f"\nEvaluating Pose Model: {name}...")
        try:
            pm = YOLO(path, task="pose")
            if hasattr(pm, "overrides"):
                pm.overrides["verbose"] = False
            
            t0 = time.perf_counter()
            pres = pm(k_frame, conf=0.25, verbose=False)[0]
            dt = (time.perf_counter() - t0) * 1000.0

            num_kpts = len(pres.keypoints) if pres.keypoints is not None else 0
            print(f"  - Inference latency: {dt:.2f} ms ({1000.0/dt:.1f} FPS)")
            print(f"  - Skeletons detected: {num_kpts}")
            if num_kpts > 0:
                print(f"  - Keypoint tensor shape: {pres.keypoints.xy.shape}")
        except Exception as e:
            print(f"  FAILED to evaluate {name}: {e}")

if __name__ == "__main__":
    run_benchmark()
