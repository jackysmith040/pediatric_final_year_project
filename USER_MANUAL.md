# Pediatric and Adult CCTV Counting System: User Manual

This manual explains how to run and control the CCTV counting system.

---

## 1. Overview and Architecture

The software counts distinct children and adults from hospital CCTV video feeds. It uses a two-stage architecture with optional pose validation:

1. **Stage 1: Person Detection and Tracking**
   - Model: `YOLO26s ONNX` (`computer_vision_models/yolo26/onnx/yolo26s.onnx`), class 0 (`person`).
   - The engine uses native AVX2 vector instructions on x86 CPUs. It processes full frames at 7.5 to 8.1 frames per second. It falls back to PyTorch `.pt` if ONNX is missing.
   - It detects people across the full frame. It maintains persistent track IDs with ByteTrack.
2. **Stage 2: Demographic Classification**
   - Model: Fine-tuned `Pediatric-Model ONNX` (`computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-model.onnx`).
   - The model evaluates standardized 224 by 224 pixel crops.
   - It checks crop coverage to prevent partial torso errors.
   - It applies environment priors:
     - **Hospital OPD (`adult_dominant`)**: Default hypothesis is Adult. The system requires high child confidence and small scale to classify a person as a child.
     - **Kindergarten and Playroom (`child_dominant`)**: Default hypothesis is Child. Toddlers on the floor are classified as children. Tall standing people with height ratio above 0.32 are classified as adults.
3. **Optional Stage 3: Skeletal Pose Validation**
   - Model: `YOLO26s Pose ONNX` (`computer_vision_models/yolo26/onnx/pose_models/yolo26s-pose.onnx`).
   - It extracts 17 body keypoints. It calculates the torso-to-leg ratio: 0.80 or higher for toddlers, and 0.70 or lower for adults.
4. **Lifecycle and Deduplication**
   - The state machine tracks states: `tentative`, `confirmed`, `lost`, and `retired`.
   - It blends 70 percent historical consensus with 30 percent rolling mean over 15 frames.
   - It merges adjacent body-part fragments for seated people.
   - It counts each confirmed person exactly once.

---

## 2. Start the System

You can start the system with two methods.

### Method A: Double-Click Launcher (Windows)

1. Open File Explorer.
2. Go to the project directory `pediatric_personal`.
3. Double-click `start.bat`.
4. The system opens the interactive menu.

### Method B: Terminal Command

1. Open a terminal in the project directory.
2. Activate the virtual environment:
   ```bash
   .venv\Scripts\activate
   ```
3. Start the interactive menu:
   ```bash
   python run.py
   ```

---

## 3. Interactive Menu Steps

When you start `run.py`, the terminal guides you through seven steps:

### Step 1: Select Input Feed
- Lists all verified recorded CCTV video files in `assets/cctv_videos/`.
- Scans for connected USB webcams and built-in cameras (`📷 USB Camera #0 [CONNECTED]`).
- Accepts custom hospital RTSP network streams (`🌐 Custom RTSP Stream URL`).
- The system processes arbitrary hospital CCTV video files.

### Step 2: Select Run Mode
- `[1]` Active movement section: Starts at frame 1450 for fast testing.
- `[2]` Quick test: Processes the first 200 frames.
- `[3]` Jump to timestamp: Starts at a specific timestamp or frame index.
- `[4]` Complete shift run: Processes the full video file.

### Step 3: Playback Mode and Live HUD
- `[1]` Real-Time Camera Sync (1.0x): Locks video display to real-world clock.
- `[2]` Maximum Processing Speed (0): Processes all frames as fast as the CPU allows.
- `[3]` Slow-Motion (0.5x): Runs at half speed for detailed observation.
- `[4]` Fast-Forward (2.0x): Runs at double speed.

### Step 4: AI Model Engine
- `[1]` YOLO26s ONNX: Recommended for CPU execution at 7.5 to 8.1 FPS.
- `[2]` YOLO26m ONNX: Higher recall in crowded areas at 2.0 to 2.7 FPS.
- `[3]` YOLO26s PyTorch: PyTorch baseline execution.

### Step 5: Skeletal Pose Estimation Toggle
- Type `y` to enable 17-point skeletal pose estimation.
- Displays body keypoints and torso-to-leg ratios on the live HUD.
- Type `n` to run standard bounding box tracking.

### Step 6: Person Re-Identification (Re-ID)
- Type `y` to enable cross-camera occupant re-identification.
- **Fast CPU Mode (Default)**: Generates 256-dimensional dual-zone HSV color descriptors. Overhead is under 0.2 milliseconds per person crop.
- **Deep Embedding Mode**: Loads deep neural embeddings with ONNX or PyTorch when a GPU is present.
- Registers and updates occupant signatures in `runs/reid_gallery/global_gallery.json`.

### Step 7: Demographic Counting
- Type `y` to count both children and adults.
- Type `n` to count children only.

---

## 4. Live Video Heads-Up Display (HUD) and Controls

When the video window opens, the system displays a semi-transparent HUD:
- **Status Pill (Top-Left)**: Shows `▶ PLAY [speed]x` in green, or `|| PAUSED` in orange.
- **Children Card**: Shows the distinct child count in green.
- **Adults Card**: Shows the distinct adult count in cyan.
- **Total Card**: Shows the total distinct count in purple.
- **Frame and Clock (Top-Right)**: Displays the frame index and CCTV timestamp.
- **Bottom Control Strip**: Displays current keyboard shortcuts.

Click the video window first to give it focus. Use these keys to control playback:

| Key | Action | Description |
| :---: | :--- | :--- |
| `Space` | Pause / Resume | Freezes or continues video playback. |
| `n` | Step Frame | Advances the video by one frame when paused. |
| `+` or `f` | Increase Speed | Increases playback speed by 1.5 times up to 16.0x. |
| `-` or `s` | Decrease Speed | Decreases playback speed down to 0.1x. |
| `r` | Reset Speed | Sets playback speed to 1.0x (real-time 25 FPS). |
| `q` | Exit | Closes the window and writes all report files. |

---

## 5. Hardware Scaling: CPU and GPU

The architecture separates core tracking from feature extraction backends.

### Current CPU Execution
- **Engines**: ONNX Runtime with `CPUExecutionProvider` and AVX2 vector instructions.
- **Throughput**: 7.5 to 8.1 FPS on full 2.7K UHD frames.
- **Latency**: ByteTrack association takes under 1.5 ms. Re-ID color descriptor takes under 0.2 ms.
- **Camera Buffer**: Threaded capture prevents frame drops on live streams.

### GPU Acceleration Setup
When you connect an NVIDIA GPU:
1. Install CUDA packages for ONNX Runtime:
   ```bash
   pip install onnxruntime-gpu
   ```
2. In Step 6 of `run.py`, choose option `[2] Deep Embedding Model`.
3. Provide the path to your ONNX or PyTorch Re-ID model (for example, OSNet).
4. The system detects `CUDAExecutionProvider` automatically and runs embeddings on the GPU.

---

## 6. Direct Command Line Usage

You can run the system directly from the terminal with the `pediatric-counter` command.

### Quick Start with Live Display
```bash
pediatric-counter run --live
```

### Fast Test with Custom Speed
```bash
pediatric-counter run -s 1450 -m 100 --live --speed 1.0
```

### Count Children Only
```bash
pediatric-counter run -s 1450 -m 100 --live --no-count-adults
```

### Run on a Specific Video File
```bash
pediatric-counter run -v "assets/cctv_videos/cctv_child_room_drawers.mp4" -m 100 --live
```

### Run Benchmark Evaluation Across Feeds
Run the evaluation suite on all CCTV videos:
```bash
python run.py evaluate
```
This command writes `runs/evaluations/evaluation_summary.md` and `evaluation_summary.json`.

---

## 7. Output Files and Results

The system writes all run outputs to `runs/otmc_opd_baseline/`:

### 1. `detection_timeline.md`
This file lists each sighting event in a table. It records start time, end time, CCTV clock time, and classification:
- `👶 CHILD`: Confirmed pediatric patient.
- `🧑 ADULT`: Confirmed adult patient, guardian, or staff member.
- `⏱️ DETECTING`: Initial observation frames before confirmation.
- `❓ UNCERTAIN`: Observation with probability close to the decision boundary.

### 2. `summary.json`
This file contains final counts in JSON format:
- `distinct_child_count`: Total distinct children observed.
- `distinct_adult_count`: Total distinct adults observed.
- `total_distinct_count`: Sum of distinct people.
- `total_frames_processed`: Total frames analyzed.

### 3. `per_frame.csv`
This file records all visible people in every frame. It stores bounding box coordinates, track IDs, and confidence values.

### 4. `annotated.mp4`
This file contains the output video with bounding boxes, track IDs, demographic labels, and count cards.

---

## 8. Plugin Architecture

The system uses a decoupled plugin architecture. The core counting engine runs independently of input and output plugins. If an output plugin fails, the core pipeline continues.

### Built-in Plugins
- `PerFrameCSVPlugin`: Writes tracking states to `per_frame.csv`.
- `JSONSummaryPlugin`: Compiles demographic totals to `summary.json`.
- `TimelineExporterPlugin`: Writes sighting intervals to `detection_timeline.md`.
- `VideoRecorderPlugin`: Encodes annotated video frames to `annotated.mp4`.
- `LiveHUDPlugin`: Renders the visualizer and handles keyboard inputs.
- `ReIDPlugin`: Generates appearance descriptors and manages the multi-camera gallery.

### Add a Custom Plugin
Subclass `PipelinePlugin` to create a custom plugin:
```python
from pediatric_counter.plugins import PipelinePlugin, FrameResult, ControlSignal

class HospitalAlertPlugin(PipelinePlugin):
    name = "hospital_alert"
    priority = 50

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if result.counted_child_count > 10:
            print("[ALERT] High pediatric patient volume detected.")
        return ControlSignal.CONTINUE
```

---

## 9. Optional Computer Vision Plugins: CLAHE and SAHI

The repository includes two optional computer vision plugins in `pediatric_counter/plugins/`. Both plugins are disabled by default.

### 1. CLAHE Plugin (`CLAHEPlugin`)
- **Function**: Contrast Limited Adaptive Histogram Equalization improves local contrast. It divides frames into contextual tiles (8 by 8 pixels) and limits contrast expansion to reduce sensor noise.
- **Use Case**: Enhances visibility in dark hallways or under bright sunlight.
- **Configuration** in `pediatric_counter/configs/room_default.yaml`:
  ```yaml
  clahe:
    enabled: false
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  ```

### 2. SAHI Plugin (`SAHIPlugin`)
- **Function**: Slicing Aided Hyper Inference slices high-resolution frames into overlapping tiles. It runs detection on each tile at native resolution and merges bounding boxes with NMS.
- **Use Case**: Detects small toddlers at long distances in 4K wide-angle feeds.
- **Configuration** in `pediatric_counter/configs/room_default.yaml`:
  ```yaml
  sahi:
    enabled: false
    slice_height: 640
    slice_width: 640
    overlap_height_ratio: 0.20
    overlap_width_ratio: 0.20
  ```

---

## 10. Developer Agent Kit

The package `pediatric_counter.agent_kit` provides functions for testing and evaluation:

```python
from pediatric_counter.agent_kit import (
    probe_video,
    probe_frame_clarity,
    inspect_frame_crops,
    quick_run,
)

# 1. Inspect video container and playable status
stats = probe_video("assets/cctv_videos/cctv_kindergarten_classroom.mp4")

# 2. Check if a frame has decoder corruption or gray blocks
clarity = probe_frame_clarity("assets/cctv_videos/cctv_child_room_drawers.mp4", frame_idx=100)
print(clarity["is_clear"])

# 3. Inspect Stage 1 boxes and Stage 2 probabilities
crops = inspect_frame_crops("assets/cctv_videos/cctv_child_room_drawers.mp4", frame_idx=700)

# 4. Run headless test on a video clip
results = quick_run("assets/cctv_videos/cctv_kindergarten_classroom.mp4", max_frames=450)
print(results)
```

---

## 11. Troubleshooting

### Video appears gray after jumping to a timestamp
The video decoder uses keyframe pre-roll for HEVC/H.265 streams. It decodes preceding reference frames before displaying the target timestamp.

### Video window does not respond to key presses
Click inside the video window with your mouse. The window must have focus to receive keyboard events.

### Terminal shows text encoding errors on Windows
Run this command in your terminal before starting:
```bash
chcp 65001
```

### Video file fails to open
Verify that the file path exists in `assets/cctv_videos/`. All six default video files in the repository open cleanly with OpenCV.

---

## 12. Academic Thesis Research Package

All academic thesis documents, LaTeX tables, mathematical formulations, and HIPAA/GDPR compliance reports are in the `thesis/` directory:
- `thesis/EXECUTIVE_THESIS_SUMMARY.md`: Summary of the clinical problem, architecture, and results.
- `thesis/METHODOLOGY_AND_MATHEMATICAL_FORMULATION.md`: Mathematical derivations for tracking, Bayesian priors, and pose ratios.
- `thesis/EMPIRICAL_BENCHMARK_RESULTS.md`: Benchmark tables, latency measurements, and case studies.
- `thesis/GROUND_TRUTH_VERIFICATION_REPORT.md`: Event sighting timelines and verification against human ground truth.
- `thesis/TABLES_AND_LATEX_SNIPPETS.tex`: LaTeX tables for thesis chapters.
- `thesis/ETHICAL_AND_PRIVACY_COMPLIANCE.md`: Healthcare privacy analysis.
