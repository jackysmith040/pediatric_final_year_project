# Pediatric and Adult CCTV Counting System

This research software counts distinct children and adults from hospital CCTV video feeds. It uses YOLO detection, ByteTrack multi-object tracking, and set-cardinality deduplication. It avoids double-counting people who remain in the camera field of view.

---

## Quick Start

### Method 1: Double-Click (Windows)
Double-click `start.bat` in File Explorer. This opens the interactive menu without typing terminal commands.

### Method 2: Interactive Terminal Menu
```bash
python run.py
```

### Method 3: Direct Command Line
```bash
# Run with live visualizer
pediatric-counter run --live

# Fast test on active movement frames
pediatric-counter run -s 1450 -m 100 --live --speed 1.0
```

---

## System Architecture

The software uses a two-stage architecture with optional pose and re-identification plugins:

1. **Stage 1 (Primary Detector)**: Full-frame person detection and tracking using `YOLO26s ONNX` (`computer_vision_models/yolo26/onnx/yolo26s.onnx`) with native AVX2 CPU acceleration (7.5 to 8.1 FPS) and ByteTrack. It falls back to PyTorch `.pt` if ONNX is missing.
2. **Stage 2 (Demographic Classifier)**: Standardized 224 by 224 pixel person crop evaluation using fine-tuned `Pediatric-Model ONNX` (`computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-model.onnx`). It uses environment-calibrated priors (`adult_dominant` for Hospital OPD vs `child_dominant` for Kindergarten and Playroom).
3. **Stage 3 (Optional Pose Estimation)**: 17-point skeletal pose estimation using `YOLO26s Pose ONNX` (`computer_vision_models/yolo26/onnx/pose_models/yolo26s-pose.onnx`). It calculates biological torso-to-leg ratios: 0.80 or higher for toddlers, and 0.70 or lower for adults.
4. **Stage 4 (Optional Person Re-Identification)**: Appearance-based occupant re-identification across long-term occlusions and multi-camera streams. It generates 256-dimensional dual-zone HSV color descriptors on CPU. It supports deep GPU neural embeddings via ONNX or PyTorch.
5. **Deduplication and Lifecycle**: Finite-state machine (`tentative` -> `confirmed` -> `lost` -> `retired`) with set-cardinality counting. It includes body-part fragmentation deduplication to prevent double-counting seated individuals.

---

## Playback Controls (Live Window)

Click inside the video window to activate keyboard controls:

| Key | Action |
| :--- | :--- |
| `+` / `f` | Increase speed (1.5x up to 16x) |
| `-` / `s` | Decrease speed (down to 0.1x slow-motion) |
| `r` | Reset speed to 1.0x real-time |
| `Space` | Pause or resume video playback |
| `n` | Advance one frame when paused |
| `q` | Exit window and save all output files |

---

## Input Feeds

The system processes recorded CCTV video files, USB webcams, and network RTSP streams:
1. `cctv_child_room_drawers.mp4` — Toddler playroom activity (1.4 MB)
2. `cctv_kindergarten_classroom.mp4` — Kindergarten classroom group activity (2.1 MB)
3. `Hospital_Old_GF_Pharmacy_...mp4` — Hospital Pharmacy Lobby 2.7K UHD (83.3 MB)
4. `2026-06-16 05:59:58` — Hospital OPD Morning Shift (509 MB)
5. `2026-06-16 09:56:00` — Hospital OPD Midday Shift (1015 MB)
6. `2026-06-16 15:07:30` — Hospital OPD Afternoon Shift (1015 MB)
7. `2026-06-25 17:59:55` — Hospital OPD Evening Shift (716 MB)
8. **Live USB / Built-in Camera and RTSP Stream**: Plug-and-play threaded capture with zero display lag.

---

## Academic Thesis Research Package

All academic thesis documents, LaTeX tables, mathematical formulations, and HIPAA/GDPR compliance reports are in the `thesis/` directory:
- `thesis/EXECUTIVE_THESIS_SUMMARY.md`: 2-page dissertation summary.
- `thesis/METHODOLOGY_AND_MATHEMATICAL_FORMULATION.md`: Mathematical derivations.
- `thesis/EMPIRICAL_BENCHMARK_RESULTS.md`: Empirical benchmarks and case studies.
- `thesis/GROUND_TRUTH_VERIFICATION_REPORT.md`: Event-level timelines and ground-truth verification.
- `thesis/TABLES_AND_LATEX_SNIPPETS.tex`: LaTeX tables for thesis chapters.
- `thesis/ETHICAL_AND_PRIVACY_COMPLIANCE.md`: Healthcare privacy analysis.

---

## Automated Unit Tests

To run the automated test suite:
```bash
python -m pytest tests/ -v
```
All 59 unit tests verify bounding box geometry, FSM transitions, distinct counting, timestamp aggregation, Re-ID descriptors, and plugin isolation.

---

## Documentation

- [`USER_MANUAL.md`](USER_MANUAL.md): Complete step-by-step user manual written in Simplified Technical English.
- [`thesis/README.md`](thesis/README.md): Guide to the academic research thesis package.
