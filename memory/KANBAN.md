# 📋 Task Kanban Board — Pediatric Counting Research Harness

## Active Sprint: Baseline Architecture & Execution

### 📥 Backlog
- [ ] Task 4.1: Evaluation Harness & Parameter Sweep (Lost timeout 5s/15s/30s, crop margin ablation, model comparison)
- [ ] Task 5.1: Interactive Web Inspection UI (Streamlit Dashboard for video inspection, crop viewer, track timeline)

### 🏗️ In Progress
*(None - Sprint Goal achieved)*

### ✅ Done
- [x] Task 3.9: Intel OpenVINO Models & Pose Estimation Benchmark (/senior-stable-delivery + /architect: extracted 4 models, installed openvino 2026.3, benchmarked yolo26s/m vs PyTorch, verified 17-point pose models; 55/55 tests pass)
- [x] Task 3.8: Base Model Heavy Lifting, Tracklet Stitching, Multi-Tracker, and Kids Sieve / Model Shadow Plugins (100% verified ground-truth on Playroom Drawers [2 kids, 1 adult] and Kindergarten [5 kids, 1 adult]; 53/53 tests pass)
- [x] Task 3.7: YOLO26s Demographic Calibration & UNKNOWN Bug Recovery (/recover + /architect: crop-area coverage filters, room demographic priors, events.py detecting promotion bugfix, kindergarten.yaml, all 50 tests passing)
- [x] Conscience OS Universal Bootloader & Neural Skills Matrix Installation
- [x] Codebase & Asset Inspection (/senior-stable-delivery + /pipeline + /architect)
- [x] Task 3.2: Classifier Integration (`pediatric-model.pt` wired for adult/child crop classification)
- [x] Task 3.3: HEVC Seek Gray-Screen Decoder Pre-roll Fix (`video_reader.py`: primes hardware/FFmpeg reference picture buffer with pre-roll grab loop)
- [x] Task 3.4: Stable v1 Core & Decoupled Plugin Architecture (`pediatric_counter/plugins/`: `base.py`, `manager.py`, `csv_exporter.py`, `json_summary.py`, `timeline_exporter.py`, `video_recorder.py`)
- [x] Task 3.5: High-End Live HUD (`live_hud.py`: alpha-blended glassmorphic status cards for Children/Adults/Total, dynamic status pill, interactive `[Space]`, `[n]`, `[+/-]`, `[r]`, `[q]` controls)
- [x] Task 3.6: Plugin Unit Tests & Full Test Suite (`tests/test_plugins.py` — 44/44 tests passed in 3.14s with strict fault isolation)
- [x] Virtual environment setup with `uv` (CPython 3.12.13)
- [x] Modern libraries installed (`ultralytics 8.4`, `supervision 0.30`, `torch 2.14`, `opencv-python 4.11`, `pydantic 2.13`, `pandas 3.0`, `pyarrow 25.0`, `pytest 9.1`)
- [x] Package scaffolding & editable install (`pediatric-counting 0.1.0`)
- [x] Task 1.1: Pure Utilities Implementation (Bounding box clipping, margin expansion, letterbox/direct resize, FPS-to-frame conversion, IoU)
- [x] Task 1.2: Unit Tests (`tests/test_crop.py`, `tests/test_lifecycle.py`, `tests/test_counting.py`, `tests/test_events.py`, `tests/test_camera.py` — 39/39 passed in 2.18s)
- [x] Task 2.1: Vision Layer (`crop.py`, `video_reader.py`, `pipeline.py`)
- [x] Task 2.2: Tracking & Lifecycle Engine (`lifecycle.py`: tentative/confirmed/lost/retired FSM + temporal label smoothing)
- [x] Task 2.3: Counting & Uncertainty Engine (`distinct_counter.py`: dual child & adult distinct set cardinality, uncertainty tracking)
- [x] Task 2.4: IO & Artifact Generation (`run_manifest.py`, CSV per-frame logger, summary JSON exporter)
- [x] Task 2.5: CLI Entry Point (`pediatric_counter/app/cli.py` with default config, `--max-frames`, `--start-frame`, `--live`, `--speed`, `--count-adults`)
- [x] Task 2.6: Recovery & Video Audit (`/recover` executed: audited all 14 CCTV files; identified 4 intact files totaling 732,816 frames; fixed moov atom error & Windows cp1252 console encoding)
- [x] Task 2.7: Warning Elimination & Live View (`ByteTrack.wrapped` eliminates FutureWarning; `--live` view added with scaled window, pause/resume, and per-track label banners)
- [x] Task 2.8: User-Friendly Interactive Runner (`run.py` guided interactive wizard + `start.bat` double-click desktop launcher)
- [x] Task 2.9: Detection Events & Special Timestamps (`events.py`: clusters sightings into discrete events with elapsed video time, CCTV wall-clock timestamps, duration, and Markdown/CSV/JSON export)
- [x] Task 2.10: Dynamic Playback Speed Controls (`pipeline.py` & `cli.py`: interactive `+`/`-` speed keys, `Space` pause, `n` frame-step, `r` reset, HUD speed indicator, `--speed` CLI flag)
- [x] Task 2.11: Simplified Technical English User Manual (`USER_MANUAL.md` following ASD-STE100 guidelines via `/ste-writing`)
- [x] Task 2.12: Live Camera & RTSP Stream Integration (`io/camera.py`: supports local USB webcams and hospital IP CCTV RTSP streams uniformly)
- [x] Task 2.13: VS Code / IDE Red Squigglies Fix (`.vscode/settings.json`: maps virtual environment interpreter path and type checking paths)
- [x] Task 2.14: HEVC POC Codec Notice Silencing (`video_reader.py`: silenced C-level FFmpeg seek warnings via `os.dup2`)
- [x] Task 3.1: Run Baseline Vertical Slice on verified OPD CCTV clip (`runs/otmc_opd_baseline` generated: `annotated.mp4`, `per_frame.csv`, `run_manifest.json`, `summary.json`, `detection_timeline.md`)
- [x] Session State Documentation (`memory/SESSION_STATE.md` via `/remember save`)
