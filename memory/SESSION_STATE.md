# 🧠 SESSION MEMORY — Pediatric Counting Research Harness
## /remember save · 2026-09-03T13:04Z

> **Restore instruction:** At the start of the next session, read this file first.
> Say: "I have restored context from `memory/SESSION_STATE.md`" and resume from **NEXT ACTIONS** below.

---

## 🎯 PROJECT MISSION

Build a **maintainable, explainable, reproducible single-room CCTV pediatric counting system** for an undergraduate mathematics thesis.

- **Primary question:** How many distinct children entered/passed through a hospital room during a video interval?
- **Mathematical formalization:** N_child = |K_child| (set cardinality, not per-frame count)
- **Ethical constraint:** Never claim to detect a fully hidden child. Use `uncertain` / `not_observable` flags.
- **Deliverable:** One `pediatric-counter run --config ...` command that produces thesis-ready metrics.

---

## 📁 WORKSPACE LAYOUT (as of this session)

```
C:\Users\doks\Desktop\CodeHouse\pediatric_personal\
├── .venv/                        ← Python 3.12 (uv-managed)
├── pyproject.toml                ← Project config, deps, CLI entry point
├── README.md
├── CONSCIENCE.md / GEMINI.md / AGENTS.md   ← Axon bootloader (active)
│
├── pediatric_counter/            ← 🏗️ MAIN PACKAGE (partially built)
│   ├── app/
│   │   ├── config.py            ✅ DONE — Pydantic schema, YAML-driven
│   │   ├── cli.py               ✅ DONE — Click CLI `pediatric-counter run`
│   │   └── pipeline.py          ✅ DONE — Full baseline vertical slice
│   ├── vision/
│   │   └── crop.py              ✅ DONE — Pure math: expand_box, letterbox, IoU, FPS utils
│   ├── counting/
│   │   ├── lifecycle.py         ✅ DONE — FSM: tentative→confirmed→lost→retired
│   │   └── distinct_counter.py  ✅ DONE — Set K_child dedup counter
│   ├── evaluation/
│   │   └── run_manifest.py      ✅ DONE — SHA-256 hashes + config snapshot
│   ├── io/
│   │   ├── video_reader.py          ✅ DONE — OpenCV VideoCapture wrapper + start_frame seeking
│   │   └── camera.py                ✅ DONE — Live USB camera & hospital RTSP network stream reader
│   └── configs/
│       └── room_default.yaml    ✅ DONE — Points to smallest test clip + bytetrack.yaml
│
├── tests/
│   ├── test_crop.py             ✅ DONE — 15 unit tests (margins, clamp, fps, IoU)
│   ├── test_lifecycle.py        ✅ DONE — 10 unit tests (FSM transitions, label smoothing)
│   ├── test_counting.py         ✅ DONE — 8 unit tests (dedup, uncertainty)
│   ├── test_events.py           ✅ DONE — 4 unit tests (timestamps, wall-clock aggregation)
│   └── test_camera.py           ✅ DONE — 2 unit tests (mocked live camera capture & error handling)
│
├── assets/
│   ├── cctv_videos/             ← 14 CCTV files in repository
│   │   ├── [1] Hospital OTMC GF OPD 2026-06-16 05:59:58 (Morning Shift, 509 MB, 190,604 frames)  ✅ VALID
│   │   ├── [2] Hospital OTMC GF OPD 2026-06-16 09:56:00 (Midday Shift, 1015 MB, 155,014 frames)   ✅ VALID
│   │   ├── [3] Hospital OTMC GF OPD 2026-06-16 15:07:30 (Afternoon Shift, 1015 MB, 134,610 frames) ✅ VALID
│   │   ├── [4] Hospital OTMC GF OPD 2026-06-25 17:59:55 (Evening Shift, 716 MB, 252,588 frames)   ✅ VALID
│   │   └── (Remaining 10 files are 1024MB empty zero-filled download placeholders — auto-skipped)
│   └── trackers/                ← 6 tracker YAML configs discovered
│       ├── bytetrack.yaml       ← BASELINE
│       ├── botsort.yaml
│       ├── deepocsort.yaml
│       ├── fasttrack.yaml
│       ├── ocsort.yaml
│       └── tracktrack.yaml
│
├── computer_vision_models/
│   └── yolo26/
│       ├── base_model/yolo26s.pt           ← BASELINE detector (Class 0 = person)
│       ├── distilled_model/best.pt
│       ├── fine_tune_model/pediatric-model.pt
│       └── onnx/onnx_distilled/best.onnx
│           onnx_fine_tuned/pediatric-*.onnx
│
└── pediatric_counting_handoff/  ← Full spec documents (READ THESE)
    ├── 00_ANTIGRAVITY_START_HERE.md
    ├── 01_learning_and_math_guide.md
    ├── 02_PRD_and_user_stories.md
    ├── 03_antigravity_technical_handoff.md
    ├── 04_evaluation_and_thesis_methods.md
    └── 05_modern_libraries_and_classifier.md
```

---

## ⚙️ ENVIRONMENT SETUP

```bash
# Virtual environment — Python 3.12 via uv
# .venv/ already created at project root

# Step 1: Install PyTorch (CUDA 12.1 — if GPU available)
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install project + all other deps
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"

# Activate
.venv\Scripts\activate
```

### Key dependency decisions (NON-NEGOTIABLE — do not change without testing):
| Package | Version pin | Reason |
|---|---|---|
| `numpy` | `<2.0` | Ultralytics + Supervision break on numpy 2.x |
| `python` | `>=3.11,<3.13` | PyTorch CUDA 12.1 wheel only ships for 3.12 |
| `torch` | installed separately | Requires `--index-url https://download.pytorch.org/whl/cu121` |
| `setuptools` | `packages.find include = ["pediatric_counter*"]` | Flat layout has assets/, memory/ dirs that confuse setuptools |

---

## 🔬 MATHEMATICAL FRAMEWORK (thesis-grade)

```
V = (I_1, ..., I_T)                        Frame sequence
D(I_t) = {(b_tj, s_tj)}                    Person detector (YOLO26s, class 0)
C_m(b) = (x1-mw, y1-mh, x2+mw, y2+mh)    Crop expansion (margin m=0.10)
P(y|z)                                     Classifier posterior (adult/child)
P̄_k(t) = mean of last L frames             Track-level smoothed probability
K_child = {k | τ_k confirmed child}         Set of distinct child track IDs
N_child = |K_child|                         FINAL COUNT (set cardinality)
```

**Track states:** `tentative` → `confirmed` → `lost` → `retired`
**Uncertain rule:** |P̄ - 0.5| < margin → `uncertain` (not counted)

---

## 🧱 ARCHITECTURAL DECISIONS (LOCKED — do not revisit without user approval)

| Decision | Choice | Rationale |
|---|---|---|
| Tracker baseline | ByteTrack (via Supervision) | Single tracker, no dynamic switching |
| Detector | `yolo26s.pt` (Ultralytics) | Already matches existing experiments |
| Detection representation | `supervision.Detections` | Unified format across all trackers |
| Crop resize | Letterbox (default), direct available | Preserves aspect ratio for CCTV geometry |
| Config system | Pydantic + YAML | All params configurable, no magic numbers |
| Count logic | Set-cardinality, confirmed only | Thesis spec: one ID = one count, ever |
| Logging | Per-frame CSV + summary JSON + run manifest | Full thesis reproducibility |

---

## 🚫 REJECTED SCOPE (DO NOT IMPLEMENT until baseline verified)

- Multi-camera ReID
- Pose estimation / carrying-relationship detectors
- SAHI / CLAHE preprocessing (ablation experiments only, post-baseline)
- Face recognition / biometric age estimation (privacy + resolution constraints)
- MobileNet classifier — test after baseline; do not assume it helps

---

## ✅ KANBAN STATE

### DONE this session
- [x] Conscience OS / Axon bootloader installed (GEMINI.md, AGENTS.md, CONSCIENCE.md)
- [x] 18 Neural Skills registered as Antigravity slash commands (global + workspace)
- [x] `pyproject.toml` with uv venv, pinned deps, CLI entry point
- [x] `pediatric_counter/app/config.py` — Pydantic PipelineConfig
- [x] `pediatric_counter/app/cli.py` — Click CLI
- [x] `pediatric_counter/app/pipeline.py` — Full baseline pipeline loop
- [x] `pediatric_counter/vision/crop.py` — Pure geometry utilities
- [x] `pediatric_counter/counting/lifecycle.py` — FSM
- [x] `pediatric_counter/counting/distinct_counter.py` — Set counter
- [x] `pediatric_counter/evaluation/run_manifest.py` — SHA-256 manifest
- [x] `pediatric_counter/io/video_reader.py` — OpenCV wrapper
- [x] `pediatric_counter/configs/room_default.yaml` — config pointing to 131MB test clip
- [x] `tests/test_crop.py`, `tests/test_lifecycle.py`, `tests/test_counting.py` — 27 unit tests written

### IN PROGRESS
- [/] `uv pip install -e ".[dev]"` — task-218 still running (deps are large, numpy/scipy/torch)
- [/] `uv pip install torch torchvision` (CUDA) — cached by previous run, re-install queued

### NEXT ACTIONS (start here next session)
1. **Verify install completed:** `uv pip list --python .venv\Scripts\python.exe | findstr torch`
2. **Run unit tests:** `.venv\Scripts\python.exe -m pytest tests/ -v`
3. **Smoke test CLI:** `pediatric-counter run --config pediatric_counter/configs/room_default.yaml`
4. **Write missing modules:**
   - `pediatric_counter/vision/detector.py` — Ultralytics wrapper returning `sv.Detections`
   - `pediatric_counter/vision/classifier.py` — PyTorch crop classifier interface
   - `pediatric_counter/vision/tracker.py` — ByteTrack wrapper (isolated state)
   - `pediatric_counter/counting/uncertainty.py` — Flag assigner
   - `pediatric_counter/evaluation/metrics.py` — MOTA, IDF1, count MAE
   - `pediatric_counter/io/artifacts.py` — Parquet/CSV per-frame logger
5. **Evaluation harness:** Lost-timeout ablation (5s / 15s / 30s) — one variable changes per run
6. **UI:** Streamlit inspection dashboard for crops + tracks

---

## 📌 KEY COMMANDS (copy-paste ready)

```bash
# Activate venv
.venv\Scripts\activate

# Run tests
python -m pytest tests/ -v --tb=short

# ─────────────────────────────────────────────────────────────
# EASIEST WAYS TO RUN (No memorizing flags needed):
# ─────────────────────────────────────────────────────────────
# 1. Interactive Menu (Asks video, mode, live view, playback speed):
python run.py

# 2. Windows Explorer:
# Just double-click start.bat in the project folder!

# 3. Direct simple run with live view and 1.0x speed:
pediatric-counter run --live --speed 1.0

# 4. Fast test on active section with custom speed (e.g. 2.0x):
pediatric-counter run -s 1450 -m 100 --live --speed 2.0

# ─────────────────────────────────────────────────────────────
# USER MANUAL & DOCUMENTATION:
# ─────────────────────────────────────────────────────────────
# Complete user guide in ASD-STE100 Simplified Technical English:
# See USER_MANUAL.md

# Check installed packages
uv pip list --python .venv\Scripts\python.exe | findstr -i "torch ultra super"
```

---

## 🔗 SPEC REFERENCE FILES

Always consult before adding any feature:
- [00_ANTIGRAVITY_START_HERE.md](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/pediatric_counting_handoff/00_ANTIGRAVITY_START_HERE.md)
- [02_PRD_and_user_stories.md](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/pediatric_counting_handoff/02_PRD_and_user_stories.md)
- [03_antigravity_technical_handoff.md](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/pediatric_counting_handoff/03_antigravity_technical_handoff.md)
- [05_modern_libraries_and_classifier.md](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/pediatric_counting_handoff/05_modern_libraries_and_classifier.md)

---

*Saved by Axon · /remember save · State 5: MEMORY_CONSOLIDATION*
