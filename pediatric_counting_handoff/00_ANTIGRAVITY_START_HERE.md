# Antigravity Agent: start here

## Mission

Turn the existing pediatric CCTV counting experiments into a maintainable, reproducible, explainable single-room software system. Preserve useful existing work, archive unclear experiments, and build a clean harness before adding more models.

## Product definition

The system processes one fixed hospital-room CCTV video. It detects visible people with a base person detector, creates a deterministic standardized crop for each person, classifies each crop as adult or child, tracks people over time with one selected tracker, manages tentative/confirmed/lost/retired track states, and counts each confirmed child track ID once over the selected interval.

A child who is fully hidden behind or carried by a caregiver is not directly observable in an RGB frame. The system may preserve an already-known child track through temporary occlusion, but must expose uncertainty and must not claim guaranteed detection of a never-observed hidden child.

## Non-negotiable engineering rules

1. Build and test the baseline before adding optional models.
2. Use one tracker at a time; compare trackers in separate runs.
3. Never use hidden runtime “magic conditions” to switch trackers or counting behavior.
4. Make every threshold and transform configurable and record it in the run manifest.
5. Keep room-specific settings in YAML/JSON, not in core source code.
6. Preserve original experiments in an archive with a short README explaining what they attempted.
7. Add unit tests before refactoring the whole application.
8. Every run must produce inspectable artifacts: configuration snapshot, model/version metadata, frame records, track records, summary metrics, and optional annotated video.
9. Do not claim ReID unless an appearance model is actually enabled and evaluated.
10. Do not claim a child was detected when the child was fully hidden; use `uncertain` or `not_observable`.

## Build order

First inspect the existing repository, notebook, UI, model paths, tracker YAML files, and current entry points. Create a short inventory and identify what can be reused.

Second implement the pure utilities: bounding-box clipping, margin expansion, crop extraction, letterbox resize, direct resize for comparison, FPS-to-frame conversion, probability aggregation, lifecycle transitions, duplicate-count prevention, and run-manifest creation. Unit-test these utilities.

Third implement the baseline vertical slice: base detector, crop transform, existing classifier, ByteTrack, track lifecycle, distinct-child count, CSV/JSON logging, and annotated output video.

Fourth implement the evaluation harness and run configuration sweeps where only one variable changes: confirmation rule, lost timeout, crop margin, resize mode, tracker backend, and optional lightweight classifier.

Fifth improve the UI so the user can select a video and configuration, inspect detections/crops/tracks, view count and uncertainty, launch evaluations, and download results. Do not hide the research parameters behind an opaque interface.

## Modern-library requirement

Prefer established libraries for infrastructure: OpenCV behind a thin video wrapper, Ultralytics for the existing detector, Supervision for standardized detections and debug overlays, one ByteTrack backend for the baseline, CVAT for persistent video ground truth, TrackEval for standard MOT metrics, and optionally FiftyOne for prediction/error inspection. Consider BoxMOT for controlled tracker/ReID comparisons and Norfair only for detector-agnostic/custom-distance experiments. Keep crop policy, lifecycle, uncertainty, and distinct-child counting local and transparent.

The compact classifier must be trained with labeled crops. MobileNet does not automatically know adult versus child; it learns from `adult` and `child` examples using supervised loss. Use transfer learning rather than training from random initialization, and test whether the model helps rather than assuming it does.

## Deliverables

- `PRD_and_user_stories.md`
- `architecture_and_harness.md`
- `evaluation_protocol.md`
- `thesis_methods.md`
- `math_defense_notes.md`
- versioned room configuration
- tested baseline pipeline
- reproducible evaluation command
- sample run artifacts
- README explaining setup, execution, interpretation, and limitations

## Definition of done

A fresh user can run one documented command on a short video, see the staged pipeline, inspect why each child ID was counted, reproduce the same result from the same configuration, change one parameter without editing source code, and obtain thesis-ready metrics. The implementation must clearly distinguish a visible child, a partially occluded child, a temporarily lost known child, and a fully hidden or unobservable child.

## Mathematical explanation expected in documentation

Use the frame sequence `I_1,...,I_T`, detector `D(I_t)`, crop function `C_m(b)`, classifier probabilities `P(y|z)`, IoU for spatial association, finite track states, and the set cardinality `N_child = |K_child|`. Explain that one person appearing across many frames contributes one ID to the set, not one count per frame. Explain that thresholds are parameters to evaluate, not truths.

## First acceptance demonstration

Run one short annotated hospital-room clip with ByteTrack. Produce a frame-by-frame record showing: frame index, detection box, crop metadata, adult/child probabilities, track ID, lifecycle state, whether the ID was counted, lost/recovered events, and uncertainty flags. Compare at least two lost-timeout settings and report distinct-child count error, duplicates, misses, ID switches, and runtime.
