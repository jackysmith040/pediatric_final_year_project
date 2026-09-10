# Antigravity technical handoff
## Build a stable, explainable research system

### Implementation directive

Build the smallest reliable vertical slice first. Do not add pose, ReID, CLAHE, SAHI, a carrying model, or multiple trackers until the baseline produces reproducible logs and metrics. Every optional component must be behind an interface and enabled by configuration, never by hidden conditions.

### Suggested package structure

```text
pediatric_counter/
  app/
    cli.py
    config.py
    pipeline.py
  vision/
    detector.py
    crop.py
    classifier.py
    tracker.py
    labels.py
  counting/
    lifecycle.py
    distinct_counter.py
    uncertainty.py
  evaluation/
    annotations.py
    run_manifest.py
    metrics.py
    report.py
  io/
    video_reader.py
    video_writer.py
    artifacts.py
  configs/
    room_default.yaml
    tracker_bytetrack.yaml
  tests/
    test_crop.py
    test_letterbox.py
    test_lifecycle.py
    test_counting.py
    test_manifest.py
  docs/
    thesis_methods.md
```

The exact language and framework may follow the existing repository, but the conceptual interfaces should remain stable. Antigravity should first inspect the existing project before replacing anything. Preserve working experiments as archived baselines rather than deleting them.

### Modern-library policy

Use established libraries for infrastructure rather than reimplementing standard algorithms. The preferred stack is OpenCV behind a thin video wrapper, Ultralytics for the existing detector, Supervision for standardized detection objects and debug overlays, one ByteTrack backend for the baseline, CVAT for persistent video annotations, and TrackEval for standard MOT metrics. FiftyOne may be added for dataset and error inspection. BoxMOT is an optional comparison layer for multiple tracker/ReID backends, while Norfair is optional when a detector-agnostic custom distance is specifically needed.

Do not let a convenience library hide the thesis-specific policy. Keep crop expansion, resize choice, track lifecycle, uncertainty labels, track-level classification aggregation, and distinct-child counting in local code with tests and logs.

The adult/child classifier must be trained on labeled person crops. A MobileNet-family model is only a candidate compact architecture: it learns from `adult` and `child` examples through supervised loss and transfer learning. It does not possess an innate adult/child concept and cannot solve a fully hidden baby without visible evidence or a separately trained relationship model.

### Configuration contract

```yaml
room:
  id: hospital_room_01
  video_path: data/videos/room_01.mp4
  fps: auto
  timezone: UTC

models:
  detector_path: models/yolo26s.pt
  classifier_path: models/adult_child_classifier.pt
  detector_class_ids: [0]
  classifier_classes: [adult, child]

crop:
  margin_fraction: 0.10
  resize_mode: letterbox
  input_size: 224
  padding_value: 114
  save_debug_crops: false

tracking:
  backend: bytetrack
  tracker_config: configs/tracker_bytetrack.yaml
  new_track_threshold: 0.50
  track_high_threshold: 0.50
  track_low_threshold: 0.10
  match_threshold: 0.80
  max_lost_time_seconds: 30

lifecycle:
  min_confirmed_observations: 5
  confirmation_window_frames: 15
  min_child_probability: 0.70
  label_window_size: 10
  uncertain_margin: 0.10

counting:
  mode: distinct_children_over_interval
  count_only_confirmed: true
  count_each_track_id_once: true
  retain_lost_ids: true

output:
  directory: runs/
  save_video: true
  save_crops: false
  save_frame_records: true
  save_metrics: true
```

Parameters are starting defaults, not truth. Every run must snapshot the resolved configuration.

### Core data records

```python
@dataclass
class Detection:
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"

@dataclass
class CropRecord:
    source_bbox: tuple[int, int, int, int]
    expanded_bbox: tuple[int, int, int, int]
    margin_fraction: float
    resize_mode: str
    input_size: int
    padding: tuple[int, int, int, int]
    transform_version: str

@dataclass
class Classification:
    probabilities: dict[str, float]
    top_label: str
    confidence: float

@dataclass
class TrackObservation:
    frame_index: int
    track_id: int
    detection: Detection
    crop: CropRecord | None
    classification: Classification | None

@dataclass
class Track:
    track_id: int
    state: str  # tentative, confirmed, lost, retired
    observations: list[TrackObservation]
    first_frame: int
    last_seen_frame: int
    counted: bool = False
    track_label: str = "uncertain"
    occlusion_events: list[dict] = field(default_factory=list)
```

### Utility functions that must be deterministic

`clip_bbox_to_image(bbox, width, height)` should return a valid integer crop rectangle and reject zero-area boxes.

`expand_bbox(bbox, margin_fraction, width, height)` should expand width and height symmetrically, clip to image boundaries, and record the original and expanded coordinates.

`crop_image(frame, bbox)` should return the crop and metadata. It must not silently resize or normalize the crop.

`letterbox_image(image, target_size, padding_value)` should scale uniformly by `min(target_width / crop_width, target_height / crop_height)`, round dimensions deterministically, place the resized image centrally, and return padding metadata.

`direct_resize_image(image, target_size)` should be available only for controlled comparison because it may distort aspect ratio.

`preprocess_crop(crop, config)` should apply the configured resize mode, color conversion, tensor conversion, and normalization exactly once.

`aggregate_track_probabilities(probability_history, method)` should support mean probability and majority vote. It must return an uncertainty flag when the top two class probabilities are too close.

`update_track_state(track, observation, config)` should implement explicit state transitions and return a transition record explaining the reason.

`should_confirm_track(track, config)` should use repeated evidence rather than movement. A still person is confirmable if matching observations persist.

`mark_track_lost(track, frame_index)` should preserve the track and record the first lost frame.

`should_retire_track(track, current_frame, fps, max_lost_time_seconds)` should compare elapsed seconds, not an arbitrary frame count.

`match_reappearing_track(detection, lost_tracks, config)` should first use spatial/motion association. Appearance/ReID must remain optional and must not be required by the baseline.

`register_count_if_new(track, counted_ids)` should increment only when a confirmed track has a child label and its ID is absent from the counted-ID set.

`build_run_manifest(config, versions, model_hashes, input_metadata)` should create a JSON manifest before processing starts.

`compute_count_metrics(predicted_ids, ground_truth_ids)` should report unique-count error, absolute error, overcount, undercount, duplicate count rate, and missed-child rate.

### Pipeline pseudocode

```python
for frame_index, frame in enumerate(video):
    detections = detector.detect(frame, class_ids=[PERSON])
    visible_observations = []

    for detection in detections:
        crop, crop_meta = build_standardized_crop(frame, detection.bbox, crop_config)
        prediction = classifier.predict(crop)
        visible_observations.append((detection, crop_meta, prediction))

    tracks, transitions = tracker.update(visible_observations)

    for track in tracks:
        update_track_state(track, frame_index, lifecycle_config)
        update_track_label_from_history(track, lifecycle_config)
        register_count_if_new(track, counted_ids)

    for track in tracks_not_matched_this_frame:
        mark_track_lost(track, frame_index)
        if should_retire_track(track, frame_index, fps, max_lost_time_seconds):
            retire(track)

    write_frame_records(frame_index, detections, tracks, transitions)
    render_debug_overlay(frame, detections, tracks, counted_ids)
```

The tracker backend should receive person detections, while the adult/child classification is attached to each resulting track. Do not pass adult/child boxes to the person tracker unless the code and experiment explicitly define that as a different architecture.

### Handling carried children

Introduce an explicit visibility field: `visible`, `partial`, `carried_possible`, `fully_hidden`, or `uncertain`. The baseline can preserve an already-known child track through temporary disappearance, but it cannot directly detect a never-observed child who is fully hidden behind a caregiver. Do not infer a new child solely because an adult’s crop looks unusual.

An optional future interface may be:

```python
class CarryingRelationshipModel(Protocol):
    def predict(self, frame_or_person_pair) -> RelationshipPrediction: ...
```

Do not implement this in the first vertical slice. Only add it after collecting representative annotated examples from the target setting and defining a separate evaluation question.

### Testing requirements

Unit-test crop clipping, expansion, letterboxing, deterministic padding, FPS-to-frame timeout conversion, state transitions, confirmation of stationary tracks, duplicate prevention, lost-track recovery, and retirement. Integration-test one short video with a frozen model and assert that output files, manifest, and summary metrics are produced.

The application should provide commands equivalent to:

```text
pediatric-counter run --config configs/room_default.yaml
pediatric-counter inspect-crops --run runs/<id>
pediatric-counter evaluate --predictions runs/<id>/tracks.csv --ground-truth data/annotations/room_01.csv
pediatric-counter compare --configs experiments/*.yaml
pediatric-counter report --run runs/<id>
```

The exact command framework may follow the existing project. The important property is that a researcher can run, inspect, evaluate, and reproduce a configuration without editing source code.

### Anti-patterns to remove

Do not switch among trackers based on unexplained runtime conditions. Do not mix detector confidence, classifier confidence, motion heuristics, and counting state in one long conditional. Do not overwrite an old ID merely because a detection was missed for one frame. Do not claim ReID when the system is only using tracker motion. Do not call a child “detected” when the child is fully hidden; use `not_observable` or `uncertain`.
