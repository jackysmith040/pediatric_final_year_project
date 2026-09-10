# Product Requirements Document
## Pediatric Counting with Interactive Computer Vision

### Product purpose

Build a maintainable, explainable, single-camera CCTV system that estimates the number of distinct children observed in one hospital room over a video interval. The system must preserve a child’s temporary track identity when possible, avoid counting the same confirmed child repeatedly, expose uncertainty, and produce evaluation evidence suitable for an undergraduate mathematics thesis.

The first release is a **research harness and stable working application**, not a safety-critical clinical system. It must make assumptions and parameters visible so another student can extend it to additional rooms after graduation.

### Definitions

| Term | Meaning in this project |
|---|---|
| Room | One fixed CCTV camera view and its associated configuration |
| Person detection | A visible person bounding box from the base detector |
| Crop | An expanded rectangle cut from the frame around one detected person |
| Adult/child classification | A label assigned to a crop and then smoothed over a track |
| Track ID | A temporary identity used to link observations within this camera stream |
| Confirmed child | A track with sufficient evidence and a track-level child label |
| Distinct-child count | Number of unique confirmed child IDs in the room during the selected video interval |
| Occlusion | A person or object blocks part or all of another person |
| Re-identification | Matching a returning person to an earlier track after a longer disappearance; optional in the first baseline |

### Primary user

The primary user is a student or researcher who needs to inspect CCTV footage, compare model configurations, understand failure cases, and produce reproducible thesis results without editing scattered magic conditions.

### Core user stories

| ID | User story | Acceptance criteria |
|---|---|---|
| US-01 | As a researcher, I want to load a video and room configuration so that the same pipeline can run on different rooms. | The system accepts a video path and a versioned YAML/JSON configuration containing camera ID, FPS policy, model paths, crop settings, tracker settings, and counting settings. |
| US-02 | As a researcher, I want to see base person detections so that I can distinguish localization failures from classification failures. | Output video and logs show person boxes, confidence, frame number, and detector model version. |
| US-03 | As a researcher, I want each detected person crop saved or inspectable so that I can understand what the classifier actually sees. | The system can save sampled crops with original box, expanded box, padding, transform version, and source frame ID. |
| US-04 | As a researcher, I want adult/child predictions attached to tracks so that a single noisy frame does not decide the identity label. | Each track stores per-frame probabilities and a track-level aggregation rule such as mean probability or majority vote. |
| US-05 | As a researcher, I want track states to be explicit so that occlusion behavior is explainable. | Every track has a state among tentative, confirmed, lost, or retired, with transition reasons in the log. |
| US-06 | As a researcher, I want a child counted only once so that long visibility does not inflate the count. | The count is based on a set of confirmed child track IDs; continuing observations do not increment it again. |
| US-07 | As a researcher, I want configurable lost-track memory so that I can test 5, 15, and 30 seconds instead of hard-coding one guess. | Lost timeout is converted from seconds to frames using the actual FPS and is recorded in every run. |
| US-08 | As a researcher, I want to compare trackers under identical conditions so that the result is scientifically meaningful. | A run manifest records detector, classifier, crop transform, tracker YAML, thresholds, and random seeds; only the selected tracker changes in a tracker comparison. |
| US-09 | As a researcher, I want ground-truth annotations and metrics so that I can measure rather than judge by appearance. | The harness reports detector, classifier, tracking, occlusion recovery, and final distinct-count metrics. |
| US-10 | As a future maintainer, I want room settings separate from core code so that I can add another hospital room safely. | Room-specific values are configuration data; no room name or coordinate is embedded in the processing logic. |
| US-11 | As a researcher, I want uncertain/unobservable cases flagged so that the system does not pretend to see fully hidden babies. | Frames and tracks can carry `uncertain` or `not_observable` status; these cases are included in separate evaluation slices. |
| US-12 | As a thesis student, I want reproducible exports so that my tables and figures can be regenerated. | Each run creates a manifest, raw per-frame CSV/Parquet, summary JSON, metrics CSV, configuration snapshot, and optional annotated video. |

### Nonfunctional requirements

The application should be deterministic when a seed and configuration are fixed. It should fail gracefully when a model path, video, or annotation file is missing. It should never silently change preprocessing between training, validation, and inference. It should log software versions and model hashes where available. It should keep personal data local and avoid facial recognition claims.

### Scope boundaries

The first release does not promise multi-camera identity matching, perfect detection of a fully hidden baby, medical diagnosis, age estimation beyond the declared adult/child classes, or automatic selection among trackers using unexplained heuristics. Pose estimation, carrying-relationship detection, ReID, CLAHE, and SAHI are optional experimental modules and are not release blockers.

### Definition of done

The product is ready for thesis evaluation when a fresh user can run one command on a room video, inspect the staged pipeline, reproduce the same metrics, change one configuration parameter, and see that the run manifest records the change. The final report must state which cases are observable, which are uncertain, and how each component affects the final count.
