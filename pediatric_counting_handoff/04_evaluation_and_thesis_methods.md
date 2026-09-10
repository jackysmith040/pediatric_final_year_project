# Evaluation protocol and thesis-ready methodology

## Research objective

Evaluate whether a modular two-stage pipeline can estimate the number of distinct children observed in one fixed hospital-room CCTV stream while preserving identity through temporary occlusion and exposing cases that are not visually observable.

## Research questions

**RQ1.** Does a CCTV-compatible base person detector followed by crop-based adult/child classification improve the reliability of child classification compared with applying the specialist model directly to the full CCTV frame?

**RQ2.** How do crop margin and resize strategy affect adult/child classification under CCTV viewpoint, distortion, small scale, and partial occlusion?

**RQ3.** How do tracker choice and lost-track timeout affect identity preservation, occlusion recovery, and distinct-child count error?

**RQ4.** What portion of carried-child cases is visible enough for direct detection, and what portion should be reported as uncertain or not observable?

**RQ5.** Does a lightweight transfer-learned crop classifier provide a useful accuracy/latency trade-off compared with the existing classifier?

## Data protocol

Divide videos by recording session rather than randomly splitting individual frames. Randomly splitting neighboring frames can place nearly identical images in both training and test sets and make the result look better than it generalizes. Use training data for model development, validation data for parameter selection, and held-out test clips for the final report.

Annotate a manageable evaluation subset. At sampled frames, label visible person boxes, adult/child status when visually observable, persistent reference IDs for children, and occlusion status. For carried-child cases, annotate the caregiver box, visible child region if present, relationship label if the relationship is actually identifiable, and an observability category.

### Annotation fields

| Field | Example |
|---|---|
| `video_id` | `room_01_clip_03` |
| `frame_index` | `1420` |
| `reference_id` | `child_03` |
| `bbox_xyxy` | `x1,y1,x2,y2` |
| `class` | `adult`, `child`, `unknown` |
| `visibility` | `visible`, `partial`, `carried_possible`, `fully_hidden` |
| `occluder_id` | `adult_02` or blank |
| `countable_from_frame` | yes/no/uncertain |

Do not force an annotator to label a fully hidden child as visible. The test data must preserve the distinction between a known child temporarily hidden and a child who was never visually detectable.

## Baseline and ablations

Freeze the base detector, video clips, annotation split, and counting policy while changing one factor at a time. Begin with the two-stage detector/classifier and ByteTrack. Then test crop margin, resize mode, lost timeout, one alternative tracker, and an optional lightweight classifier.

| Run | Detector | Classifier | Crop | Tracker | Purpose |
|---|---|---|---|---|---|
| B0 | Base person detector | Existing classifier | Declared default | ByteTrack | Primary baseline |
| B1 | Base person detector | Existing classifier | Direct resize | ByteTrack | Test geometric distortion |
| B2 | Base person detector | Existing classifier | Letterbox | ByteTrack | Test aspect-ratio preservation |
| B3 | Base person detector | Existing classifier | Margin sweep | ByteTrack | Test contextual crop |
| B4 | Base person detector | Existing classifier | Default | ByteTrack, timeout sweep | Test occlusion memory |
| B5 | Base person detector | Existing classifier | Default | One alternative tracker | Test tracker sensitivity |
| B6 | Base person detector | Lightweight transfer classifier | Default | ByteTrack | Test compact neural classifier |

## Metrics

### Detection and classification

For visible person detection, report precision, recall, and false-negative rate. For adult/child classification, report a confusion matrix, accuracy, adult recall, child recall, macro-F1, and performance by visibility category.

### Tracking

Report identity switches, track fragmentation, IDF1, and HOTA where feasible. HOTA was proposed to balance detection, association, and localization rather than overemphasize only one of them [1]. TrackEval provides implementations for HOTA, CLEAR MOT, and identity metrics [2].

### Counting

Let `G` be the set of reference child IDs and `P` the set of predicted confirmed child IDs for a video interval. The unique-count error is:

\[
E_{count}=|\lvert P\rvert-\lvert G\rvert|.
\]

Also report signed error `|P| - |G|`, overcount, undercount, duplicate count rate, and missed-child rate. The final count should be computed over IDs, not over detections or frames.

### Occlusion

For each occlusion event, record whether the old ID was recovered, whether a new duplicate ID was created, whether the child was missed, and whether the result was marked uncertain. Useful quantities include recovery rate, false re-association rate, identity-switch rate after occlusion, and count impact per occlusion event.

### Efficiency

Measure average inference time per frame, frames per second, memory use where practical, and crop-classification latency. A smaller network is not better merely because it is smaller; it must maintain acceptable classification and count performance.

## Statistical reporting

Report results per video as well as an aggregate mean and median. Use the same clips for paired comparisons between configurations. If the number of clips is small, avoid overstating statistical significance; show per-clip results and confidence intervals or bootstrap intervals where appropriate.

## Thesis-ready methodology draft

This study developed a modular computer-vision pipeline for estimating the number of distinct children observed in a fixed hospital-room CCTV stream. The system uses a tracking-by-detection architecture. A pretrained person detector first identifies visible person regions in each frame. Each region is expanded by a configurable margin, cropped, and transformed into a fixed classifier input while preserving aspect ratio through letterboxing. A specialist classifier then assigns adult or child probabilities to each crop. These observations are associated over time by a multi-object tracker, initially ByteTrack, which assigns temporary track IDs.

Track management is modeled as a finite-state process with tentative, confirmed, lost, and retired states. A tentative track is not counted immediately. It becomes confirmed after repeated matching observations. A confirmed track may enter the lost state during a temporary detection failure or occlusion and remains eligible for recovery for a configurable period. If a matching observation returns before the timeout, the track retains its ID; otherwise it is retired. The distinct-child count is the cardinality of the set of confirmed child track IDs, so repeated observations of one child do not increase the count.

The crop transformation is defined explicitly rather than selected as an undocumented constant. For a detected box with width `w` and height `h`, the system expands each side by `m` times the corresponding dimension, clips the result to the image boundaries, and extracts the resulting region. The crop is then resized to a fixed square input either by direct resizing or by aspect-ratio-preserving letterboxing. The margin, target size, padding value, resize mode, and transform version are recorded in the run manifest.

The evaluation separates visible, partially occluded, carried-possible, and fully hidden cases. A fully hidden child is not treated as directly detectable from an RGB frame. Instead, an existing child identity may be retained temporarily through the lost-track mechanism, while an unseen child is labeled not observable or uncertain unless a separate carrying-relationship model is developed and evaluated.

Experiments compare crop transforms, lost-track timeouts, tracker backends, and an optional lightweight transfer-learned classifier while holding the remaining pipeline constant. Performance is measured using detection and classification metrics, identity and association metrics, occlusion recovery, computational cost, and final distinct-child count error.

## Defense notes

**Why is the system two-stage?** The first model is good at locating visible people in the CCTV domain. The second model receives a standardized person crop and specializes in adult/child classification. This separation lets us identify whether an error came from localization or classification.

**Why not count every detection?** A detection exists for a frame; a count represents a person over an interval. Counting every frame would count one stationary child many times. The set of persistent track IDs solves that difference.

**Why wait before confirmation if people are sitting still?** Confirmation uses repeated observations, not movement. A still person can produce a stable box in many consecutive frames.

**How does the system handle a child behind a mother?** If the child was already tracked, the system keeps the identity in a lost state for a configured period and tries to associate the child when visible again. If the child is fully hidden and was never detected, the system cannot guarantee detection from an RGB frame.

**Why might ReID fail for a carried child?** The appearance crop may contain mostly the caregiver or a mixture of caregiver and child. The feature is therefore not necessarily a reliable signature of the child [3].

**Why use ByteTrack first?** It is a simple baseline and its method uses lower-confidence detections to recover existing tracks during difficult frames [4]. It gives a transparent starting point before adding appearance features.

**What does a small neural network contribute?** It is an optional lightweight classifier for standardized crops. Its purpose is to test whether a compact transfer-learned model can classify adult versus child accurately enough at lower computational cost. It is not a solution to full occlusion.

**What is the mathematical contribution?** The system defines functions from frames to boxes, boxes to crops, crops to probabilities, probabilities to track labels, and track IDs to a set-based count. It also uses a finite-state model for track lifecycle and explicitly defined error metrics.

**What is the limitation?** Results are valid for the tested camera conditions and annotated data. The system should not be presented as guaranteed child detection, universal age estimation, or a clinical safety device.

## References

[1]: https://arxiv.org/abs/2009.07736 "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking"

[2]: https://github.com/JonathonLuiten/TrackEval "TrackEval: Code for evaluating object tracking"

[3]: https://openaccess.thecvf.com/content/CVPR2021/html/Stadler_Improving_Multiple_Pedestrian_Tracking_by_Track_Management_and_Occlusion_Handling_CVPR_2021_paper.html "Improving Multiple Pedestrian Tracking by Track Management and Occlusion Handling"

[4]: https://arxiv.org/abs/2110.06864 "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
