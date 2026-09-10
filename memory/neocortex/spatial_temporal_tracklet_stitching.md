---
neuron_id: spatial-temporal-tracklet-stitching
title: Spatial-Temporal Tracklet Stitching & History-Prioritized Containment
synaptic_weight: 48
corpus_callosum: two-stage-demographic-spatial-prior
blindspot: false
summary: Explains how history-prioritized duplicate bounding box containment and posture-tolerant spatial-temporal stitching resolve multi-tracker fragmentation and count inflation in crowded and low-resolution CCTV feeds.
---

# Spatial-Temporal Tracklet Stitching & History-Prioritized Containment

## Conceptual Foundation
Standard multi-object trackers (e.g., ByteTrack, BoT-SORT) optimize frame-to-frame Kalman state updates but frequently fragment tracklets during posture shifts (sitting to standing) or emit overlapping boxes (torso vs. full-body) on small subjects. A robust lifecycle manager resolves this via a two-tier spatial-temporal state machine:

1. **History-Prioritized Duplicate Containment**:
   - When person detectors generate nested bounding boxes ($\text{containment} \ge 0.55$ or $\text{IoU} \ge 0.35$) in the same frame, selecting the primary ID based solely on bounding box area causes newly spawned tracklets to usurp confirmed tracks.
   - Ordering by historical observations (`total_hits`) preserves the established identity while merging redundant concurrent tracklets into a single canonical record.

2. **Immediate & Cross-Posture Tracklet Stitching**:
   - Tracklets dropped by the tracker in the current frame remain in `CONFIRMED` state during detection ingestion; candidate lookup must inspect current active IDs to allow immediate frame-to-frame continuation.
   - Relaxing bounding box height ratio thresholds ($h_{\text{ratio}} \ge 0.35$) accommodates real-world child behaviors (floor play, sitting, climbing, standing) where vertical extent dynamically doubles without displacement.

3. **Cumulative Demographic Consensus**:
   - Tracks with extended histories (`total_hits > 2 * label_window`) resist end-of-track occlusion artifacts by blending cumulative historical mean ($70\%$) with local rolling-window mean ($30\%$).
