"""
Finite track lifecycle state machine.

States and transitions (thesis specification):

    ┌───────────┐   enough hits   ┌─────────────┐
    │ tentative │ ──────────────► │  confirmed  │
    └───────────┘                 └──────┬──────┘
          │  missed                      │ missed
          ▼                             ▼
      (pruned)                   ┌─────────────┐   timeout   ┌─────────┐
                                 │    lost     │ ──────────► │ retired │
                                 └─────────────┘             └─────────┘
                                       │ recovered
                                       ▼
                                 ┌─────────────┐
                                 │  confirmed  │  (same track ID reused)
                                 └─────────────┘

Lifecycle transitions are deterministic — every parameter comes from config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST      = "lost"
    RETIRED   = "retired"


@dataclass
class TrackRecord:
    """All per-track state needed by the lifecycle manager."""

    track_id: int
    state: TrackState = TrackState.TENTATIVE

    # Classification history — list of child probability per frame
    child_probs: list[float] = field(default_factory=list)
    # Label assigned at track-level (child / adult / uncertain)
    track_label: Literal["child", "adult", "uncertain"] | None = None

    # How many consecutive frames the track was observed
    hit_streak: int = 0
    # How many consecutive frames the track was *not* matched
    lost_frames: int = 0
    # Total frames where this track was matched (for manifest)
    total_hits: int = 0

    # Whether this track ID has already been counted
    counted: bool = False
    # Uncertainty flag — set when evidence is ambiguous
    uncertain: bool = False

    # Observation logging
    first_seen_frame: int = 0
    last_seen_frame:  int = 0

    # Spatial bounding box and scale tracking for anthropometrics and stitching
    last_bbox: tuple[float, float, float, float] | None = None
    max_height_ratio: float = 0.0
    raw_track_ids: set[int] = field(default_factory=set)


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle Manager
# ──────────────────────────────────────────────────────────────────────────────

class LifecycleManager:
    """
    Manages track state transitions for all active tracks.

    Includes spatial-temporal tracklet stitching to eliminate track fragmentation
    when individuals are temporarily occluded or change posture.
    """

    def __init__(
        self,
        min_confirmed_observations: int = 5,
        confirmation_window_frames: int = 15,
        min_child_probability: float = 0.70,
        label_window_size: int = 10,
        uncertain_margin: float = 0.10,
        max_lost_frames: int = 30,        # pre-converted seconds → frames
        enable_stitching: bool = True,
        stitching_max_distance_ratio: float = 0.18,
        stitching_min_iou: float = 0.10,
    ) -> None:
        self.min_confirmed_obs    = min_confirmed_observations
        self.confirm_window       = confirmation_window_frames
        self.min_child_prob       = min_child_probability
        self.label_window         = label_window_size
        self.uncertain_margin     = uncertain_margin
        self.max_lost_frames      = max_lost_frames
        self.enable_stitching     = enable_stitching
        self.stitching_max_dist   = stitching_max_distance_ratio
        self.stitching_min_iou    = stitching_min_iou

        self._tracks: dict[int, TrackRecord] = {}
        # Mapping of raw tracker ID → canonical stitched TrackRecord ID
        self._raw_to_canonical: dict[int, int] = {}

    def get_canonical_id(self, raw_tid: int) -> int:
        """Return the stitched canonical track ID for a raw tracker ID."""
        return self._raw_to_canonical.get(raw_tid, raw_tid)

    # ── Public interface ──────────────────────────────────────────────────────

    def update(
        self,
        active_ids: set[int],
        child_probs: dict[int, float],
        frame_idx: int,
        bounding_boxes: dict[int, tuple[float, float, float, float]] | None = None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> dict[int, TrackRecord]:
        """
        Advance the state machine one frame with optional tracklet stitching.

        Args:
            active_ids:      Track IDs matched by the tracker in this frame.
            child_probs:     Map of track_id → classifier P(child) for matched tracks.
            frame_idx:       Current frame index (0-based).
            bounding_boxes:  Map of track_id → (x1, y1, x2, y2) bounding box.
            frame_shape:     (height, width, channels) of current video frame.

        Returns:
            Snapshot of all non-retired track records keyed by canonical ID.
        """
        active_canonical_ids: set[int] = set()
        h_frame = float(frame_shape[0]) if frame_shape else 1080.0
        w_frame = float(frame_shape[1]) if frame_shape else 1920.0
        diag = (h_frame ** 2 + w_frame ** 2) ** 0.5

        # 0. Check concurrent active boxes for duplicate box containment (e.g. torso vs full-body splits)
        if self.enable_stitching and bounding_boxes and len(active_ids) > 1:
            raw_id_list = list(active_ids)
            for i in range(len(raw_id_list)):
                id_a = raw_id_list[i]
                box_a = bounding_boxes.get(id_a)
                if not box_a:
                    continue
                area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
                for j in range(i + 1, len(raw_id_list)):
                    id_b = raw_id_list[j]
                    box_b = bounding_boxes.get(id_b)
                    if not box_b:
                        continue
                    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

                    w_a = box_a[2] - box_a[0]
                    h_a = box_a[3] - box_a[1]
                    w_b = box_b[2] - box_b[0]
                    h_b = box_b[3] - box_b[1]

                    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
                    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih

                    is_duplicate = False

                    if inter > 0:
                        containment = inter / min(area_a, area_b)
                        iou = inter / (area_a + area_b - inter)

                        center_a_x = (box_a[0] + box_a[2]) / 2.0
                        center_b_x = (box_b[0] + box_b[2]) / 2.0
                        center_a_y = (box_a[1] + box_a[3]) / 2.0
                        center_b_y = (box_b[1] + box_b[3]) / 2.0

                        w_min = min(w_a, w_b)
                        h_min = min(h_a, h_b)

                        norm_dx = abs(center_a_x - center_b_x) / max(1.0, w_min)
                        norm_dy = abs(center_a_y - center_b_y) / max(1.0, h_min)

                        # Suppress duplicate detector proposals on the same person (e.g. concentric nested boxes)
                        # while preserving separate tracks for distinct individuals standing in line
                        if (iou >= 0.65) or (containment >= 0.80 and norm_dx < 0.35 and norm_dy < 0.35):
                            is_duplicate = True

                    # Also detect body-part fragmentation on seated occupants
                    # (e.g. detector fracturing a seated person into adjacent head/torso and knee/lap fragments)
                    if not is_duplicate:
                        gap_x = max(0.0, max(box_a[0], box_b[0]) - min(box_a[2], box_b[2]))
                        overlap_y = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
                        rel_overlap_y = overlap_y / max(1.0, min(h_a, h_b))

                        gap_y = max(0.0, max(box_a[1], box_b[1]) - min(box_a[3], box_b[3]))
                        overlap_x = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
                        rel_overlap_x = overlap_x / max(1.0, min(w_a, w_b))

                        ux1, uy1 = min(box_a[0], box_b[0]), min(box_a[1], box_b[1])
                        ux2, uy2 = max(box_a[2], box_b[2]), max(box_a[3], box_b[3])
                        union_area = max(1.0, (ux2 - ux1) * (uy2 - uy1))
                        area_fill = (area_a + area_b) / union_area

                        aspect_a = h_a / max(1.0, w_a)
                        aspect_b = h_b / max(1.0, w_b)

                        # Horizontal fragmentation: side-by-side touching fragments of seated occupant
                        h_frag = (
                            gap_x <= 25.0
                            and rel_overlap_y >= 0.70
                            and area_fill >= 0.75
                            and aspect_a <= 1.60
                            and aspect_b <= 1.60
                        )
                        # Vertical fragmentation: head/upper body stacked above lap/knees
                        v_frag = (
                            gap_y <= 25.0
                            and rel_overlap_x >= 0.70
                            and area_fill >= 0.75
                            and aspect_a <= 1.60
                            and aspect_b <= 1.60
                        )
                        if h_frag or v_frag:
                            is_duplicate = True

                    if is_duplicate:
                        # Prefer established track with longer history over newly spawned tracklet
                            canon_a = self.get_canonical_id(id_a)
                            canon_b = self.get_canonical_id(id_b)
                            rec_a = self._tracks.get(canon_a)
                            rec_b = self._tracks.get(canon_b)
                            hits_a = rec_a.total_hits if rec_a else 0
                            hits_b = rec_b.total_hits if rec_b else 0

                            if hits_a != hits_b:
                                primary_canon = canon_a if hits_a > hits_b else canon_b
                                secondary_raw = id_b if hits_a > hits_b else id_a
                            else:
                                primary_canon = canon_a if area_a >= area_b else canon_b
                                secondary_raw = id_b if area_a >= area_b else id_a

                            self._raw_to_canonical[secondary_raw] = primary_canon

        # 1. Resolve raw IDs to canonical IDs (with spatial-temporal stitching)
        for raw_tid in active_ids:
            box = bounding_boxes.get(raw_tid) if bounding_boxes else None

            if raw_tid in self._raw_to_canonical:
                canon_id = self._raw_to_canonical[raw_tid]
            else:
                canon_id = raw_tid
                # Attempt stitching against recently lost tracks if box is available
                if self.enable_stitching and box is not None:
                    best_match_id = self._find_stitching_candidate(box, frame_idx, diag, active_ids=active_ids)
                    if best_match_id is not None:
                        canon_id = best_match_id
                self._raw_to_canonical[raw_tid] = canon_id

            if canon_id not in self._tracks:
                rec = TrackRecord(track_id=canon_id, first_seen_frame=frame_idx)
                rec.raw_track_ids.add(raw_tid)
                self._tracks[canon_id] = rec
            else:
                self._tracks[canon_id].raw_track_ids.add(raw_tid)

            active_canonical_ids.add(canon_id)

            # Update spatial and height ratio metrics
            rec = self._tracks[canon_id]
            if box is not None:
                rec.last_bbox = box
                h_box = max(1.0, box[3] - box[1])
                h_ratio = h_box / max(1.0, h_frame)
                rec.max_height_ratio = max(rec.max_height_ratio, h_ratio)

        # 2. Update matched tracks
        for canon_id, rec in self._tracks.items():
            if canon_id in active_canonical_ids:
                prob = child_probs.get(canon_id)
                if prob is None:
                    # Look up by any associated raw tracker ID
                    for rtid in rec.raw_track_ids:
                        if rtid in child_probs:
                            prob = child_probs[rtid]
                            break
                if prob is None:
                    prob = 0.5
                self._on_hit(rec, prob, frame_idx)
            else:
                self._on_miss(rec)

        # 3. Compute track-level labels for confirmed tracks
        for rec in self._tracks.values():
            if rec.state is TrackState.CONFIRMED:
                rec.track_label = self._compute_label(rec)

        return {
            tid: rec for tid, rec in self._tracks.items()
            if rec.state is not TrackState.RETIRED
        }

    def _find_stitching_candidate(
        self,
        new_box: tuple[float, float, float, float],
        frame_idx: int,
        frame_diag: float,
        active_ids: set[int] | None = None,
    ) -> int | None:
        """Find the best recently lost or retired tracklet (or track dropped this frame) to stitch with new_box."""
        nx1, ny1, nx2, ny2 = new_box
        ncx, ncy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
        nh = max(1.0, ny2 - ny1)
        best_id: int | None = None
        best_dist = float("inf")

        for tid, rec in self._tracks.items():
            if active_ids is not None and (rec.raw_track_ids & active_ids):
                # Track is already matched to another detection in this frame
                continue
            if rec.state not in (TrackState.LOST, TrackState.RETIRED):
                # If still CONFIRMED/TENTATIVE, only stitch if it was not detected this frame
                if active_ids is None or rec.last_seen_frame >= frame_idx:
                    continue
            if rec.last_bbox is None:
                continue
            frames_lost = frame_idx - rec.last_seen_frame
            if frames_lost > self.max_lost_frames:
                continue

            ox1, oy1, ox2, oy2 = rec.last_bbox
            ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
            oh = max(1.0, oy2 - oy1)

            # Height ratio consistency (accommodates sitting to standing posture changes)
            height_ratio = min(nh, oh) / max(nh, oh)
            if height_ratio < 0.35:
                continue

            # Compute IoU
            ix1, iy1 = max(nx1, ox1), max(ny1, oy1)
            ix2, iy2 = min(nx2, ox2), min(ny2, oy2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            intersection = iw * ih
            union = (nx2 - nx1) * (ny2 - ny1) + (ox2 - ox1) * (oy2 - oy1) - intersection
            iou = intersection / max(1.0, union)

            # Center Euclidean distance normalized by frame diagonal
            center_dist = (((ncx - ocx) ** 2 + (ncy - ocy) ** 2) ** 0.5) / max(1.0, frame_diag)

            if iou >= self.stitching_min_iou or center_dist <= self.stitching_max_dist:
                if center_dist < best_dist:
                    best_dist = center_dist
                    best_id = tid

        return best_id

    def retire_all(self) -> None:
        """Retire every track (call at end of video)."""
        for rec in self._tracks.values():
            rec.state = TrackState.RETIRED

    # ── Internal transition helpers ──────────────────────────────────────────

    def _on_hit(self, rec: TrackRecord, child_prob: float, frame_idx: int) -> None:
        rec.hit_streak   += 1
        rec.total_hits   += 1
        rec.lost_frames   = 0
        rec.last_seen_frame = frame_idx
        rec.child_probs.append(child_prob)

        if rec.state is TrackState.RETIRED:
            rec.state = TrackState.TENTATIVE
            rec.hit_streak = 1

        if rec.state is TrackState.TENTATIVE:
            if rec.hit_streak >= self.min_confirmed_obs or rec.total_hits >= (self.min_confirmed_obs * 2):
                rec.state = TrackState.CONFIRMED
        elif rec.state is TrackState.LOST:
            rec.state = TrackState.CONFIRMED  # recovered

    def _on_miss(self, rec: TrackRecord) -> None:
        rec.hit_streak  = 0
        rec.lost_frames += 1

        if rec.state is TrackState.TENTATIVE:
            rec.state = TrackState.RETIRED
        elif rec.state is TrackState.CONFIRMED:
            rec.state = TrackState.LOST
        elif rec.state is TrackState.LOST:
            if rec.lost_frames >= self.max_lost_frames:
                rec.state = TrackState.RETIRED


    def _compute_label(
        self, rec: TrackRecord
    ) -> Literal["child", "adult", "uncertain"]:
        """
        Rolling-window mean over the last *label_window_size* frames.

        Returns 'uncertain' when the probability falls within *uncertain_margin*
        of the decision boundary (0.5) or within margin of min_child_prob.
        """
        window = rec.child_probs[-self.label_window :]
        if not window:
            return "uncertain"
        mean_p = sum(window) / len(window)

        # For long-confirmed tracks, blend window mean with cumulative mean to resist edge artifacts
        if len(rec.child_probs) > self.label_window * 2:
            overall_p = sum(rec.child_probs) / len(rec.child_probs)
            mean_p = 0.70 * overall_p + 0.30 * mean_p

        if abs(mean_p - 0.5) < self.uncertain_margin:
            return "uncertain"
        if mean_p >= self.min_child_prob:
            return "child"
        return "adult"

