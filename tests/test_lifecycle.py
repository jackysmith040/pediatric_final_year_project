"""
Unit tests for the track lifecycle state machine.

These tests verify all state transitions deterministically:
  tentative → confirmed (enough hits)
  confirmed → lost      (misses begin)
  lost      → retired   (timeout exceeded)
  lost      → confirmed (recovery)
  tentative → retired   (early miss)
"""
import pytest
from pediatric_counter.counting.lifecycle import LifecycleManager, TrackState


def make_manager(**kwargs) -> LifecycleManager:
    defaults = dict(
        min_confirmed_observations=3,
        confirmation_window_frames=10,
        min_child_probability=0.70,
        label_window_size=5,
        uncertain_margin=0.10,
        max_lost_frames=5,
    )
    defaults.update(kwargs)
    return LifecycleManager(**defaults)


class TestTentativeToConfirmed:
    def test_track_starts_tentative(self):
        mgr = make_manager()
        mgr.update(active_ids={1}, child_probs={1: 0.8}, frame_idx=0)
        assert mgr._tracks[1].state is TrackState.TENTATIVE

    def test_confirmed_after_min_hits(self):
        mgr = make_manager(min_confirmed_observations=3)
        for i in range(3):
            mgr.update(active_ids={1}, child_probs={1: 0.8}, frame_idx=i)
        assert mgr._tracks[1].state is TrackState.CONFIRMED

    def test_not_confirmed_before_min_hits(self):
        mgr = make_manager(min_confirmed_observations=5)
        for i in range(4):
            mgr.update(active_ids={1}, child_probs={1: 0.8}, frame_idx=i)
        assert mgr._tracks[1].state is TrackState.TENTATIVE


class TestConfirmedToLost:
    def test_confirmed_becomes_lost_on_miss(self):
        mgr = make_manager(min_confirmed_observations=2)
        # Confirm the track
        for i in range(2):
            mgr.update(active_ids={1}, child_probs={1: 0.8}, frame_idx=i)
        assert mgr._tracks[1].state is TrackState.CONFIRMED
        # One miss → lost
        mgr.update(active_ids=set(), child_probs={}, frame_idx=2)
        assert mgr._tracks[1].state is TrackState.LOST

    def test_lost_recovers_on_hit(self):
        mgr = make_manager(min_confirmed_observations=2)
        for i in range(2):
            mgr.update(active_ids={1}, child_probs={1: 0.8}, frame_idx=i)
        mgr.update(active_ids=set(), child_probs={}, frame_idx=2)
        assert mgr._tracks[1].state is TrackState.LOST
        mgr.update(active_ids={1}, child_probs={1: 0.8}, frame_idx=3)
        assert mgr._tracks[1].state is TrackState.CONFIRMED  # recovered

    def test_lost_retires_after_timeout(self):
        mgr = make_manager(min_confirmed_observations=2, max_lost_frames=3)
        for i in range(2):
            mgr.update(active_ids={1}, child_probs={1: 0.8}, frame_idx=i)
        # Miss 3 frames
        for i in range(2, 5):
            mgr.update(active_ids=set(), child_probs={}, frame_idx=i)
        assert mgr._tracks[1].state is TrackState.RETIRED


class TestTentativeRetired:
    def test_tentative_retires_on_first_miss(self):
        mgr = make_manager(min_confirmed_observations=5)
        mgr.update(active_ids={1}, child_probs={1: 0.8}, frame_idx=0)
        mgr.update(active_ids=set(), child_probs={}, frame_idx=1)
        assert mgr._tracks[1].state is TrackState.RETIRED


class TestLabelComputation:
    def test_high_child_prob_labels_child(self):
        mgr = make_manager(min_confirmed_observations=3, min_child_probability=0.70)
        for i in range(3):
            mgr.update(active_ids={1}, child_probs={1: 0.90}, frame_idx=i)
        assert mgr._tracks[1].track_label == "child"

    def test_low_child_prob_labels_adult(self):
        mgr = make_manager(min_confirmed_observations=3, min_child_probability=0.70)
        for i in range(3):
            mgr.update(active_ids={1}, child_probs={1: 0.20}, frame_idx=i)
        assert mgr._tracks[1].track_label == "adult"

    def test_borderline_prob_labels_uncertain(self):
        mgr = make_manager(
            min_confirmed_observations=3,
            min_child_probability=0.70,
            uncertain_margin=0.15,
        )
        # 0.5 is right on the boundary → uncertain
        for i in range(3):
            mgr.update(active_ids={1}, child_probs={1: 0.50}, frame_idx=i)
        assert mgr._tracks[1].track_label == "uncertain"


class TestTrackletStitching:
    def test_stitching_reassociates_split_tracklet(self):
        mgr = make_manager(min_confirmed_observations=2, max_lost_frames=10, enable_stitching=True)
        box1 = (100.0, 100.0, 150.0, 200.0)
        
        # Confirm Track 1
        mgr.update(active_ids={1}, child_probs={1: 0.85}, frame_idx=0, bounding_boxes={1: box1})
        mgr.update(active_ids={1}, child_probs={1: 0.85}, frame_idx=1, bounding_boxes={1: box1})
        assert mgr._tracks[1].state is TrackState.CONFIRMED

        # Track 1 misses for 2 frames -> becomes LOST
        mgr.update(active_ids=set(), child_probs={}, frame_idx=2)
        mgr.update(active_ids=set(), child_probs={}, frame_idx=3)
        assert mgr._tracks[1].state is TrackState.LOST

        # Tracker assigns new raw ID 2 nearby (overlapping or close to previous location)
        box2 = (105.0, 102.0, 152.0, 201.0)
        mgr.update(active_ids={2}, child_probs={2: 0.88}, frame_idx=4, bounding_boxes={2: box2})

        # Track 2 should be stitched back to Track 1!
        assert mgr.get_canonical_id(2) == 1
        assert mgr._tracks[1].state is TrackState.CONFIRMED
        assert 2 in mgr._tracks[1].raw_track_ids
        assert mgr._tracks[1].last_seen_frame == 4

