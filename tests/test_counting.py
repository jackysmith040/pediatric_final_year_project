"""
Unit tests for the distinct child and adult counter.

Verifies:
  - same track ID is never counted twice
  - uncertain tracks are not counted
  - adult tracks counted when enabled and excluded when disabled
  - total count correctness
"""
import pytest
from pediatric_counter.counting.distinct_counter import DistinctChildCounter
from pediatric_counter.counting.lifecycle import (
    TrackRecord,
    TrackState,
)


def confirmed_child(tid: int) -> TrackRecord:
    rec = TrackRecord(track_id=tid, state=TrackState.CONFIRMED)
    rec.track_label = "child"
    return rec


def confirmed_adult(tid: int) -> TrackRecord:
    rec = TrackRecord(track_id=tid, state=TrackState.CONFIRMED)
    rec.track_label = "adult"
    return rec


def uncertain_track(tid: int) -> TrackRecord:
    rec = TrackRecord(track_id=tid, state=TrackState.CONFIRMED)
    rec.track_label = "uncertain"
    return rec


class TestDistinctChildCounter:
    def test_single_child_counted_once(self):
        ctr = DistinctChildCounter()
        rec = confirmed_child(1)
        ctr.update({1: rec})
        assert ctr.distinct_child_count == 1
        assert ctr.count == 1
        # Present again — must not increment
        ctr.update({1: rec})
        assert ctr.distinct_child_count == 1

    def test_two_different_children_counted(self):
        ctr = DistinctChildCounter()
        ctr.update({1: confirmed_child(1), 2: confirmed_child(2)})
        assert ctr.distinct_child_count == 2
        assert ctr.distinct_adult_count == 0

    def test_adult_counted_when_enabled(self):
        ctr = DistinctChildCounter(count_adults=True)
        ctr.update({1: confirmed_adult(1)})
        assert ctr.distinct_child_count == 0
        assert ctr.distinct_adult_count == 1
        assert ctr.total_count == 1

    def test_adult_not_counted_when_disabled(self):
        ctr = DistinctChildCounter(count_adults=False)
        ctr.update({1: confirmed_adult(1)})
        assert ctr.distinct_child_count == 0
        assert ctr.distinct_adult_count == 0
        assert ctr.total_count == 0

    def test_mixed_children_and_adults_count(self):
        ctr = DistinctChildCounter(count_adults=True, count_children=True)
        ctr.update({
            1: confirmed_child(1),
            2: confirmed_adult(2),
            3: confirmed_adult(3),
        })
        assert ctr.distinct_child_count == 1
        assert ctr.distinct_adult_count == 2
        assert ctr.total_count == 3
        assert ctr.state.counted_child_ids == [1]
        assert ctr.state.counted_adult_ids == [2, 3]

    def test_uncertain_not_counted(self):
        ctr = DistinctChildCounter()
        ctr.update({1: uncertain_track(1)})
        assert ctr.distinct_child_count == 0
        assert ctr.distinct_adult_count == 0
        assert 1 in ctr.state.uncertain_ids

    def test_uncertain_resolves_to_child(self):
        ctr = DistinctChildCounter()
        ctr.update({1: uncertain_track(1)})
        assert ctr.distinct_child_count == 0
        # Now resolves to child
        child_rec = confirmed_child(1)
        ctr.update({1: child_rec})
        assert ctr.distinct_child_count == 1
        assert 1 not in ctr.state.uncertain_ids

    def test_demographic_refinement_from_adult_to_child(self):
        """If a track is tentatively confirmed as adult but subsequent evidence confirms child, it should reclassify cleanly."""
        ctr = DistinctChildCounter(count_adults=True, count_children=True)
        adult_rec = confirmed_adult(1)
        ctr.update({1: adult_rec})
        assert ctr.distinct_adult_count == 1
        assert ctr.distinct_child_count == 0

        # Refined evidence flips track 1 to child
        child_rec = confirmed_child(1)
        ctr.update({1: child_rec})
        assert ctr.distinct_adult_count == 0
        assert ctr.distinct_child_count == 1
        assert ctr.total_count == 1

    def test_reset_clears_state(self):
        ctr = DistinctChildCounter()
        ctr.update({1: confirmed_child(1), 2: confirmed_adult(2)})
        assert ctr.total_count == 2
        ctr.reset()
        assert ctr.distinct_child_count == 0
        assert ctr.distinct_adult_count == 0
        assert ctr.total_count == 0

