"""
Distinct-child and distinct-adult counter — set cardinality implementation.

Mathematical definition (thesis spec):
    K_child = { k | τ_k achieved confirmed-child status }
    K_adult = { k | τ_k achieved confirmed-adult status }
    N_child = |K_child|
    N_adult = |K_adult|

A track ID is added to its respective set exactly once upon confirmation,
preventing per-frame count inflation.

A track ID is never removed from the set once confirmed — even if the
track is later lost or retired.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CounterState:
    """Snapshot of the counter for inclusion in run manifests and telemetry."""
    distinct_child_count: int
    distinct_adult_count: int
    counted_child_ids: list[int]
    counted_adult_ids: list[int]
    uncertain_ids: list[int]

    @property
    def counted_ids(self) -> list[int]:
        """Backward-compatible alias for counted child IDs."""
        return self.counted_child_ids


class DistinctChildCounter:
    """
    Maintains the sets K_child and K_adult of confirmed track IDs.

    - Confirmed tracks labeled 'child' enter K_child.
    - Confirmed tracks labeled 'adult' enter K_adult (if count_adults is True).
    - Uncertain tracks are logged separately and never counted until resolved.
    """

    def __init__(self, count_adults: bool = True, count_children: bool = True) -> None:
        self.count_adults = count_adults
        self.count_children = count_children

        self._counted_child_ids: set[int] = set()  # K_child
        self._counted_adult_ids: set[int] = set()  # K_adult
        self._uncertain_ids:     set[int] = set()  # flagged but unresolved

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, track_records: dict) -> None:
        """
        Ingest the latest track snapshots and update the demographic sets.

        Args:
            track_records: mapping of track_id → TrackRecord.
        """
        from pediatric_counter.counting.lifecycle import TrackState

        for tid, rec in track_records.items():
            if rec.state is TrackState.CONFIRMED:
                if rec.track_label == "child" and self.count_children:
                    if tid in self._counted_adult_ids:
                        self._counted_adult_ids.discard(tid)
                    self._counted_child_ids.add(tid)
                    rec.counted = True
                    self._uncertain_ids.discard(tid)
                elif rec.track_label == "adult" and self.count_adults:
                    if tid in self._counted_child_ids:
                        self._counted_child_ids.discard(tid)
                    self._counted_adult_ids.add(tid)
                    rec.counted = True
                    self._uncertain_ids.discard(tid)
                elif rec.track_label == "uncertain":
                    if not rec.counted:
                        self._uncertain_ids.add(tid)

    @property
    def count(self) -> int:
        """Backward-compatible count property returning N_child."""
        return len(self._counted_child_ids)

    @property
    def distinct_child_count(self) -> int:
        """Return N_child = |K_child|."""
        return len(self._counted_child_ids)

    @property
    def distinct_adult_count(self) -> int:
        """Return N_adult = |K_adult|."""
        return len(self._counted_adult_ids)

    @property
    def total_count(self) -> int:
        """Return total distinct confirmed individuals = |K_child| + |K_adult|."""
        return len(self._counted_child_ids) + len(self._counted_adult_ids)

    @property
    def state(self) -> CounterState:
        return CounterState(
            distinct_child_count=self.distinct_child_count,
            distinct_adult_count=self.distinct_adult_count,
            counted_child_ids=sorted(self._counted_child_ids),
            counted_adult_ids=sorted(self._counted_adult_ids),
            uncertain_ids=sorted(self._uncertain_ids),
        )

    def reset(self) -> None:
        """Reset for a new video run."""
        self._counted_child_ids.clear()
        self._counted_adult_ids.clear()
        self._uncertain_ids.clear()
