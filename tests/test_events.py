"""
Unit tests for detection events and timestamp indexer.
"""
from datetime import datetime
import pytest
from pediatric_counter.counting.events import (
    EventIndexer,
    format_seconds,
    parse_cctv_start_time,
)


def test_format_seconds():
    assert format_seconds(0.0) == "00:00.0"
    assert format_seconds(58.5) == "00:58.5"
    assert format_seconds(3661.0) == "01:01:01.0"


def test_parse_cctv_start_time():
    filename = "Hospital OTMC GF OPD_Hospital_Hospital_20260616055958_20260616080702_117596445 - Copy.mp4"
    dt = parse_cctv_start_time(filename)
    assert dt == datetime(2026, 6, 16, 5, 59, 58)


def test_parse_cctv_start_time_none_on_invalid():
    assert parse_cctv_start_time("invalid_video.mp4") is None


def test_event_indexer_aggregation():
    indexer = EventIndexer(fps=25.0, cctv_start_time=datetime(2026, 6, 16, 6, 0, 0))
    # Track 1 across 10 frames (0.4s)
    for f in range(100, 110):
        indexer.update(frame_idx=f, track_id=1, demographic="adult", confidence=0.9, box=[10, 10, 50, 50])
    
    events = indexer.finalize_all()
    assert len(events) == 1
    ev = events[0]
    assert ev.track_id == 1
    assert ev.demographic == "adult"
    assert ev.start_frame == 100
    assert ev.end_frame == 109
    assert ev.frame_count == 10
    assert ev.duration_seconds == pytest.approx(0.36, abs=0.05)
    assert ev.wall_clock_start == "06:00:04"


def test_event_indexer_detecting_promotion():
    indexer = EventIndexer(fps=25.0)
    # Frames 0-4: track is in initial 'detecting' state
    for f in range(5):
        indexer.update(frame_idx=f, track_id=2, demographic="detecting", confidence=0.5, box=[20, 20, 60, 60])
    # Frames 5-10: track is confirmed as 'child'
    for f in range(5, 11):
        indexer.update(frame_idx=f, track_id=2, demographic="child", confidence=0.85, box=[20, 20, 60, 60])

    events = indexer.finalize_all()
    assert len(events) == 1
    ev = events[0]
    assert ev.track_id == 2
    assert ev.demographic == "child"
    assert ev.start_frame == 0
    assert ev.end_frame == 10
