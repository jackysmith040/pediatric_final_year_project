"""
Detection events and special timestamp indexer.

Clusters continuous tracking observations of confirmed individuals
into discrete chronological events with elapsed video timestamps,
CCTV wall-clock timestamps, duration, and peak bounding boxes.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def parse_cctv_start_time(video_path: str | Path) -> datetime | None:
    """
    Extract starting timestamp from CCTV filename pattern.
    Example: ..._20260616055958_20260616080702_... -> 2026-06-16 05:59:58
    """
    name = Path(video_path).name
    # Look for 14-digit pattern YYYYMMDDHHMMSS
    matches = re.findall(r"(20\d{12})", name)
    if matches:
        try:
            return datetime.strptime(matches[0], "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:04.1f}"
    return f"{mins:02d}:{secs:04.1f}"


@dataclass
class DetectionEvent:
    """A continuous sighting event of a tracked individual."""
    event_id: int
    track_id: int
    demographic: str           # "child" | "adult" | "uncertain"
    start_frame: int
    end_frame: int
    start_time: str            # Video elapsed time "HH:MM:SS"
    end_time: str              # Video elapsed time "HH:MM:SS"
    duration_seconds: float
    frame_count: int
    wall_clock_start: str | None = None  # Real CCTV time if available
    wall_clock_end: str | None = None
    avg_confidence: float = 0.0
    sample_box: list[int] = field(default_factory=list)


class EventIndexer:
    """
    Indexes and aggregates per-frame detections into continuous sighting events.
    """

    def __init__(self, fps: float = 25.0, cctv_start_time: datetime | None = None) -> None:
        self.fps = fps
        self.cctv_start_time = cctv_start_time

        self._active_events: dict[int, dict[str, Any]] = {}
        self._completed_events: list[DetectionEvent] = []
        self._next_event_id = 1

    def update(
        self,
        frame_idx: int,
        track_id: int,
        demographic: str,
        confidence: float,
        box: list[int],
    ) -> None:
        """Record observation of a track in a frame."""
        elapsed_sec = frame_idx / self.fps

        if track_id not in self._active_events:
            # Start new sighting event
            self._active_events[track_id] = {
                "track_id": track_id,
                "demographic": demographic,
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "start_sec": elapsed_sec,
                "end_sec": elapsed_sec,
                "confs": [confidence],
                "sample_box": box,
            }
        else:
            # Continue active sighting
            ev = self._active_events[track_id]
            ev["end_frame"] = frame_idx
            ev["end_sec"] = elapsed_sec
            ev["confs"].append(confidence)
            # Update label if it became confirmed
            if ev["demographic"] in ("unknown", "uncertain", "detecting") and demographic in ("child", "adult"):
                ev["demographic"] = demographic

    def end_track(self, track_id: int) -> None:
        """Close sighting event when a track is lost or retired."""
        if track_id in self._active_events:
            raw = self._active_events.pop(track_id)
            self._finalize_event(raw)

    def finalize_all(self) -> list[DetectionEvent]:
        """Finalize all remaining active events at end of video."""
        for raw in list(self._active_events.values()):
            self._finalize_event(raw)
        self._active_events.clear()
        return self._completed_events

    def _finalize_event(self, raw: dict[str, Any]) -> None:
        duration = max(0.04, round(raw["end_sec"] - raw["start_sec"], 2))
        avg_conf = sum(raw["confs"]) / len(raw["confs"]) if raw["confs"] else 0.0

        wc_start = None
        wc_end = None
        if self.cctv_start_time:
            t0 = self.cctv_start_time + timedelta(seconds=raw["start_sec"])
            t1 = self.cctv_start_time + timedelta(seconds=raw["end_sec"])
            wc_start = t0.strftime("%H:%M:%S")
            wc_end = t1.strftime("%H:%M:%S")

        ev = DetectionEvent(
            event_id=self._next_event_id,
            track_id=raw["track_id"],
            demographic=raw["demographic"],
            start_frame=raw["start_frame"],
            end_frame=raw["end_frame"],
            start_time=format_seconds(raw["start_sec"]),
            end_time=format_seconds(raw["end_sec"]),
            duration_seconds=duration,
            frame_count=len(raw["confs"]),
            wall_clock_start=wc_start,
            wall_clock_end=wc_end,
            avg_confidence=round(avg_conf, 3),
            sample_box=raw["sample_box"],
        )
        self._completed_events.append(ev)
        self._next_event_id += 1

    # ── Export methods ────────────────────────────────────────────────────────

    def save_json(self, path: Path) -> None:
        """Save detection events to JSON."""
        data = [asdict(ev) for ev in self._completed_events]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, path: Path) -> None:
        """Save detection events to CSV."""
        if not self._completed_events:
            return
        fields = [
            "event_id", "track_id", "demographic", "start_frame", "end_frame",
            "start_time", "end_time", "duration_seconds", "frame_count",
            "wall_clock_start", "wall_clock_end", "avg_confidence",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for ev in self._completed_events:
                writer.writerow(asdict(ev))

    def save_markdown(self, path: Path, title: str = "CCTV Detection Event Timeline") -> None:
        """Save clean human-readable markdown table of timestamps."""
        lines = [
            f"# ⏱️ {title}\n",
            f"Total Detection Events: **{len(self._completed_events)}**\n",
            "| Event # | Track ID | Demographic | Video Time | Wall-Clock Time | Frames | Duration | Conf |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for ev in self._completed_events:
            wc = f"{ev.wall_clock_start} - {ev.wall_clock_end}" if ev.wall_clock_start else "N/A"
            lines.append(
                f"| #{ev.event_id} | #{ev.track_id} | **{ev.demographic.upper()}** | "
                f"`{ev.start_time}` - `{ev.end_time}` | `{wc}` | "
                f"{ev.start_frame}-{ev.end_frame} | {ev.duration_seconds}s | {ev.avg_confidence:.0%} |"
            )
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
