"""
Run manifest — captures every parameter, version, and hash for reproducibility.

Every run creates a unique manifest.json so that results can be traced
back to their exact configuration months later (thesis requirement).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str | None:
    """Return hex SHA-256 of a file, or None if the file is inaccessible."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def create_manifest(
    config: Any,          # PipelineConfig — typed loosely to avoid circular import
    video_fps: float,
    video_total_frames: int,
    run_id: str | None = None,
) -> dict:
    """
    Build the manifest dictionary.

    Args:
        config:              Validated PipelineConfig instance.
        video_fps:           Detected FPS of the source video.
        video_total_frames:  Total frame count of the source video.
        run_id:              Optional caller-supplied run identifier.

    Returns:
        dict ready to be serialised to JSON.
    """
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "video": {
            "path": str(config.room.video_path),
            "detected_fps": video_fps,
            "total_frames": video_total_frames,
        },
        "models": {
            "detector_path": str(config.models.detector_path),
            "detector_sha256": _file_sha256(config.models.detector_path),
            "classifier_path": str(config.models.classifier_path)
            if config.models.classifier_path else None,
            "classifier_sha256": _file_sha256(config.models.classifier_path)
            if config.models.classifier_path else None,
        },
        "config_hash": config.config_hash(),
        "config_snapshot": json.loads(config.model_dump_json(exclude_none=True)),
    }
    return manifest


def save_manifest(manifest: dict, output_dir: Path) -> Path:
    """Write manifest to {output_dir}/run_manifest.json and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "run_manifest.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return out
