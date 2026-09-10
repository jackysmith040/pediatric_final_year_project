"""
Configuration schema and loader.

All thresholds, paths, and hyper-parameters are defined here and sourced
exclusively from YAML files.  No magic numbers live in processing code.
"""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────────────

class RoomConfig(BaseModel):
    id: str
    video_path: Path
    fps: float | Literal["auto"] = "auto"
    timezone: str = "UTC"
    start_frame: int = 0
    max_frames: int | None = None
    demographic_prior: Literal["adult_dominant", "child_dominant", "neutral"] = "adult_dominant"


class ModelsConfig(BaseModel):
    detector_path: Path
    classifier_path: Path | None = None
    detector_class_ids: list[int] = Field(default_factory=lambda: [0])
    classifier_classes: list[str] = Field(default_factory=lambda: ["adult", "child"])


class CropConfig(BaseModel):
    margin_fraction: float = 0.10
    resize_mode: Literal["letterbox", "direct"] = "letterbox"
    input_size: int = 224
    padding_value: int = 114
    save_debug_crops: bool = False


class TrackingConfig(BaseModel):
    backend: str = "bytetrack"
    tracker_config: Path = Path("configs/tracker_bytetrack.yaml")
    new_track_threshold: float = 0.50
    track_high_threshold: float = 0.50
    track_low_threshold: float = 0.10
    match_threshold: float = 0.80
    max_lost_time_seconds: float = 30.0
    tracker_config: str = "assets/trackers/bytetrack.yaml"


class LifecycleConfig(BaseModel):
    min_confirmed_observations: int = 5
    confirmation_window_frames: int = 15
    min_child_probability: float = 0.70
    label_window_size: int = 10
    uncertain_margin: float = 0.10
    enable_stitching: bool = True


class CountingConfig(BaseModel):
    mode: str = "distinct_children_over_interval"
    count_only_confirmed: bool = True
    count_each_track_id_once: bool = True
    retain_lost_ids: bool = True
    count_adults: bool = True
    count_children: bool = True


class ArtifactsConfig(BaseModel):
    output_dir: Path = Path("runs")
    save_annotated_video: bool = True
    save_debug_crops: bool = False
    save_per_frame_csv: bool = True
    save_summary_json: bool = True
    live_view: bool = False
    live_view_scale: float = 0.5
    playback_speed: float = 1.0
    realtime_sync: bool = False


class ClaheConfig(BaseModel):
    enabled: bool = False
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


class SahiConfig(BaseModel):
    enabled: bool = False
    slice_height: int = 640
    slice_width: int = 640
    overlap_height_ratio: float = 0.20
    overlap_width_ratio: float = 0.20
    iou_threshold: float = 0.50


class KidsSieveConfig(BaseModel):
    enabled: bool = False
    model_path: str = "computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-kids-only.onnx"
    confidence_threshold: float = 0.25

    @field_validator("model_path", mode="before")
    @classmethod
    def coerce_path(cls, v):
        return str(v) if v is not None else ""


class ModelShadowConfig(BaseModel):
    enabled: bool = False
    model_path: str = "computer_vision_models/yolo26/onnx/onnx_fine_tuned/pediatric-smaller-dataset-trained.onnx"
    confidence_threshold: float = 0.20
    output_filename: str = "model_comparison.json"

    @field_validator("model_path", mode="before")
    @classmethod
    def coerce_path(cls, v):
        return str(v) if v is not None else ""


class PoseConfig(BaseModel):
    enabled: bool = False
    model_path: str = "computer_vision_models/yolo26/onnx/pose_models/yolo26s-pose.onnx"
    confidence_threshold: float = 0.25
    draw_on_frame: bool = True
    output_filename: str = "pose_analysis.json"

    @field_validator("model_path", mode="before")
    @classmethod
    def coerce_path(cls, v):
        return str(v) if v is not None else ""


class ReIDConfig(BaseModel):
    enabled: bool = False
    mode: Literal["advisory", "canonical_stitching"] = "advisory"
    similarity_threshold: float = 0.78
    gallery_path: str = "runs/reid_gallery/global_gallery.json"
    output_filename: str = "reid_analysis.json"
    deep_model_path: str = ""
    use_gpu: bool = False

    @field_validator("gallery_path", "deep_model_path", mode="before")
    @classmethod
    def coerce_path(cls, v):
        return str(v) if v is not None else ""


# ──────────────────────────────────────────────────────────────────────────────
# Root config
# ──────────────────────────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    room: RoomConfig
    models: ModelsConfig
    crop: CropConfig = Field(default_factory=CropConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    counting: CountingConfig = Field(default_factory=CountingConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    clahe: ClaheConfig = Field(default_factory=ClaheConfig)
    sahi: SahiConfig = Field(default_factory=SahiConfig)
    kids_sieve: KidsSieveConfig = Field(default_factory=KidsSieveConfig)
    model_shadow: ModelShadowConfig = Field(default_factory=ModelShadowConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)


    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate a YAML configuration file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def config_hash(self) -> str:
        """SHA-256 of the serialised config — used in run manifests."""
        raw = self.model_dump_json(exclude_none=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def load_config(path: str | Path) -> PipelineConfig:
    """Load and return a validated PipelineConfig from YAML."""
    return PipelineConfig.from_yaml(path)
