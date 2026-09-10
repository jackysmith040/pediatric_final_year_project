"""
Plugins package for Pediatric & Adult CCTV Counter.
"""
from pediatric_counter.plugins.base import (
    ControlSignal,
    FrameResult,
    PipelinePlugin,
    PipelineSummary,
)
from pediatric_counter.plugins.csv_exporter import PerFrameCSVPlugin
from pediatric_counter.plugins.json_summary import JSONSummaryPlugin
from pediatric_counter.plugins.live_hud import LiveHUDPlugin
from pediatric_counter.plugins.manager import PluginManager
from pediatric_counter.plugins.timeline_exporter import TimelineExporterPlugin
from pediatric_counter.plugins.video_recorder import VideoRecorderPlugin
from pediatric_counter.plugins.clahe_plugin import CLAHEPlugin
from pediatric_counter.plugins.sahi_plugin import SAHIPlugin
from pediatric_counter.plugins.kids_sieve import KidsSievePlugin
from pediatric_counter.plugins.model_shadow import ModelShadowPlugin
from pediatric_counter.plugins.pose import PosePlugin
from pediatric_counter.plugins.reid import ReIDPlugin

__all__ = [
    "ControlSignal",
    "FrameResult",
    "PipelinePlugin",
    "PipelineSummary",
    "PluginManager",
    "PerFrameCSVPlugin",
    "JSONSummaryPlugin",
    "TimelineExporterPlugin",
    "VideoRecorderPlugin",
    "LiveHUDPlugin",
    "CLAHEPlugin",
    "SAHIPlugin",
    "KidsSievePlugin",
    "ModelShadowPlugin",
    "PosePlugin",
    "ReIDPlugin",
]

