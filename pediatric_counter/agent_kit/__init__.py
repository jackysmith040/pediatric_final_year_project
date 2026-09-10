"""
Agent Kit - Developer helpers and diagnostic tools for rapid inspection and token conservation.
"""
from pediatric_counter.agent_kit.diagnostics import probe_video, probe_frame_clarity
from pediatric_counter.agent_kit.crop_inspector import inspect_frame_crops
from pediatric_counter.agent_kit.runner import quick_run

__all__ = [
    "probe_video",
    "probe_frame_clarity",
    "inspect_frame_crops",
    "quick_run",
]
