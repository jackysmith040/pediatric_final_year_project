"""
Plugin Manager — safely orchestrates plugins with strict fault isolation.

Invariants:
- A crash or exception in ANY plugin NEVER causes the counting core to fail.
- Plugins are sorted by priority: lower numbers execute first.
- Plugins that fail repeatedly are safely disabled with an informative warning.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from pediatric_counter.plugins.base import (
    ControlSignal,
    FrameResult,
    PipelinePlugin,
    PipelineSummary,
)

logger = logging.getLogger("pediatric_counter.plugins")


class PluginManager:
    """Manages lifecycle events and dispatches calls to registered PipelinePlugins."""

    def __init__(self) -> None:
        self._plugins: List[PipelinePlugin] = []
        self._failed_plugins: set[str] = set()

    def register(self, plugin: PipelinePlugin) -> None:
        """Register a new plugin, maintaining priority order."""
        self._plugins.append(plugin)
        self._plugins.sort(key=lambda p: getattr(p, "priority", 50))
        logger.debug(f"Registered plugin '{plugin.name}' with priority {getattr(plugin, 'priority', 50)}")

    @property
    def registered_plugins(self) -> List[PipelinePlugin]:
        """Return list of active registered plugins."""
        return [p for p in self._plugins if p.name not in self._failed_plugins]

    def broadcast_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        """Broadcast session start to all registered plugins."""
        for plugin in list(self._plugins):
            if plugin.name in self._failed_plugins:
                continue
            try:
                plugin.on_start(
                    output_dir=output_dir,
                    fps=fps,
                    total_frames=total_frames,
                    frame_shape=frame_shape,
                    config=config,
                )
            except Exception as e:
                logger.warning(f"[PLUGIN ERROR] Plugin '{plugin.name}' failed in on_start(): {e}. Disabling.")
                self._failed_plugins.add(plugin.name)

    def broadcast_frame(self, result: FrameResult) -> ControlSignal:
        """
        Broadcast per-frame analytical payload to all plugins.
        Aggregates ControlSignals. Highest priority signals: QUIT > STEP > PAUSE > SPEED_CHANGE > CONTINUE.
        """
        aggregated_signal = ControlSignal.CONTINUE

        for plugin in list(self._plugins):
            if plugin.name in self._failed_plugins:
                continue
            try:
                sig = plugin.on_frame(result)
                if sig == ControlSignal.QUIT:
                    aggregated_signal = ControlSignal.QUIT
                elif sig == ControlSignal.STEP and aggregated_signal != ControlSignal.QUIT:
                    aggregated_signal = ControlSignal.STEP
                elif sig == ControlSignal.PAUSE and aggregated_signal not in (ControlSignal.QUIT, ControlSignal.STEP):
                    aggregated_signal = ControlSignal.PAUSE
                elif sig == ControlSignal.SPEED_CHANGE and aggregated_signal == ControlSignal.CONTINUE:
                    aggregated_signal = ControlSignal.SPEED_CHANGE
            except Exception as e:
                logger.warning(f"[PLUGIN ERROR] Plugin '{plugin.name}' failed in on_frame(): {e}. Disabling.")
                self._failed_plugins.add(plugin.name)

        return aggregated_signal

    def broadcast_finish(self, summary: PipelineSummary) -> None:
        """Broadcast session completion to all active plugins."""
        for plugin in list(self._plugins):
            try:
                plugin.on_finish(summary)
            except Exception as e:
                logger.warning(f"[PLUGIN ERROR] Plugin '{plugin.name}' failed in on_finish(): {e}")

    def broadcast_error(self, error: Exception) -> None:
        """Broadcast an unhandled pipeline error to plugins."""
        for plugin in list(self._plugins):
            try:
                plugin.on_error(error)
            except Exception as e:
                logger.warning(f"[PLUGIN ERROR] Plugin '{plugin.name}' failed in on_error(): {e}")

    def close_all(self) -> None:
        """Release resources on all plugins."""
        for plugin in list(self._plugins):
            try:
                plugin.close()
            except Exception as e:
                logger.warning(f"[PLUGIN ERROR] Plugin '{plugin.name}' failed in close(): {e}")
