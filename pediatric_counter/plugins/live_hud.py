"""
Live HUD Plugin — High-End Heads-Up Display with Glassmorphic Metrics and Interactive Controls.

Design:
- Alpha-blended semi-transparent glass cards for Children, Adults, and Total counts.
- Dynamic status pill (▶ PLAYING in Emerald, ⏸ PAUSED in Amber).
- Responsive hotkey handling:
    [Space] Pause / Resume
    [n]     Single-step forward 1 frame while paused
    [+] / [-] Adjust playback speed (0.1x to 16.0x)
    [r]     Reset playback speed to 1.0x
    [q]     Safe graceful exit
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

from pediatric_counter.plugins.base import (
    ControlSignal,
    FrameResult,
    PipelinePlugin,
    PipelineSummary,
)


class LiveHUDPlugin(PipelinePlugin):
    """Modern OpenCV HUD visualizer with glassmorphism and interactive playback controls."""

    name: str = "live_hud"
    priority: int = 90

    # Color Palette (BGR format)
    COLOR_CHILD = (118, 230, 0)      # Emerald Green (#00E676)
    COLOR_ADULT = (255, 176, 0)      # Vivid Cyan (#00B0FF)
    COLOR_TOTAL = (255, 77, 124)     # Electric Violet (#7C4DFF)
    COLOR_AMBER = (0, 165, 255)      # Warning Amber
    COLOR_TEXT_WHITE = (245, 245, 250)
    COLOR_TEXT_MUTED = (170, 180, 195)
    COLOR_CARD_BG = (22, 18, 16)     # Dark Slate Navy
    COLOR_CARD_BORDER = (55, 48, 42)

    def __init__(
        self,
        window_name: str = "Pediatric & Adult Counter - Live HUD",
        scale: float = 0.6,
        initial_speed: float = 1.0,
    ) -> None:
        self.window_name = window_name
        self.scale = max(0.2, min(2.0, float(scale)))
        self.playback_speed = max(0.1, min(16.0, float(initial_speed)))
        self.is_paused = False
        self._fps = 25.0
        self._window_opened = False
        self._last_frame_timestamp: Optional[float] = None

    def on_start(
        self,
        output_dir: Path,
        fps: float,
        total_frames: int,
        frame_shape: Tuple[int, int, int],
        config: Any = None,
    ) -> None:
        self._fps = float(fps) if fps > 0 else 25.0
        if config and hasattr(config, "artifacts"):
            if hasattr(config.artifacts, "live_view_scale"):
                self.scale = float(config.artifacts.live_view_scale)
            if hasattr(config.artifacts, "playback_speed"):
                self.playback_speed = float(config.artifacts.playback_speed)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._window_opened = True

    def on_frame(self, result: FrameResult) -> ControlSignal:
        if not self._window_opened:
            return ControlSignal.CONTINUE

        # Prepare normalized display frame: ensure low-res feeds (e.g. 480x270) are upscaled
        # to at least 1280px wide for sharp viewing and spacious HUD metric cards.
        raw_h, raw_w = result.annotated_frame.shape[:2]
        if raw_w < 1280:
            target_w = 1280
            target_h = int(raw_h * (1280.0 / raw_w))
            frame_to_show = cv2.resize(result.annotated_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        elif self.scale != 1.0:
            target_w = max(640, int(raw_w * self.scale))
            target_h = max(360, int(raw_h * self.scale))
            frame_to_show = cv2.resize(result.annotated_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_to_show = result.annotated_frame.copy()

        now = time.perf_counter()
        elapsed_proc_ms = 0.0
        if self._last_frame_timestamp is not None:
            elapsed_proc_ms = (now - self._last_frame_timestamp) * 1000.0

        # Render modern glassmorphic HUD directly on the display canvas
        self._render_hud(frame_to_show, result)

        cv2.imshow(self.window_name, frame_to_show)

        # Handle keyboard interactions with rate limiting
        sig = self._handle_keys(elapsed_proc_ms)
        self._last_frame_timestamp = time.perf_counter()
        return sig

    def _render_hud(self, img: np.ndarray, result: FrameResult) -> None:
        """Render modern glassmorphic status cards and bottom control bar."""
        h, w = img.shape[:2]

        # ── 1. Top Glassmorphic Status Bar ─────────────────────────────────────
        top_bar_h = 74
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_bar_h), self.COLOR_CARD_BG, -1)
        cv2.line(overlay, (0, top_bar_h), (w, top_bar_h), self.COLOR_CARD_BORDER, 2)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

        # Status Pill: [▶ PLAYING 1.00x] or [⏸ PAUSED]
        pill_x = 18
        pill_y = 15
        pill_w = 175
        pill_h = 42

        status_color = self.COLOR_AMBER if self.is_paused else self.COLOR_CHILD
        status_label = "|| PAUSED" if self.is_paused else f"> PLAY {self.playback_speed:.2f}x"

        # Pill background
        cv2.rectangle(img, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (35, 28, 25), -1)
        cv2.rectangle(img, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), status_color, 2)
        cv2.putText(
            img,
            status_label,
            (pill_x + 14, pill_y + 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Metric Cards (Children, Adults, Total)
        card_x = pill_x + pill_w + 24
        spacing = 210

        # Children Card
        self._draw_metric_card(
            img,
            x=card_x,
            y=15,
            w=190,
            h=42,
            label="CHILDREN",
            value=str(result.counted_child_count),
            accent_color=self.COLOR_CHILD,
        )

        # Adults Card
        self._draw_metric_card(
            img,
            x=card_x + spacing,
            y=15,
            w=190,
            h=42,
            label="ADULTS",
            value=str(result.counted_adult_count),
            accent_color=self.COLOR_ADULT,
        )

        # Total Card
        self._draw_metric_card(
            img,
            x=card_x + spacing * 2,
            y=15,
            w=190,
            h=42,
            label="TOTAL",
            value=str(result.total_count),
            accent_color=self.COLOR_TOTAL,
        )

        # Frame and CCTV Timestamp on top right (measured right-aligned)
        meta_str = f"Frame {result.frame_idx}"
        if result.cctv_timestamp_str:
            meta_str += f" | {result.cctv_timestamp_str}"
        (tw, _), _ = cv2.getTextSize(meta_str, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        x_meta = max(card_x + spacing * 3 + 10, w - tw - 24)
        cv2.putText(
            img,
            meta_str,
            (x_meta, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.COLOR_TEXT_MUTED,
            2,
            cv2.LINE_AA,
        )

        # ── 2. Bottom Glassmorphic Control Strip ───────────────────────────────
        bot_bar_h = 44
        bot_overlay = img.copy()
        cv2.rectangle(bot_overlay, (0, h - bot_bar_h), (w, h), (14, 12, 10), -1)
        cv2.line(bot_overlay, (0, h - bot_bar_h), (w, h - bot_bar_h), (40, 36, 32), 1)
        cv2.addWeighted(bot_overlay, 0.85, img, 0.15, 0, img)

        footer_text = (
            f"[Space] Pause/Play    [n] Step Frame    [+/-] Speed ({self.playback_speed:.2f}x)    "
            f"[r] Reset 1x    [q] Exit"
        )
        cv2.putText(
            img,
            footer_text,
            (24, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            self.COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )

    def _draw_metric_card(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        value: str,
        accent_color: Tuple[int, int, int],
    ) -> None:
        """Render a compact metric badge with colored indicator strip."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (32, 26, 24), -1)
        cv2.rectangle(img, (x, y), (x + 5, y + h), accent_color, -1)  # accent bar
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 52, 48), 1)

        cv2.putText(
            img,
            label,
            (x + 14, y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            value,
            (x + w - 45, y + 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            accent_color,
            2,
            cv2.LINE_AA,
        )

    def _handle_keys(self, elapsed_proc_ms: float = 0.0) -> ControlSignal:
        """Process keyboard events with adaptive real-time responsiveness without artificial lag."""
        if self.playback_speed <= 0:
            wait_ms = 1
        else:
            base_frame_ms = max(1.0, 1000.0 / self._fps)
            target_period_ms = base_frame_ms / self.playback_speed
            remaining_ms = target_period_ms - elapsed_proc_ms
            wait_ms = max(1, int(round(remaining_ms)))

        # If currently paused, loop until unpaused, stepped, or quit
        if self.is_paused:
            while self.is_paused:
                k = cv2.waitKey(25) & 0xFF
                if k == ord("q"):
                    return ControlSignal.QUIT
                elif k == ord(" "):
                    self.is_paused = False
                    return ControlSignal.CONTINUE
                elif k == ord("n"):
                    return ControlSignal.STEP
                elif k in (ord("+"), ord("="), ord("f")):
                    self.playback_speed = min(16.0, round(self.playback_speed * 1.5, 2))
                    return ControlSignal.SPEED_CHANGE
                elif k in (ord("-"), ord("_"), ord("s")):
                    self.playback_speed = max(0.1, round(self.playback_speed / 1.5, 2))
                    return ControlSignal.SPEED_CHANGE
                elif k == ord("r"):
                    self.playback_speed = 1.0
                    return ControlSignal.SPEED_CHANGE
                elif k == 255:  # no key
                    continue

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            return ControlSignal.QUIT
        elif key == ord(" "):
            self.is_paused = True
            return ControlSignal.PAUSE
        elif key in (ord("+"), ord("="), ord("f")):
            self.playback_speed = min(16.0, round(self.playback_speed * 1.5, 2))
            return ControlSignal.SPEED_CHANGE
        elif key in (ord("-"), ord("_"), ord("s")):
            self.playback_speed = max(0.1, round(self.playback_speed / 1.5, 2))
            return ControlSignal.SPEED_CHANGE
        elif key == ord("r"):
            self.playback_speed = 1.0
            return ControlSignal.SPEED_CHANGE

        return ControlSignal.CONTINUE

    def close(self) -> None:
        if self._window_opened:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._window_opened = False
