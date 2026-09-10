"""
Live camera and RTSP stream reader for real-time hospital deployment.

Supports:
  - Local laptop webcam / USB camera (integer index e.g. 0, 1, 2)
  - Hospital IP CCTV network stream (RTSP / HTTP stream URL)
  - Zero-latency threaded frame-grabbing (eliminates OpenCV driver buffer lag)
  - Auto-reconnection for intermittent network RTSP streams
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger("pediatric_counter.io.camera")


class LiveCameraReader:
    """
    Reads live frames from a local USB camera or hospital RTSP network stream.
    Employs a dedicated background grabber thread to ensure zero driver-buffer
    latency during compute-heavy neural inference.
    """

    def __init__(
        self,
        source: int | str = 0,
        target_fps: float = 25.0,
        width: int | None = None,
        height: int | None = None,
        use_threading: bool = True,
    ) -> None:
        self.source = int(source) if str(source).isdigit() else str(source)
        self.target_fps = target_fps
        self.requested_width = width
        self.requested_height = height
        self.use_threading = use_threading

        self.cap: cv2.VideoCapture | None = None
        self.fps: float = target_fps
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int | None = None  # Live streams are indefinite
        self.start_wall_clock: datetime = datetime.now()

        # Threading primitives for zero-latency buffer flushing
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_idx: int = 0
        self._new_frame_event = threading.Event()
        self._consecutive_failures: int = 0

    def __enter__(self) -> LiveCameraReader:
        self._connect()
        if self.use_threading:
            self._start_grabber_thread()
        return self

    def _connect(self) -> None:
        """Initialize connection to USB camera or RTSP stream."""
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # On Windows, DirectShow backend initializes USB webcams significantly faster
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.source)
        else:
            # For RTSP, set buffer size to 1 to reduce network latency
            self.cap = cv2.VideoCapture(str(self.source))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap or not self.cap.isOpened():
            src_desc = f"Camera #{self.source}" if isinstance(self.source, int) else f"Stream '{self.source}'"
            raise RuntimeError(
                f"Cannot connect to live source: {src_desc}. "
                "Make sure the camera is plugged in and not used by another application."
            )

        if self.requested_width and self.requested_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        hw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = hw_fps if hw_fps and hw_fps > 0 else self.target_fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.start_wall_clock = datetime.now()
        self._consecutive_failures = 0

    def _start_grabber_thread(self) -> None:
        """Start background daemon thread that continually consumes frames."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._grabber_worker, daemon=True)
        self._thread.start()

        # Wait briefly for first frame to arrive
        self._new_frame_event.wait(timeout=2.0)

    def _grabber_worker(self) -> None:
        """Continuously pulls latest frames from OpenCV into memory buffer."""
        idx = 0
        while not self._stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._consecutive_failures += 1
                if self._consecutive_failures > 50:
                    # Attempt silent auto-reconnection for network streams
                    try:
                        self._connect()
                    except Exception:
                        pass
                time.sleep(0.01)
                continue

            self._consecutive_failures = 0
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_idx = idx
                idx += 1
            self._new_frame_event.set()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def frames(self, max_frames: int | None = None) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame_bgr) continuously with zero driver buffer lag."""
        if not self.cap:
            raise RuntimeError("LiveCameraReader is not opened. Use within a 'with' block.")

        frame_idx = 0
        last_yielded_idx = -1

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            if self.use_threading:
                # Wait for a fresh frame from the background thread
                if not self._new_frame_event.wait(timeout=0.5):
                    if self._stop_event.is_set():
                        break
                    continue
                self._new_frame_event.clear()

                with self._lock:
                    if self._latest_frame is None:
                        continue
                    frame = self._latest_frame.copy()
                    raw_idx = self._latest_frame_idx

                # Avoid yielding exact same frame twice in tight loops
                if raw_idx == last_yielded_idx:
                    time.sleep(0.005)
                    continue
                last_yielded_idx = raw_idx

            else:
                # Direct non-threaded read
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

            yield frame_idx, frame
            frame_idx += 1


def detect_available_cameras(max_to_test: int = 4) -> list[dict[str, Any]]:
    """Test integer camera indices to find connected USB cameras and probe properties."""
    available = []
    for idx in range(max_to_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                available.append({
                    "index": idx,
                    "resolution": f"{w}x{h}",
                    "fps": round(fps, 1),
                    "label": f"USB / Built-in Camera #{idx} ({w}x{h} @ {fps:.0f}fps)",
                })
            cap.release()
    return available
