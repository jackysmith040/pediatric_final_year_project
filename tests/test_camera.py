"""
Unit tests for live camera and stream reader.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from pediatric_counter.io.camera import LiveCameraReader, detect_available_cameras


def test_live_camera_reader_mocked():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, fake_frame)
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else (640 if prop == 3 else 480)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with LiveCameraReader(source=0) as reader:
            assert reader.width == 640
            assert reader.height == 480
            assert reader.fps == 30.0

            frames_collected = []
            for f_idx, frame in reader.frames(max_frames=5):
                frames_collected.append((f_idx, frame.shape))

            assert len(frames_collected) == 5
            assert frames_collected[0][0] == 0
            assert frames_collected[4][0] == 4


def test_live_camera_unopened_raises():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="Cannot connect to live source"):
            with LiveCameraReader(source=99):
                pass


def test_detect_available_cameras():
    mock_cap = MagicMock()
    mock_cap.isOpened.side_effect = [True, False, False, False]
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, fake_frame)
    mock_cap.get.return_value = 30.0

    with patch("cv2.VideoCapture", return_value=mock_cap):
        cameras = detect_available_cameras(max_to_test=2)
        assert len(cameras) >= 1
        assert cameras[0]["index"] == 0
        assert "640x480" in cameras[0]["resolution"]

