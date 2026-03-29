"""Tests for vertex.pose_hub — PoseProvider protocol and StaticFrameDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vertex.pose_hub import MediaPipePoseProvider, StaticFrameDetector, PoseProvider


class TestPoseProviderProtocol:
    def test_media_pipe_implements_protocol(self):
        """MediaPipePoseProvider satisfies PoseProvider structural typing."""
        assert isinstance(MediaPipePoseProvider(), PoseProvider)

    def test_static_frame_detector_does_not_implement_protocol(self):
        """StaticFrameDetector is NOT a PoseProvider — it has no start()/stop()."""
        assert not isinstance(StaticFrameDetector.__new__(StaticFrameDetector), PoseProvider)


class TestStaticFrameDetector:
    def test_detect_returns_none_when_no_pose(self):
        """detect() returns None when MediaPipe finds no landmarks."""
        empty_result = MagicMock()
        empty_result.pose_landmarks = []

        mock_landmarker = MagicMock()
        mock_landmarker.detect.return_value = empty_result

        with patch("vertex.pose_hub.mp") as mock_mp:
            mock_mp.tasks.vision.PoseLandmarkerOptions = MagicMock()
            mock_mp.tasks.vision.PoseLandmarker.create_from_options.return_value = mock_landmarker
            mock_mp.tasks.vision.RunningMode.IMAGE = "IMAGE"
            mock_mp.tasks.BaseOptions = MagicMock()
            mock_mp.Image = MagicMock()
            mock_mp.ImageFormat.SRGB = "SRGB"

            detector = StaticFrameDetector.__new__(StaticFrameDetector)
            detector._landmarker = mock_landmarker

            import cv2  # noqa: PLC0415
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            result = detector.detect(frame)
            assert result is None

    def test_detect_returns_landmarks_when_pose_found(self):
        """detect() returns first landmark list when MediaPipe finds a pose."""
        fake_lms = [MagicMock()]
        pose_result = MagicMock()
        pose_result.pose_landmarks = [fake_lms]

        mock_landmarker = MagicMock()
        mock_landmarker.detect.return_value = pose_result

        with patch("vertex.pose_hub.mp") as mock_mp:
            mock_mp.tasks.vision.RunningMode.IMAGE = "IMAGE"
            mock_mp.ImageFormat.SRGB = "SRGB"
            mock_mp.Image = MagicMock()

            detector = StaticFrameDetector.__new__(StaticFrameDetector)
            detector._landmarker = mock_landmarker

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            result = detector.detect(frame)
            assert result == fake_lms

    def test_close_sets_landmarker_to_none(self):
        """close() releases the underlying MediaPipe landmarker."""
        detector = StaticFrameDetector.__new__(StaticFrameDetector)
        detector._landmarker = MagicMock()
        detector.close()
        assert detector._landmarker is None
