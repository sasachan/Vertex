"""Tests for vertex.streamer — input source abstraction."""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from vertex.streamer import (
    CameraSource, VideoSource, ImageSource,
    create_source, _IMAGE_EXTS,
)


class TestCreateSource:
    def test_digit_returns_camera(self):
        src = create_source("0")
        assert isinstance(src, CameraSource)

    def test_jpg_returns_image(self):
        src = create_source("photo.jpg")
        assert isinstance(src, ImageSource)

    def test_png_returns_image(self):
        src = create_source("photo.png")
        assert isinstance(src, ImageSource)

    def test_mp4_returns_video(self):
        src = create_source("clip.mp4")
        assert isinstance(src, VideoSource)

    def test_avi_returns_video(self):
        src = create_source("clip.avi")
        assert isinstance(src, VideoSource)

    def test_url_returns_video(self):
        src = create_source("https://example.com/video.mp4")
        assert isinstance(src, VideoSource)


class TestCameraSource:
    def test_is_live(self):
        src = CameraSource(index=0)
        assert src.is_live() is True
        assert src.fps() == 30.0
        assert src.frame_count() == -1


class TestImageSource:
    def test_reads_once(self):
        # Create a temp image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = (0, 255, 0)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            path = f.name

        try:
            src = ImageSource(path)
            assert src.open() is True
            assert src.is_live() is False
            assert src.frame_count() == 1
            assert src.fps() == 30.0

            ok, frame = src.read()
            assert ok is True
            assert frame is not None
            assert frame.shape == (100, 100, 3)

            ok2, frame2 = src.read()
            assert ok2 is False  # second read returns nothing

            src.release()
        finally:
            os.unlink(path)


class TestVideoSource:
    def test_opens_valid_video(self):
        # Create a tiny test video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, 15.0, (64, 64))
            for _ in range(10):
                frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            src = VideoSource(path)
            assert src.open() is True
            assert src.is_live() is False
            assert src.fps() == 15.0
            assert src.frame_count() == 10

            ok, frame = src.read()
            assert ok is True
            assert frame is not None

            src.release()
        finally:
            os.unlink(path)

    def test_invalid_path(self):
        src = VideoSource("nonexistent_video.mp4")
        assert src.open() is False
