"""
Vertex — VertexStreamer: Input source abstraction.

Provides a unified interface for live camera, video file, image file,
and URL video inputs. Factory function auto-detects source type.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@runtime_checkable
class InputSource(Protocol):
    """Unified input source protocol."""

    def open(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...
    def fps(self) -> float: ...
    def is_live(self) -> bool: ...
    def frame_count(self) -> int: ...


class CameraSource:
    """Live webcam via cv2.VideoCapture(int)."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def fps(self) -> float:
        return 30.0

    def is_live(self) -> bool:
        return True

    def frame_count(self) -> int:
        return -1


class VideoSource:
    """Video file or direct URL via cv2.VideoCapture(str)."""

    def __init__(self, path_or_url: str):
        self._path = path_or_url
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._total: int = 0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            return False
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = raw_fps if raw_fps > 0 else 30.0
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def fps(self) -> float:
        return self._fps

    def is_live(self) -> bool:
        return False

    def frame_count(self) -> int:
        return self._total


class ImageSource:
    """Single image file. Yields one frame, then stops."""

    def __init__(self, path: str):
        self._path = path
        self._frame: np.ndarray | None = None
        self._read_once: bool = False

    def open(self) -> bool:
        self._frame = cv2.imread(self._path)
        return self._frame is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._frame is not None and not self._read_once:
            self._read_once = True
            return True, self._frame.copy()
        return False, None

    def release(self) -> None:
        self._frame = None

    def fps(self) -> float:
        return 30.0

    def is_live(self) -> bool:
        return False

    def frame_count(self) -> int:
        return 1


def create_source(arg: str) -> InputSource:
    """Auto-detect input type and return appropriate source.

    - Digits only → CameraSource(int)
    - Image extension → ImageSource
    - http:// or https:// → VideoSource (cv2 handles URLs)
    - Otherwise → VideoSource (file path)
    """
    if arg.isdigit():
        return CameraSource(index=int(arg))

    ext = os.path.splitext(arg)[1].lower()
    if ext in _IMAGE_EXTS:
        return ImageSource(arg)

    # Video file or direct URL
    return VideoSource(arg)
