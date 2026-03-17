"""
Vertex — VertexPoseHub: Pose estimation provider abstraction.

Strategy pattern — PoseProvider protocol with MediaPipe implementation.
Swap to YOLOv8-pose in Phase 3 by implementing the same protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import mediapipe as mp
import numpy as np

from .models import MODEL_PATH


# ---------------------------------------------------------------------------
# Provider protocol — structural subtyping (no ABC ceremony)
# ---------------------------------------------------------------------------
@runtime_checkable
class PoseProvider(Protocol):
    def start(self) -> None: ...
    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> list | None: ...
    def stop(self) -> None: ...

    @property
    def landmark_count(self) -> int:
        """Number of landmarks returned by this provider.

        MediaPipe Pose: 33.  RTMPose-WholeBody (Phase 1.5): 133.
        Downstream code gates KSL steps 2-3 (Nocking, Hook & Grip) behind
        landmark_count >= 133 (requires hand landmarks).
        """
        ...


# ---------------------------------------------------------------------------
# MediaPipe Pose Landmarker implementation
# ---------------------------------------------------------------------------
class MediaPipePoseProvider:
    """Wraps MediaPipe Tasks API PoseLandmarker in VIDEO running mode."""

    landmark_count: int = 33  # MediaPipe Pose 33-landmark model

    def __init__(self, model_path: str = MODEL_PATH,
                 num_poses: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._model_path = model_path
        self._num_poses = num_poses
        self._det_conf = min_detection_confidence
        self._track_conf = min_tracking_confidence
        self._landmarker = None
        # Re-export for HUD fallback drawing
        self.PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils

    def start(self) -> None:
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=self._num_poses,
            min_pose_detection_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> list | None:
        """Run pose detection. Returns list of landmark lists, or None."""
        if self._landmarker is None:
            return None
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(img, timestamp_ms)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks
        return None

    def stop(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    @staticmethod
    def compute_jaw_anchor_proxy(landmarks) -> "np.ndarray":
        """Compute jaw anchor proxy: 40% R_EAR + 60% MOUTH_R.

        Exposes bio_lab.jaw_proxy() as a named PoseHub method per S3 Phase 1 spec.
        Allows callers to resolve the anchor point without importing bio_lab directly.
        """
        from .bio_lab import jaw_proxy
        return jaw_proxy(landmarks)


# ---------------------------------------------------------------------------
# Static-frame detector — IMAGE mode (developer / extraction tool only)
# ---------------------------------------------------------------------------
class StaticFrameDetector:
    """MediaPipe IMAGE mode detector for per-frame analysis.

    Not a PoseProvider: IMAGE mode is stateless per-frame with no session lifecycle.
    Use MediaPipePoseProvider for live VIDEO-mode inference.
    Input frames are BGR (OpenCV native); conversion is handled internally.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame_bgr: np.ndarray) -> list | None:
        """Run pose detection on a BGR frame. Returns landmark list or None."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(img)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks[0]
        return None

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
