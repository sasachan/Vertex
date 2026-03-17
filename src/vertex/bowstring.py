"""
Vertex — BowstringDetector: CV-based bowstring detection for release confirmation.

Phase 1: ROI + Gaussian Blur + Canny Edge + Hough Line Transform (string presence).
Phase 2: + MOG2 Background Subtraction + Dense Optical Flow at 120 FPS.

The bowstring under tension is geometrically straight — Hough Transform
reconstructs string position even through motion blur or lighting gaps.

Supplementary signal: fused with pose-based release detection in action_logic.
Optional module — StateMachine works without it (falls back to pose-only).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import (
    ShotState, StringState,
    BOWSTRING_CANNY_LOW, BOWSTRING_CANNY_HIGH,
    BOWSTRING_HOUGH_THRESHOLD, BOWSTRING_HOUGH_MIN_LENGTH,
    BOWSTRING_HOUGH_MAX_GAP, BOWSTRING_ANGLE_TOLERANCE,
    BOWSTRING_ROI_MARGIN_PCT,
    MIN_LANDMARK_VISIBILITY,
    L_SHOULDER, L_WRIST,
)
from .bio_lab import lm_xy, shoulder_width


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------
@dataclass
class ROIBox:
    """Pixel-space bounding box for the bow arm region."""
    x1: int
    y1: int
    x2: int
    y2: int


def compute_roi(landmarks, frame_h: int, frame_w: int,
                sw: float) -> ROIBox | None:
    """Derive ROI from bow arm landmarks (L_SHOULDER → L_WRIST + margin).

    Returns None if key landmarks are not visible.
    """
    ls = landmarks[L_SHOULDER]
    lw = landmarks[L_WRIST]
    if ls.visibility < MIN_LANDMARK_VISIBILITY or lw.visibility < MIN_LANDMARK_VISIBILITY:
        return None

    margin = int(max(sw, 0.01) * frame_w * BOWSTRING_ROI_MARGIN_PCT)

    ls_px = (int(ls.x * frame_w), int(ls.y * frame_h))
    lw_px = (int(lw.x * frame_w), int(lw.y * frame_h))

    x1 = max(0, min(ls_px[0], lw_px[0]) - margin)
    y1 = max(0, min(ls_px[1], lw_px[1]) - margin)
    x2 = min(frame_w, max(ls_px[0], lw_px[0]) + margin)
    y2 = min(frame_h, max(ls_px[1], lw_px[1]) + margin)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return ROIBox(x1=x1, y1=y1, x2=x2, y2=y2)


# ---------------------------------------------------------------------------
# String line detection via Canny + Hough
# ---------------------------------------------------------------------------
def _detect_string_lines(roi_gray: np.ndarray) -> list:
    """Run Canny edge detection + Hough Line Transform on a grayscale ROI.

    Returns list of (x1, y1, x2, y2) line segments that are near-vertical
    (within BOWSTRING_ANGLE_TOLERANCE degrees of vertical).
    """
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, BOWSTRING_CANNY_LOW, BOWSTRING_CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=BOWSTRING_HOUGH_THRESHOLD,
        minLineLength=BOWSTRING_HOUGH_MIN_LENGTH,
        maxLineGap=BOWSTRING_HOUGH_MAX_GAP,
    )
    if lines is None:
        return []

    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 1:
            continue
        angle_from_vertical = abs(float(np.degrees(np.arctan2(abs(dx), abs(dy)))))
        if angle_from_vertical <= BOWSTRING_ANGLE_TOLERANCE:
            vertical_lines.append((x1, y1, x2, y2))

    return vertical_lines


def _compute_string_angle(lines: list) -> float:
    """Compute the median angle from vertical of detected string lines."""
    if not lines:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 1:
            continue
        angle = float(np.degrees(np.arctan2(dx, dy)))
        angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


# ---------------------------------------------------------------------------
# BowstringDetector
# ---------------------------------------------------------------------------
class BowstringDetector:
    """CV-based bowstring detection for dual-signal release confirmation.

    Phase 1 pipeline (30 FPS):
        1. ROI extraction (dynamic from bow arm landmarks)
        2. Gaussian Blur (5x5)
        3. Canny Edge Detection
        4. Hough Line Transform — detects near-vertical lines

    Release detection: string present in frame N-1 but absent in frame N.
    """

    def __init__(self) -> None:
        self._prev_detected: bool = False
        self._last_roi: ROIBox | None = None

    def reset(self) -> None:
        """Clear state between shots."""
        self._prev_detected = False
        self._last_roi = None

    def feed_frame(self, frame_bgr: np.ndarray,
                   landmarks, sw: float,
                   state: ShotState) -> StringState:
        """Process one frame through the bowstring detection pipeline.

        Only active during AIM and ANCHOR states (when string is expected
        to be visible and taut). Returns default StringState for other states.

        Args:
            frame_bgr: Full BGR frame from camera/video.
            landmarks: MediaPipe pose landmarks.
            sw: Shoulder width (normalised).
            state: Current shot-cycle state.

        Returns:
            StringState with detection results.
        """
        # Only process during relevant states
        if state not in (ShotState.ANCHOR, ShotState.AIM):
            self._prev_detected = False
            return StringState()

        h, w_frame = frame_bgr.shape[:2]

        # Compute ROI from landmarks; fall back to last known
        roi = compute_roi(landmarks, h, w_frame, sw)
        if roi is None:
            roi = self._last_roi
        if roi is None:
            return StringState()
        self._last_roi = roi

        # Extract and convert ROI to grayscale
        roi_img = frame_bgr[roi.y1:roi.y2, roi.x1:roi.x2]
        if roi_img.size == 0:
            return StringState()
        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # Detect string lines
        lines = _detect_string_lines(roi_gray)
        detected = len(lines) > 0
        angle = _compute_string_angle(lines) if detected else 0.0
        confidence = min(len(lines) / 3.0, 1.0) if detected else 0.0

        # Release signal: string was present, now gone
        release_signal = self._prev_detected and not detected
        self._prev_detected = detected

        return StringState(
            detected=detected,
            angle=angle,
            velocity=0.0,  # Phase 2: optical flow
            confidence=confidence,
            release_signal=release_signal,
        )
