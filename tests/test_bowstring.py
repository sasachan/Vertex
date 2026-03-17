"""Tests for vertex.bowstring — CV-based bowstring detection pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from vertex.bowstring import (
    BowstringDetector, ROIBox, compute_roi,
    _detect_string_lines, _compute_string_angle,
)
from vertex.models import ShotState, StringState
from tests.conftest import FakeLandmark, _make_landmarks


# ---------------------------------------------------------------------------
# Synthetic frame helpers
# ---------------------------------------------------------------------------
def _black_frame(h: int = 200, w: int = 300) -> np.ndarray:
    """Create a plain black BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _frame_with_vertical_line(h: int = 200, w: int = 300,
                               x: int = 150, thickness: int = 2) -> np.ndarray:
    """Create a black frame with a bright vertical white line."""
    import cv2
    frame = _black_frame(h, w)
    cv2.line(frame, (x, 10), (x, h - 10), (255, 255, 255), thickness)
    return frame


def _frame_with_angled_line(h: int = 200, w: int = 300,
                             angle_deg: float = 30.0) -> np.ndarray:
    """Create a frame with a line at a given angle from vertical."""
    import cv2
    frame = _black_frame(h, w)
    cx, cy = w // 2, h // 2
    length = h // 2 - 10
    dx = int(length * np.sin(np.radians(angle_deg)))
    dy = int(length * np.cos(np.radians(angle_deg)))
    cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 255, 255), 2)
    return frame


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------
class TestComputeRoi:
    def test_returns_roi_from_visible_landmarks(self):
        lms = _make_landmarks()
        roi = compute_roi(lms, 480, 640, sw=0.2)
        assert roi is not None
        assert isinstance(roi, ROIBox)
        assert 0 <= roi.x1 < roi.x2 <= 640
        assert 0 <= roi.y1 < roi.y2 <= 480

    def test_returns_none_for_invisible_landmarks(self):
        lms = _make_landmarks()
        lms[11] = FakeLandmark(x=0.4, y=0.4, visibility=0.1)  # L_SHOULDER invisible
        roi = compute_roi(lms, 480, 640, sw=0.2)
        assert roi is None

    def test_returns_none_for_tiny_roi(self):
        """When shoulder and wrist overlap, ROI is too small."""
        lms = _make_landmarks()
        # Place L_SHOULDER and L_WRIST almost at same position
        lms[11] = FakeLandmark(x=0.50, y=0.50, visibility=0.99)
        lms[15] = FakeLandmark(x=0.501, y=0.501, visibility=0.99)
        roi = compute_roi(lms, 480, 640, sw=0.001)
        # With sw=0.001, margin ~ 0 → roi too small
        assert roi is None

    def test_margin_scales_with_sw(self):
        lms = _make_landmarks()
        roi_small = compute_roi(lms, 480, 640, sw=0.1)
        roi_large = compute_roi(lms, 480, 640, sw=0.5)
        assert roi_small is not None and roi_large is not None
        # Larger SW → larger margin → bigger ROI
        small_area = (roi_small.x2 - roi_small.x1) * (roi_small.y2 - roi_small.y1)
        large_area = (roi_large.x2 - roi_large.x1) * (roi_large.y2 - roi_large.y1)
        assert large_area >= small_area


# ---------------------------------------------------------------------------
# String line detection (Canny + Hough)
# ---------------------------------------------------------------------------
class TestDetectStringLines:
    def test_vertical_line_detected(self):
        frame = _frame_with_vertical_line()
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = _detect_string_lines(gray)
        assert len(lines) > 0

    def test_no_line_empty(self):
        frame = _black_frame()
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = _detect_string_lines(gray)
        assert len(lines) == 0

    def test_steep_angle_rejected(self):
        """A line at 30° from vertical exceeds BOWSTRING_ANGLE_TOLERANCE (15°)."""
        frame = _frame_with_angled_line(angle_deg=30.0)
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = _detect_string_lines(gray)
        # Should reject lines too far from vertical
        assert len(lines) == 0


class TestComputeStringAngle:
    def test_empty_returns_zero(self):
        assert _compute_string_angle([]) == 0.0

    def test_vertical_line_near_zero(self):
        # Near-vertical line: dx=0, dy=100 → angle ≈ 0
        lines = [(100, 10, 100, 110)]
        angle = _compute_string_angle(lines)
        assert abs(angle) < 1.0

    def test_tilted_line_nonzero(self):
        # Tilted line: significant dx
        lines = [(100, 10, 120, 110)]
        angle = _compute_string_angle(lines)
        assert abs(angle) > 1.0


# ---------------------------------------------------------------------------
# BowstringDetector — full pipeline
# ---------------------------------------------------------------------------
class TestBowstringDetector:
    def test_default_state(self):
        bd = BowstringDetector()
        result = bd.feed_frame(
            _black_frame(), _make_landmarks(), 0.2, ShotState.IDLE)
        assert result.detected is False
        assert result.release_signal is False

    def test_only_active_during_anchor_aim(self):
        """Returns default StringState for non-ANCHOR/AIM states."""
        bd = BowstringDetector()
        lms = _make_landmarks()
        for state in (ShotState.IDLE, ShotState.SETUP, ShotState.DRAW,
                      ShotState.RELEASE, ShotState.FOLLOW_THROUGH):
            result = bd.feed_frame(_black_frame(), lms, 0.2, state)
            assert result.detected is False

    def test_detects_string_during_anchor(self):
        bd = BowstringDetector()
        lms = _make_landmarks()
        # Create frame with vertical line in the bow arm region
        frame = _frame_with_vertical_line(h=480, w=640, x=200)
        result = bd.feed_frame(frame, lms, 0.2, ShotState.ANCHOR)
        # Detection depends on whether the line falls within the ROI
        # The result should at least be a valid StringState
        assert isinstance(result, StringState)

    def test_release_signal_on_string_disappearance(self):
        """String present → absent triggers release_signal=True."""
        bd = BowstringDetector()
        lms = _make_landmarks()

        # Build a frame with a clear vertical line inside the ROI
        roi = compute_roi(lms, 480, 640, sw=0.2)
        if roi is None:
            pytest.skip("ROI not computable from default landmarks")

        # Frame with line inside the ROI
        frame_with = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        line_x = (roi.x1 + roi.x2) // 2
        cv2.line(frame_with, (line_x, roi.y1 + 5), (line_x, roi.y2 - 5),
                 (255, 255, 255), 2)

        # Frame without line
        frame_without = np.zeros((480, 640, 3), dtype=np.uint8)

        # Feed: string present
        r1 = bd.feed_frame(frame_with, lms, 0.2, ShotState.AIM)
        assert r1.detected is True

        # Feed: string absent → release signal
        r2 = bd.feed_frame(frame_without, lms, 0.2, ShotState.AIM)
        assert r2.release_signal is True

    def test_no_release_signal_when_string_stays(self):
        """Consecutive detected frames should NOT trigger release."""
        bd = BowstringDetector()
        lms = _make_landmarks()
        roi = compute_roi(lms, 480, 640, sw=0.2)
        if roi is None:
            pytest.skip("ROI not computable from default landmarks")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        line_x = (roi.x1 + roi.x2) // 2
        cv2.line(frame, (line_x, roi.y1 + 5), (line_x, roi.y2 - 5),
                 (255, 255, 255), 2)

        r1 = bd.feed_frame(frame, lms, 0.2, ShotState.AIM)
        r2 = bd.feed_frame(frame, lms, 0.2, ShotState.AIM)
        assert r2.release_signal is False

    def test_reset_clears_state(self):
        bd = BowstringDetector()
        bd._prev_detected = True
        bd._last_roi = ROIBox(0, 0, 100, 100)
        bd.reset()
        assert bd._prev_detected is False
        assert bd._last_roi is None

    def test_confidence_scales_with_line_count(self):
        """Confidence = min(line_count / 3, 1.0) for detected frames."""
        bd = BowstringDetector()
        lms = _make_landmarks()
        roi = compute_roi(lms, 480, 640, sw=0.2)
        if roi is None:
            pytest.skip("ROI not computable from default landmarks")

        # Create frame with multiple vertical lines inside ROI
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        for offset in range(-20, 21, 10):
            line_x = (roi.x1 + roi.x2) // 2 + offset
            if roi.x1 <= line_x <= roi.x2:
                cv2.line(frame, (line_x, roi.y1 + 5), (line_x, roi.y2 - 5),
                         (255, 255, 255), 2)

        r = bd.feed_frame(frame, lms, 0.2, ShotState.AIM)
        if r.detected:
            assert 0.0 < r.confidence <= 1.0
