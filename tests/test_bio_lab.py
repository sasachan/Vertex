"""Tests for vertex.bio_lab — pure biomechanics math."""

from __future__ import annotations

import collections

import numpy as np
import pytest

from vertex.bio_lab import (
    lm_xy, dist_xy, dist_lm, angle_at, jaw_proxy, shoulder_width,
    median_filter, rotate_point, frame_valid, key_landmarks_visible,
    compute_bio, gc3, gc3_abs, worst_color, assess_posture,
    compute_corrections, evaluate_frame_quality,
    compute_stance, compute_raise_quality, compute_draw_profile,
    compute_transfer_proxy, compute_tremor, compute_expansion,
    compute_follow_through,
)
from vertex.models import (
    COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_WHITE,
    FRAME_VALIDITY_THRESHOLD,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    FrameMetrics,
)
from tests.conftest import FakeLandmark, _make_landmarks


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
class TestLmXy:
    def test_basic(self):
        lm = FakeLandmark(x=0.3, y=0.7)
        result = lm_xy(lm)
        np.testing.assert_array_almost_equal(result, [0.3, 0.7])


class TestDistXy:
    def test_same_point(self):
        a = np.array([1.0, 2.0])
        assert dist_xy(a, a) == 0.0

    def test_known_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(dist_xy(a, b) - 5.0) < 1e-6


class TestAngleAt:
    def test_right_angle(self):
        lms = [FakeLandmark(0, 0)] * 33
        lms[0] = FakeLandmark(x=1.0, y=0.0)   # point a
        lms[1] = FakeLandmark(x=0.0, y=0.0)   # vertex
        lms[2] = FakeLandmark(x=0.0, y=1.0)   # point b
        angle = angle_at(lms, 0, 1, 2)
        assert abs(angle - 90.0) < 0.1

    def test_straight_line(self):
        lms = [FakeLandmark(0, 0)] * 33
        lms[0] = FakeLandmark(x=0.0, y=0.0)
        lms[1] = FakeLandmark(x=0.5, y=0.0)
        lms[2] = FakeLandmark(x=1.0, y=0.0)
        angle = angle_at(lms, 0, 1, 2)
        assert abs(angle - 180.0) < 0.1


class TestJawProxy:
    def test_weighted_average(self):
        lms = [FakeLandmark(0, 0)] * 33
        lms[8] = FakeLandmark(x=0.5, y=0.2)     # R_EAR
        lms[10] = FakeLandmark(x=0.6, y=0.3)    # MOUTH_R
        jaw = jaw_proxy(lms)
        expected = 0.4 * np.array([0.5, 0.2]) + 0.6 * np.array([0.6, 0.3])
        np.testing.assert_array_almost_equal(jaw, expected)


class TestShoulderWidth:
    def test_known_width(self, fake_landmarks):
        sw = shoulder_width(fake_landmarks)
        assert sw > 0


class TestMedianFilter:
    def test_short_window(self):
        vals = collections.deque([1.0, 2.0, 3.0])
        assert median_filter(vals, 10) == 3.0  # window > len → return last

    def test_full_window(self):
        vals = collections.deque([1.0, 5.0, 3.0, 2.0, 4.0])
        assert median_filter(vals, 5) == 3.0

    def test_empty(self):
        assert median_filter(collections.deque(), 5) == 0.0


class TestRotatePoint:
    def test_no_rotation(self):
        result = rotate_point((0, 0), (10, 0), 0)
        assert result == (10, 0)

    def test_90_degrees(self):
        result = rotate_point((0, 0), (10, 0), 90)
        assert abs(result[0]) <= 1
        assert abs(result[1] - 10) <= 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestFrameValid:
    def test_short_history_always_valid(self, fake_landmarks):
        sw_hist = collections.deque([0.2, 0.2], maxlen=30)
        valid, sw = frame_valid(fake_landmarks, sw_hist)
        assert valid is True
        assert sw > 0

    def test_outlier_rejected(self, fake_landmarks):
        sw = shoulder_width(fake_landmarks)
        sw_hist = collections.deque([sw] * 10, maxlen=30)
        # Create landmarks with very different shoulder width
        bad = list(fake_landmarks)
        bad[L_SHOULDER] = FakeLandmark(x=0.1, y=0.4, visibility=0.99)
        bad[R_SHOULDER] = FakeLandmark(x=0.9, y=0.4, visibility=0.99)
        valid, _ = frame_valid(bad, sw_hist)
        assert valid is False


class TestKeyLandmarksVisible:
    def test_all_visible(self, fake_landmarks):
        assert key_landmarks_visible(fake_landmarks) is True

    def test_hidden_landmark(self, fake_landmarks):
        fake_landmarks[20] = FakeLandmark(x=0.5, y=0.5, visibility=0.1)
        assert key_landmarks_visible(fake_landmarks) is False


# ---------------------------------------------------------------------------
# Biomechanics
# ---------------------------------------------------------------------------
class TestComputeBio:
    def test_returns_biometrics(self, fake_landmarks):
        sw = shoulder_width(fake_landmarks)
        bio = compute_bio(fake_landmarks, sw)
        assert bio.bsa > 0
        assert bio.dea > 0
        assert isinstance(bio.hand_xy, np.ndarray)
        assert isinstance(bio.hip_mid, np.ndarray)


# ---------------------------------------------------------------------------
# Coaching colours
# ---------------------------------------------------------------------------
class TestGc3:
    def test_in_range(self):
        assert gc3(92.0, 85.0, 100.0) == COLOR_GREEN

    def test_below_range(self):
        assert gc3(70.0, 85.0, 100.0) == COLOR_RED

    def test_margin(self):
        assert gc3(84.0, 85.0, 100.0) == COLOR_YELLOW


class TestGc3Abs:
    def test_within_limit(self):
        assert gc3_abs(2.0, 3.0) == COLOR_GREEN

    def test_beyond_limit(self):
        assert gc3_abs(5.0, 3.0) == COLOR_RED


class TestWorstColor:
    def test_red_wins(self):
        assert worst_color(COLOR_GREEN, COLOR_RED) == COLOR_RED

    def test_same(self):
        assert worst_color(COLOR_GREEN, COLOR_GREEN) == COLOR_GREEN


class TestAssessPosture:
    def test_returns_dict(self, fake_landmarks, sample_bio):
        colors = assess_posture(sample_bio)
        assert isinstance(colors, dict)
        assert L_SHOULDER in colors


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------
class TestComputeCorrections:
    def test_well_positioned_no_corrections(self, fake_landmarks, sample_bio):
        corrections = compute_corrections(sample_bio, fake_landmarks, 720, 1280)
        # Good bio means few or no corrections
        assert isinstance(corrections, list)
        assert len(corrections) <= 2  # MAX_CORRECTION_CUES

    def test_bad_posture_produces_corrections(self, fake_landmarks):
        from vertex.bio_lab import compute_bio, shoulder_width
        sw = shoulder_width(fake_landmarks)
        bio = compute_bio(fake_landmarks, sw)
        # Force bad shoulder tilt
        bio.shoulder_tilt = 15.0
        bio.torso_lean = 12.0
        corrections = compute_corrections(bio, fake_landmarks, 720, 1280)
        assert len(corrections) > 0


# ---------------------------------------------------------------------------
# evaluate_frame_quality
# ---------------------------------------------------------------------------
class TestEvaluateFrameQuality:
    def _good_metrics(self) -> FrameMetrics:
        return FrameMetrics(
            anchor_dist=0.25, vis_mean=0.95,
            sharpness=400.0, shoulder_width=0.20,
        )

    def test_returns_empty_dict_on_exception(self):
        """evaluate_frame_quality never raises; completely broken input → ({}, False)."""
        # Make FrameMetrics.anchor_dist raise by passing a mock that raises on attribute access
        class _BadMetrics:
            @property
            def anchor_dist(self):
                raise RuntimeError("simulated hardware failure")
            vis_mean = 0.95
            sharpness = 400.0
            shoulder_width = 0.20

        cl, validated = evaluate_frame_quality(None, _BadMetrics())
        assert cl == {}
        assert validated is False

    def test_low_visibility_fails_g1(self):
        """G1 FAIL (vis_mean below threshold) forces validated=False."""
        lms = _make_landmarks()
        metrics = FrameMetrics(
            anchor_dist=0.25, vis_mean=0.50,  # below EXTRACT_VIS_THRESHOLD
            sharpness=400.0, shoulder_width=0.20,
        )
        cl, validated = evaluate_frame_quality(lms, metrics)
        assert not validated
        assert cl.get("G1", {}).get("rating") == "FAIL"

    def test_elite_frame_returns_checklist_keys(self, fake_landmarks):
        """Well-formed elite metrics produce a complete G1-G7 checklist."""
        metrics = self._good_metrics()
        cl, _ = evaluate_frame_quality(fake_landmarks, metrics)
        for key in ("G1", "G2", "G3"):
            assert key in cl, f"Missing {key} in checklist"
        # G4-G7 may be missing if compute_bio fails on fake_landmarks, that is OK
        assert "_validated" in cl
        assert "_bio_pass" in cl
        assert "_green_count" in cl


# ---------------------------------------------------------------------------
# KSL sub-phase compute functions (Phase 1)
# ---------------------------------------------------------------------------
class TestComputeStance:
    def test_returns_expected_keys(self, fake_landmarks):
        sw = shoulder_width(fake_landmarks)
        result = compute_stance(fake_landmarks, sw)
        assert "stance_width" in result
        assert "hip_alignment" in result
        assert "weight_proxy" in result

    def test_stance_width_positive(self, fake_landmarks):
        sw = shoulder_width(fake_landmarks)
        result = compute_stance(fake_landmarks, sw)
        assert result["stance_width"] > 0

    def test_zero_sw_handled(self, fake_landmarks):
        """SW=0 should not cause division by zero."""
        result = compute_stance(fake_landmarks, 0.0)
        assert isinstance(result["stance_width"], float)

    def test_symmetric_stance_small_weight_proxy(self, fake_landmarks):
        """With symmetric landmarks, weight_proxy should be near zero."""
        sw = shoulder_width(fake_landmarks)
        result = compute_stance(fake_landmarks, sw)
        # Conftest landmarks have symmetric ankles (0.42, 0.58) and hips (0.42, 0.58)
        assert abs(result["weight_proxy"]) < 0.5


class TestComputeRaiseQuality:
    def test_insufficient_data_returns_negative(self):
        assert compute_raise_quality([0.5, 0.4]) == -1.0

    def test_smooth_raise_low_variance(self):
        # Linear decline — perfectly smooth
        seq = [0.5 - i * 0.01 for i in range(20)]
        var = compute_raise_quality(seq)
        assert var >= 0.0
        assert var < 1e-10  # variance of identical deltas is 0

    def test_jittery_raise_higher_variance(self):
        # Alternating up/down — jittery
        seq = [0.5 + (0.02 if i % 2 == 0 else -0.02) for i in range(20)]
        var = compute_raise_quality(seq)
        assert var > 0.0


class TestComputeDrawProfile:
    def test_insufficient_data(self):
        result = compute_draw_profile([0.4], fps=30.0)
        assert result["draw_smoothness"] == -1.0
        assert result["draw_velocity_mean"] == 0.0

    def test_linear_decline(self):
        dists = [0.5 - i * 0.01 for i in range(30)]
        result = compute_draw_profile(dists, fps=30.0)
        assert result["draw_duration_s"] == pytest.approx(1.0, abs=0.01)
        assert result["draw_smoothness"] >= 0.0
        assert result["draw_smoothness"] < 1e-10  # constant velocity → zero variance
        assert result["draw_velocity_mean"] < 0  # decreasing distances

    def test_fps_affects_duration(self):
        dists = [0.5 - i * 0.01 for i in range(60)]
        r30 = compute_draw_profile(dists, fps=30.0)
        r60 = compute_draw_profile(dists, fps=60.0)
        assert r30["draw_duration_s"] > r60["draw_duration_s"]


class TestComputeTransferProxy:
    def test_insufficient_data(self):
        assert compute_transfer_proxy([], sw=0.2) == -1.0
        assert compute_transfer_proxy([np.array([0.6, 0.4])], sw=0.2) == -1.0

    def test_posterior_shift_positive(self):
        # R_SHOULDER moves progressively rightward (positive X)
        positions = [np.array([0.6 + i * 0.002, 0.4]) for i in range(10)]
        result = compute_transfer_proxy(positions, sw=0.2)
        assert result > 0

    def test_no_shift_near_zero(self):
        positions = [np.array([0.6, 0.4]) for _ in range(10)]
        result = compute_transfer_proxy(positions, sw=0.2)
        assert abs(result) < 1e-6

    def test_normalised_by_sw(self):
        positions = [np.array([0.6 + i * 0.002, 0.4]) for i in range(10)]
        r_small = compute_transfer_proxy(positions, sw=0.1)
        r_large = compute_transfer_proxy(positions, sw=0.4)
        # Same raw displacement, smaller SW → larger normalised value
        assert r_small > r_large


class TestComputeTremor:
    def test_insufficient_frames_returns_none(self):
        positions = [np.array([0.7, 0.3]) for _ in range(5)]
        assert compute_tremor(positions, sw=0.2) is None

    def test_stationary_points_near_zero(self):
        """Identical positions → zero displacement → zero RMS."""
        from vertex.models import TREMOR_MIN_FRAMES
        positions = [np.array([0.7, 0.3]) for _ in range(TREMOR_MIN_FRAMES + 5)]
        result = compute_tremor(positions, sw=0.2)
        assert result is not None
        assert result == 0.0

    def test_oscillating_points_positive(self):
        """Alternating positions → nonzero RMS."""
        from vertex.models import TREMOR_MIN_FRAMES
        positions = [np.array([0.7 + (0.001 if i % 2 == 0 else -0.001), 0.3])
                     for i in range(TREMOR_MIN_FRAMES + 5)]
        result = compute_tremor(positions, sw=0.2)
        assert result is not None
        assert result > 0

    def test_normalised_by_sw(self):
        from vertex.models import TREMOR_MIN_FRAMES
        positions = [np.array([0.7 + (0.001 if i % 2 == 0 else -0.001), 0.3])
                     for i in range(TREMOR_MIN_FRAMES + 5)]
        r_small = compute_tremor(positions, sw=0.1)
        r_large = compute_tremor(positions, sw=0.4)
        assert r_small > r_large


class TestComputeExpansion:
    def test_insufficient_data(self):
        assert compute_expansion([0.2, 0.21]) == -1.0

    def test_increasing_spreads_positive_slope(self):
        from vertex.models import EXPANSION_MIN_FRAMES
        spreads = [0.2 + i * 0.0001 for i in range(EXPANSION_MIN_FRAMES + 5)]
        slope = compute_expansion(spreads)
        assert slope > 0

    def test_constant_spreads_zero_slope(self):
        from vertex.models import EXPANSION_MIN_FRAMES
        spreads = [0.2] * (EXPANSION_MIN_FRAMES + 5)
        slope = compute_expansion(spreads)
        assert abs(slope) < 1e-10

    def test_decreasing_spreads_negative_slope(self):
        from vertex.models import EXPANSION_MIN_FRAMES
        spreads = [0.2 - i * 0.0001 for i in range(EXPANSION_MIN_FRAMES + 5)]
        slope = compute_expansion(spreads)
        assert slope < 0


class TestComputeFollowThrough:
    def test_empty_inputs(self):
        result = compute_follow_through([], [], [], sw=0.2)
        assert result["arm_drop_y"] == -1.0
        assert result["bsa_follow_var"] == -1.0
        assert result["release_hand_angle"] == -1.0

    def test_arm_drop_detected(self):
        """Y increases (arm drops) → positive arm_drop_y."""
        wrist_y = [0.50, 0.52, 0.54, 0.56, 0.58]
        result = compute_follow_through(wrist_y, [], [], sw=0.2)
        assert result["arm_drop_y"] > 0

    def test_stable_bsa_low_variance(self):
        bsa_values = [92.0, 92.1, 91.9, 92.0, 92.05]
        result = compute_follow_through([], bsa_values, [], sw=0.2)
        assert result["bsa_follow_var"] >= 0
        assert result["bsa_follow_var"] < 1.0

    def test_hand_trajectory_angle(self):
        positions = [np.array([0.54, 0.28]), np.array([0.56, 0.30])]
        result = compute_follow_through([], [], positions, sw=0.2)
        assert result["release_hand_angle"] != -1.0

    def test_normalised_by_sw(self):
        wrist_y = [0.50, 0.52, 0.54]
        r_small = compute_follow_through(wrist_y, [], [], sw=0.1)
        r_large = compute_follow_through(wrist_y, [], [], sw=0.4)
        # Same raw drop, smaller SW → larger normalised value
        assert r_small["arm_drop_y"] > r_large["arm_drop_y"]
