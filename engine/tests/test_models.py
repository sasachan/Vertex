"""Tests for vertex.models — schema and constants."""

from __future__ import annotations

import numpy as np
import pytest

from vertex.models import (
    ShotState, BioMetrics, ShotRecord, CSV_HEADERS, GOLD,
    TARGET_FPS, DRAW_WINDOW, ANCHOR_WINDOW,
    L_SHOULDER, R_SHOULDER, COLOR_GREEN, COLOR_RED,
    FrameMetrics, GoldCheck, FrameQuality,
    EXTRACT_VIS_THRESHOLD, EXTRACT_SHARPNESS_MIN, EXTRACT_SHARPNESS_ELITE,
    EXTRACT_ANCHOR_CAMERA_MAX, EXTRACT_ANCHOR_DRAW_MAX,
    EXTRACT_ANCHOR_IDLE_MAX, EXTRACT_ANCHOR_DEEP_FLAG,
    StringState,
    BOWSTRING_ROI_MARGIN_PCT, BOWSTRING_CANNY_LOW, BOWSTRING_CANNY_HIGH,
    BOWSTRING_HOUGH_THRESHOLD, BOWSTRING_HOUGH_MIN_LENGTH,
    BOWSTRING_HOUGH_MAX_GAP, BOWSTRING_ANGLE_TOLERANCE,
    TREMOR_MIN_FRAMES, TRANSFER_WINDOW_FRAMES, EXPANSION_MIN_FRAMES,
)
from vertex.models.schema import ShotState as SchemaShotState
from vertex.models.constants import TARGET_FPS as ConstTargetFPS
from vertex.models.gold import GOLD as GoldDict
from vertex.models.display import COLOR_GREEN as DisplayGreen


class TestShotState:
    def test_values(self):
        assert ShotState.IDLE.value == "IDLE"
        assert ShotState.DRAW.value == "DRAW"
        assert ShotState.ANCHOR.value == "ANCHOR"
        assert ShotState.RELEASE.value == "RELEASE"

    def test_phase1_states_exist(self):
        # Phase 1 — 7-state machine
        assert ShotState.SETUP.value == "SETUP"
        assert ShotState.AIM.value == "AIM"
        assert ShotState.FOLLOW_THROUGH.value == "FOLLOW_THROUGH"

    def test_string_enum(self):
        assert str(ShotState.IDLE) == "ShotState.IDLE"
        assert ShotState.IDLE == "IDLE"


class TestBioMetrics:
    def test_construction(self):
        bio = BioMetrics(
            anchor_dist=0.15,
            hand_xy=np.array([0.5, 0.3]),
            jaw_xy=np.array([0.5, 0.28]),
            bsa=92.0, dea=142.0,
            shoulder_tilt=1.0, torso_lean=0.5,
            draw_length=1.8, dfl_angle=2.0,
            hip_mid=np.array([0.5, 0.7]),
            draw_arm_angle=142.0,
        )
        assert bio.bsa == 92.0
        assert bio.anchor_dist == 0.15


class TestShotRecord:
    def test_fields_match_csv_headers(self):
        headers = set(CSV_HEADERS)
        # ShotRecord has all CSV fields except timestamp_utc (added at write time)
        # plus the extra 'flags' field
        record_fields = {f.name for f in ShotRecord.__dataclass_fields__.values()}
        csv_only = headers - record_fields - {"timestamp_utc"}
        assert csv_only == set(), f"CSV headers not in ShotRecord: {csv_only}"

    def test_vertex_score_defaults_to_minus_one(self):
        shot = ShotRecord(
            shot_number=1, hold_seconds=2.5,
            anchor_distance_mean=0.15, anchor_distance_var=0.0005,
            release_jump_x=0.0, release_jump_y=0.0, release_jump_mag=0.0,
            bow_shoulder_angle=92.0, draw_elbow_angle=140.0,
            shoulder_tilt_deg=1.0, torso_lean_deg=0.5,
            draw_length_norm=1.8, dfl_angle=2.0,
            sway_range_x=0.01, sway_range_y=0.008, sway_velocity=0.002,
            is_snap_shot=False, is_overtime=False, is_valid=True,
            state_sequence="DRAW\u2192ANCHOR\u2192AIM\u2192RELEASE",
        )
        assert shot.vertex_score == -1.0


class TestGold:
    def test_hold_range(self):
        assert GOLD["hold_min"] < GOLD["hold_max"]

    def test_anchor_var_ordering(self):
        assert GOLD["anchor_var_elite"] < GOLD["anchor_var_good"] < GOLD["anchor_var_poor"]

    def test_shoulder_range(self):
        assert GOLD["bow_shoulder_min"] < GOLD["bow_shoulder_max"]


class TestConstants:
    def test_target_fps(self):
        assert TARGET_FPS == 30

    def test_frame_windows_positive(self):
        assert DRAW_WINDOW > 0
        assert ANCHOR_WINDOW > 0

    def test_landmark_indices(self):
        assert L_SHOULDER == 11
        assert R_SHOULDER == 12


class TestExtractConstants:
    def test_threshold_ordering(self):
        """DEEP_FLAG < DRAW_MAX < IDLE_MAX < CAMERA_MAX."""
        assert EXTRACT_ANCHOR_DEEP_FLAG < EXTRACT_ANCHOR_DRAW_MAX
        assert EXTRACT_ANCHOR_DRAW_MAX  < EXTRACT_ANCHOR_IDLE_MAX
        assert EXTRACT_ANCHOR_IDLE_MAX  < EXTRACT_ANCHOR_CAMERA_MAX

    def test_sharpness_ordering(self):
        assert EXTRACT_SHARPNESS_MIN < EXTRACT_SHARPNESS_ELITE

    def test_vis_threshold_value(self):
        assert 0.0 < EXTRACT_VIS_THRESHOLD < 1.0


class TestFrameMetrics:
    def test_construction(self):
        m = FrameMetrics(anchor_dist=0.25, vis_mean=0.90,
                         sharpness=350.0, shoulder_width=0.20)
        assert m.anchor_dist == 0.25
        assert m.sharpness == 350.0


class TestGoldCheck:
    def test_construction(self):
        gc = GoldCheck(label="Bow shoulder angle", value="92.0deg",
                       target="85-100deg", rating="GREEN", source="Shinohara 2018")
        assert gc.rating == "GREEN"
        assert gc.source == "Shinohara 2018"


class TestFrameQuality:
    def test_construction(self):
        m = FrameMetrics(anchor_dist=0.20, vis_mean=0.92,
                         sharpness=400.0, shoulder_width=0.20)
        fq = FrameQuality(
            score=0.85, metrics=m, phase_hint="ANCHOR/AIM",
            quality_flag="OK", checklist={}, validated=True, landmarks=[],
        )
        assert fq.validated is True
        assert fq.score == 0.85
        assert fq.metrics.anchor_dist == 0.20


class TestReExports:
    """Verify models/__init__.py re-exports work from all sub-modules."""

    def test_schema_reexport(self):
        assert ShotState is SchemaShotState

    def test_constants_reexport(self):
        assert TARGET_FPS == ConstTargetFPS

    def test_gold_reexport(self):
        assert GOLD is GoldDict

    def test_display_reexport(self):
        assert COLOR_GREEN == DisplayGreen


# ---------------------------------------------------------------------------
# Phase 1 — KSL additions
# ---------------------------------------------------------------------------
class TestStringState:
    def test_default_construction(self):
        ss = StringState()
        assert ss.detected is False
        assert ss.angle == 0.0
        assert ss.velocity == 0.0
        assert ss.confidence == 0.0
        assert ss.release_signal is False

    def test_custom_values(self):
        ss = StringState(detected=True, angle=3.5, velocity=12.0,
                         confidence=0.8, release_signal=True)
        assert ss.detected is True
        assert ss.angle == 3.5
        assert ss.confidence == 0.8


class TestBowstringConstants:
    def test_roi_margin_positive(self):
        assert BOWSTRING_ROI_MARGIN_PCT > 0

    def test_canny_thresholds_ordered(self):
        assert BOWSTRING_CANNY_LOW < BOWSTRING_CANNY_HIGH

    def test_hough_params_positive(self):
        assert BOWSTRING_HOUGH_THRESHOLD > 0
        assert BOWSTRING_HOUGH_MIN_LENGTH > 0
        assert BOWSTRING_HOUGH_MAX_GAP > 0

    def test_angle_tolerance_reasonable(self):
        assert 0 < BOWSTRING_ANGLE_TOLERANCE < 90


class TestTremorTransferExpansionConstants:
    def test_tremor_min_frames_positive(self):
        assert TREMOR_MIN_FRAMES > 0

    def test_transfer_window_positive(self):
        assert TRANSFER_WINDOW_FRAMES > 0

    def test_expansion_min_frames_positive(self):
        assert EXPANSION_MIN_FRAMES > 0


class TestNewGoldRanges:
    def test_tremor_rms_ordering(self):
        assert GOLD["tremor_rms_elite"] < GOLD["tremor_rms_good"]

    def test_arm_drop_ordering(self):
        assert GOLD["arm_drop_elite"] < GOLD["arm_drop_good"]

    def test_draw_smoothness_ordering(self):
        assert GOLD["draw_smoothness_elite"] < GOLD["draw_smoothness_good"]

    def test_stance_width_range(self):
        assert GOLD["stance_width_min"] < GOLD["stance_width_max"]

    def test_expansion_ordering(self):
        assert GOLD["expansion_rate_elite"] > GOLD["expansion_rate_good"]


class TestShotRecordKSLFields:
    def test_new_fields_exist(self):
        shot = ShotRecord(
            shot_number=1, hold_seconds=2.0,
            anchor_distance_mean=0.15, anchor_distance_var=0.001,
            release_jump_x=0.0, release_jump_y=0.0, release_jump_mag=0.0,
            bow_shoulder_angle=92.0, draw_elbow_angle=140.0,
            shoulder_tilt_deg=1.0, torso_lean_deg=0.5,
            draw_length_norm=1.8, dfl_angle=2.0,
            sway_range_x=0.01, sway_range_y=0.008, sway_velocity=0.002,
            is_snap_shot=False, is_overtime=False, is_valid=True,
            state_sequence="DRAW\u2192ANCHOR",
        )
        # All KSL fields should have default values
        assert shot.draw_duration_s == -1.0
        assert shot.draw_smoothness == -1.0
        assert shot.draw_alignment_score == -1.0
        assert shot.stance_width == -1.0
        assert shot.setup_posture_score == -1.0
        assert shot.raise_smoothness == -1.0
        assert shot.tremor_rms_wrist == -1.0
        assert shot.tremor_rms_elbow == -1.0
        assert shot.transfer_shift == -1.0
        assert shot.expansion_rate == -1.0
        assert shot.arm_drop_y == -1.0
        assert shot.bsa_follow_var == -1.0
        assert shot.release_hand_angle == -1.0
        assert shot.cv_release_detected is False
        assert shot.release_confidence == "MEDIUM"

    def test_fields_match_csv_headers_with_ksl(self):
        """Updated CSV_HEADERS includes all KSL fields."""
        headers = set(CSV_HEADERS)
        record_fields = {f.name for f in ShotRecord.__dataclass_fields__.values()}
        csv_only = headers - record_fields - {"timestamp_utc"}
        assert csv_only == set(), f"CSV headers not in ShotRecord: {csv_only}"
