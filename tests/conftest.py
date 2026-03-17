"""Shared test fixtures for Vertex tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from vertex.models import BioMetrics, ShotRecord


# ---------------------------------------------------------------------------
# Fake MediaPipe-style landmark
# ---------------------------------------------------------------------------
@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.99


def _make_landmarks(n: int = 33) -> list[FakeLandmark]:
    """Generate a 33-landmark array with plausible normalised coords."""
    lms = [FakeLandmark(x=0.5, y=0.5) for _ in range(n)]
    # Key landmarks for archery analysis
    lms[8] = FakeLandmark(x=0.55, y=0.25, visibility=0.95)     # R_EAR
    lms[10] = FakeLandmark(x=0.52, y=0.30, visibility=0.90)    # MOUTH_R
    lms[11] = FakeLandmark(x=0.40, y=0.40, visibility=0.99)    # L_SHOULDER
    lms[12] = FakeLandmark(x=0.60, y=0.40, visibility=0.99)    # R_SHOULDER
    lms[13] = FakeLandmark(x=0.30, y=0.50, visibility=0.98)    # L_ELBOW
    lms[14] = FakeLandmark(x=0.70, y=0.42, visibility=0.98)    # R_ELBOW
    lms[15] = FakeLandmark(x=0.22, y=0.50, visibility=0.97)    # L_WRIST
    lms[16] = FakeLandmark(x=0.75, y=0.38, visibility=0.97)    # R_WRIST
    lms[20] = FakeLandmark(x=0.54, y=0.28, visibility=0.96)    # R_INDEX (near jaw)
    lms[23] = FakeLandmark(x=0.42, y=0.70, visibility=0.99)    # L_HIP
    lms[24] = FakeLandmark(x=0.58, y=0.70, visibility=0.99)    # R_HIP
    lms[27] = FakeLandmark(x=0.42, y=0.90, visibility=0.95)    # L_ANKLE
    lms[28] = FakeLandmark(x=0.58, y=0.90, visibility=0.95)    # R_ANKLE
    return lms


@pytest.fixture
def fake_landmarks():
    """33 plausible normalised landmarks (right-handed archer)."""
    return _make_landmarks()


@pytest.fixture
def sample_bio() -> BioMetrics:
    """A plausible BioMetrics for a well-positioned archer."""
    return BioMetrics(
        anchor_dist=0.15,
        hand_xy=np.array([0.54, 0.28]),
        jaw_xy=np.array([0.535, 0.280]),
        bsa=92.0,
        dea=142.0,
        shoulder_tilt=1.5,
        torso_lean=0.8,
        draw_length=1.85,
        dfl_angle=2.0,
        hip_mid=np.array([0.50, 0.70]),
        draw_arm_angle=142.0,
    )


@pytest.fixture
def sample_shot() -> ShotRecord:
    """A sample completed shot record."""
    return ShotRecord(
        shot_number=1,
        hold_seconds=3.2,
        anchor_distance_mean=0.15,
        anchor_distance_var=0.0005,
        release_jump_x=0.01,
        release_jump_y=-0.005,
        release_jump_mag=0.011,
        bow_shoulder_angle=92.0,
        draw_elbow_angle=142.0,
        shoulder_tilt_deg=1.5,
        torso_lean_deg=0.8,
        draw_length_norm=1.85,
        dfl_angle=2.0,
        sway_range_x=0.012,
        sway_range_y=0.008,
        sway_velocity=0.002,
        is_snap_shot=False,
        is_overtime=False,
        is_valid=True,
        vertex_score=78.0,
        state_sequence="DRAW\u2192ANCHOR\u2192AIM\u2192RELEASE",
    )
