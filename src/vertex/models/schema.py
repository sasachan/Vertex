"""
Vertex — Data schema: enums, dataclasses, CSV headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# State enum — S3 VertexActionLogic canonical names (Phase 1 — 7-state)
# ---------------------------------------------------------------------------
class ShotState(str, Enum):
    IDLE = "IDLE"
    SETUP = "SETUP"
    DRAW = "DRAW"
    ANCHOR = "ANCHOR"
    AIM = "AIM"
    RELEASE = "RELEASE"
    FOLLOW_THROUGH = "FOLLOW_THROUGH"


# ---------------------------------------------------------------------------
# Frame extraction types (used by VertexBioLab.evaluate_frame_quality)
# ---------------------------------------------------------------------------
@dataclass
class FrameMetrics:
    """Derived frame-level scalar metrics passed to gold evaluation."""
    anchor_dist: float
    vis_mean: float
    sharpness: float
    shoulder_width: float


@dataclass
class GoldCheck:
    """Single gold-standard checklist entry."""
    label: str
    value: str
    target: str
    rating: str   # GREEN | YELLOW | RED | PASS | FAIL | WARN | N/A
    source: str


@dataclass
class FrameQuality:
    """Full quality assessment for one extracted frame."""
    score: float
    metrics: FrameMetrics
    phase_hint: str           # ANCHOR/AIM | REST/DRAW
    quality_flag: str         # OK | ANCHORED_DEEP
    checklist: dict           # str -> GoldCheck-like dict; G1–G7
    validated: bool
    landmarks: list


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------
@dataclass
class BioMetrics:
    """Biomechanical measurements for a single frame."""
    anchor_dist: float
    hand_xy: np.ndarray
    jaw_xy: np.ndarray
    bsa: float            # bow shoulder angle
    dea: float            # draw elbow angle
    shoulder_tilt: float
    torso_lean: float
    draw_length: float
    dfl_angle: float
    hip_mid: np.ndarray
    draw_arm_angle: float


@dataclass
class StringState:
    """Per-frame bowstring detection result."""
    detected: bool = False         # string line found in frame
    angle: float = 0.0             # string angle from vertical (degrees)
    velocity: float = 0.0          # optical flow magnitude in string region
    confidence: float = 0.0        # Hough line vote count / expected
    release_signal: bool = False   # velocity spike or string disappearance


@dataclass
class ShotRecord:
    """Aggregated data for one completed shot."""
    shot_number: int
    hold_seconds: float
    anchor_distance_mean: float
    anchor_distance_var: float
    release_jump_x: float
    release_jump_y: float
    release_jump_mag: float
    bow_shoulder_angle: float
    draw_elbow_angle: float
    shoulder_tilt_deg: float
    torso_lean_deg: float
    draw_length_norm: float
    dfl_angle: float
    sway_range_x: float
    sway_range_y: float
    sway_velocity: float
    is_snap_shot: bool
    is_overtime: bool
    is_valid: bool
    state_sequence: str
    vertex_score: float = -1.0   # -1 = not computed; Phase 1 = anchor stability only
    flags: str = ""
    # --- KSL Phase 1 additions ---
    draw_duration_s: float = -1.0          # seconds in DRAW state
    draw_smoothness: float = -1.0          # variance of per-frame distance deltas
    draw_alignment_score: float = -1.0     # % frames BSA+DEA within GOLD during draw
    stance_width: float = -1.0             # ankle-to-ankle distance (SW-normalised)
    setup_posture_score: float = -1.0      # gc3 resting posture assessment (0-10)
    raise_smoothness: float = -1.0         # L_WRIST elevation variance during raise
    tremor_rms_wrist: float = -1.0         # RMS displacement of R_WRIST in AIM (SW-norm)
    tremor_rms_elbow: float = -1.0         # RMS displacement of R_ELBOW in AIM (SW-norm)
    transfer_shift: float = -1.0           # R_SHOULDER posterior displacement proxy (SW-norm)
    expansion_rate: float = -1.0           # linear slope of shoulder spread in AIM
    arm_drop_y: float = -1.0              # bow arm Y displacement post-release (SW-norm)
    bsa_follow_var: float = -1.0          # BSA variance during follow-through
    release_hand_angle: float = -1.0      # R_WRIST lateral trajectory angle post-release
    cv_release_detected: bool = False      # bowstring CV confirmed release
    release_confidence: str = "MEDIUM"     # HIGH / MEDIUM / LOW


CSV_HEADERS = [
    "shot_number", "timestamp_utc", "hold_seconds",
    "anchor_distance_mean", "anchor_distance_var",
    "release_jump_x", "release_jump_y", "release_jump_mag",
    "bow_shoulder_angle", "draw_elbow_angle",
    "shoulder_tilt_deg", "torso_lean_deg",
    "draw_length_norm", "dfl_angle",
    "sway_range_x", "sway_range_y", "sway_velocity",
    "is_snap_shot", "is_overtime", "is_valid",
    "vertex_score", "state_sequence",
    # KSL Phase 1 additions
    "draw_duration_s", "draw_smoothness", "draw_alignment_score",
    "stance_width", "setup_posture_score", "raise_smoothness",
    "tremor_rms_wrist", "tremor_rms_elbow",
    "transfer_shift", "expansion_rate",
    "arm_drop_y", "bsa_follow_var", "release_hand_angle",
    "cv_release_detected", "release_confidence",
]
