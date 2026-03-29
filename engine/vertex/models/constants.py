"""
Vertex — Constants: landmark indices, thresholds, frame windows, paths.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (right-handed archer)
# ---------------------------------------------------------------------------
R_WRIST = 16
R_INDEX = 20
R_EAR = 8
MOUTH_R = 10
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
L_HIP = 23
R_HIP = 24
L_ANKLE = 27
R_ANKLE = 28

# ---------------------------------------------------------------------------
# Thresholds — tuned for Logitech webcam at desk distance
# ---------------------------------------------------------------------------
WRIST_JAW_DRAW_THRESHOLD = 0.50
WRIST_JAW_ANCHOR_THRESHOLD = 0.35
ANCHOR_VARIANCE_THRESHOLD = 0.003
RELEASE_JUMP_THRESHOLD = 0.08
RELEASE_CONFIRM_FRAMES = 3
MIN_ANCHOR_HOLD = 0.5

# Frame-window constants (calibrated at TARGET_FPS)
TARGET_FPS = 30
DRAW_WINDOW = 4
ANCHOR_WINDOW = 8
MEDIAN_FILTER_WINDOW = 5

IDLE_COOLDOWN = 1.0
MAX_ANCHOR_HOLD = 10.0
SNAP_SHOT_THRESHOLD = 0.5
FRAME_VALIDITY_THRESHOLD = 0.15
MIN_LANDMARK_VISIBILITY = 0.5
FOLLOW_THROUGH_FRAMES = 15

# Phase 1 — 7-state machine
SETUP_COLLECT_FRAMES = 15   # calibration sample window in SETUP (not FPS-scaled)
SETUP_TIMEOUT = 8.0         # seconds before SETUP returns to IDLE (no draw)
AIM_ENTRY_FRAMES = 10       # stable ANCHOR frames before entering AIM (FPS-scaled)

# ---------------------------------------------------------------------------
# Frame extraction quality thresholds (used by extract_frames.py dev tool)
# ---------------------------------------------------------------------------
EXTRACT_VIS_THRESHOLD    = 0.70   # all key landmarks must exceed this
EXTRACT_SHARPNESS_MIN    = 100.0  # minimum Laplacian variance — blurry frames rejected
EXTRACT_SHARPNESS_ELITE  = 300.0  # elite sharpness (G2 GREEN band)
EXTRACT_ANCHOR_CAMERA_MAX = 1.5   # above = front-facing camera — hard reject
EXTRACT_ANCHOR_DRAW_MAX  = 0.5    # below = confirmed draw/anchor phase (score bonus)
EXTRACT_ANCHOR_IDLE_MAX  = 0.6    # REST/DRAW above = idle stance, discard
EXTRACT_ANCHOR_DEEP_FLAG = 0.05   # below = ANCHORED_DEEP (compound anchor or occlusion)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_VERTEX_PKG_DIR = os.path.dirname(os.path.dirname(__file__))   # src/vertex/
SESSIONS_DIR = os.path.join(_VERTEX_PKG_DIR, "..", "..", "sessions")
MODEL_PATH = os.path.join(_VERTEX_PKG_DIR, "pose_landmarker_lite.task")

# Audio
BEEP_FREQ = 880
BEEP_DURATION_MS = 150

# ---------------------------------------------------------------------------
# Bowstring CV detection (Phase 1: Canny+Hough; Phase 2: +Optical Flow)
# ---------------------------------------------------------------------------
BOWSTRING_ROI_MARGIN_PCT = 0.15
BOWSTRING_CANNY_LOW = 50
BOWSTRING_CANNY_HIGH = 150
BOWSTRING_HOUGH_THRESHOLD = 30
BOWSTRING_HOUGH_MIN_LENGTH = 20
BOWSTRING_HOUGH_MAX_GAP = 10
BOWSTRING_ANGLE_TOLERANCE = 15.0   # degrees from vertical to count as "string"
BOWSTRING_FLOW_RELEASE_THRESHOLD = 5.0  # Phase 2 only (120 FPS optical flow)

# ---------------------------------------------------------------------------
# Tremor analysis (AIM phase — KSL step 9)
# ---------------------------------------------------------------------------
TREMOR_MIN_FRAMES = 15  # minimum AIM frames before computing tremor RMS

# ---------------------------------------------------------------------------
# Transfer proxy (AIM phase — KSL step 8)
# ---------------------------------------------------------------------------
TRANSFER_WINDOW_FRAMES = 10  # frames after ANCHOR→AIM to measure transfer

# ---------------------------------------------------------------------------
# Expansion tracking (AIM phase — KSL step 10)
# ---------------------------------------------------------------------------
EXPANSION_MIN_FRAMES = 20  # minimum AIM frames for regression fit
