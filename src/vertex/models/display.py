"""
Vertex — Display constants: colours, HUD parameters, reference pose config.
"""

from __future__ import annotations

from .constants import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    R_INDEX, L_HIP, R_HIP,
)

# ---------------------------------------------------------------------------
# Display colours (BGR)
# ---------------------------------------------------------------------------
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 220, 220)
COLOR_RED = (0, 0, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (160, 160, 160)
COLOR_PANEL_BG = (20, 20, 20)
COLOR_ACCENT = (255, 140, 0)

YELLOW_MARGIN_PCT = 0.15
_COLOR_SEVERITY: dict[tuple, int] = {COLOR_RED: 3, COLOR_YELLOW: 2, COLOR_GREEN: 1}

# ---------------------------------------------------------------------------
# Hold timer thresholds (HUD bar colours)
# ---------------------------------------------------------------------------
HOLD_GREEN_SECONDS = 2.0
HOLD_YELLOW_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Correction guidance constants
# ---------------------------------------------------------------------------
MAX_CORRECTION_CUES = 2
GHOST_ALPHA = 0.4
GHOST_RADIUS = 10
MIN_CORRECTION_DEG = 5.0

# ---------------------------------------------------------------------------
# Reference pose constants
# ---------------------------------------------------------------------------
REF_ALPHA = 0.35
REF_COLOR = (0, 220, 0)
REF_JOINT_RADIUS = 7
_REF_CONNECTIONS = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_INDEX),
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
]
