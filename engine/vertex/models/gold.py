"""
Vertex — Gold Standard Reference Constants (Tier 1 — Literature).

Each value is backed by peer-reviewed sports-science research.
See REQUIREMENTS.md § POC Findings for full citation list.
"""

from __future__ import annotations

GOLD: dict[str, float] = {
    # Hold duration — Hennessy & Parker 1990
    "hold_min": 2.0,
    "hold_max": 5.0,
    # Anchor stability — Soylu et al. 2006
    "anchor_var_elite": 0.0003,
    "anchor_var_good": 0.001,
    "anchor_var_poor": 0.003,
    # Bow shoulder angle — Shinohara 2018 (99.0 ± 4.5), Lin 2010 (90° opt)
    "bow_shoulder_min": 85,
    "bow_shoulder_max": 100,
    # Draw elbow angle — Shinohara 2018 (138.3 ± 5.0)
    "draw_elbow_min": 130,
    "draw_elbow_max": 155,
    # Posture alignment
    "shoulder_tilt_max": 3.0,
    "torso_lean_max": 3.0,
    # Body sway — Jacquot et al. 2025
    "sway_velocity_elite": 0.003,
    "sway_velocity_good": 0.008,
    "sway_range_elite": 0.015,
    "sway_range_good": 0.035,
    # Draw arm — Lee et al. 2024
    "draw_arm_angle_anchor": 25,
    "draw_arm_angle_delta": 1.0,
    "draw_arm_angle_release": 5.0,
    # Tremor — Lin 2010, Jacquot 2025 (CoP velocity as proxy)
    # TODO(Phase 1): tune from range data
    "tremor_rms_elite": 0.002,     # SW-normalised, initial estimate
    "tremor_rms_good": 0.005,
    # Follow-through arm drop
    # TODO(Phase 1): tune from range data
    "arm_drop_elite": 0.02,        # SW-normalised max Y drop
    "arm_drop_good": 0.05,         # larger = collapse/pluck
    # Draw profile smoothness
    # TODO(Phase 1): tune from range data
    "draw_smoothness_elite": 0.0005,   # low variance = smooth draw
    "draw_smoothness_good": 0.002,
    # Stance width — shoulder-width ratio
    "stance_width_min": 0.8,       # SW-normalised (shoulder-width apart)
    "stance_width_max": 1.2,       # slightly wider = optimal
    # Expansion rate — back-tension marker spread
    # TODO(Phase 1): tune from range data
    "expansion_rate_elite": 0.0001,    # SW/frame, subtle posterior shift
    "expansion_rate_good": 0.00005,
}
