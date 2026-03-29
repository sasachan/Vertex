"""
Vertex — VertexHUD: All cv2 rendering — skeleton, overlays, HUD, corrections.

Every draw_* function lives here. No biomechanical math — only visualisation.
Imports cv2 and numpy for rendering; all data arrives via BioMetrics or dicts.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .models import (
    GOLD, BioMetrics,
    WRIST_JAW_ANCHOR_THRESHOLD, WRIST_JAW_DRAW_THRESHOLD,
    MIN_LANDMARK_VISIBILITY, HOLD_GREEN_SECONDS, HOLD_YELLOW_SECONDS,
    GHOST_ALPHA, GHOST_RADIUS, REF_ALPHA, REF_COLOR, REF_JOINT_RADIUS,
    _REF_CONNECTIONS,
    COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_WHITE,
    COLOR_GRAY, COLOR_PANEL_BG, COLOR_ACCENT,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST,
    L_HIP, R_HIP, R_INDEX,
)
from .bio_lab import gc3_abs, worst_color


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------
def draw_skeleton(frame, landmarks, landmark_colors: dict, connections) -> None:
    """Draw pose skeleton with per-landmark assessment colours."""
    h, w = frame.shape[:2]
    pts = {}
    for i, lm in enumerate(landmarks):
        if lm.visibility > MIN_LANDMARK_VISIBILITY:
            pts[i] = (int(lm.x * w), int(lm.y * h))
    for conn in connections:
        s, e = conn.start, conn.end
        if s in pts and e in pts:
            c1 = landmark_colors.get(s, COLOR_WHITE)
            c2 = landmark_colors.get(e, COLOR_WHITE)
            cv2.line(frame, pts[s], pts[e], worst_color(c1, c2), 2, cv2.LINE_AA)
    for i, pt in pts.items():
        color = landmark_colors.get(i, COLOR_WHITE)
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Coaching overlay — DFL line, T-bar, anchor proximity
# ---------------------------------------------------------------------------
def draw_coaching_overlay(frame, landmarks, bio: BioMetrics) -> None:
    h, w = frame.shape[:2]

    def px(idx):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    def vis(idx):
        return landmarks[idx].visibility > MIN_LANDMARK_VISIBILITY

    # DFL line
    if vis(L_WRIST) and vis(R_ELBOW):
        dfl_abs = abs(bio.dfl_angle)
        dfl_c = COLOR_GREEN if dfl_abs < 5 else (
            COLOR_YELLOW if dfl_abs < 10 else COLOR_RED)
        cv2.line(frame, px(L_WRIST), px(R_ELBOW), dfl_c, 3, cv2.LINE_AA)

    # Torso T-bar
    if vis(L_SHOULDER) and vis(R_SHOULDER) and vis(L_HIP) and vis(R_HIP):
        tilt_c = gc3_abs(bio.shoulder_tilt, GOLD["shoulder_tilt_max"])
        lean_c = gc3_abs(bio.torso_lean, GOLD["torso_lean_max"])
        ls, rs = px(L_SHOULDER), px(R_SHOULDER)
        cv2.line(frame, ls, rs, tilt_c, 3, cv2.LINE_AA)
        smid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        lh, rh = px(L_HIP), px(R_HIP)
        hmid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        cv2.line(frame, smid, hmid, lean_c, 3, cv2.LINE_AA)

    # Anchor proximity label
    if vis(R_INDEX):
        pt = px(R_INDEX)
        ad = bio.anchor_dist
        if ad < WRIST_JAW_ANCHOR_THRESHOLD:
            pc = COLOR_GREEN
        elif ad < WRIST_JAW_DRAW_THRESHOLD:
            pc = COLOR_YELLOW
        else:
            pc = COLOR_GRAY
        cv2.putText(frame, f"d={ad:.2f}", (pt[0] + 12, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Correction guides — ghost targets + arrows
# ---------------------------------------------------------------------------
def draw_correction_guides(frame, corrections: list[dict]) -> None:
    for corr in corrections:
        cur = corr["current_px"]
        tgt = corr["target_px"]
        color = corr["color"]

        overlay = frame.copy()
        cv2.circle(overlay, tgt, GHOST_RADIUS, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, GHOST_ALPHA, frame, 1 - GHOST_ALPHA, 0, frame)
        cv2.circle(frame, tgt, GHOST_RADIUS, color, 1, cv2.LINE_AA)

        dx = tgt[0] - cur[0]
        dy = tgt[1] - cur[1]
        dist = max(1, int(np.sqrt(dx * dx + dy * dy)))
        thickness = min(3, max(1, dist // 40))
        if dist > 8:
            cv2.arrowedLine(frame, cur, tgt, color, thickness,
                            cv2.LINE_AA, tipLength=0.25)


def draw_correction_cues(frame, corrections: list[dict], w: int) -> None:
    if not corrections:
        return
    rx = w - 320
    ry = 75
    panel_h = 8 + len(corrections) * 22 + 8
    ov = frame.copy()
    cv2.rectangle(ov, (rx - 10, ry), (w - 10, ry + panel_h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    for i, corr in enumerate(corrections):
        cv2.putText(frame, corr["cue_text"], (rx, ry + 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, corr["color"], 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Reference pose ghost
# ---------------------------------------------------------------------------
def draw_reference_pose(frame, ref_pts: dict, live_hip_mid=None,
                        snap_hip_mid=None) -> None:
    h, w = frame.shape[:2]
    offset = np.zeros(2)
    if live_hip_mid is not None and snap_hip_mid is not None:
        offset = live_hip_mid - snap_hip_mid

    def to_px(pt):
        shifted = pt + offset
        return (int(shifted[0] * w), int(shifted[1] * h))

    overlay = frame.copy()
    for a, b in _REF_CONNECTIONS:
        if a in ref_pts and b in ref_pts:
            cv2.line(overlay, to_px(ref_pts[a]), to_px(ref_pts[b]),
                     REF_COLOR, 2, cv2.LINE_AA)
    for idx, pt in ref_pts.items():
        cv2.circle(overlay, to_px(pt), REF_JOINT_RADIUS,
                   REF_COLOR, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, REF_ALPHA, frame, 1 - REF_ALPHA, 0, frame)

    if R_SHOULDER in ref_pts:
        cv2.putText(frame, "IDEAL", (to_px(ref_pts[R_SHOULDER])[0] + 15,
                    to_px(ref_pts[R_SHOULDER])[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, REF_COLOR, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Progress bar for video playback
# ---------------------------------------------------------------------------
def draw_progress_bar(frame, current: int, total: int) -> None:
    """Draw a thin progress bar at the bottom of the frame."""
    if total <= 0:
        return
    h, w = frame.shape[:2]
    bar_h = 4
    y = h - bar_h
    progress = min(current / total, 1.0)
    filled = int(w * progress)
    cv2.rectangle(frame, (0, y), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, y), (filled, h), COLOR_ACCENT, -1)


# ---------------------------------------------------------------------------
# Main HUD
# ---------------------------------------------------------------------------
def draw_hud(frame, state: str, hold_s: float, shots: int, avg_hold: float,
             best_hold: float, fps: float, sess_path: str,
             bio: BioMetrics | None, debug: bool, flags_str: str,
             corrections: list[dict] | None = None,
             vertex_score: float = -1.0,
             string_detected: bool = False,
             tremor_rms: float = -1.0,
             release_confidence: str = "",
             setup_posture_score: float = -1.0) -> None:
    h, w = frame.shape[:2]

    # State panel
    ph = 140 if state == "ANCHOR" else 90
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (380, ph), COLOR_PANEL_BG, -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "VERTEX", (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, "TheSighter", (100, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, COLOR_GRAY, 1, cv2.LINE_AA)

    sc = {"IDLE": COLOR_GRAY, "SETUP": COLOR_ACCENT,
          "DRAW": COLOR_YELLOW, "ANCHOR": COLOR_GREEN,
          "AIM": (0, 255, 120), "RELEASE": COLOR_RED,
          "FOLLOW_THROUGH": (180, 180, 255)}.get(state, COLOR_WHITE)
    cv2.circle(frame, (30, 62), 8, sc, -1)
    cv2.putText(frame, state, (50, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.6, sc, 3, cv2.LINE_AA)

    if state == "ANCHOR" and hold_s > 0:
        tc = COLOR_GREEN if hold_s >= HOLD_GREEN_SECONDS else (
            COLOR_YELLOW if hold_s >= HOLD_YELLOW_SECONDS else COLOR_RED)
        bx, by, bw = 18, 100, 350
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 24), (60, 60, 60), -1)
        fw = int(bw * min(hold_s / 5.0, 1.0))
        cv2.rectangle(frame, (bx, by), (bx + fw, by + 24), tc, -1)
        cv2.putText(frame, f"{hold_s:.1f}s", (bx + bw + 8, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tc, 2, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:.0f} FPS", (w - 130, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, COLOR_GREEN if fps >= 25 else COLOR_RED, 2, cv2.LINE_AA)

    # Debug bio gauges
    if bio and debug:
        rx, ry = w - 320, 90
        ov2 = frame.copy()
        cv2.rectangle(ov2, (rx - 10, ry - 10), (w - 10, ry + 178), COLOR_PANEL_BG, -1)
        cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)

        def gc(v, lo, hi):
            return COLOR_GREEN if lo <= v <= hi else COLOR_RED

        items = [
            (f"Bow Shldr: {bio.bsa:.0f} deg",
             gc(bio.bsa, GOLD["bow_shoulder_min"], GOLD["bow_shoulder_max"])),
            (f"Draw Elbow: {bio.dea:.0f} deg",
             gc(bio.dea, GOLD["draw_elbow_min"], GOLD["draw_elbow_max"])),
            (f"DFL Angle: {bio.dfl_angle:.1f} deg",
             COLOR_GREEN if abs(bio.dfl_angle) < 10 else COLOR_YELLOW),
            (f"Shldr Tilt: {abs(bio.shoulder_tilt):.1f} deg",
             COLOR_GREEN if abs(bio.shoulder_tilt) < GOLD["shoulder_tilt_max"] else COLOR_RED),
            (f"Torso Lean: {abs(bio.torso_lean):.1f} deg",
             COLOR_GREEN if abs(bio.torso_lean) < GOLD["torso_lean_max"] else COLOR_RED),
            (f"Draw Len: {bio.draw_length:.2f}",
             COLOR_WHITE),
            (f"Hip CoP: ({bio.hip_mid[0]:.3f}, {bio.hip_mid[1]:.3f})",
             COLOR_GRAY),
        ]
        for i, (txt, col) in enumerate(items):
            cv2.putText(frame, txt, (rx, ry + 20 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # Setup posture feedback (during SETUP state)
    if state == "SETUP" and setup_posture_score >= 0:
        ps_c = (COLOR_GREEN if setup_posture_score >= 7
                else COLOR_YELLOW if setup_posture_score >= 4
                else COLOR_RED)
        cv2.putText(frame, f"Posture: {setup_posture_score:.0f}/10",
                    (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ps_c, 1, cv2.LINE_AA)

    # Bowstring detection indicator (during AIM)
    if state in ("AIM", "ANCHOR") and string_detected:
        cv2.putText(frame, "STRING", (w - 130, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1, cv2.LINE_AA)

    # Tremor gauge (during AIM)
    if state == "AIM" and tremor_rms >= 0:
        tr_c = (COLOR_GREEN if tremor_rms < GOLD.get("tremor_rms_elite", 0.002)
                else COLOR_YELLOW if tremor_rms < GOLD.get("tremor_rms_good", 0.005)
                else COLOR_RED)
        cv2.putText(frame, f"Tremor: {tremor_rms:.4f}",
                    (18, 95 if state != "SETUP" else 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tr_c, 1, cv2.LINE_AA)

    # Release confidence badge (briefly after RELEASE)
    if state in ("RELEASE", "FOLLOW_THROUGH") and release_confidence:
        rc_c = {"HIGH": COLOR_GREEN, "MEDIUM": COLOR_YELLOW, "LOW": COLOR_RED}.get(
            release_confidence, COLOR_GRAY)
        cv2.putText(frame, f"Release: {release_confidence}",
                    (18, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc_c, 1, cv2.LINE_AA)

    # Bottom bar
    by2 = h - 50
    ov3 = frame.copy()
    cv2.rectangle(ov3, (0, by2), (w, h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(ov3, 0.75, frame, 0.25, 0, frame)

    yt = h - 18
    cv2.putText(frame, f"Shots: {shots}", (20, yt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)
    at = f"{avg_hold:.1f}s" if avg_hold > 0 else "--"
    bt = f"{best_hold:.1f}s" if best_hold > 0 else "--"
    cv2.putText(frame, f"Avg Hold: {at}", (200, yt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Best: {bt}", (430, yt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 1, cv2.LINE_AA)
    if flags_str:
        cv2.putText(frame, flags_str, (600, yt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1, cv2.LINE_AA)
    if vertex_score >= 0:
        vs_c = (COLOR_GREEN if vertex_score >= 70
                else COLOR_YELLOW if vertex_score >= 40
                else COLOR_RED)
        cv2.putText(frame, f"VS:{vertex_score:.0f}", (750, yt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, vs_c, 2, cv2.LINE_AA)
    if sess_path:
        cv2.putText(frame, os.path.basename(sess_path), (w - 280, yt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)

    if corrections:
        draw_correction_cues(frame, corrections, w)

    cv2.putText(frame, "Q:quit R:reset D:debug C:coach S:ref", (w - 310, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Consent banner — shown before session starts (GDPR Article 9)
# ---------------------------------------------------------------------------
def draw_consent_banner(frame) -> None:
    """Overlay consent information on frame. Caller loops until ENTER or Q."""
    h, w = frame.shape[:2]
    cx = max(20, w // 2 - 300)
    cy = max(30, h // 2 - 100)
    panel_w = min(620, w - cx - 20)
    ov = frame.copy()
    cv2.rectangle(ov, (cx - 20, cy - 40), (cx + panel_w, cy + 180), COLOR_PANEL_BG, -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    cv2.putText(frame, "VERTEX TheSighter", (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, "This session captures biometric pose data (skeleton only).",
                (cx, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "All processing runs on-device. No data leaves this machine.",
                (cx, cy + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Session data is saved locally in sessions/ for coaching review.",
                (cx, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Press ENTER to start  |  Q to quit",
                (cx, cy + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2, cv2.LINE_AA)
