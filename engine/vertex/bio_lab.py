"""
Vertex — VertexBioLab: Biomechanical analysis engine.

Pure numerical functions — ZERO cv2 imports.
Computes joint angles, posture metrics, corrections, and reference poses.
All coordinate inputs are normalised MediaPipe landmarks or pixel tuples.
"""

from __future__ import annotations

import numpy as np

from .models import (
    GOLD, WRIST_JAW_ANCHOR_THRESHOLD, WRIST_JAW_DRAW_THRESHOLD,
    YELLOW_MARGIN_PCT, MIN_LANDMARK_VISIBILITY, MIN_CORRECTION_DEG,
    MAX_CORRECTION_CUES, FRAME_VALIDITY_THRESHOLD,
    COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_WHITE, COLOR_GRAY,
    _COLOR_SEVERITY,
    R_WRIST, R_INDEX, R_EAR, MOUTH_R,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST,
    L_HIP, R_HIP, L_ANKLE, R_ANKLE,
    BioMetrics, FrameMetrics,
    EXTRACT_VIS_THRESHOLD, EXTRACT_SHARPNESS_MIN, EXTRACT_SHARPNESS_ELITE,
    EXTRACT_ANCHOR_DRAW_MAX,
    TREMOR_MIN_FRAMES, TRANSFER_WINDOW_FRAMES, EXPANSION_MIN_FRAMES,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def lm_xy(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y])


def dist_xy(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def dist_lm(landmarks, a: int, b: int) -> float:
    return dist_xy(lm_xy(landmarks[a]), lm_xy(landmarks[b]))


def angle_at(landmarks, a: int, vertex: int, b: int) -> float:
    va = lm_xy(landmarks[a]) - lm_xy(landmarks[vertex])
    vb = lm_xy(landmarks[b]) - lm_xy(landmarks[vertex])
    cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def jaw_proxy(landmarks) -> np.ndarray:
    return 0.4 * lm_xy(landmarks[R_EAR]) + 0.6 * lm_xy(landmarks[MOUTH_R])


def shoulder_width(landmarks) -> float:
    return dist_lm(landmarks, L_SHOULDER, R_SHOULDER)


def median_filter(values, window: int) -> float:
    if len(values) < window:
        return values[-1] if values else 0.0
    return float(np.median(list(values)[-window:]))


def rotate_point(origin, point, angle_deg):
    """Rotate pixel-coord 'point' around 'origin' by angle_deg."""
    rad = np.radians(angle_deg)
    ox, oy = origin
    px, py = point
    dx, dy = px - ox, py - oy
    rx = dx * np.cos(rad) - dy * np.sin(rad)
    ry = dx * np.sin(rad) + dy * np.cos(rad)
    return (int(ox + rx), int(oy + ry))


def _rotate_norm(origin, point, angle_deg):
    """Rotate normalised 2D point around origin by angle_deg."""
    rad = np.radians(angle_deg)
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    rx = dx * np.cos(rad) - dy * np.sin(rad)
    ry = dx * np.sin(rad) + dy * np.cos(rad)
    return np.array([origin[0] + rx, origin[1] + ry])


def _extend_limb(joint, parent, length):
    """Place joint at 'length' distance from parent, same direction."""
    d = joint - parent
    n = np.linalg.norm(d)
    if n < 1e-6:
        return joint
    return parent + d / n * length


# ---------------------------------------------------------------------------
# Frame / landmark validation
# ---------------------------------------------------------------------------
def frame_valid(landmarks, sw_hist) -> tuple[bool, float]:
    sw = shoulder_width(landmarks)
    if len(sw_hist) < 5:
        return True, sw
    med = float(np.median(list(sw_hist)))
    if med > 0 and abs(sw - med) / med > FRAME_VALIDITY_THRESHOLD:
        return False, sw
    return True, sw


def key_landmarks_visible(landmarks) -> bool:
    keys = [R_INDEX, R_EAR, MOUTH_R, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST]
    return all(landmarks[i].visibility > MIN_LANDMARK_VISIBILITY for i in keys)


# ---------------------------------------------------------------------------
# Core biomechanics computation
# ---------------------------------------------------------------------------
def compute_bio(landmarks, sw: float) -> BioMetrics:
    jaw = jaw_proxy(landmarks)
    hand = lm_xy(landmarks[R_INDEX])
    anchor_dist = dist_xy(hand, jaw) / max(sw, 0.01)
    bsa = angle_at(landmarks, L_ELBOW, L_SHOULDER, L_HIP)
    dea = angle_at(landmarks, R_SHOULDER, R_ELBOW, R_WRIST)

    ls, rs = lm_xy(landmarks[L_SHOULDER]), lm_xy(landmarks[R_SHOULDER])
    shoulder_tilt = float(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0])))

    smid = (ls + rs) / 2
    hmid = (lm_xy(landmarks[L_HIP]) + lm_xy(landmarks[R_HIP])) / 2
    trunk = smid - hmid
    torso_lean = float(np.degrees(np.arctan2(trunk[0], -trunk[1])))

    draw_len = dist_lm(landmarks, L_WRIST, R_WRIST) / max(sw, 0.01)

    bw = lm_xy(landmarks[L_WRIST])
    de = lm_xy(landmarks[R_ELBOW])
    dfl_vec = de - bw
    dfl_angle = float(np.degrees(np.arctan2(dfl_vec[1], dfl_vec[0])))

    hip_mid = (lm_xy(landmarks[L_HIP]) + lm_xy(landmarks[R_HIP])) / 2
    draw_arm_angle = angle_at(landmarks, R_SHOULDER, R_ELBOW, R_WRIST)

    return BioMetrics(
        anchor_dist=anchor_dist, hand_xy=hand, jaw_xy=jaw,
        bsa=bsa, dea=dea,
        shoulder_tilt=shoulder_tilt, torso_lean=torso_lean,
        draw_length=draw_len, dfl_angle=dfl_angle,
        hip_mid=hip_mid, draw_arm_angle=draw_arm_angle,
    )


# ---------------------------------------------------------------------------
# Coaching colour assessment
# ---------------------------------------------------------------------------
def gc3(value: float, lo: float, hi: float, margin_pct: float = YELLOW_MARGIN_PCT):
    """GREEN if in range, YELLOW if near boundary, RED if out."""
    if lo <= value <= hi:
        return COLOR_GREEN
    margin = (hi - lo) * margin_pct
    if (lo - margin) <= value < lo or hi < value <= (hi + margin):
        return COLOR_YELLOW
    return COLOR_RED


def gc3_abs(value: float, limit: float, margin_pct: float = YELLOW_MARGIN_PCT):
    """Three-tier for absolute-value metrics (tilt, lean)."""
    av = abs(value)
    if av <= limit:
        return COLOR_GREEN
    if av <= limit * (1 + margin_pct):
        return COLOR_YELLOW
    return COLOR_RED


def worst_color(c1, c2):
    return c1 if _COLOR_SEVERITY.get(c1, 0) >= _COLOR_SEVERITY.get(c2, 0) else c2


def assess_posture(bio: BioMetrics) -> dict[int, tuple]:
    """Map landmark indices to BGR colours based on biomechanics vs GOLD."""
    colors: dict[int, tuple] = {}
    bsa_c = gc3(bio.bsa, GOLD["bow_shoulder_min"], GOLD["bow_shoulder_max"])
    colors[L_SHOULDER] = bsa_c
    colors[L_ELBOW] = bsa_c
    colors[L_WRIST] = bsa_c

    dea_c = gc3(bio.dea, GOLD["draw_elbow_min"], GOLD["draw_elbow_max"])
    colors[R_ELBOW] = dea_c
    colors[R_WRIST] = dea_c
    colors[R_SHOULDER] = dea_c

    tilt_c = gc3_abs(bio.shoulder_tilt, GOLD["shoulder_tilt_max"])
    lean_c = gc3_abs(bio.torso_lean, GOLD["torso_lean_max"])
    posture_c = worst_color(tilt_c, lean_c)
    colors[L_HIP] = posture_c
    colors[R_HIP] = posture_c
    colors[L_SHOULDER] = worst_color(colors[L_SHOULDER], posture_c)
    colors[R_SHOULDER] = worst_color(colors[R_SHOULDER], posture_c)

    ad = bio.anchor_dist
    if ad < WRIST_JAW_ANCHOR_THRESHOLD:
        colors[R_INDEX] = COLOR_GREEN
    elif ad < WRIST_JAW_DRAW_THRESHOLD:
        colors[R_INDEX] = COLOR_YELLOW
    else:
        colors[R_INDEX] = COLOR_WHITE
    return colors


# ---------------------------------------------------------------------------
# Posture correction computation
# ---------------------------------------------------------------------------
def compute_corrections(bio: BioMetrics, landmarks, h: int, w: int) -> list[dict]:
    """Compute top-N corrections sorted by severity. Returns list of dicts."""
    corrections = []

    def px(idx):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    def vis(idx):
        return landmarks[idx].visibility > MIN_LANDMARK_VISIBILITY

    # Bow shoulder angle
    bsa = bio.bsa
    bsa_target = (GOLD["bow_shoulder_min"] + GOLD["bow_shoulder_max"]) / 2
    bsa_dev = bsa - bsa_target
    bsa_c = gc3(bsa, GOLD["bow_shoulder_min"], GOLD["bow_shoulder_max"])
    if bsa_c != COLOR_GREEN and abs(bsa_dev) >= MIN_CORRECTION_DEG and vis(L_SHOULDER) and vis(L_ELBOW):
        cur = px(L_ELBOW)
        tgt = rotate_point(px(L_SHOULDER), cur, -bsa_dev)
        verb = "Lower" if bsa_dev > 0 else "Raise"
        arrow = "\u2193" if bsa_dev > 0 else "\u2191"
        corrections.append({
            "metric": "bow_shoulder", "severity": abs(bsa_dev), "color": bsa_c,
            "current_px": cur, "target_px": tgt,
            "cue_text": f"{arrow} {verb} bow shoulder {abs(bsa_dev):.0f}\u00b0",
        })

    # Draw elbow angle
    dea = bio.dea
    dea_target = (GOLD["draw_elbow_min"] + GOLD["draw_elbow_max"]) / 2
    dea_dev = dea - dea_target
    dea_c = gc3(dea, GOLD["draw_elbow_min"], GOLD["draw_elbow_max"])
    if dea_c != COLOR_GREEN and abs(dea_dev) >= MIN_CORRECTION_DEG and vis(R_ELBOW) and vis(R_WRIST):
        cur = px(R_WRIST)
        tgt = rotate_point(px(R_ELBOW), cur, -dea_dev)
        verb = "Close" if dea_dev > 0 else "Open"
        arrow = "\u2190" if dea_dev > 0 else "\u2192"
        corrections.append({
            "metric": "draw_elbow", "severity": abs(dea_dev), "color": dea_c,
            "current_px": cur, "target_px": tgt,
            "cue_text": f"{arrow} {verb} draw elbow {abs(dea_dev):.0f}\u00b0",
        })

    # Shoulder tilt
    tilt = bio.shoulder_tilt
    tilt_c = gc3_abs(tilt, GOLD["shoulder_tilt_max"])
    if tilt_c != COLOR_GREEN and abs(tilt) >= MIN_CORRECTION_DEG / 2 and vis(L_SHOULDER) and vis(R_SHOULDER):
        if tilt > 0:
            cur = px(R_SHOULDER)
            tgt = rotate_point(px(L_SHOULDER), cur, -tilt)
            side = "right"
        else:
            cur = px(L_SHOULDER)
            tgt = rotate_point(px(R_SHOULDER), cur, tilt)
            side = "left"
        corrections.append({
            "metric": "shoulder_tilt", "severity": abs(tilt), "color": tilt_c,
            "current_px": cur, "target_px": tgt,
            "cue_text": f"\u2193 Level {side} shoulder {abs(tilt):.0f}\u00b0",
        })

    # Torso lean
    lean = bio.torso_lean
    lean_c = gc3_abs(lean, GOLD["torso_lean_max"])
    if lean_c != COLOR_GREEN and abs(lean) >= MIN_CORRECTION_DEG / 2 and vis(L_SHOULDER) and vis(R_SHOULDER) and vis(L_HIP) and vis(R_HIP):
        ls, rs = px(L_SHOULDER), px(R_SHOULDER)
        smid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        lh, rh = px(L_HIP), px(R_HIP)
        hmid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        tgt = (hmid[0], smid[1])
        direction = "\u2190" if lean > 0 else "\u2192"
        corrections.append({
            "metric": "torso_lean", "severity": abs(lean), "color": lean_c,
            "current_px": smid, "target_px": tgt,
            "cue_text": f"{direction} Stand straighter {abs(lean):.0f}\u00b0",
        })

    # DFL angle
    dfl = bio.dfl_angle
    dfl_c = COLOR_GREEN if abs(dfl) < 5 else (COLOR_YELLOW if abs(dfl) < 10 else COLOR_RED)
    if dfl_c != COLOR_GREEN and abs(dfl) >= MIN_CORRECTION_DEG and vis(L_WRIST) and vis(R_ELBOW):
        bw = px(L_WRIST)
        de = px(R_ELBOW)
        mid = ((bw[0] + de[0]) // 2, (bw[1] + de[1]) // 2)
        tgt = rotate_point(mid, de, -dfl)
        corrections.append({
            "metric": "dfl", "severity": abs(dfl), "color": dfl_c,
            "current_px": de, "target_px": tgt,
            "cue_text": f"\u21bb Level force line {abs(dfl):.0f}\u00b0",
        })

    corrections.sort(key=lambda c: c["severity"], reverse=True)
    return corrections[:MAX_CORRECTION_CUES]


# ---------------------------------------------------------------------------
# Reference pose computation
# ---------------------------------------------------------------------------
def capture_reference(landmarks) -> dict[int, np.ndarray]:
    """Compute ideal skeleton from current landmarks + GOLD angles."""
    pts = {}
    for i, lm in enumerate(landmarks):
        pts[i] = np.array([lm.x, lm.y])

    ref: dict[int, np.ndarray] = {}

    ls, rs = pts[L_SHOULDER].copy(), pts[R_SHOULDER].copy()
    lh, rh = pts[L_HIP].copy(), pts[R_HIP].copy()

    # Level shoulders
    smid = (ls + rs) / 2
    half_sw = np.linalg.norm(rs - ls) / 2
    ref[L_SHOULDER] = np.array([smid[0] - half_sw, smid[1]])
    ref[R_SHOULDER] = np.array([smid[0] + half_sw, smid[1]])

    # Level torso
    hmid = (lh + rh) / 2
    half_hw = np.linalg.norm(rh - lh) / 2
    ref[L_HIP] = np.array([hmid[0] - half_hw, hmid[1]])
    ref[R_HIP] = np.array([hmid[0] + half_hw, hmid[1]])

    # Bow arm at GOLD BSA midpoint
    bsa_target = (GOLD["bow_shoulder_min"] + GOLD["bow_shoulder_max"]) / 2
    bsa_current = angle_at(landmarks, L_ELBOW, L_SHOULDER, L_HIP)
    bsa_delta = bsa_current - bsa_target
    limb_upper = np.linalg.norm(pts[L_ELBOW] - pts[L_SHOULDER])
    limb_lower = np.linalg.norm(pts[L_WRIST] - pts[L_ELBOW])
    ref[L_ELBOW] = _rotate_norm(ref[L_SHOULDER], pts[L_ELBOW], -bsa_delta)
    ref[L_ELBOW] = _extend_limb(ref[L_ELBOW], ref[L_SHOULDER], limb_upper)
    ref[L_WRIST] = _extend_limb(
        ref[L_ELBOW] + (ref[L_ELBOW] - ref[L_SHOULDER]),
        ref[L_ELBOW], limb_lower)

    # Draw arm at GOLD DEA midpoint
    dea_target = (GOLD["draw_elbow_min"] + GOLD["draw_elbow_max"]) / 2
    dea_current = angle_at(landmarks, R_SHOULDER, R_ELBOW, R_WRIST)
    dea_delta = dea_current - dea_target
    limb_upper_r = np.linalg.norm(pts[R_ELBOW] - pts[R_SHOULDER])
    limb_lower_r = np.linalg.norm(pts[R_WRIST] - pts[R_ELBOW])
    ref[R_ELBOW] = _rotate_norm(ref[R_SHOULDER], pts[R_ELBOW], -dea_delta)
    ref[R_ELBOW] = _extend_limb(ref[R_ELBOW], ref[R_SHOULDER], limb_upper_r)
    ref[R_WRIST] = _rotate_norm(ref[R_ELBOW], pts[R_WRIST], -dea_delta)
    ref[R_WRIST] = _extend_limb(ref[R_WRIST], ref[R_ELBOW], limb_lower_r)

    # Anchor hand at jaw
    jaw = 0.4 * pts[R_EAR] + 0.6 * pts[MOUTH_R]
    ref[R_INDEX] = jaw.copy()

    return ref


# ---------------------------------------------------------------------------
# KSL sub-phase compute functions (Phase 1)
# ---------------------------------------------------------------------------
def compute_stance(landmarks, sw: float) -> dict:
    """KSL step 1 — Stance baseline from lower-body landmarks.

    Returns dict with stance_width, hip_alignment, weight_proxy.
    All distances SW-normalised.
    """
    sw_safe = max(sw, 0.01)
    la = lm_xy(landmarks[L_ANKLE])
    ra = lm_xy(landmarks[R_ANKLE])
    stance_w = dist_xy(la, ra) / sw_safe

    lh = lm_xy(landmarks[L_HIP])
    rh = lm_xy(landmarks[R_HIP])
    hip_align = float(np.degrees(np.arctan2(rh[1] - lh[1], rh[0] - lh[0])))

    hip_mid = (lh + rh) / 2
    ankle_mid = (la + ra) / 2
    weight_proxy = (hip_mid[0] - ankle_mid[0]) / sw_safe

    return {
        "stance_width": float(stance_w),
        "hip_alignment": float(hip_align),
        "weight_proxy": float(weight_proxy),
    }


def compute_raise_quality(wrist_y_sequence: list[float]) -> float:
    """KSL step 5 — Set-Up (Raise) smoothness.

    Measures variance of frame-to-frame deltas in L_WRIST Y position.
    Lower variance = smoother bow raise. Returns -1.0 if insufficient data.
    """
    if len(wrist_y_sequence) < 3:
        return -1.0
    deltas = np.diff(wrist_y_sequence)
    return float(np.var(deltas))


def compute_draw_profile(distances: list[float], fps: float) -> dict:
    """KSL step 6 — Draw curve profiling.

    Returns dict with draw_duration_s, draw_smoothness, draw_velocity_mean.
    """
    n = len(distances)
    duration_s = n / max(fps, 1.0)
    if n < 2:
        return {
            "draw_duration_s": duration_s,
            "draw_smoothness": -1.0,
            "draw_velocity_mean": 0.0,
        }
    deltas = np.diff(distances)
    return {
        "draw_duration_s": float(duration_s),
        "draw_smoothness": float(np.var(deltas)),
        "draw_velocity_mean": float(np.mean(deltas)),
    }


def compute_transfer_proxy(shoulder_positions: list[np.ndarray],
                           sw: float) -> float:
    """KSL step 8 — Transfer proxy via R_SHOULDER displacement.

    Tracks mean X-axis shift of R_SHOULDER over the transfer window.
    Posterior shift (positive X toward camera in oblique setup) = correct engagement.
    Returns displacement normalised by SW, or -1.0 if insufficient data.

    Confidence: 6/10 — true transfer detection requires EMG/force plate (Phase 3+).
    """
    if len(shoulder_positions) < 2:
        return -1.0
    sw_safe = max(sw, 0.01)
    deltas = [float(shoulder_positions[i + 1][0] - shoulder_positions[i][0])
              for i in range(len(shoulder_positions) - 1)]
    return float(np.mean(deltas)) / sw_safe


def compute_tremor(positions: list[np.ndarray], sw: float) -> float | None:
    """KSL step 9 — Right-arm micro-tremor RMS.

    Computes RMS of frame-to-frame displacements (SW-normalised).
    Returns None if fewer than TREMOR_MIN_FRAMES samples.
    """
    if len(positions) < TREMOR_MIN_FRAMES:
        return None
    sw_safe = max(sw, 0.01)
    displacements = [float(np.linalg.norm(positions[i + 1] - positions[i]))
                     for i in range(len(positions) - 1)]
    rms = float(np.sqrt(np.mean(np.square(displacements)))) / sw_safe
    return rms


def compute_expansion(shoulder_spreads: list[float]) -> float:
    """KSL step 10 — Aim & Expand via shoulder spread linear regression.

    Returns the slope of inter-shoulder distance over AIM frames.
    Positive slope = scapular adduction (correct back tension expansion).
    Returns -1.0 if insufficient data.
    """
    if len(shoulder_spreads) < EXPANSION_MIN_FRAMES:
        return -1.0
    x = np.arange(len(shoulder_spreads), dtype=float)
    y = np.array(shoulder_spreads, dtype=float)
    # Simple linear regression slope: cov(x,y) / var(x)
    x_mean = x.mean()
    y_mean = y.mean()
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-12))
    return slope


def compute_follow_through(bow_wrist_y: list[float],
                           bsa_values: list[float],
                           release_hand_positions: list[np.ndarray],
                           sw: float) -> dict:
    """KSL step 12 — Follow-through quality metrics.

    Returns dict with arm_drop_y, bsa_follow_var, release_hand_angle.
    arm_drop_y: max downward displacement of bow wrist Y (normalised by SW).
    bsa_follow_var: variance of BSA during follow-through.
    release_hand_angle: angle of R_WRIST trajectory vector (degrees from horizontal).
    """
    sw_safe = max(sw, 0.01)
    result: dict = {
        "arm_drop_y": -1.0,
        "bsa_follow_var": -1.0,
        "release_hand_angle": -1.0,
    }

    if len(bow_wrist_y) >= 2:
        # In normalised coords, Y increases downward → positive diff = arm drop
        baseline_y = bow_wrist_y[0]
        max_drop = max(v - baseline_y for v in bow_wrist_y)
        result["arm_drop_y"] = float(max_drop) / sw_safe

    if len(bsa_values) >= 2:
        result["bsa_follow_var"] = float(np.var(bsa_values))

    if len(release_hand_positions) >= 2:
        start = release_hand_positions[0]
        end = release_hand_positions[-1]
        delta = end - start
        angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
        result["release_hand_angle"] = angle

    return result


# ---------------------------------------------------------------------------
# Frame gold-standard evaluation (used by extract_frames dev tool)
# ---------------------------------------------------------------------------
_RATING_LABELS: dict[tuple, str] = {
    COLOR_GREEN: "GREEN", COLOR_YELLOW: "YELLOW", COLOR_RED: "RED",
}


def _build_bio_checks(landmarks, sw: float) -> dict:
    """G4–G7 biomechanical gold checks. Returns {} if compute_bio fails."""
    try:
        bio = compute_bio(landmarks, sw)
    except Exception:
        return {}
    rl = _RATING_LABELS
    return {
        "G4": {"label": "Bow shoulder angle", "value": f"{bio.bsa:.1f}deg",
               "target": f"{GOLD['bow_shoulder_min']}-{GOLD['bow_shoulder_max']}deg",
               "rating": rl.get(gc3(bio.bsa, GOLD["bow_shoulder_min"], GOLD["bow_shoulder_max"]), "RED"),
               "source": "Shinohara 2018"},
        "G5": {"label": "Draw elbow angle",   "value": f"{bio.dea:.1f}deg",
               "target": f"{GOLD['draw_elbow_min']}-{GOLD['draw_elbow_max']}deg",
               "rating": rl.get(gc3(bio.dea, GOLD["draw_elbow_min"], GOLD["draw_elbow_max"]), "RED"),
               "source": "Shinohara 2018"},
        "G6": {"label": "Shoulder tilt",      "value": f"{bio.shoulder_tilt:.1f}deg",
               "target": f"<{GOLD['shoulder_tilt_max']}deg",
               "rating": rl.get(gc3_abs(bio.shoulder_tilt, GOLD["shoulder_tilt_max"]), "RED"),
               "source": "Posture alignment"},
        "G7": {"label": "Torso lean",         "value": f"{bio.torso_lean:.1f}deg",
               "target": f"<{GOLD['torso_lean_max']}deg",
               "rating": rl.get(gc3_abs(bio.torso_lean, GOLD["torso_lean_max"]), "RED"),
               "source": "Posture alignment"},
    }


def evaluate_frame_quality(landmarks, metrics: FrameMetrics) -> tuple[dict, bool]:
    """Gold standard G1–G7 evaluation for one extracted frame. Never raises."""
    try:
        ad, vm, sharp, sw = (
            metrics.anchor_dist, metrics.vis_mean,
            metrics.sharpness, metrics.shoulder_width,
        )
        g1_r = "PASS" if vm >= EXTRACT_VIS_THRESHOLD else "FAIL"
        g2_r = "GREEN" if sharp >= EXTRACT_SHARPNESS_ELITE else (
            "YELLOW" if sharp >= EXTRACT_SHARPNESS_MIN else "RED")
        g3_r = "GREEN" if ad < 0.35 else ("YELLOW" if ad < EXTRACT_ANCHOR_DRAW_MAX else "RED")
        cl: dict = {
            "G1": {"label": "Landmark visibility", "value": f"{vm:.3f}",
                   "target": f">={EXTRACT_VIS_THRESHOLD} all key",
                   "rating": g1_r, "source": "MediaPipe"},
            "G2": {"label": "Frame sharpness",     "value": f"{sharp:.0f}",
                   "target": ">=300 elite / >=100 min",
                   "rating": g2_r, "source": "Laplacian var"},
            "G3": {"label": "Anchor position",     "value": f"{ad:.3f} SW",
                   "target": "<0.35 SW elite  <0.50 SW good",
                   "rating": g3_r, "source": "Soylu 2006"},
        }
        cl.update(_build_bio_checks(landmarks, sw))
        bio_pass = sum(
            1 for k in ("G4", "G5", "G6", "G7")
            if cl.get(k, {}).get("rating") in ("GREEN", "YELLOW")
        )
        validated = g1_r == "PASS" and g2_r != "RED" and g3_r != "RED" and bio_pass >= 2
        cl["_validated"]   = validated
        cl["_bio_pass"]    = bio_pass
        cl["_green_count"] = sum(
            1 for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
            if cl.get(k, {}).get("rating") in ("GREEN", "PASS")
        )
        return cl, validated
    except Exception:
        return {}, False
