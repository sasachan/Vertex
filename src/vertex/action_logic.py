"""
Vertex — VertexActionLogic: Shot lifecycle state machine.

State pattern — each archery state is a class with enter/update/exit.
Archery profile: IDLE → SETUP → DRAW → ANCHOR → AIM → RELEASE → FOLLOW_THROUGH → IDLE.
Other sport profiles load different state sets via sport_map.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from .models import (
    ShotState, BioMetrics, ShotRecord, GOLD,
    WRIST_JAW_DRAW_THRESHOLD, WRIST_JAW_ANCHOR_THRESHOLD,
    ANCHOR_VARIANCE_THRESHOLD, RELEASE_JUMP_THRESHOLD,
    RELEASE_CONFIRM_FRAMES, MIN_ANCHOR_HOLD,
    DRAW_WINDOW, ANCHOR_WINDOW, MEDIAN_FILTER_WINDOW,
    IDLE_COOLDOWN, MAX_ANCHOR_HOLD, SNAP_SHOT_THRESHOLD,
    FOLLOW_THROUGH_FRAMES, TARGET_FPS,
    SETUP_COLLECT_FRAMES, SETUP_TIMEOUT, AIM_ENTRY_FRAMES,
    TREMOR_MIN_FRAMES, TRANSFER_WINDOW_FRAMES, EXPANSION_MIN_FRAMES,
    L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, R_ELBOW,
    StringState,
)
from .bio_lab import (
    median_filter, compute_bio, compute_stance, compute_raise_quality,
    compute_draw_profile, compute_transfer_proxy, compute_tremor,
    compute_expansion, compute_follow_through, assess_posture,
    gc3, shoulder_width, lm_xy, angle_at,
)


# ---------------------------------------------------------------------------
# Shared context passed to every state's update()
# ---------------------------------------------------------------------------
@dataclass
class StateContext:
    """Mutable frame context shared between states and the main loop."""
    now: float = 0.0
    dt: float = 0.0
    bio: BioMetrics | None = None
    anchor_dist_raw: float = 0.0      # unfiltered wrist-jaw distance
    anchor_dist_filtered: float = 0.0  # median-filtered
    sw: float = 0.0                    # shoulder width


# ---------------------------------------------------------------------------
# State base
# ---------------------------------------------------------------------------
class ArcheryState:
    """Base class for archery shot-cycle states."""
    name: ShotState

    def enter(self, machine: StateMachine) -> None:
        pass

    def update(self, machine: StateMachine, ctx: StateContext) -> ShotState | None:
        """Process one frame. Return a new ShotState to transition, or None to stay."""
        return None

    def exit(self, machine: StateMachine) -> None:
        pass


# ---------------------------------------------------------------------------
# Concrete states
# ---------------------------------------------------------------------------
class IdleState(ArcheryState):
    name = ShotState.IDLE

    def enter(self, machine: StateMachine) -> None:
        machine.idle_entry = machine.last_now
        machine.raw_hist.clear()
        machine.filt_hist.clear()

    def update(self, machine: StateMachine, ctx: StateContext) -> ShotState | None:
        if ctx.now - machine.idle_entry < IDLE_COOLDOWN:
            return None
        if ctx.bio is None:
            return None
        return ShotState.SETUP


class SetupState(ArcheryState):
    """Pre-draw stance assessment. Calibrates per-session distance thresholds.

    Collects SETUP_COLLECT_FRAMES of resting jaw-hand distance (SW-normalised).
    Derives dynamic draw/anchor thresholds from the measured resting baseline.
    Falls back to constants if calibration produces implausible values.
    """
    name = ShotState.SETUP

    def enter(self, machine: StateMachine) -> None:
        machine.idle_entry = machine.last_now
        machine.calib_dists = []
        machine.calib_done = False
        machine.setup_wrist_y = []  # KSL step 5: raise tracking

    def update(self, machine: StateMachine, ctx: StateContext) -> ShotState | None:
        # Track bow arm raise (KSL step 5)
        if ctx.bio is not None:
            machine.setup_wrist_y.append(ctx.bio.hand_xy[1])  # Y position of wrist

        # Collect resting-position calibration samples
        if len(machine.calib_dists) < machine.setup_collect_frames:
            machine.calib_dists.append(ctx.anchor_dist_filtered)
            return None

        # Compute dynamic thresholds once after the collection window
        if not machine.calib_done:
            rest = float(np.median(machine.calib_dists))
            # Only override defaults when hand is clearly away from jaw (> 0.8 SW)
            if rest > 0.8:
                machine.dynamic_draw_threshold = min(rest * 0.65, WRIST_JAW_DRAW_THRESHOLD)
                machine.dynamic_anchor_threshold = min(rest * 0.45, WRIST_JAW_ANCHOR_THRESHOLD)

            # KSL step 1 (Stance) + step 4 (Set Posture) — baseline assessment
            if machine._setup_landmarks is not None:
                sw = ctx.sw if ctx.sw > 0 else 1.0
                stance = compute_stance(machine._setup_landmarks, sw)
                machine.baseline_stance = stance

                bio = ctx.bio
                if bio is not None:
                    colors = assess_posture(bio)
                    from .models import COLOR_GREEN, COLOR_YELLOW
                    green_count = sum(1 for c in colors.values()
                                      if c == COLOR_GREEN or c == COLOR_YELLOW)
                    machine.setup_posture_score = round(green_count / max(len(colors), 1) * 10, 1)
                    machine.baseline_bio = bio

            machine.calib_done = True
            print(f"  [setup] draw_thresh={machine.dynamic_draw_threshold:.3f} "
                  f"anchor_thresh={machine.dynamic_anchor_threshold:.3f}"
                  f" posture={machine.setup_posture_score:.1f}/10")

        # Timeout: no draw attempt within SETUP_TIMEOUT -> return to IDLE
        if ctx.now - machine.idle_entry > SETUP_TIMEOUT:
            print("  [setup] Timeout -> IDLE")
            return ShotState.IDLE

        # Draw-initiation detection (mirrors former IdleState logic)
        recent = list(machine.filt_hist)
        if len(recent) < machine.draw_window:
            return None
        win = recent[-machine.draw_window:]
        dec = sum(1 for i in range(len(win) - 1) if win[i] > win[i + 1])
        thresh = machine.dynamic_draw_threshold
        if dec >= machine.draw_window - 1 and ctx.anchor_dist_filtered < thresh:
            return ShotState.DRAW
        return None


class DrawState(ArcheryState):
    name = ShotState.DRAW

    def enter(self, machine: StateMachine) -> None:
        machine.state_log = ["DRAW"]
        machine.release_cnt = 0
        machine.draw_dists = []       # KSL step 6: draw profile accumulator
        machine.draw_bios = []        # KSL step 6: per-frame bio during draw

    def update(self, machine: StateMachine, ctx: StateContext) -> ShotState | None:
        df = ctx.anchor_dist_filtered

        # KSL step 6 — accumulate draw curve
        machine.draw_dists.append(df)
        if ctx.bio is not None:
            machine.draw_bios.append(ctx.bio)

        # Let-down: hand moves away from jaw
        if df > machine.dynamic_draw_threshold:
            print("  [let-down] Draw abandoned -> IDLE")
            return ShotState.IDLE

        # Transition to ANCHOR: stable low distance
        recent = list(machine.filt_hist)
        if len(recent) >= machine.anchor_window:
            aw = recent[-machine.anchor_window:]
            var = np.var(aw)
            if df < machine.dynamic_anchor_threshold and var < ANCHOR_VARIANCE_THRESHOLD:
                return ShotState.ANCHOR

        return None


class AnchorState(ArcheryState):
    name = ShotState.ANCHOR

    def enter(self, machine: StateMachine) -> None:
        machine.anchor_t0 = machine.last_now
        machine.anchor_dists = []
        machine.anchor_bios = []
        machine.aim_frames = 0
        machine.state_log.append("ANCHOR")
        if machine.last_ctx_bio is not None:
            machine.last_hand_xy = machine.last_ctx_bio.hand_xy.copy()

    def update(self, machine: StateMachine, ctx: StateContext) -> ShotState | None:
        bio = ctx.bio
        if bio is None:
            return None

        hold = ctx.now - machine.anchor_t0
        machine.hold = hold
        machine.anchor_dists.append(ctx.anchor_dist_filtered)
        machine.anchor_bios.append(bio)
        machine.last_hand_xy = bio.hand_xy.copy()

        if hold > MAX_ANCHOR_HOLD and int(hold) != int(hold - ctx.dt):
            print(f"  [overtime] Anchor {hold:.0f}s")

        # Let-down from anchor (rare but must be caught)
        if ctx.anchor_dist_filtered > machine.dynamic_draw_threshold:
            print("  [let-down] Anchor abandoned -> IDLE")
            return ShotState.IDLE

        # Accumulate stable frames then hand off to AIM
        machine.aim_frames += 1
        if machine.aim_frames >= machine.aim_entry_frames:
            return ShotState.AIM

        return None


class AimState(ArcheryState):
    """Active aiming / back-tension phase. Release detection lives here.

    Entered from ANCHOR after aim_entry_frames of stable hold.
    Continues accumulating anchor_bios for shot aggregation.
    Exits to RELEASE when departure is detected across release_confirm_frames.
    """
    name = ShotState.AIM

    def enter(self, machine: StateMachine) -> None:
        machine.state_log.append("AIM")
        machine.release_cnt = 0
        machine.aim_wrist_positions = []     # KSL step 9: tremor tracking
        machine.aim_elbow_positions = []     # KSL step 9: tremor tracking
        machine.aim_shoulder_positions = []  # KSL step 8: transfer proxy
        machine.aim_shoulder_spreads = []    # KSL step 10: expansion tracking
        machine.aim_frame_count = 0

    def update(self, machine: StateMachine, ctx: StateContext) -> ShotState | None:
        bio = ctx.bio
        if bio is None:
            return None

        hold = ctx.now - machine.anchor_t0
        machine.hold = hold
        machine.anchor_dists.append(ctx.anchor_dist_filtered)
        machine.anchor_bios.append(bio)
        machine.last_hand_xy = bio.hand_xy.copy()
        machine.aim_frame_count += 1

        # KSL step 8 — Transfer proxy (early AIM frames)
        if machine.aim_frame_count <= machine.transfer_window_frames:
            if machine._aim_landmarks is not None:
                machine.aim_shoulder_positions.append(
                    lm_xy(machine._aim_landmarks[R_SHOULDER]))

        # KSL step 9 — Tremor accumulation
        if machine._aim_landmarks is not None:
            machine.aim_wrist_positions.append(
                lm_xy(machine._aim_landmarks[R_WRIST]))
            machine.aim_elbow_positions.append(
                lm_xy(machine._aim_landmarks[R_ELBOW]))

        # KSL step 10 — Expansion tracking
        if machine._aim_landmarks is not None:
            spread = shoulder_width(machine._aim_landmarks)
            machine.aim_shoulder_spreads.append(spread)

        if hold > MAX_ANCHOR_HOLD and int(hold) != int(hold - ctx.dt):
            print(f"  [overtime] Still aiming {hold:.0f}s")

        # Release detection: departure from anchor baseline (raw signal)
        baseline = float(np.mean(machine.anchor_dists))
        departure = ctx.anchor_dist_raw - baseline
        if departure > RELEASE_JUMP_THRESHOLD:
            machine.release_cnt += 1
        else:
            machine.release_cnt = max(0, machine.release_cnt - 1)

        # Secondary: arm angle delta — Lee et al. 2024
        aa = list(machine.arm_angle_hist)
        if len(aa) >= 5:
            aa_delta = float(np.mean(np.abs(np.diff(aa[-5:]))))
            if aa_delta > GOLD["draw_arm_angle_release"]:
                machine.release_cnt += 1

        # Dual-signal release fusion with bowstring CV
        cv_signal = machine.last_string_state.release_signal
        pose_signal = machine.release_cnt >= machine.release_confirm_frames

        # Fusion: CV confirmation reduces required confirm frames
        required = machine.release_confirm_frames
        if cv_signal and pose_signal:
            # Both agree → HIGH confidence, immediate release
            machine.cv_release_detected = True
            machine.release_confidence = "HIGH"
            if hold >= MIN_ANCHOR_HOLD:
                return ShotState.RELEASE
        elif pose_signal:
            # Pose only → MEDIUM confidence, standard confirmation
            machine.release_confidence = "MEDIUM"
            if hold >= MIN_ANCHOR_HOLD:
                return ShotState.RELEASE
        elif cv_signal and machine.release_cnt >= 1:
            # CV + partial pose → reduce by 1
            machine.cv_release_detected = True
            machine.release_confidence = "MEDIUM"
            if hold >= MIN_ANCHOR_HOLD:
                return ShotState.RELEASE

        return None


class ReleaseState(ArcheryState):
    name = ShotState.RELEASE

    def enter(self, machine: StateMachine) -> None:
        machine.state_log.append("RELEASE")
        machine._build_shot_record()

    def update(self, machine: StateMachine, ctx: StateContext) -> ShotState | None:
        # Immediately hand off to FOLLOW_THROUGH for hip-sway tracking
        return ShotState.FOLLOW_THROUGH


class FollowThroughState(ArcheryState):
    """Post-release follow-through tracking.

    Collects hip-sway data for FOLLOW_THROUGH_FRAMES then returns to IDLE.
    Sway magnitude logged as a coaching cue (not yet persisted to ShotRecord).
    """
    name = ShotState.FOLLOW_THROUGH

    def enter(self, machine: StateMachine) -> None:
        machine.state_log.append("FOLLOW_THROUGH")
        machine.ft_frames = 0
        machine.ft_hip_positions = []
        machine.ft_bow_wrist_y = []           # KSL step 12: arm drop
        machine.ft_bsa_values = []            # KSL step 12: BSA consistency
        machine.ft_release_hand_positions = [] # KSL step 12: hand trajectory

    def update(self, machine: StateMachine, ctx: StateContext) -> ShotState | None:
        bio = ctx.bio
        if bio is not None:
            machine.ft_hip_positions.append(bio.hip_mid.copy())
            # KSL step 12 accumulators
            machine.ft_bow_wrist_y.append(bio.hand_xy[1])  # bow wrist Y in norm coords
            machine.ft_bsa_values.append(bio.bsa)
            machine.ft_release_hand_positions.append(bio.hand_xy.copy())
        machine.ft_frames += 1

        if machine.ft_frames >= machine.follow_through_frames:
            if machine.ft_hip_positions and machine.last_anchor_hip is not None:
                ft_drifts = [float(np.linalg.norm(p - machine.last_anchor_hip))
                             for p in machine.ft_hip_positions]
                ref_sw = float(np.median(list(machine.sw_hist))) if machine.sw_hist else 1.0
                ft_sway = float(np.mean(ft_drifts)) / max(ref_sw, 0.01)
                print(f"  [follow-through] sway={ft_sway:.4f} frames={machine.ft_frames}")

            # KSL step 12 — populate follow-through metrics on the shot record
            if machine.last_shot is not None:
                ref_sw = float(np.median(list(machine.sw_hist))) if machine.sw_hist else 1.0
                ft = compute_follow_through(
                    machine.ft_bow_wrist_y,
                    machine.ft_bsa_values,
                    machine.ft_release_hand_positions,
                    ref_sw,
                )
                machine.last_shot.arm_drop_y = round(ft["arm_drop_y"], 6)
                machine.last_shot.bsa_follow_var = round(ft["bsa_follow_var"], 8)
                machine.last_shot.release_hand_angle = round(ft["release_hand_angle"], 2)

            return ShotState.IDLE

        return None


# ---------------------------------------------------------------------------
# State map — 7-state Phase 1 machine
# ---------------------------------------------------------------------------
_STATE_MAP: dict[ShotState, ArcheryState] = {
    ShotState.IDLE: IdleState(),
    ShotState.SETUP: SetupState(),
    ShotState.DRAW: DrawState(),
    ShotState.ANCHOR: AnchorState(),
    ShotState.AIM: AimState(),
    ShotState.RELEASE: ReleaseState(),
    ShotState.FOLLOW_THROUGH: FollowThroughState(),
}


# ---------------------------------------------------------------------------
# Shot-record computation helpers (extracted from _build_shot_record per E3)
# ---------------------------------------------------------------------------
def _compute_bio_means(bios: list[BioMetrics]) -> tuple[float, float, float, float, float, float]:
    """Return mean BSA, DEA, shoulder_tilt, torso_lean, draw_length, DFL angle."""
    if not bios:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean([b.bsa for b in bios])),
        float(np.mean([b.dea for b in bios])),
        float(np.mean([b.shoulder_tilt for b in bios])),
        float(np.mean([b.torso_lean for b in bios])),
        float(np.mean([b.draw_length for b in bios])),
        float(np.mean([b.dfl_angle for b in bios])),
    )


def _compute_sway(hip_mids: list, ref_sw: float) -> tuple[float, float, float]:
    """Return (sway_range_x, sway_range_y, sway_velocity) normalised by ref_sw."""
    if len(hip_mids) < 2:
        return 0.0, 0.0, 0.0
    hx = [h[0] for h in hip_mids]
    hy = [h[1] for h in hip_mids]
    sw = max(ref_sw, 0.01)
    sway_rx = (max(hx) - min(hx)) / sw
    sway_ry = (max(hy) - min(hy)) / sw
    deltas = [float(np.linalg.norm(hip_mids[i + 1] - hip_mids[i]))
              for i in range(len(hip_mids) - 1)]
    sway_vel = float(np.mean(deltas)) / sw
    return sway_rx, sway_ry, sway_vel


def _compute_release_jump(bio: BioMetrics | None, last_hand_xy: np.ndarray
                          ) -> tuple[float, float, float]:
    """Return (dx, dy, magnitude) of hand displacement at the moment of release."""
    rh = bio.hand_xy if bio is not None else np.zeros(2)
    rdx = float(rh[0] - last_hand_xy[0])
    rdy = float(rh[1] - last_hand_xy[1])
    return rdx, rdy, float(np.sqrt(rdx ** 2 + rdy ** 2))


def _anchor_stability_score(av: float) -> float:
    """Vertex Score — anchor stability component (Phase 1, 100% weight).

    Maps anchor_distance_var to 0–100 using a quadratic decay against the
    GOLD poor-threshold (0.003). Elite (<0.0003) scores ~99; poor (>=0.003) = 0.
    """
    poor = GOLD["anchor_var_poor"]
    if poor <= 0:
        return 0.0
    ratio = min(av / poor, 1.0)
    return round(max(0.0, (1.0 - ratio) ** 2 * 100.0), 1)


class StateMachine:
    """Manages the archery shot lifecycle — Phase 1 7-state machine."""

    def __init__(self, fps: float = 30.0,
                 bowstring_detector=None):
        self.state: ShotState = ShotState.IDLE
        self._state_obj: ArcheryState = _STATE_MAP[ShotState.IDLE]
        self.bowstring_detector = bowstring_detector  # optional BowstringDetector

        # FPS-scaled frame-window constants
        scale = fps / TARGET_FPS
        self.draw_window = max(2, round(DRAW_WINDOW * scale))
        self.anchor_window = max(2, round(ANCHOR_WINDOW * scale))
        self.median_filter_window = max(2, round(MEDIAN_FILTER_WINDOW * scale))
        self.release_confirm_frames = max(1, round(RELEASE_CONFIRM_FRAMES * scale))
        self.follow_through_frames = max(1, round(FOLLOW_THROUGH_FRAMES * scale))
        self.aim_entry_frames = max(3, round(AIM_ENTRY_FRAMES * scale))
        self.setup_collect_frames = SETUP_COLLECT_FRAMES  # wall-time based — not FPS-scaled
        self.transfer_window_frames = max(3, round(TRANSFER_WINDOW_FRAMES * scale))
        self.fps = fps

        buf_len = max(30, round(30 * scale))

        # Shared buffers
        self.raw_hist: collections.deque = collections.deque(maxlen=buf_len)
        self.filt_hist: collections.deque = collections.deque(maxlen=buf_len)
        self.sw_hist: collections.deque = collections.deque(maxlen=buf_len)
        self.arm_angle_hist: collections.deque = collections.deque(maxlen=max(10, round(10 * scale)))

        # State-local data
        self.last_now: float = 0.0
        self.idle_entry: float = 0.0
        self.anchor_t0: float = 0.0
        self.hold: float = 0.0
        self.anchor_dists: list[float] = []
        self.anchor_bios: list[BioMetrics] = []
        self.last_hand_xy: np.ndarray = np.zeros(2)
        self.release_cnt: int = 0
        self.aim_frames: int = 0
        self.state_log: list[str] = []
        self.ft_hip_positions: list[np.ndarray] = []
        self.ft_frames: int = 0
        self.last_anchor_hip: np.ndarray | None = None
        self.last_ctx_bio: BioMetrics | None = None

        # KSL sub-phase accumulators
        self._setup_landmarks = None     # stashed landmarks for stance computation
        self._aim_landmarks = None       # stashed landmarks for tremor/transfer
        self.baseline_stance: dict = {}
        self.baseline_bio: BioMetrics | None = None
        self.setup_posture_score: float = -1.0
        self.setup_wrist_y: list[float] = []
        self.draw_dists: list[float] = []
        self.draw_bios: list[BioMetrics] = []
        self.aim_wrist_positions: list[np.ndarray] = []
        self.aim_elbow_positions: list[np.ndarray] = []
        self.aim_shoulder_positions: list[np.ndarray] = []
        self.aim_shoulder_spreads: list[float] = []
        self.aim_frame_count: int = 0
        self.ft_bow_wrist_y: list[float] = []
        self.ft_bsa_values: list[float] = []
        self.ft_release_hand_positions: list[np.ndarray] = []
        self.cv_release_detected: bool = False
        self.release_confidence: str = "MEDIUM"
        self.last_string_state: StringState = StringState()

        # Phase 1 — per-session calibration (computed in SETUP state)
        self.calib_dists: list[float] = []
        self.calib_done: bool = False
        self.dynamic_draw_threshold: float = WRIST_JAW_DRAW_THRESHOLD
        self.dynamic_anchor_threshold: float = WRIST_JAW_ANCHOR_THRESHOLD

        # Shot output
        self.shot_count: int = 0
        self.hold_times: list[float] = []
        self.best_hold: float = 0.0
        self.last_shot: ShotRecord | None = None
        self.flags_str: str = ""

        # Callbacks — registered externally
        self._callbacks: dict[str, list] = {
            "on_state_change": [],
            "on_shot_end": [],
        }

    def add_callback(self, event: str, fn) -> None:
        if event in self._callbacks:
            self._callbacks[event].append(fn)

    def _run_callbacks(self, event: str, **kwargs) -> None:
        for fn in self._callbacks.get(event, []):
            fn(**kwargs)

    def feed_frame(self, bio: BioMetrics | None, anchor_dist_raw: float,
                   sw: float, now: float, dt: float,
                   landmarks=None, frame_bgr=None) -> None:
        """Process one frame through the state machine.

        Optional landmarks/frame_bgr enable KSL sub-phase and bowstring detection.
        """
        self.last_ctx_bio = bio
        self.last_now = now
        self._setup_landmarks = landmarks  # stash for SETUP stance computation
        self._aim_landmarks = landmarks    # stash for AIM tremor/transfer

        # Update shared buffers
        self.raw_hist.append(anchor_dist_raw)
        df = median_filter(self.raw_hist, self.median_filter_window)
        self.filt_hist.append(df)
        if bio is not None:
            self.arm_angle_hist.append(bio.draw_arm_angle)

        # Bowstring CV detection (optional)
        if self.bowstring_detector is not None and landmarks is not None and frame_bgr is not None:
            self.last_string_state = self.bowstring_detector.feed_frame(
                frame_bgr, landmarks, sw, self.state)
        else:
            self.last_string_state = StringState()

        ctx = StateContext(
            now=now, dt=dt, bio=bio,
            anchor_dist_raw=anchor_dist_raw,
            anchor_dist_filtered=df, sw=sw,
        )

        next_state = self._state_obj.update(self, ctx)
        if next_state is not None and next_state != self.state:
            old = self.state
            self._state_obj.exit(self)
            self.state = next_state
            self._state_obj = _STATE_MAP[next_state]
            self._state_obj.enter(self)
            self._run_callbacks("on_state_change", old_state=old, new_state=next_state)

    def _build_shot_record(self) -> None:
        """Aggregate anchor/aim-phase data into a ShotRecord and fire on_shot_end."""
        hold = self.hold
        self.shot_count += 1
        self.hold_times.append(hold)
        if hold > self.best_hold:
            self.best_hold = hold

        am = float(np.mean(self.anchor_dists)) if self.anchor_dists else 0.0
        av = float(np.var(self.anchor_dists)) if self.anchor_dists else 0.0

        rdx, rdy, rmag = _compute_release_jump(self.last_ctx_bio, self.last_hand_xy)
        absa, adea, astilt, atlean, adlen, adfl = _compute_bio_means(self.anchor_bios)

        ref_sw = float(np.median(list(self.sw_hist))) if self.sw_hist else 1.0
        hip_mids = [b.hip_mid for b in self.anchor_bios]
        sway_rx, sway_ry, sway_vel = _compute_sway(hip_mids, ref_sw)
        self.last_anchor_hip = np.mean(hip_mids, axis=0) if hip_mids else None

        snap = hold < SNAP_SHOT_THRESHOLD
        overtime = hold > MAX_ANCHOR_HOLD
        is_valid = not snap and av < 0.005

        fl = []
        if snap:
            fl.append("SNAP")
        if overtime:
            fl.append("OVERTIME")
        if not is_valid:
            fl.append("FLAGGED")
        self.flags_str = " ".join(fl)

        vs = _anchor_stability_score(av)

        # --- KSL sub-phase metrics ---
        # KSL step 6: Draw profile
        dp = compute_draw_profile(self.draw_dists, self.fps)
        draw_duration_s = round(dp["draw_duration_s"], 3)
        draw_smoothness = round(dp["draw_smoothness"], 8) if dp["draw_smoothness"] >= 0 else -1.0
        # Draw alignment: % frames where BSA+DEA within GOLD during draw
        if self.draw_bios:
            aligned = sum(
                1 for b in self.draw_bios
                if GOLD["bow_shoulder_min"] <= b.bsa <= GOLD["bow_shoulder_max"]
                and GOLD["draw_elbow_min"] <= b.dea <= GOLD["draw_elbow_max"]
            )
            draw_alignment = round(aligned / len(self.draw_bios), 3)
        else:
            draw_alignment = -1.0

        # KSL step 1: Stance width
        stance_w = round(self.baseline_stance.get("stance_width", -1.0), 4)

        # KSL step 4: Setup posture
        setup_ps = round(self.setup_posture_score, 1)

        # KSL step 5: Raise smoothness
        raise_sm = round(compute_raise_quality(self.setup_wrist_y), 8)

        # KSL step 8: Transfer proxy
        transfer = round(compute_transfer_proxy(
            self.aim_shoulder_positions, ref_sw), 6)

        # KSL step 9: Tremor
        tremor_w = compute_tremor(self.aim_wrist_positions, ref_sw)
        tremor_e = compute_tremor(self.aim_elbow_positions, ref_sw)
        tremor_wrist = round(tremor_w, 6) if tremor_w is not None else -1.0
        tremor_elbow = round(tremor_e, 6) if tremor_e is not None else -1.0

        # KSL step 10: Expansion
        expansion = round(compute_expansion(self.aim_shoulder_spreads), 8)

        shot = ShotRecord(
            shot_number=self.shot_count,
            hold_seconds=round(hold, 3),
            anchor_distance_mean=round(am, 6),
            anchor_distance_var=round(av, 8),
            release_jump_x=round(rdx, 6),
            release_jump_y=round(rdy, 6),
            release_jump_mag=round(rmag, 6),
            bow_shoulder_angle=round(absa, 1),
            draw_elbow_angle=round(adea, 1),
            shoulder_tilt_deg=round(astilt, 2),
            torso_lean_deg=round(atlean, 2),
            draw_length_norm=round(adlen, 4),
            dfl_angle=round(adfl, 2),
            sway_range_x=round(sway_rx, 6),
            sway_range_y=round(sway_ry, 6),
            sway_velocity=round(sway_vel, 6),
            is_snap_shot=snap,
            is_overtime=overtime,
            is_valid=is_valid,
            vertex_score=vs,
            state_sequence="\u2192".join(self.state_log),
            flags=self.flags_str,
            draw_duration_s=draw_duration_s,
            draw_smoothness=draw_smoothness,
            draw_alignment_score=draw_alignment,
            stance_width=stance_w,
            setup_posture_score=setup_ps,
            raise_smoothness=raise_sm,
            tremor_rms_wrist=tremor_wrist,
            tremor_rms_elbow=tremor_elbow,
            transfer_shift=transfer,
            expansion_rate=expansion,
            arm_drop_y=-1.0,       # populated in FOLLOW_THROUGH exit
            bsa_follow_var=-1.0,   # populated in FOLLOW_THROUGH exit
            release_hand_angle=-1.0,  # populated in FOLLOW_THROUGH exit
            cv_release_detected=self.cv_release_detected,
            release_confidence=self.release_confidence,
        )
        self.last_shot = shot

        sway_tag = (
            "ELITE" if sway_vel < GOLD["sway_velocity_elite"]
            else "GOOD" if sway_vel < GOLD["sway_velocity_good"]
            else "HIGH")
        fs = f"  [{self.flags_str}]" if self.flags_str else ""
        print(f"  Shot #{self.shot_count:>3d}: Hold {hold:.1f}s  "
              f"BSA {absa:.0f} DEA {adea:.0f} DFL {adfl:.0f} "
              f"Sway:{sway_tag} var {av:.6f}  VS:{vs:.0f}{fs}")

        self._run_callbacks("on_shot_end", shot=shot)

    def reset(self) -> None:
        """Reset for a new session."""
        self.state = ShotState.IDLE
        self._state_obj = _STATE_MAP[ShotState.IDLE]
        self.last_now = 0.0
        self.idle_entry = 0.0
        self.raw_hist.clear()
        self.filt_hist.clear()
        self.sw_hist.clear()
        self.arm_angle_hist.clear()
        self.shot_count = 0
        self.hold_times.clear()
        self.best_hold = 0.0
        self.hold = 0.0
        self.aim_frames = 0
        self.flags_str = ""
        self.last_shot = None
        self.calib_dists = []
        self.calib_done = False
        self.dynamic_draw_threshold = WRIST_JAW_DRAW_THRESHOLD
        self.dynamic_anchor_threshold = WRIST_JAW_ANCHOR_THRESHOLD
        # KSL accumulators
        self.baseline_stance = {}
        self.baseline_bio = None
        self.setup_posture_score = -1.0
        self.setup_wrist_y = []
        self.draw_dists = []
        self.draw_bios = []
        self.aim_wrist_positions = []
        self.aim_elbow_positions = []
        self.aim_shoulder_positions = []
        self.aim_shoulder_spreads = []
        self.aim_frame_count = 0
        self.ft_bow_wrist_y = []
        self.ft_bsa_values = []
        self.ft_release_hand_positions = []
        self.cv_release_detected = False
        self.release_confidence = "MEDIUM"
        self.last_string_state = StringState()
        if self.bowstring_detector is not None:
            self.bowstring_detector.reset()
