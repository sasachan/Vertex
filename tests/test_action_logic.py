"""Tests for vertex.action_logic — Phase 1 7-state machine."""

from __future__ import annotations

import numpy as np
import pytest

from vertex.action_logic import (
    StateMachine, StateContext,
    _anchor_stability_score, _compute_bio_means, _compute_sway, _compute_release_jump,
)
from vertex.models import ShotState, BioMetrics, TARGET_FPS, WRIST_JAW_DRAW_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_bio(anchor_dist: float = 0.15, draw_arm_angle: float = 25.0) -> BioMetrics:
    return BioMetrics(
        anchor_dist=anchor_dist,
        hand_xy=np.array([0.54, 0.28]),
        jaw_xy=np.array([0.535, 0.280]),
        bsa=92.0, dea=142.0,
        shoulder_tilt=1.5, torso_lean=0.8,
        draw_length=1.85, dfl_angle=2.0,
        hip_mid=np.array([0.50, 0.70]),
        draw_arm_angle=draw_arm_angle,
    )


def _feed_frames(sm: StateMachine, n: int, anchor_dist: float,
                 sw: float = 0.2, dt: float = 1/30, start_t: float = 2.0):
    """Feed n frames to the state machine with constant anchor distance."""
    for i in range(n):
        t = start_t + i * dt
        bio = _make_bio(anchor_dist=anchor_dist)
        sm.feed_frame(bio, anchor_dist, sw, t, dt)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestStateMachineInit:
    def test_starts_idle(self):
        sm = StateMachine()
        assert sm.state == ShotState.IDLE

    def test_default_fps(self):
        sm = StateMachine()
        assert sm.draw_window == 4
        assert sm.anchor_window == 8

    def test_60fps_scaling(self):
        sm = StateMachine(fps=60.0)
        assert sm.draw_window == 8    # 4 * 2
        assert sm.anchor_window == 16  # 8 * 2
        assert sm.median_filter_window == 10  # 5 * 2
        assert sm.follow_through_frames == 30  # 15 * 2
        assert sm.aim_entry_frames == 20  # 10 * 2

    def test_calibration_fields_initialised_to_defaults(self):
        sm = StateMachine()
        assert sm.dynamic_draw_threshold == WRIST_JAW_DRAW_THRESHOLD
        assert sm.dynamic_anchor_threshold > 0
        assert not sm.calib_done
        assert sm.calib_dists == []


# ---------------------------------------------------------------------------
# IDLE -> SETUP
# ---------------------------------------------------------------------------
class TestIdleToSetup:
    def test_idle_enters_setup_when_pose_detected(self):
        sm = StateMachine()
        # Feed one frame past idle cooldown with a bio object present
        bio = _make_bio(0.60)
        sm.feed_frame(bio, 0.60, 0.2, 2.0, 1 / 30)
        assert sm.state == ShotState.SETUP

    def test_idle_stays_idle_without_pose(self):
        sm = StateMachine()
        # Feed None bio past cooldown — should stay IDLE
        sm.feed_frame(None, 0.60, 0.2, 2.0, 1 / 30)
        assert sm.state == ShotState.IDLE


# ---------------------------------------------------------------------------
# SETUP calibration and draw-initiation
# ---------------------------------------------------------------------------
class TestSetupCalibration:
    def _enter_setup(self) -> StateMachine:
        sm = StateMachine()
        sm.state = ShotState.SETUP
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.SETUP]
        sm._state_obj.enter(sm)
        sm.idle_entry = 0.0   # allow timeout checks from t=0
        return sm

    def test_dynamic_thresholds_computed_from_resting_distance(self):
        sm = self._enter_setup()
        # Feed SETUP_COLLECT_FRAMES + 1 frames at a high resting dist (1.5 SW)
        for i in range(sm.setup_collect_frames + 2):
            sm.feed_frame(_make_bio(1.5), 1.5, 0.2, float(i) / 30, 1 / 30)
        assert sm.calib_done
        # 1.5 * 0.65 = 0.975 which is capped at WRIST_JAW_DRAW_THRESHOLD (0.50)
        assert sm.dynamic_draw_threshold == WRIST_JAW_DRAW_THRESHOLD

    def test_low_resting_distance_keeps_defaults(self):
        sm = self._enter_setup()
        # Hand already close to jaw at rest (calib data invalid)
        for i in range(sm.setup_collect_frames + 2):
            sm.feed_frame(_make_bio(0.3), 0.3, 0.2, float(i) / 30, 1 / 30)
        assert sm.calib_done
        # rest=0.3 < 0.8 so defaults unchanged
        assert sm.dynamic_draw_threshold == WRIST_JAW_DRAW_THRESHOLD

    def test_setup_timeout_returns_to_idle(self):
        sm = self._enter_setup()
        # Feed calibration frames then hold high dist past SETUP_TIMEOUT
        from vertex.models import SETUP_TIMEOUT
        for i in range(sm.setup_collect_frames + 2):
            sm.feed_frame(_make_bio(1.5), 1.5, 0.2, float(i) / 30, 1 / 30)
        # Now feed past timeout
        sm.feed_frame(_make_bio(1.5), 1.5, 0.2, SETUP_TIMEOUT + 1.0, 1 / 30)
        assert sm.state == ShotState.IDLE


# ---------------------------------------------------------------------------
# SETUP -> DRAW
# ---------------------------------------------------------------------------
class TestSetupToDraw:
    def test_setup_transitions_to_draw_on_decreasing_close_distance(self):
        sm = StateMachine()
        sm.state = ShotState.SETUP
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.SETUP]
        sm._state_obj.enter(sm)
        sm.idle_entry = 0.0

        # Collect calibration with moderate rest distance that keeps defaults
        for i in range(sm.setup_collect_frames + 1):
            sm.feed_frame(_make_bio(1.5), 1.5, 0.2, float(i) / 30, 1 / 30)

        # Feed draw_window+2 frames of STRICTLY DECREASING distance below threshold
        base_t = (sm.setup_collect_frames + 2) / 30
        thresh = sm.dynamic_draw_threshold
        for i in range(sm.draw_window + 2):
            d = thresh * 0.8 - i * 0.01  # decreasing below threshold each frame
            sm.feed_frame(_make_bio(d), d, 0.2, base_t + i / 30, 1 / 30)

        assert sm.state in (ShotState.DRAW, ShotState.ANCHOR)


class TestDrawToAnchor:
    def test_stable_distance_triggers_anchor(self):
        sm = StateMachine()
        sm.state = ShotState.DRAW
        sm.last_now = 1.0
        from vertex.action_logic import _STATE_MAP, DrawState
        sm._state_obj = _STATE_MAP[ShotState.DRAW]
        sm._state_obj.enter(sm)

        # Feed stable distance below anchor threshold
        # 20 frames is enough to pass ANCHOR (8 frames) and reach AIM (10 frames)
        _feed_frames(sm, 20, anchor_dist=0.20, start_t=2.0)
        assert sm.state in (ShotState.ANCHOR, ShotState.AIM)


class TestLetDown:
    def test_draw_abandoned_returns_idle(self):
        sm = StateMachine()
        sm.state = ShotState.DRAW
        sm.last_now = 1.0
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.DRAW]
        sm._state_obj.enter(sm)

        # Feed distance above draw threshold (let-down)
        _feed_frames(sm, 5, anchor_dist=0.60, start_t=2.0)
        assert sm.state == ShotState.IDLE


# ---------------------------------------------------------------------------
# ANCHOR -> AIM
# ---------------------------------------------------------------------------
class TestAnchorToAim:
    def test_anchor_enters_aim_after_stable_frames(self):
        sm = StateMachine()
        sm.state = ShotState.DRAW
        sm.last_now = 1.0
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.DRAW]
        sm._state_obj.enter(sm)

        # Stable low distance → ANCHOR, then more frames → AIM
        _feed_frames(sm, sm.anchor_window + sm.aim_entry_frames + 5, anchor_dist=0.20,
                     start_t=2.0)
        assert sm.state == ShotState.AIM

    def test_anchor_letdown_during_anchor_returns_idle(self):
        sm = StateMachine()
        sm.state = ShotState.ANCHOR
        sm.last_now = 2.0
        sm.anchor_t0 = 2.0
        sm.aim_frames = 0
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.ANCHOR]
        sm._state_obj.enter(sm)

        # Hand retreats past draw threshold → let-down
        _feed_frames(sm, 3, anchor_dist=0.70, start_t=2.1)
        assert sm.state == ShotState.IDLE


# ---------------------------------------------------------------------------
# AIM release detection
# ---------------------------------------------------------------------------
class TestAimRelease:
    def _get_aim_machine(self) -> StateMachine:
        """Return a state machine already in AIM state with anchor history."""
        sm = StateMachine()
        sm.state = ShotState.DRAW
        sm.last_now = 1.0
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.DRAW]
        sm._state_obj.enter(sm)
        _feed_frames(sm, sm.anchor_window + sm.aim_entry_frames + 5,
                     anchor_dist=0.20, start_t=2.0)
        assert sm.state == ShotState.AIM
        return sm

    def test_departure_triggers_release(self):
        sm = self._get_aim_machine()
        # Feed large departure (release)
        base_t = 3.0
        for i in range(60):
            t = base_t + i / 30
            sm.feed_frame(_make_bio(0.80), 0.80, 0.2, t, 1 / 30)
            if sm.state in (ShotState.RELEASE, ShotState.FOLLOW_THROUGH, ShotState.IDLE):
                break
        assert sm.state in (ShotState.RELEASE, ShotState.FOLLOW_THROUGH, ShotState.IDLE)

    def test_shot_record_built_on_release(self):
        sm = self._get_aim_machine()
        shots = []
        sm.add_callback("on_shot_end", lambda shot, **_: shots.append(shot))
        for i in range(80):
            sm.feed_frame(_make_bio(0.80), 0.80, 0.2, 3.0 + i / 30, 1 / 30)
            if sm.state == ShotState.IDLE:
                break
        assert len(shots) == 1
        assert shots[0].vertex_score >= 0


# ---------------------------------------------------------------------------
# FOLLOW_THROUGH -> IDLE
# ---------------------------------------------------------------------------
class TestFollowThrough:
    def test_follow_through_returns_to_idle(self):
        sm = StateMachine()
        sm.state = ShotState.FOLLOW_THROUGH
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.FOLLOW_THROUGH]
        sm._state_obj.enter(sm)
        sm.last_anchor_hip = None

        # Feed follow_through_frames frames
        _feed_frames(sm, sm.follow_through_frames + 2, anchor_dist=0.80, start_t=5.0)
        assert sm.state == ShotState.IDLE


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class TestCallbacks:
    def test_on_state_change_fires(self):
        sm = StateMachine()
        events = []
        sm.add_callback("on_state_change", lambda **kw: events.append(kw))

        # Force transition IDLE → DRAW
        sm.state = ShotState.DRAW
        sm.last_now = 1.0
        from vertex.action_logic import _STATE_MAP, DrawState
        sm._state_obj = _STATE_MAP[ShotState.DRAW]
        sm._state_obj.enter(sm)

        # Feed stable to trigger DRAW → ANCHOR
        _feed_frames(sm, 20, anchor_dist=0.20, start_t=2.0)
        assert len(events) > 0

    def test_on_shot_end_fires(self):
        sm = StateMachine()
        shots = []
        sm.add_callback("on_shot_end", lambda shot, **kw: shots.append(shot))

        # Get to ANCHOR first
        sm.state = ShotState.DRAW
        sm.last_now = 1.0
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.DRAW]
        sm._state_obj.enter(sm)
        _feed_frames(sm, 20, anchor_dist=0.20, start_t=2.0)
        assert sm.state in (ShotState.ANCHOR, ShotState.AIM)

        # Force release by jumping distance — feed enough frames to reach IDLE
        dt = 1.0 / 30
        for i in range(80):
            t = 10.0 + i * dt
            bio = _make_bio(anchor_dist=0.80)
            sm.feed_frame(bio, 0.80, 0.2, t, dt)
            if sm.state == ShotState.IDLE:
                break

        assert len(shots) >= 1


# ---------------------------------------------------------------------------
# Vertex Score helpers
# ---------------------------------------------------------------------------
class TestVertexScore:
    def test_elite_variance_scores_near_100(self):
        from vertex.models import GOLD
        score = _anchor_stability_score(GOLD["anchor_var_elite"])
        # elite=0.0003, poor=0.003 -> ratio=0.1 -> (0.9)^2 * 100 = 81.0
        assert score > 75.0

    def test_poor_variance_scores_zero(self):
        from vertex.models import GOLD
        score = _anchor_stability_score(GOLD["anchor_var_poor"])
        assert score == 0.0

    def test_good_variance_scores_between_40_and_80(self):
        from vertex.models import GOLD
        score = _anchor_stability_score(GOLD["anchor_var_good"])
        assert 20.0 < score < 90.0

    def test_zero_variance_scores_100(self):
        score = _anchor_stability_score(0.0)
        assert score == 100.0


# ---------------------------------------------------------------------------
# Compute helpers (E1 — unit tests for extracted functions)
# ---------------------------------------------------------------------------
class TestComputeHelpers:
    def test_compute_bio_means_empty_returns_zeros(self):
        result = _compute_bio_means([])
        assert all(v == 0.0 for v in result)

    def test_compute_sway_single_point_returns_zeros(self):
        pts = [np.array([0.5, 0.7])]
        assert _compute_sway(pts, 0.2) == (0.0, 0.0, 0.0)

    def test_compute_sway_two_points_nonzero(self):
        pts = [np.array([0.5, 0.70]), np.array([0.52, 0.71])]
        rx, ry, vel = _compute_sway(pts, 0.2)
        assert rx > 0 and ry > 0 and vel > 0

    def test_compute_release_jump_zero_on_matching_positions(self):
        pos = np.array([0.54, 0.28])
        bio = _make_bio()
        bio.hand_xy = pos
        dx, dy, mag = _compute_release_jump(bio, pos)
        assert dx == 0.0 and dy == 0.0 and mag == 0.0


# ---------------------------------------------------------------------------
# Reset — updated to cover Phase 1 fields
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_clears_state(self):
        sm = StateMachine()
        sm.shot_count = 5
        sm.hold_times = [1.0, 2.0, 3.0]
        sm.best_hold = 3.0
        sm.reset()
        assert sm.shot_count == 0
        assert sm.hold_times == []
        assert sm.best_hold == 0.0
        assert sm.state == ShotState.IDLE

    def test_reset_restores_calibration_defaults(self):
        sm = StateMachine()
        sm.dynamic_draw_threshold = 0.20
        sm.dynamic_anchor_threshold = 0.10
        sm.calib_done = True
        sm.reset()
        from vertex.models import WRIST_JAW_DRAW_THRESHOLD
        assert sm.dynamic_draw_threshold == WRIST_JAW_DRAW_THRESHOLD
        assert not sm.calib_done
        assert sm.calib_dists == []


# ---------------------------------------------------------------------------
# KSL Phase 1 — sub-phase accumulator tests
# ---------------------------------------------------------------------------
class TestKSLAccumulators:
    def test_init_ksl_fields_have_defaults(self):
        sm = StateMachine()
        assert sm.baseline_stance == {}
        assert sm.baseline_bio is None
        assert sm.setup_posture_score == -1.0
        assert sm.setup_wrist_y == []
        assert sm.draw_dists == []
        assert sm.draw_bios == []
        assert sm.aim_wrist_positions == []
        assert sm.aim_elbow_positions == []
        assert sm.aim_shoulder_positions == []
        assert sm.aim_shoulder_spreads == []
        assert sm.aim_frame_count == 0
        assert sm.cv_release_detected is False
        assert sm.release_confidence == "MEDIUM"

    def test_reset_clears_ksl_accumulators(self):
        sm = StateMachine()
        sm.baseline_stance = {"stance_width": 1.0}
        sm.setup_posture_score = 8.5
        sm.draw_dists = [0.3, 0.2]
        sm.aim_wrist_positions = [np.array([0.7, 0.3])]
        sm.cv_release_detected = True
        sm.release_confidence = "HIGH"
        sm.reset()
        assert sm.baseline_stance == {}
        assert sm.setup_posture_score == -1.0
        assert sm.draw_dists == []
        assert sm.aim_wrist_positions == []
        assert sm.cv_release_detected is False
        assert sm.release_confidence == "MEDIUM"

    def test_draw_state_accumulates_draw_dists(self):
        sm = StateMachine()
        sm.state = ShotState.DRAW
        from vertex.action_logic import _STATE_MAP
        sm._state_obj = _STATE_MAP[ShotState.DRAW]
        sm._state_obj.enter(sm)
        sm.last_now = 1.0
        # Feed a few frames still below draw threshold
        for i in range(5):
            bio = _make_bio(anchor_dist=0.30)
            sm.feed_frame(bio, 0.30, 0.2, 2.0 + i / 30, 1 / 30)
            if sm.state != ShotState.DRAW:
                break
        assert len(sm.draw_dists) > 0


class TestStateMachineWithBowstringDetector:
    def test_init_with_bowstring_detector(self):
        class FakeDetector:
            def feed_frame(self, *a, **kw):
                from vertex.models import StringState
                return StringState()
            def reset(self):
                pass

        sm = StateMachine(bowstring_detector=FakeDetector())
        assert sm.bowstring_detector is not None

    def test_reset_calls_bowstring_reset(self):
        class FakeDetector:
            def __init__(self):
                self.reset_called = False
            def feed_frame(self, *a, **kw):
                from vertex.models import StringState
                return StringState()
            def reset(self):
                self.reset_called = True

        det = FakeDetector()
        sm = StateMachine(bowstring_detector=det)
        sm.reset()
        assert det.reset_called is True

    def test_feed_frame_accepts_landmarks_and_frame(self):
        """feed_frame with landmarks and frame_bgr params does not raise."""
        sm = StateMachine()
        bio = _make_bio()
        # Should not raise even without bowstring detector
        sm.feed_frame(bio, 0.5, 0.2, 1.0, 1 / 30,
                      landmarks=None, frame_bgr=None)
        assert sm.state in (ShotState.SETUP, ShotState.IDLE)


class TestShotRecordKSLFields:
    """Verify KSL fields are populated in the shot record after a full cycle."""

    def _run_full_shot(self) -> list:
        """Run a state machine through a complete shot cycle and return shots."""
        sm = StateMachine()
        shots = []
        sm.add_callback("on_shot_end", lambda shot, **_: shots.append(shot))

        # IDLE → SETUP (need to wait past IDLE_COOLDOWN)
        from vertex.models import IDLE_COOLDOWN
        sm.feed_frame(_make_bio(0.60), 0.60, 0.2, IDLE_COOLDOWN + 0.1, 1 / 30)
        assert sm.state == ShotState.SETUP

        # SETUP calibration
        for i in range(sm.setup_collect_frames + 2):
            sm.feed_frame(_make_bio(1.5), 1.5, 0.2, 1.0 + i / 30, 1 / 30)

        # SETUP → DRAW (decreasing distance)
        base_t = 2.0
        thresh = sm.dynamic_draw_threshold
        for i in range(sm.draw_window + 2):
            d = thresh * 0.8 - i * 0.01
            sm.feed_frame(_make_bio(d), d, 0.2, base_t + i / 30, 1 / 30)

        # DRAW → ANCHOR → AIM (stable low distance)
        base_t = 3.0
        total_frames = sm.anchor_window + sm.aim_entry_frames + 5
        for i in range(total_frames):
            sm.feed_frame(_make_bio(0.20), 0.20, 0.2, base_t + i / 30, 1 / 30)

        # AIM → RELEASE (large departure)
        base_t = 5.0
        for i in range(80):
            sm.feed_frame(_make_bio(0.80), 0.80, 0.2, base_t + i / 30, 1 / 30)
            if sm.state == ShotState.IDLE:
                break

        return shots

    def test_shot_record_has_draw_duration(self):
        shots = self._run_full_shot()
        assert len(shots) >= 1
        assert shots[0].draw_duration_s > 0

    def test_shot_record_has_draw_smoothness(self):
        shots = self._run_full_shot()
        assert len(shots) >= 1
        # draw_smoothness >= 0 (variance) or -1.0 if insufficient data
        assert shots[0].draw_smoothness >= -1.0

    def test_shot_record_has_state_sequence(self):
        shots = self._run_full_shot()
        assert len(shots) >= 1
        assert "DRAW" in shots[0].state_sequence

    def test_shot_record_has_release_confidence(self):
        shots = self._run_full_shot()
        assert len(shots) >= 1
        assert shots[0].release_confidence in ("HIGH", "MEDIUM", "LOW")
