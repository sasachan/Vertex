"""
Vertex — Pipeline: reusable frame-processing session.

Shared by CLI (core.py) and web server (server.py).
Owns the source→pose→bio→state machine data flow.
Does NOT own display, I/O format, or transport.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Protocol, Callable

import cv2
import numpy as np

from .models import BioMetrics, ShotRecord, StringState
from .streamer import create_source, InputSource, ImageSource
from .pose_hub import MediaPipePoseProvider
from .bio_lab import (
    compute_bio, shoulder_width, frame_valid, key_landmarks_visible,
    assess_posture, compute_corrections, capture_reference,
)
from .action_logic import StateMachine
from .hud import (
    draw_skeleton, draw_coaching_overlay, draw_correction_guides,
    draw_reference_pose, draw_hud, draw_progress_bar,
)
from .bowstring import BowstringDetector


# ---------------------------------------------------------------------------
# Frame result — one processed frame's output
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    """Output of processing a single frame."""
    frame_bgr: np.ndarray
    bio: BioMetrics | None = None
    state: str = "IDLE"
    hold: float = 0.0
    shot: ShotRecord | None = None          # set when a shot completes this frame
    fps: float = 30.0
    shot_count: int = 0
    avg_hold: float = 0.0
    best_hold: float = 0.0
    vertex_score: float = -1.0
    flags: str = ""
    string_state: StringState = field(default_factory=StringState)
    tremor_rms: float = -1.0
    release_confidence: str = ""
    setup_posture_score: float = -1.0
    frame_idx: int = 0
    total_frames: int = 0
    valid: bool = True


# ---------------------------------------------------------------------------
# Pipeline session — owns one capture→analysis lifecycle
# ---------------------------------------------------------------------------
class PipelineSession:
    """Encapsulates the Vertex processing loop.

    Callers (CLI, server) drive the loop by calling ``process_frame()``
    repeatedly, or ``run()`` for a blocking loop with a callback per frame.
    """

    def __init__(
        self,
        input_arg: str = "0",
        *,
        draw_overlays: bool = True,
        coaching: bool = True,
        debug: bool = False,
    ) -> None:
        self.input_arg = input_arg
        self.draw_overlays = draw_overlays
        self.coaching = coaching
        self.debug = debug

        # Components — created in start()
        self.source: InputSource | None = None
        self.pose: MediaPipePoseProvider | None = None
        self.sm: StateMachine | None = None
        self.bowstring: BowstringDetector | None = None

        # State
        self._bio: BioMetrics | None = None
        self._corrections: list[dict] | None = None
        self._ref_pose: dict | None = None
        self._ref_hip_mid: np.ndarray | None = None
        self._ftimes: collections.deque = collections.deque(maxlen=30)
        self._prev_t: float = 0.0
        self._ts_ms: int = 0
        self._frame_idx: int = 0
        self._total_frames: int = 0
        self._live: bool = False
        self._source_fps: float = 30.0
        self._started: bool = False

        # Shot accumulator — server/CLI can read this
        self.shots: list[ShotRecord] = []
        self._pending_shot: ShotRecord | None = None

        # External callbacks
        self._on_shot: list[Callable[[ShotRecord], None]] = []

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        """Initialise source, pose, state machine. Returns True on success."""
        self.source = create_source(self.input_arg)
        if not self.source.open():
            return False

        self._live = self.source.is_live()
        self._source_fps = self.source.fps()
        self._total_frames = self.source.frame_count()

        self.pose = MediaPipePoseProvider()
        self.pose.start()

        self.bowstring = BowstringDetector()
        self.sm = StateMachine(
            fps=self._source_fps, bowstring_detector=self.bowstring,
        )

        # Wire internal shot callback
        self.sm.add_callback("on_shot_end", self._handle_shot)

        self._prev_t = time.time()
        self._started = True
        return True

    def stop(self) -> None:
        """Release all resources."""
        if self.pose:
            self.pose.stop()
        if self.source:
            self.source.release()
        self._started = False

    def on_shot(self, fn: Callable[[ShotRecord], None]) -> None:
        """Register an external shot-completion callback."""
        self._on_shot.append(fn)

    def reset(self) -> None:
        """Reset session state (shot count, hold times, calibration)."""
        if self.sm:
            self.sm.reset()
        self.shots.clear()
        self._bio = None
        self._corrections = None

    # -- properties ----------------------------------------------------------

    @property
    def is_live(self) -> bool:
        return self._live

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def is_image(self) -> bool:
        return isinstance(self.source, ImageSource)

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # -- frame processing ----------------------------------------------------

    def process_frame(self) -> FrameResult | None:
        """Read and process one frame. Returns None when source is exhausted."""
        if not self._started or self.source is None or self.pose is None:
            return None

        ret, frame = self.source.read()
        if not ret or frame is None:
            return None

        self._frame_idx += 1
        self._pending_shot = None

        # Mirror live camera
        if self._live:
            frame = cv2.flip(frame, 1)

        # Timestamp
        if self._live:
            now = time.time()
            dt = now - self._prev_t
            self._prev_t = now
            self._ts_ms += int(dt * 1000) if dt > 0 else 33
        else:
            dt = 1.0 / self._source_fps
            now = self._frame_idx * dt
            self._ts_ms = int(self._frame_idx * 1000.0 / self._source_fps)

        if dt > 0:
            self._ftimes.append(1.0 / dt)
        fps = float(np.mean(self._ftimes)) if self._ftimes else self._source_fps

        # Pose detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_landmarks = self.pose.detect(rgb, self._ts_ms)

        bio = self._bio
        sm = self.sm
        valid_frame = True

        if all_landmarks is not None:
            lms = all_landmarks[0]

            if not key_landmarks_visible(lms):
                # Fallback draw
                if self.draw_overlays:
                    self.pose.mp_drawing.draw_landmarks(
                        frame, lms,
                        self.pose.PoseLandmarksConnections.POSE_LANDMARKS)
                valid_frame = False
            else:
                valid, sw = frame_valid(lms, sm.sw_hist)
                sm.sw_hist.append(sw)

                if not valid:
                    if self.draw_overlays:
                        self.pose.mp_drawing.draw_landmarks(
                            frame, lms,
                            self.pose.PoseLandmarksConnections.POSE_LANDMARKS)
                    valid_frame = False
                else:
                    bio = compute_bio(lms, sw)
                    self._bio = bio

                    # Reference ghost
                    if self.draw_overlays and self._ref_pose is not None:
                        draw_reference_pose(
                            frame, self._ref_pose,
                            bio.hip_mid, self._ref_hip_mid)

                    # Coaching
                    if self.draw_overlays:
                        if self.coaching:
                            lm_colors = assess_posture(bio)
                            draw_skeleton(
                                frame, lms, lm_colors,
                                self.pose.PoseLandmarksConnections.POSE_LANDMARKS)
                            draw_coaching_overlay(frame, lms, bio)
                            self._corrections = compute_corrections(
                                bio, lms, frame.shape[0], frame.shape[1])
                            draw_correction_guides(frame, self._corrections)
                        else:
                            self._corrections = None
                            self.pose.mp_drawing.draw_landmarks(
                                frame, lms,
                                self.pose.PoseLandmarksConnections.POSE_LANDMARKS)

                    # State machine
                    sm.feed_frame(
                        bio, bio.anchor_dist, sw, now, dt,
                        landmarks=lms, frame_bgr=frame)

        # HUD overlay
        if self.draw_overlays:
            avg_h = float(np.mean(sm.hold_times)) if sm.hold_times else 0.0
            cues = self._corrections if self.coaching else None
            vs = sm.last_shot.vertex_score if sm.last_shot else -1.0
            draw_hud(
                frame, sm.state.value, sm.hold, sm.shot_count, avg_h,
                sm.best_hold, fps, "", bio, self.debug, sm.flags_str, cues,
                vertex_score=vs,
                string_detected=sm.last_string_state.detected,
                tremor_rms=(sm.last_shot.tremor_rms_wrist
                            if sm.last_shot else -1.0),
                release_confidence=sm.release_confidence,
                setup_posture_score=sm.setup_posture_score,
            )
            if not self._live and self._total_frames > 0:
                draw_progress_bar(frame, self._frame_idx, self._total_frames)

        # Build result
        avg_h = float(np.mean(sm.hold_times)) if sm.hold_times else 0.0
        return FrameResult(
            frame_bgr=frame,
            bio=bio,
            state=sm.state.value,
            hold=sm.hold,
            shot=self._pending_shot,
            fps=fps,
            shot_count=sm.shot_count,
            avg_hold=avg_h,
            best_hold=sm.best_hold,
            vertex_score=(sm.last_shot.vertex_score
                          if sm.last_shot else -1.0),
            flags=sm.flags_str,
            string_state=sm.last_string_state,
            tremor_rms=(sm.last_shot.tremor_rms_wrist
                        if sm.last_shot else -1.0),
            release_confidence=sm.release_confidence,
            setup_posture_score=sm.setup_posture_score,
            frame_idx=self._frame_idx,
            total_frames=self._total_frames,
            valid=valid_frame,
        )

    def capture_reference(self) -> bool:
        """Capture/clear reference pose snapshot. Returns True if toggled."""
        if self._ref_pose is not None:
            self._ref_pose = None
            self._ref_hip_mid = None
            return True
        if self._bio is not None and self.pose is not None:
            # Need current landmarks — not stored, so skip if not available
            return False
        return False

    # -- internal ------------------------------------------------------------

    def _handle_shot(self, shot: ShotRecord, **_) -> None:
        self._pending_shot = shot
        self.shots.append(shot)
        for fn in self._on_shot:
            fn(shot)
