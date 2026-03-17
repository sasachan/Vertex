"""
Vertex — VertexCore: Session orchestrator (Mediator + Callback system).

Wires Streamer, PoseHub, ActionLogic, BioLab, HUD, and SessionIO.
Entry point: python -m vertex [input] [--output DIR] [--no-display]
"""

from __future__ import annotations

import argparse
import collections
import os
import time

import cv2
import numpy as np

from .models import (
    GOLD, SESSIONS_DIR, BioMetrics, ShotRecord, ANCHOR_WINDOW,
)
from .streamer import create_source, ImageSource
from .pose_hub import MediaPipePoseProvider
from .bio_lab import (
    compute_bio, shoulder_width, frame_valid, key_landmarks_visible,
    assess_posture, compute_corrections, capture_reference,
)
from .action_logic import StateMachine
from .hud import (
    draw_skeleton, draw_coaching_overlay, draw_correction_guides,
    draw_reference_pose, draw_hud, draw_progress_bar, draw_consent_banner,
)
from .session_io import (
    create_session_csv, write_shot_csv, write_session_json, play_beep,
)
from .bowstring import BowstringDetector


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vertex",
        description="Vertex TheSighter — archery biomechanics analysis",
    )
    p.add_argument(
        "input", nargs="?", default="0",
        help="Input source: camera index (default 0), video file path, "
             "image file path, or direct video URL",
    )
    p.add_argument(
        "-o", "--output", default=SESSIONS_DIR,
        help="Output directory for session data (default: sessions/)",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Headless mode — no cv2 window, output files only",
    )
    return p.parse_args()


def _analyze_image(args: argparse.Namespace, source, pose) -> None:
    """Single-frame analysis for image input."""
    ret, frame = source.read()
    if not ret or frame is None:
        print("ERROR: Cannot read image.")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    all_landmarks = pose.detect(rgb, 0)

    if all_landmarks is None:
        print("No pose detected in image.")
        if not args.no_display:
            cv2.imshow("Vertex - TheSighter", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    lms = all_landmarks[0]

    if not key_landmarks_visible(lms):
        print("Key landmarks not visible in image.")
        pose.mp_drawing.draw_landmarks(
            frame, lms, pose.PoseLandmarksConnections.POSE_LANDMARKS)
    else:
        sw = shoulder_width(lms)
        bio = compute_bio(lms, sw)
        lm_colors = assess_posture(bio)
        draw_skeleton(frame, lms, lm_colors,
                      pose.PoseLandmarksConnections.POSE_LANDMARKS)
        draw_coaching_overlay(frame, lms, bio)
        corrections = compute_corrections(bio, lms, frame.shape[0], frame.shape[1])
        draw_correction_guides(frame, corrections)

        # Print analysis
        print(f"  Bow Shoulder Angle: {bio.bsa:.1f}°")
        print(f"  Draw Elbow Angle:   {bio.dea:.1f}°")
        print(f"  Shoulder Tilt:      {bio.shoulder_tilt:.1f}°")
        print(f"  Torso Lean:         {bio.torso_lean:.1f}°")
        print(f"  DFL Angle:          {bio.dfl_angle:.1f}°")
        print(f"  Draw Length (norm):  {bio.draw_length:.3f}")
        for corr in corrections:
            print(f"  Correction: {corr['cue_text']}")

    # Save annotated image
    basename = os.path.splitext(os.path.basename(source._path))[0]
    out_path = os.path.join(args.output, f"{basename}_analyzed.jpg")
    os.makedirs(args.output, exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"  Saved: {out_path}")

    if not args.no_display:
        cv2.imshow("Vertex - TheSighter", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    args = _parse_args()

    # --- Input source ---
    source = create_source(args.input)
    if not source.open():
        print(f"ERROR: Cannot open input: {args.input}")
        return

    live = source.is_live()
    source_fps = source.fps()

    # --- Pose provider (Strategy) ---
    pose = MediaPipePoseProvider()
    pose.start()

    # --- Image mode: single-frame analysis ---
    if isinstance(source, ImageSource):
        try:
            _analyze_image(args, source, pose)
        finally:
            pose.stop()
            source.release()
        return

    # --- Session CSV ---
    sess_path, csv_wr, csv_fh = create_session_csv(args.output)

    # --- State machine (FPS-scaled) + Bowstring CV detector ---
    bowstring = BowstringDetector()
    sm = StateMachine(fps=source_fps, bowstring_detector=bowstring)

    # Register callbacks
    shots_list: list[ShotRecord] = []

    def on_shot_end(shot: ShotRecord, **_):
        write_shot_csv(csv_wr, shot)
        csv_fh.flush()
        shots_list.append(shot)
        play_beep()

    sm.add_callback("on_shot_end", on_shot_end)

    # --- Video output writer (live and video sources) ---
    video_writer: cv2.VideoWriter | None = None
    if live:
        ts_tag = time.strftime('%Y%m%d_%H%M%S')
        out_video = os.path.join(args.output, f"session_{ts_tag}_recorded.mp4")
    else:
        basename = os.path.splitext(os.path.basename(args.input))[0]
        out_video = os.path.join(args.output, f"{basename}_analyzed.mp4")
    os.makedirs(args.output, exist_ok=True)

    # --- Display state ---
    debug = False
    coaching = True
    bio: BioMetrics | None = None
    corrections: list[dict] | None = None
    ref_pose: dict | None = None
    ref_hip_mid: np.ndarray | None = None
    diag_t = 0.0
    paused = False

    ftimes: collections.deque = collections.deque(maxlen=30)
    prev_t = time.time()
    ts_ms = 0
    frame_idx = 0
    total_frames = source.frame_count()

    # --- Banner ---
    input_desc = f"Camera({args.input})" if live else os.path.basename(args.input)
    print("=" * 60)
    print("  VERTEX — TheSighter (Phase 1)")
    print("  Benchmarks: Hennessy 1990 | Soylu 2006 | Ertan 2003")
    print("              Shinohara 2018 | Jacquot 2025 | Lin 2010")
    print("=" * 60)
    print(f"  Session: {os.path.basename(sess_path)}")
    print(f"  Input:   {input_desc} @ {source_fps:.0f} FPS")
    print(f"  Model:   pose_landmarker_lite (CPU)")
    print("-" * 60)
    controls = "Q:quit R:reset D:debug C:coach S:ref"
    if not live:
        controls += " SPACE:pause <->:step"
    print(f"  Controls: {controls}")
    print(f"  Flow: IDLE -> SETUP -> DRAW -> ANCHOR -> AIM -> RELEASE -> FOLLOW_THROUGH")
    print("-" * 60)

    # --- Consent gate (GDPR Article 9 — biometric data) ---
    if not args.no_display:
        print("  Waiting for consent (ENTER to start, Q to quit)...")
        if live:
            while True:
                ret_c, frame_c = source.read()
                if ret_c and frame_c is not None:
                    frame_c = cv2.flip(frame_c, 1)
                    draw_consent_banner(frame_c)
                    cv2.imshow("Vertex - TheSighter", frame_c)
                key_c = cv2.waitKey(30) & 0xFF
                if key_c == 13:  # Enter
                    break
                elif key_c == ord("q"):
                    csv_fh.close()
                    pose.stop()
                    source.release()
                    return
        else:
            # Show consent over the first frame of the video/image
            ret_c, frame_c = source.read()
            if ret_c and frame_c is not None:
                draw_consent_banner(frame_c)
                cv2.imshow("Vertex - TheSighter", frame_c)
            else:
                blank = np.zeros((480, 640, 3), np.uint8)
                draw_consent_banner(blank)
                cv2.imshow("Vertex - TheSighter", blank)
            while True:
                key_c = cv2.waitKey(30) & 0xFF
                if key_c == 13:
                    break
                elif key_c == ord("q"):
                    csv_fh.close()
                    pose.stop()
                    source.release()
                    return
            # Rewind to start so the main loop processes all frames from the beginning
            if hasattr(source, "_cap") and source._cap is not None:
                source._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
        print("  Consent confirmed — session started.")

    # --- Main loop ---
    try:
        while True:
            # Pause handling (video only)
            if paused and not live:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key == ord("q"):
                    break
                elif key == 83:  # Right arrow
                    paused = False  # process one frame, re-pause below
                continue

            ret, frame = source.read()
            if not ret or frame is None:
                break

            frame_idx += 1

            # Mirror only for live camera
            if live:
                frame = cv2.flip(frame, 1)

            # Timestamp strategy
            if live:
                now = time.time()
                dt = now - prev_t
                prev_t = now
                ts_ms += int(dt * 1000) if dt > 0 else 33
            else:
                dt = 1.0 / source_fps
                now = frame_idx * dt
                ts_ms = int(frame_idx * 1000.0 / source_fps)

            if dt > 0:
                ftimes.append(1.0 / dt)
            fps = np.mean(ftimes) if ftimes else source_fps

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_landmarks = pose.detect(rgb, ts_ms)

            if all_landmarks is not None:
                lms = all_landmarks[0]

                # Visibility gate
                if not key_landmarks_visible(lms):
                    pose.mp_drawing.draw_landmarks(
                        frame, lms, pose.PoseLandmarksConnections.POSE_LANDMARKS)
                    draw_hud(frame, sm.state.value, sm.hold, sm.shot_count,
                             float(np.mean(sm.hold_times)) if sm.hold_times else 0.0,
                             sm.best_hold, fps, sess_path, bio, debug, sm.flags_str,
                             vertex_score=sm.last_shot.vertex_score if sm.last_shot else -1.0)
                    if not live and total_frames > 0:
                        draw_progress_bar(frame, frame_idx, total_frames)
                    if video_writer is not None:
                        video_writer.write(frame)
                    if not args.no_display:
                        cv2.imshow("Vertex - TheSighter", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                # Frame validity
                valid, sw = frame_valid(lms, sm.sw_hist)
                sm.sw_hist.append(sw)
                if not valid:
                    pose.mp_drawing.draw_landmarks(
                        frame, lms, pose.PoseLandmarksConnections.POSE_LANDMARKS)
                    draw_hud(frame, sm.state.value, sm.hold, sm.shot_count,
                             float(np.mean(sm.hold_times)) if sm.hold_times else 0.0,
                             sm.best_hold, fps, sess_path, bio, debug, sm.flags_str,
                             vertex_score=sm.last_shot.vertex_score if sm.last_shot else -1.0)
                    if not live and total_frames > 0:
                        draw_progress_bar(frame, frame_idx, total_frames)
                    if video_writer is not None:
                        video_writer.write(frame)
                    if not args.no_display:
                        cv2.imshow("Vertex - TheSighter", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                # Biomechanics
                bio = compute_bio(lms, sw)

                # Reference pose ghost
                if ref_pose is not None:
                    draw_reference_pose(frame, ref_pose, bio.hip_mid, ref_hip_mid)

                # Coaching overlay
                if coaching:
                    lm_colors = assess_posture(bio)
                    draw_skeleton(frame, lms, lm_colors,
                                  pose.PoseLandmarksConnections.POSE_LANDMARKS)
                    draw_coaching_overlay(frame, lms, bio)
                    corrections = compute_corrections(
                        bio, lms, frame.shape[0], frame.shape[1])
                    draw_correction_guides(frame, corrections)
                else:
                    corrections = None
                    pose.mp_drawing.draw_landmarks(
                        frame, lms, pose.PoseLandmarksConnections.POSE_LANDMARKS)

                # State machine
                sm.feed_frame(bio, bio.anchor_dist, sw, now, dt,
                              landmarks=lms, frame_bgr=frame)

                # Diagnostics (once per second)
                if live:
                    diag_check = time.time()
                else:
                    diag_check = now
                if diag_check - diag_t >= 1.0:
                    diag_t = diag_check
                    rd = list(sm.filt_hist)
                    vd = np.var(rd[-sm.anchor_window:]) if len(rd) >= sm.anchor_window else -1
                    aa = list(sm.arm_angle_hist)
                    aa_delta = float(np.mean(np.abs(np.diff(aa)))) if len(aa) > 1 else -1
                    df = rd[-1] if rd else 0.0
                    print(f"  [diag] {sm.state.value:>8s}  d={df:.4f}  v={vd:.6f}  "
                          f"bsa={bio.bsa:.0f}  dea={bio.dea:.0f}  "
                          f"arm={bio.draw_arm_angle:.0f}  aa_d={aa_delta:.2f}")

            # HUD
            avg_h = float(np.mean(sm.hold_times)) if sm.hold_times else 0.0
            cues = corrections if coaching else None
            vs = sm.last_shot.vertex_score if sm.last_shot else -1.0
            draw_hud(frame, sm.state.value, sm.hold, sm.shot_count, avg_h,
                     sm.best_hold, fps, sess_path, bio, debug, sm.flags_str, cues,
                     vertex_score=vs,
                     string_detected=sm.last_string_state.detected,
                     tremor_rms=sm.last_shot.tremor_rms_wrist if sm.last_shot else -1.0,
                     release_confidence=sm.release_confidence,
                     setup_posture_score=sm.setup_posture_score)

            # Progress bar for video
            if not live and total_frames > 0:
                draw_progress_bar(frame, frame_idx, total_frames)

            # Video writer — initialise on first processed frame (live and video)
            if video_writer is None:
                h_f, w_f = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(out_video, fourcc, source_fps, (w_f, h_f))
            if video_writer is not None:
                video_writer.write(frame)

            # Display
            if not args.no_display:
                cv2.imshow("Vertex - TheSighter", frame)

                wait_ms = 1 if live else max(1, int(1000 / source_fps))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    sm.reset()
                    bio = None
                    corrections = None
                    csv_fh.close()
                    sess_path, csv_wr, csv_fh = create_session_csv(args.output)
                    print(f"  Reset -> {os.path.basename(sess_path)}")
                elif key == ord("d"):
                    debug = not debug
                elif key == ord("c"):
                    coaching = not coaching
                elif key == ord("s"):
                    if ref_pose is not None:
                        ref_pose = None
                        ref_hip_mid = None
                        print("  [snapshot] Reference pose cleared")
                    elif bio is not None and all_landmarks is not None:
                        ref_pose = capture_reference(all_landmarks[0])
                        ref_hip_mid = bio.hip_mid.copy()
                        print("  [snapshot] Reference pose captured")
                elif key == ord(" ") and not live:
                    paused = True
                elif key == 81 and not live:  # Left arrow: step back (re-process)
                    pass  # cv2 VideoCapture doesn't support backward seek reliably

    finally:
        csv_fh.close()
        if video_writer is not None:
            video_writer.release()
        pose.stop()
        source.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Session summary
    print()
    print("=" * 60)
    print("  SESSION SUMMARY")
    print(f"  Total shots:  {sm.shot_count}")
    if sm.hold_times:
        print(f"  Avg hold:     {np.mean(sm.hold_times):.1f}s")
        print(f"  Best hold:    {sm.best_hold:.1f}s")
        print(f"  Worst hold:   {min(sm.hold_times):.1f}s")
        print(f"  Consistency:  \u03c3 = {np.std(sm.hold_times):.2f}s")
        in_gold = sum(1 for h in sm.hold_times if GOLD["hold_min"] <= h <= GOLD["hold_max"])
        print(f"  In gold range ({GOLD['hold_min']}-{GOLD['hold_max']}s): "
              f"{in_gold}/{len(sm.hold_times)}")
    print(f"  Data saved:   {sess_path}")

    # JSON report
    if shots_list:
        avg_vs = float(np.mean([s.vertex_score for s in shots_list if s.vertex_score >= 0]))
        summary = {
            "total_shots": sm.shot_count,
            "avg_hold": float(np.mean(sm.hold_times)) if sm.hold_times else 0.0,
            "best_hold": sm.best_hold,
            "hold_std": float(np.std(sm.hold_times)) if sm.hold_times else 0.0,
            "avg_vertex_score": round(avg_vs, 1),
        }
        json_path = sess_path.replace(".csv", ".json")
        write_session_json(json_path, shots_list, summary)
        print(f"  Avg VS:       {avg_vs:.1f}  (anchor stability, Phase 1)")
        print(f"  JSON report:  {json_path}")

    if video_writer is not None:
        print(f"  Video saved:  {out_video}")

    print("=" * 60)


if __name__ == "__main__":
    main()
