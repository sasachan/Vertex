"""
Vertex -- Validation Frame Extractor (thin orchestrator)

Scans archery video/image folders, scores frames by biomechanical quality,
saves the top N per video as annotated PNGs, and writes a manifest JSON.

Components used:
  extract_frames_viz.annotate_frame     -- Annotation rendering (dev-tool only)
  VertexPoseHub.StaticFrameDetector     -- MediaPipe IMAGE mode detection
  VertexBioLab.evaluate_frame_quality   -- Gold standard G1-G7 scoring
  VertexSessionIO.SHARegistry           -- SHA-256 dedup registry
  VertexSessionIO.file_sha256           -- File hashing
  VertexSessionIO.move_to_processed     -- Safe file lifecycle management
  models.constants (EXTRACT_*)          -- All quality thresholds

Usage:
    python extract_frames.py <video_folder> [--output <dir>] [--max <N>]
    python extract_frames.py <video_folder> --flush
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from extract_frames_viz import annotate_frame
from vertex.bio_lab import evaluate_frame_quality
from vertex.models import (
    EXTRACT_VIS_THRESHOLD, EXTRACT_SHARPNESS_MIN,
    EXTRACT_ANCHOR_CAMERA_MAX, EXTRACT_ANCHOR_DRAW_MAX,
    EXTRACT_ANCHOR_IDLE_MAX, EXTRACT_ANCHOR_DEEP_FLAG,
    FrameMetrics,
    R_EAR, MOUTH_R, L_SHOULDER, R_SHOULDER, R_INDEX,
    MODEL_PATH,
)
from vertex.pose_hub import StaticFrameDetector
from vertex.session_io import SHARegistry, file_sha256, move_to_processed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_KEY_LANDMARKS = [8, 10, 11, 12, 13, 14, 15, 16, 20, 23, 24]
_VIDEO_EXTS    = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}
_IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".PNG", ".JPG", ".JPEG"}
_SKIP_DIRS     = {"extracted", "processed"}


@dataclass
class ExtractionConfig:
    input_path: str
    output_dir: str
    max_frames: int = 8
    sample_every_sec: float = 2.0
    flush_first: bool = False

    def __post_init__(self) -> None:
        for label, path_str in (("input_path", self.input_path), ("output_dir", self.output_dir)):
            if "\x00" in path_str:
                raise ValueError(f"ExtractionConfig: null byte in {label}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract best biomechanically-valid frames from archery videos."
    )
    ap.add_argument("input",               help="Folder or single file path")
    ap.add_argument("--output",            default=None)
    ap.add_argument("--max",               type=int,   default=8)
    ap.add_argument("--sample-every",      type=float, default=2.0)
    ap.add_argument("--flush",             action="store_true")
    return ap.parse_args()


def _locate_model(script_dir: str) -> str:
    path = os.path.join(script_dir, "src", "vertex", "pose_landmarker_lite.task")
    if not os.path.exists(path):
        print(f"ERROR: Model not found at {path}")
        sys.exit(1)
    return path


# ---------------------------------------------------------------------------
# Frame detection and scoring
# ---------------------------------------------------------------------------

def _sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _detect_frame(frame_bgr: np.ndarray, detector: StaticFrameDetector
                  ) -> tuple[list | None, float]:
    """Sharpness gate then MediaPipe detection. Returns (landmarks_or_None, sharpness)."""
    sharp = _sharpness(frame_bgr)
    if sharp < EXTRACT_SHARPNESS_MIN:
        return None, sharp
    return detector.detect(frame_bgr), sharp


def _frame_metrics(lms: list) -> FrameMetrics | None:
    """Compute and validate spatial metrics from landmarks. Returns None if disqualified."""
    vis_vals = [lms[i].visibility for i in _KEY_LANDMARKS if i < len(lms)]
    if not vis_vals or float(np.min(vis_vals)) < EXTRACT_VIS_THRESHOLD:
        return None
    ls  = np.array([lms[L_SHOULDER].x, lms[L_SHOULDER].y])
    rs  = np.array([lms[R_SHOULDER].x, lms[R_SHOULDER].y])
    sw  = float(np.linalg.norm(ls - rs))
    if sw < 0.01:
        return None
    jaw  = (0.4 * np.array([lms[R_EAR].x,  lms[R_EAR].y])
          + 0.6 * np.array([lms[MOUTH_R].x, lms[MOUTH_R].y]))
    hand = np.array([lms[R_INDEX].x, lms[R_INDEX].y])
    ad   = float(np.linalg.norm(hand - jaw) / sw)
    if ad > EXTRACT_ANCHOR_CAMERA_MAX or ad > EXTRACT_ANCHOR_IDLE_MAX:
        return None
    return FrameMetrics(anchor_dist=round(ad, 3), vis_mean=round(float(np.mean(vis_vals)), 3),
                        sharpness=0.0, shoulder_width=round(sw, 4))


def _score_landmarks(lms: list, sharpness: float) -> dict | None:
    """Assemble scoring dict from valid landmarks. Returns None if disqualified."""
    metrics = _frame_metrics(lms)
    if metrics is None:
        return None
    metrics.sharpness = round(sharpness, 1)
    ad            = metrics.anchor_dist
    phase_hint    = "ANCHOR/AIM" if ad < EXTRACT_ANCHOR_DRAW_MAX else "REST/DRAW"
    quality_flag  = "ANCHORED_DEEP" if ad < EXTRACT_ANCHOR_DEEP_FLAG else "OK"
    angle_fit     = max(0.0, 1.0 - ad / EXTRACT_ANCHOR_CAMERA_MAX)
    sharp_score   = min(sharpness / 3000.0, 1.0)
    base_score    = metrics.vis_mean * 0.5 + angle_fit * 0.3 + sharp_score * 0.2
    if ad < EXTRACT_ANCHOR_DRAW_MAX:
        base_score += 0.25
    checklist, validated = evaluate_frame_quality(lms, metrics)
    return {"score": round(base_score, 4), "anchor_dist": ad,
            "visibility_mean": metrics.vis_mean, "sharpness": metrics.sharpness,
            "phase_hint": phase_hint, "quality_flag": quality_flag,
            "checklist": checklist, "validated": validated, "landmarks": lms}


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _checklist_for_manifest(cl: dict) -> dict:
    out = {k: v for k, v in cl.items() if not k.startswith("_")}
    out["validated"]   = cl.get("_validated", False)
    out["bio_pass"]    = cl.get("_bio_pass", 0)
    out["green_count"] = cl.get("_green_count", 0)
    return out


@dataclass
class _ProcessConfig:
    path: str
    detector: StaticFrameDetector
    output_dir: str
    file_idx: int
    max_frames: int = 8
    sample_every_sec: float = 2.0
    prefix: str = "v"


def _process_video(cfg: _ProcessConfig) -> list[dict]:
    cap   = cv2.VideoCapture(cfg.path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name  = os.path.splitext(os.path.basename(cfg.path))[0]
    print(f"\n  [{cfg.file_idx}] {name}  {int(cap.get(3))}x{int(cap.get(4))}  {fps:.0f}fps  {total/fps:.0f}s")
    step  = max(1, int(fps * cfg.sample_every_sec))
    cands: list[dict] = []
    for fidx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            break
        lms, sharp = _detect_frame(frame, cfg.detector)
        if lms is None:
            continue
        scored = _score_landmarks(lms, sharp)
        if scored:
            scored.update({"frame_idx": fidx, "time_sec": round(fidx / fps, 2),
                           "source": name, "frame": frame})
            cands.append(scored)
    cap.release()
    cands.sort(key=lambda x: x["score"], reverse=True)
    kept: list[dict] = []
    for c in cands:
        if all(abs(c["time_sec"] - k["time_sec"]) > 2.0 for k in kept):
            kept.append(c)
        if len(kept) >= cfg.max_frames:
            break
    return _save_frames(kept, cfg)


def _process_image(cfg: _ProcessConfig) -> list[dict]:
    name  = os.path.splitext(os.path.basename(cfg.path))[0]
    frame = cv2.imread(cfg.path)
    if frame is None:
        print(f"\n  [{cfg.file_idx}] {name}: cannot read")
        return []
    print(f"\n  [{cfg.file_idx}] {name}  ({frame.shape[1]}x{frame.shape[0]})")
    lms, sharp = _detect_frame(frame, cfg.detector)
    if lms is None:
        print("       SKIPPED: blurry or no pose detected")
        return []
    scored = _score_landmarks(lms, sharp)
    if scored is None:
        print("       SKIPPED: disqualified (camera angle / visibility)")
        return []
    scored.update({"frame_idx": 0, "time_sec": 0.0, "source": name, "frame": frame})
    return _save_frames([scored], cfg)


def _save_frames(frames: list[dict], cfg: _ProcessConfig) -> list[dict]:
    name  = os.path.splitext(os.path.basename(cfg.path))[0]
    saved: list[dict] = []
    if not frames:
        print(f"       0 frames saved. HINT: check camera is side-profile "
              f"(anchor > {EXTRACT_ANCHOR_IDLE_MAX}SW = idle stance)")
        return saved
    for rank, r in enumerate(frames, 1):
        ts_part  = f"_t{r['time_sec']:.0f}s" if r.get("time_sec") else ""
        out_name = f"{cfg.prefix}{cfg.file_idx:02d}_{name}_rank{rank:02d}{ts_part}.png"
        cv2.imwrite(os.path.join(cfg.output_dir, out_name), annotate_frame(r["frame"], r, rank, name))
        entry = {k: v for k, v in r.items() if k not in ("frame", "landmarks", "checklist")}
        entry["saved_as"]  = out_name
        entry["validated"] = r.get("validated", False)
        entry["checklist"] = _checklist_for_manifest(r.get("checklist", {}))
        saved.append(entry)
        val  = " VALIDATED" if r.get("validated") else " [review]"
        flag = " [ANCHORED_DEEP]" if r.get("quality_flag") == "ANCHORED_DEEP" else ""
        print(f"       Rank {rank}: t={r.get('time_sec','-')}s  "
              f"score={r['score']:.3f}  anchor={r['anchor_dist']:.3f}SW{val}{flag}")
    return saved


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def _collect_files(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    found: list[str] = []
    for root, _dirs, files in os.walk(input_path):
        if set(root.replace("\\", "/").split("/")) & _SKIP_DIRS:
            continue
        for fname in files:
            if os.path.splitext(fname)[1] in _VIDEO_EXTS | _IMAGE_EXTS:
                found.append(os.path.join(root, fname))
    return found


def _flush_and_rewalk(cfg: ExtractionConfig, registry: SHARegistry) -> list[str]:
    print("\n  [FLUSH] Restoring processed files...")
    restored = registry.flush(cfg.input_path)
    for r in restored:
        print(f"         Restored: {os.path.basename(r)}")
    cleared = sum(1 for f in Path(cfg.output_dir).iterdir()
                  if f.suffix == ".png" or f.name == "manifest.json"
                  for _ in [f.unlink()])
    print(f"  [FLUSH] {len(restored)} restored, {cleared} PNGs removed. Ready.\n")
    return _collect_files(cfg.input_path)


def _extract_all(cfg: ExtractionConfig, detector: StaticFrameDetector,
                 registry: SHARegistry, input_files: list[str]) -> tuple[list, list, int]:
    """Process all files. Returns (all_results, manifest_files, skipped_count)."""
    all_results, manifest_files, skipped = [], [], 0
    video_idx = img_idx = 0
    base         = cfg.input_path if os.path.isdir(cfg.input_path) else os.path.dirname(cfg.input_path)
    allowed_base = str(Path(base).resolve())
    for fpath in sorted(input_files):
        sha = file_sha256(fpath)
        if registry.is_registered(sha):
            prev = registry.get(sha)
            print(f"\n  [SKIP] {os.path.basename(fpath)}")
            print(f"         SHA {sha[:12]}...  processed {prev.get('processed_at','?')}")
            skipped += 1
            continue
        is_video  = os.path.splitext(fpath)[1] in _VIDEO_EXTS
        video_idx, img_idx = (video_idx + 1, img_idx) if is_video else (video_idx, img_idx + 1)
        fidx, pfx = (video_idx, "v") if is_video else (img_idx, "i")
        pcfg  = _ProcessConfig(path=fpath, detector=detector, output_dir=cfg.output_dir,
                               file_idx=fidx, max_frames=cfg.max_frames,
                               sample_every_sec=cfg.sample_every_sec, prefix=pfx)
        saved = _process_video(pcfg) if is_video else _process_image(pcfg)
        all_results.extend(saved)
        manifest_files.append({"source": fpath, "sha256": sha, "frames": saved})
        moved_to = move_to_processed(fpath, allowed_base)
        registry.register(sha, {"original_path": fpath, "sha256": sha,
                                 "filename": os.path.basename(fpath),
                                 "processed_at": datetime.now(timezone.utc).isoformat(),
                                 "frames_extracted": len(saved), "moved_to": moved_to or fpath})
        registry.save()
        if moved_to:
            print(f"       Moved -> processed/{os.path.basename(moved_to)}  [{sha[:12]}...]")
    return all_results, manifest_files, skipped


def _build_manifest_summary(all_results: list[dict], registry: SHARegistry,
                             skipped: int, elapsed: float) -> dict:
    return {
        "total_frames":              len(all_results),
        "anchor_aim_frames":         sum(1 for r in all_results if r["phase_hint"] == "ANCHOR/AIM"),
        "rest_draw_frames":          sum(1 for r in all_results if r["phase_hint"] == "REST/DRAW"),
        "validated_frames":          sum(1 for r in all_results if r.get("validated")),
        "green_ratings_total":       sum(r.get("checklist", {}).get("green_count", 0) for r in all_results),
        "skipped_already_processed": skipped,
        "registry_total":            len(registry._data),
        "elapsed_sec":               round(elapsed, 1),
    }


def _print_summary(summary: dict, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print(f"  COMPLETE -- {summary['total_frames']} frames in {summary['elapsed_sec']:.0f}s")
    print(f"  ANCHOR/AIM: {summary['anchor_aim_frames']}   REST/DRAW: {summary['rest_draw_frames']}")
    print(f"  VALIDATED:  {summary['validated_frames']}/{summary['total_frames']}")
    print(f"  Total GREEN:{summary['green_ratings_total']}")
    if summary.get("skipped_already_processed"):
        print(f"  Skipped: {summary['skipped_already_processed']}  (registry: {summary['registry_total']})")
    print(f"  Manifest:   {os.path.join(output_dir, 'manifest.json')}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args       = _parse_args()
    base_dir   = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    output_dir = args.output or os.path.join(base_dir, "extracted")
    os.makedirs(output_dir, exist_ok=True)
    cfg = ExtractionConfig(input_path=args.input, output_dir=output_dir,
                           max_frames=args.max, sample_every_sec=args.sample_every,
                           flush_first=args.flush)
    registry    = SHARegistry(output_dir)
    input_files = _collect_files(cfg.input_path)
    if cfg.flush_first:
        input_files = _flush_and_rewalk(cfg, registry)
    if not input_files:
        print(f"No video/image files found in: {cfg.input_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"  VERTEX -- Frame Extractor  ({len(input_files)} file(s))")
    print(f"  Input:  {cfg.input_path}\n  Output: {output_dir}")
    if cfg.flush_first:
        print("  Mode:   FLUSH + full reprocess")
    print("=" * 60)

    t0       = time.time()
    model    = _locate_model(os.path.dirname(os.path.abspath(__file__)))
    detector = StaticFrameDetector(model)
    try:
        all_results, manifest_files, skipped = _extract_all(cfg, detector, registry, input_files)
    finally:
        detector.close()

    elapsed  = time.time() - t0
    summary  = _build_manifest_summary(all_results, registry, skipped, elapsed)
    manifest = {"files": manifest_files, "summary": summary}
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    _print_summary(summary, output_dir)


if __name__ == "__main__":
    main()