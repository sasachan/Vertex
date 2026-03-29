"""
Vertex — Session I/O: CSV creation, shot writing, JSON export, audio.

Extracted from core.py to isolate persistence concerns.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import CSV_HEADERS, SESSIONS_DIR, ShotRecord, BEEP_FREQ, BEEP_DURATION_MS


# ---------------------------------------------------------------------------
# Audio callback
# ---------------------------------------------------------------------------
def play_beep() -> None:
    try:
        import winsound
        winsound.Beep(BEEP_FREQ, BEEP_DURATION_MS)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session CSV management
# ---------------------------------------------------------------------------
def create_session_csv(sessions_dir: str = SESSIONS_DIR):
    os.makedirs(sessions_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(sessions_dir, f"session_{ts}.csv")
    fh = open(path, "w", newline="", encoding="utf-8")
    wr = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
    wr.writeheader()
    return path, wr, fh


def write_shot_csv(csv_wr, shot: ShotRecord) -> None:
    csv_wr.writerow({
        "shot_number": shot.shot_number,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hold_seconds": shot.hold_seconds,
        "anchor_distance_mean": shot.anchor_distance_mean,
        "anchor_distance_var": shot.anchor_distance_var,
        "release_jump_x": shot.release_jump_x,
        "release_jump_y": shot.release_jump_y,
        "release_jump_mag": shot.release_jump_mag,
        "bow_shoulder_angle": shot.bow_shoulder_angle,
        "draw_elbow_angle": shot.draw_elbow_angle,
        "shoulder_tilt_deg": shot.shoulder_tilt_deg,
        "torso_lean_deg": shot.torso_lean_deg,
        "draw_length_norm": shot.draw_length_norm,
        "dfl_angle": shot.dfl_angle,
        "sway_range_x": shot.sway_range_x,
        "sway_range_y": shot.sway_range_y,
        "sway_velocity": shot.sway_velocity,
        "is_snap_shot": shot.is_snap_shot,
        "is_overtime": shot.is_overtime,
        "is_valid": shot.is_valid,
        "vertex_score": shot.vertex_score,
        "state_sequence": shot.state_sequence,
        # KSL Phase 1 additions
        "draw_duration_s": shot.draw_duration_s,
        "draw_smoothness": shot.draw_smoothness,
        "draw_alignment_score": shot.draw_alignment_score,
        "stance_width": shot.stance_width,
        "setup_posture_score": shot.setup_posture_score,
        "raise_smoothness": shot.raise_smoothness,
        "tremor_rms_wrist": shot.tremor_rms_wrist,
        "tremor_rms_elbow": shot.tremor_rms_elbow,
        "transfer_shift": shot.transfer_shift,
        "expansion_rate": shot.expansion_rate,
        "arm_drop_y": shot.arm_drop_y,
        "bsa_follow_var": shot.bsa_follow_var,
        "release_hand_angle": shot.release_hand_angle,
        "cv_release_detected": shot.cv_release_detected,
        "release_confidence": shot.release_confidence,
    })


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------
def write_session_json(path: str, shots: list[ShotRecord],
                       summary: dict) -> None:
    """Write session data as JSON report."""
    data = {
        "version": "0.2.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "shots": [asdict(s) for s in shots],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# File hashing and safe path operations (used by extraction tool)
# ---------------------------------------------------------------------------

def _sanitise_str(value: str) -> str:
    """Remove null bytes and non-printable control characters from a string."""
    cleaned = value.replace("\x00", "")
    return "".join(c for c in cleaned if unicodedata.category(c)[0] != "C" or c in "\t\n\r")


def file_sha256(path: str) -> str:
    """Return hex SHA-256 digest of a file (chunked — memory safe for large videos)."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def move_to_processed(fpath: str, allowed_base: str) -> str | None:
    """Move fpath into a processed/ subfolder. Returns destination path or None.

    Security: resolves symlinks, rejects paths that escape allowed_base.
    """
    allowed = Path(allowed_base).resolve()
    src = Path(fpath).resolve()
    if not src.is_file():
        return None
    if not src.is_relative_to(allowed):
        raise ValueError(f"move_to_processed: path outside allowed base — {src}")

    proc_dir = src.parent / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    dest = proc_dir / src.name
    if dest.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        dest = proc_dir / f"{src.stem}_{ts}{src.suffix}"
    try:
        shutil.move(str(src), str(dest))
        return str(dest)
    except OSError as exc:
        print(f"WARNING: could not move file — {exc}")
        return None


# ---------------------------------------------------------------------------
# SHA registry — tracks processed files across extraction runs
# ---------------------------------------------------------------------------
_REGISTRY_FILENAME = "processed_registry.json"


class SHARegistry:
    """Tracks processed files by SHA-256 to avoid duplicate extractions.

    All public methods are safe to call multiple times without side-effects.
    """

    def __init__(self, output_dir: str) -> None:
        self._dir = output_dir
        self._path = os.path.join(output_dir, _REGISTRY_FILENAME)
        self._data: dict = {}
        self.load()

    def load(self) -> None:
        """Load registry from disk; no-op if file does not exist."""
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                self._data = json.load(f)

    def save(self) -> None:
        """Persist registry to disk (call after each file for crash-safety)."""
        os.makedirs(self._dir, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def is_registered(self, sha: str) -> bool:
        return sha in self._data

    def get(self, sha: str) -> dict | None:
        return self._data.get(sha)

    def register(self, sha: str, metadata: dict) -> None:
        """Register a SHA with associated metadata. String values are sanitised."""
        clean = {
            k: _sanitise_str(v) if isinstance(v, str) else v
            for k, v in metadata.items()
        }
        self._data[sha] = clean

    def flush(self, source_dir: str) -> list[str]:
        """Restore all processed/ files to their parent folders. Returns list of restored paths.

        After flush the registry is cleared. Output PNGs and manifest are NOT removed here
        — that responsibility belongs to the orchestrator (extract_frames.py).
        """
        source_dir_path = Path(source_dir).resolve()
        restored: list[str] = []
        for root, _dirs, files in os.walk(source_dir):
            norm = root.replace("\\", "/").rstrip("/")
            if not norm.endswith("/processed"):
                continue
            parent = os.path.dirname(root)
            for fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join(parent, fname)
                if os.path.exists(dst):
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                    stem, ext = os.path.splitext(fname)
                    dst = os.path.join(parent, f"{stem}_restored_{ts}{ext}")
                try:
                    shutil.move(src, dst)
                    restored.append(dst)
                except OSError as exc:
                    print(f"WARNING: could not restore {fname} — {exc}")
        self._data = {}
        if os.path.exists(self._path):
            os.remove(self._path)
        return restored
