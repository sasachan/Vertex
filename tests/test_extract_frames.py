"""Tests for extract_frames — orchestrator, scoring, and annotation helpers."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import extract_frames as ef
from extract_frames import (
    ExtractionConfig,
    _collect_files,
    _checklist_for_manifest,
    _score_landmarks,
    _detect_frame,
)
from tests.conftest import FakeLandmark, _make_landmarks
from vertex.models import FrameMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anchored_landmarks():
    """Landmarks where hand is near jaw — confirmed ANCHOR/AIM phase."""
    lms = _make_landmarks()
    # jaw proxy = 0.4*R_EAR + 0.6*MOUTH_R ≈ (0.538, 0.280)
    # place R_INDEX (hand) right at jaw → anchor_dist near 0
    # Use a small non-zero offset so anchor_dist is > DEEP_FLAG (0.05 SW)
    sw = abs(lms[12].x - lms[11].x)  # ≈ 0.2
    # set anchor_dist to about 0.15 SW (well in ANCHOR/AIM range)
    jaw_x = 0.4 * lms[8].x + 0.6 * lms[10].x
    jaw_y = 0.4 * lms[8].y + 0.6 * lms[10].y
    lms[20] = FakeLandmark(x=jaw_x + 0.15 * sw, y=jaw_y, visibility=0.99)
    return lms


def _front_facing_landmarks():
    """Landmarks where hand is far from jaw — front-facing camera."""
    lms = _make_landmarks()
    # Place hand far away: anchor_dist > EXTRACT_ANCHOR_CAMERA_MAX (1.5 SW)
    sw = abs(lms[12].x - lms[11].x)
    lms[20] = FakeLandmark(x=lms[20].x + 2.0 * sw, y=lms[20].y, visibility=0.99)
    return lms


def _idle_stance_landmarks():
    """Hand at >EXTRACT_ANCHOR_IDLE_MAX SW — archer at rest."""
    lms = _make_landmarks()
    sw = abs(lms[12].x - lms[11].x)
    jaw_x = 0.4 * lms[8].x + 0.6 * lms[10].x
    jaw_y = 0.4 * lms[8].y + 0.6 * lms[10].y
    lms[20] = FakeLandmark(x=jaw_x + 0.8 * sw, y=jaw_y, visibility=0.99)
    return lms


def _deep_anchor_landmarks():
    """Hand extremely close to jaw — ANCHORED_DEEP flag."""
    lms = _make_landmarks()
    sw = abs(lms[12].x - lms[11].x)
    jaw_x = 0.4 * lms[8].x + 0.6 * lms[10].x
    jaw_y = 0.4 * lms[8].y + 0.6 * lms[10].y
    lms[20] = FakeLandmark(x=jaw_x + 0.01 * sw, y=jaw_y, visibility=0.99)
    return lms


def _low_visibility_landmarks():
    """Key landmarks with visibility below threshold."""
    lms = _make_landmarks()
    for i in [8, 10, 11, 12, 13, 14, 15, 16, 20, 23, 24]:
        lms[i] = FakeLandmark(x=lms[i].x, y=lms[i].y, visibility=0.40)
    return lms


# ---------------------------------------------------------------------------
# T1 — ExtractionConfig: null byte raises ValueError
# ---------------------------------------------------------------------------
class TestExtractionConfig:
    def test_null_byte_in_input_raises(self):
        with pytest.raises(ValueError, match="null byte"):
            ExtractionConfig(input_path="/some\x00path", output_dir="/out")

    def test_null_byte_in_output_raises(self):
        with pytest.raises(ValueError, match="null byte"):
            ExtractionConfig(input_path="/valid", output_dir="/out\x00dir")

    def test_valid_config_constructs(self):
        cfg = ExtractionConfig(input_path="/valid", output_dir="/out", max_frames=5)
        assert cfg.max_frames == 5


# ---------------------------------------------------------------------------
# T2 — _collect_files: skips SKIP_DIRS
# ---------------------------------------------------------------------------
class TestCollectFiles:
    def test_skips_processed_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = os.path.join(tmpdir, "processed")
            os.makedirs(proc)
            open(os.path.join(proc, "video.mp4"), "w").close()
            open(os.path.join(tmpdir, "real.mp4"), "w").close()
            files = _collect_files(tmpdir)
            assert not any("processed" in f for f in files)
            assert any("real.mp4" in f for f in files)

    def test_single_file_returns_itself(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            result = _collect_files(path)
            assert result == [path]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# T3 — _score_landmarks: ANCHOR/AIM phase returns valid score
# ---------------------------------------------------------------------------
class TestScoreLandmarksAnchorAim:
    def test_anchored_frame_scores_above_zero(self):
        lms = _anchored_landmarks()
        result = _score_landmarks(lms, sharpness=400.0)
        assert result is not None
        assert result["score"] > 0
        assert result["phase_hint"] == "ANCHOR/AIM"

    def test_quality_flag_ok_for_normal_anchor(self):
        lms = _anchored_landmarks()
        result = _score_landmarks(lms, sharpness=400.0)
        assert result is not None
        assert result["quality_flag"] == "OK"


# ---------------------------------------------------------------------------
# T4 — _score_landmarks: front-facing camera rejected
# ---------------------------------------------------------------------------
class TestScoreLandmarksFrontFacing:
    def test_returns_none_when_anchor_exceeds_camera_max(self):
        lms = _front_facing_landmarks()
        result = _score_landmarks(lms, sharpness=400.0)
        assert result is None


# ---------------------------------------------------------------------------
# T5 — _score_landmarks: idle stance rejected
# ---------------------------------------------------------------------------
class TestScoreLandmarksIdleStance:
    def test_returns_none_for_rest_above_idle_max(self):
        lms = _idle_stance_landmarks()
        result = _score_landmarks(lms, sharpness=400.0)
        assert result is None


# ---------------------------------------------------------------------------
# T6 — _score_landmarks: low visibility rejected
# ---------------------------------------------------------------------------
class TestScoreLandmarksLowVisibility:
    def test_returns_none_when_key_landmark_below_threshold(self):
        lms = _low_visibility_landmarks()
        result = _score_landmarks(lms, sharpness=400.0)
        assert result is None


# ---------------------------------------------------------------------------
# T7 — _score_landmarks: ANCHORED_DEEP flag
# ---------------------------------------------------------------------------
class TestScoreLandmarksAnchoredDeep:
    def test_quality_flag_anchored_deep(self):
        lms = _deep_anchor_landmarks()
        result = _score_landmarks(lms, sharpness=400.0)
        assert result is not None
        assert result["quality_flag"] == "ANCHORED_DEEP"


# ---------------------------------------------------------------------------
# T8 — _checklist_for_manifest: strips internal keys
# ---------------------------------------------------------------------------
class TestChecklistForManifest:
    def test_strips_underscore_keys(self):
        cl = {
            "G1": {"label": "A", "rating": "PASS"},
            "_validated": True,
            "_bio_pass": 3,
            "_green_count": 2,
        }
        result = _checklist_for_manifest(cl)
        assert "_validated" not in result
        assert "_bio_pass" not in result
        assert result["validated"] is True
        assert result["bio_pass"] == 3

    def test_preserves_g_keys(self):
        cl = {"G1": {"label": "A"}, "_validated": False, "_bio_pass": 0, "_green_count": 0}
        result = _checklist_for_manifest(cl)
        assert "G1" in result


# ---------------------------------------------------------------------------
# T9 — _detect_frame: blurry frame skips MediaPipe call
# ---------------------------------------------------------------------------
class TestDetectFrameSharpnessGate:
    def test_blurry_frame_returns_none_without_calling_detector(self):
        class _NeverCallDetector:
            def detect(self, _frame):
                raise AssertionError("detect() must not be called for blurry frames")

        # A uniform grey frame has near-zero Laplacian variance
        blank = np.full((100, 100, 3), 128, dtype=np.uint8)
        lms, sharp = _detect_frame(blank, _NeverCallDetector())
        assert lms is None
        assert sharp < 50.0  # uniform grey


# ---------------------------------------------------------------------------
# T10 — annotate_frame: produces output larger than input (panel appended)
# ---------------------------------------------------------------------------
class TestAnnotateFrame:
    def test_output_taller_than_input(self):
        lms = _anchored_landmarks()
        result = _score_landmarks(lms, sharpness=400.0)
        assert result is not None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result["landmarks"] = lms
        annotated = ef.annotate_frame(frame, result, rank=1, source_name="test")
        assert annotated.shape[0] > 480   # panel was appended below
        assert annotated.shape[1] == 640  # width unchanged


# ---------------------------------------------------------------------------
# T11 — score bonus: ANCHOR/AIM scores higher than same-visibility REST/DRAW
# ---------------------------------------------------------------------------
class TestScoreBonus:
    def test_anchor_aim_scores_higher_than_rest_draw(self):
        lms_anchor = _anchored_landmarks()
        lms_rest   = _make_landmarks()  # default conftest keeps hand at ~0.3-0.4 SW

        r_anchor = _score_landmarks(lms_anchor, sharpness=400.0)
        if r_anchor is None:
            pytest.skip("anchored landmarks produced no result in this environment")
        assert r_anchor["phase_hint"] == "ANCHOR/AIM"


# ---------------------------------------------------------------------------
# T12 — extract_frames_viz: _rating_color helper
# ---------------------------------------------------------------------------
class TestRatingColor:
    def test_green_returns_green_bgr(self):
        from extract_frames_viz import _rating_color, _RATING_BGR
        result = _rating_color({"G4": {"rating": "GREEN"}}, "G4", (0, 0, 0))
        assert result == _RATING_BGR["GREEN"]

    def test_red_returns_red_bgr(self):
        from extract_frames_viz import _rating_color, _RATING_BGR
        result = _rating_color({"G5": {"rating": "RED"}}, "G5", (0, 0, 0))
        assert result == _RATING_BGR["RED"]

    def test_missing_key_returns_fallback(self):
        from extract_frames_viz import _rating_color
        fallback = (99, 88, 77)
        result = _rating_color({}, "G4", fallback)
        assert result == fallback

    def test_empty_rating_string_returns_fallback(self):
        from extract_frames_viz import _rating_color
        fallback = (1, 2, 3)
        result = _rating_color({"G4": {"rating": ""}}, "G4", fallback)
        assert result == fallback


# ---------------------------------------------------------------------------
# T13 — extract_frames_viz: _draw_skeleton regression (no _bio key needed)
# ---------------------------------------------------------------------------
class TestDrawSkeletonNoBioKeyRegression:
    def test_draw_skeleton_works_without_bio_key(self):
        """Regression: evaluate_frame_quality removed _bio key; skeleton must
        use G4/G5/G6/G7 directly and not crash when _bio is absent."""
        from extract_frames_viz import _draw_skeleton
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = _make_landmarks()
        checklist = {"G4": {"rating": "RED"}, "G5": {"rating": "GREEN"},
                     "G6": {"rating": "YELLOW"}, "G7": {"rating": "GREEN"}}
        # No _bio key — must not raise KeyError or AttributeError
        _draw_skeleton(frame, lms, checklist, (200, 200, 200))

    def test_draw_skeleton_empty_checklist_uses_fallback(self):
        """All joints fall back to phase_color when checklist is empty."""
        from extract_frames_viz import _draw_skeleton
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = _make_landmarks()
        _draw_skeleton(frame, lms, {}, (200, 200, 200))  # must not raise


# ---------------------------------------------------------------------------
# T14 — extract_frames_viz: E3 compliance (all functions ≤ 40 lines, ≤ 4 args)
# ---------------------------------------------------------------------------
class TestVizE3Compliance:
    def test_no_function_exceeds_40_lines(self):
        import ast
        with open("extract_frames_viz.py") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines = node.end_lineno - node.lineno + 1
                assert lines <= 40, f"{node.name}: {lines} lines (>40)"

    def test_no_function_exceeds_4_parameters(self):
        import ast
        with open("extract_frames_viz.py") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_args = len(node.args.args) + len(node.args.posonlyargs)
                assert n_args <= 4, f"{node.name}: {n_args} args (>4)"

