"""
Vertex -- Frame Extractor Visualization (developer tool)

Annotation rendering for extracted archery frames.
Called only by extract_frames.py -- not part of the runtime pipeline.

All cv2 calls are intentionally here.
S3 rule "ALL cv2 rendering -> hud.py" applies to the runtime pipeline only.
This module is a dev-tool annotation helper and does not participate in the
Streamer -> PoseHub -> BioLab -> HUD -> SessionIO runtime chain.

Public API: annotate_frame(frame, result, rank, source_name) -> np.ndarray
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Mapping: skeleton connection -> gold checklist key (G4-G7)
# Reads ratings directly from the checklist dict -- no _bio private key needed.
# ---------------------------------------------------------------------------
_CONN_KEY = {
    (11, 13): "G4", (13, 15): "G4",                      # bow arm (bow shoulder angle)
    (12, 14): "G5", (14, 16): "G5", (16, 20): "G5",      # draw arm (draw elbow angle)
    (11, 12): "G6",                                       # shoulder line (shoulder tilt)
    (11, 23): "G7", (12, 24): "G7", (23, 24): "G7",      # torso/hips (torso lean)
}
_JOINT_KEY = {
    11: "G4", 13: "G4", 15: "G4",
    12: "G5", 14: "G5", 16: "G5", 20: "G5",
    23: "G7", 24: "G7",
}
_RATING_BGR = {
    "GREEN": (0, 200, 0), "YELLOW": (0, 200, 200), "RED": (0, 50, 220),
    "PASS":  (0, 200, 0), "FAIL":   (0, 50, 220),  "WARN": (50, 150, 255),
}
_PHASE_BGR     = {"ANCHOR/AIM": (0, 220, 0), "REST/DRAW": (0, 220, 220)}
_CHECKLIST_KEYS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]


def _rating_color(checklist: dict, key: str, fallback: tuple) -> tuple:
    """Resolve BGR color from checklist[key]['rating']. Returns fallback if absent."""
    rating = checklist.get(key, {}).get("rating", "")
    return _RATING_BGR.get(rating, fallback) if rating else fallback


def _draw_skeleton(out: np.ndarray, lms: list, checklist: dict, phase_color: tuple) -> None:
    """Draw gold-standard coloured skeleton using G4/G5/G6/G7 checklist ratings."""
    h, w = out.shape[:2]
    pts = {i: (int(lm.x * w), int(lm.y * h))
           for i, lm in enumerate(lms) if lm.visibility > 0.5}
    for (a, b), gkey in _CONN_KEY.items():
        if a in pts and b in pts:
            cv2.line(out, pts[a], pts[b], _rating_color(checklist, gkey, phase_color), 2, cv2.LINE_AA)
    for i, pt in pts.items():
        jc = _rating_color(checklist, _JOINT_KEY.get(i, ""), phase_color)
        cv2.circle(out, pt, 5, jc, -1, cv2.LINE_AA)
        cv2.circle(out, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_banner(out: np.ndarray, result: dict, rank: int, source_name: str) -> None:
    phase_color = _PHASE_BGR.get(result["phase_hint"], (200, 200, 200))
    banner_h = 130 if result.get("quality_flag") == "ANCHORED_DEEP" else 108
    ov = out.copy()
    cv2.rectangle(ov, (8, 8), (520, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.75, out, 0.25, 0, out)
    cv2.putText(out, "VERTEX  TheSighter  Gold Standard Review", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 140, 0), 1, cv2.LINE_AA)
    t_lbl = f"t={result.get('time_sec', '')}s  " if result.get("time_sec") is not None else ""
    cv2.putText(out, f"Rank #{rank}  Score: {result['score']:.3f}  {t_lbl}{source_name}",
                (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Phase: {result['phase_hint']}  anchor={result['anchor_dist']:.3f}SW",
                (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, phase_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"Vis: {result['visibility_mean']:.3f}  Sharp: {result['sharpness']:.0f}",
                (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)
    if result.get("quality_flag") == "ANCHORED_DEEP":
        cv2.putText(out, "ANCHORED_DEEP: verify compound anchor or landmark occlusion",
                    (18, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 80, 255), 1, cv2.LINE_AA)


def _draw_badge(out: np.ndarray, checklist: dict, validated: bool) -> None:
    bx = out.shape[1] - 268
    bov = out.copy()
    cv2.rectangle(bov, (bx, 8), (bx + 260, 52),
                  (0, 130, 0) if validated else (0, 70, 160), -1)
    cv2.addWeighted(bov, 0.88, out, 0.12, 0, out)
    cv2.putText(out, "VALIDATED" if validated else "REVIEW NEEDED",
                (bx + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Gold: {checklist.get('_green_count', 0)} GREEN   "
                     f"{checklist.get('_bio_pass', 0)}/4 bio checks",
                (bx + 10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_checklist_panel(w: int, checklist: dict) -> np.ndarray:
    row_h, hdr_h = 26, 36
    panel = np.full((hdr_h + len(_CHECKLIST_KEYS) * row_h + 6, w, 3), (16, 16, 16), dtype=np.uint8)
    cx = {"id": 8, "label": 44, "value": 250, "target": 370, "rating": 538, "source": 628}
    cv2.putText(panel, "GOLD STANDARD CHECKLIST", (cx["id"], 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 140, 0), 1, cv2.LINE_AA)
    for col, txt in [("label", "CHECK"), ("value", "VALUE"),
                     ("target", "GOLD TARGET"), ("rating", "RESULT"), ("source", "SOURCE")]:
        cv2.putText(panel, txt, (cx[col], 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(panel, (0, hdr_h - 2), (w, hdr_h - 2), (50, 50, 50), 1)
    for ri, gkey in enumerate(_CHECKLIST_KEYS):
        item = checklist.get(gkey)
        if not item:
            continue
        y  = hdr_h + ri * row_h + row_h - 6
        rc = _RATING_BGR.get(item["rating"], (100, 100, 100))
        cv2.rectangle(panel, (cx["id"], y - 16), (cx["id"] + 30, y + 4), rc, -1)
        cv2.putText(panel, gkey,           (cx["id"] + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, item["label"],  (cx["label"],  y),     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, item["value"],  (cx["value"],  y),     cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, item["target"], (cx["target"], y),     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(panel, item["rating"], (cx["rating"], y),     cv2.FONT_HERSHEY_SIMPLEX, 0.46, rc, 1, cv2.LINE_AA)
        cv2.putText(panel, item["source"], (cx["source"], y),     cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    return panel


def annotate_frame(frame: np.ndarray, result: dict, rank: int, source_name: str) -> np.ndarray:
    """Compose annotated frame: gold-coloured skeleton + banner + badge + checklist panel."""
    out = frame.copy()
    _, w     = out.shape[:2]
    checklist   = result.get("checklist", {})
    phase_color = _PHASE_BGR.get(result["phase_hint"], (200, 200, 200))
    _draw_skeleton(out, result["landmarks"], checklist, phase_color)
    _draw_banner(out, result, rank, source_name)
    _draw_badge(out, checklist, result.get("validated", False))
    return np.vstack([out, _draw_checklist_panel(w, checklist)])