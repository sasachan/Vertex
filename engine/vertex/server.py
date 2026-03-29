"""
Vertex — FastAPI server: MJPEG video feed + WebSocket metrics + REST API.

Start: python -m vertex.server [--input 0] [--port 8000]
   or: uvicorn vertex.server:app --reload
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .models import SESSIONS_DIR, ShotRecord
from .pipeline import PipelineSession, FrameResult
from .session_io import create_session_csv, write_shot_csv, write_session_json


# ---------------------------------------------------------------------------
# Global pipeline state (singleton — one active session)
# ---------------------------------------------------------------------------
_pipeline: PipelineSession | None = None
_pipeline_lock = threading.Lock()
_latest_frame: bytes = b""
_latest_result: FrameResult | None = None
_ws_clients: list = []
_running = False


def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode BGR frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _result_to_json(r: FrameResult) -> str:
    """Serialize a FrameResult to JSON for WebSocket push."""
    payload = {
        "type": "frame",
        "state": r.state,
        "hold": round(r.hold, 2),
        "fps": round(r.fps, 1),
        "shot_count": r.shot_count,
        "avg_hold": round(r.avg_hold, 2),
        "best_hold": round(r.best_hold, 2),
        "vertex_score": round(r.vertex_score, 1),
        "flags": r.flags,
        "tremor_rms": round(r.tremor_rms, 4),
        "release_confidence": r.release_confidence,
        "setup_posture_score": round(r.setup_posture_score, 1),
        "frame_idx": r.frame_idx,
        "total_frames": r.total_frames,
    }
    if r.bio is not None:
        payload["bio"] = r.bio.to_dict()
    if r.shot is not None:
        payload["shot"] = r.shot.to_dict()
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------
def _pipeline_loop(input_arg: str, output_dir: str) -> None:
    """Blocking loop that runs in a background thread."""
    global _pipeline, _latest_frame, _latest_result, _running

    pipeline = PipelineSession(input_arg, draw_overlays=True)
    if not pipeline.start():
        print(f"[server] ERROR: Cannot open input: {input_arg}")
        _running = False
        return

    with _pipeline_lock:
        _pipeline = pipeline

    # Session CSV
    sess_path, csv_wr, csv_fh = create_session_csv(output_dir)

    def on_shot(shot: ShotRecord) -> None:
        write_shot_csv(csv_wr, shot)
        csv_fh.flush()

    pipeline.on_shot(on_shot)
    _running = True

    try:
        while _running:
            result = pipeline.process_frame()
            if result is None:
                if pipeline.is_live:
                    time.sleep(0.01)
                    continue
                break

            _latest_frame = _encode_jpeg(result.frame_bgr)
            _latest_result = result

            # Throttle to ~30 FPS for non-live sources
            if not pipeline.is_live:
                time.sleep(1.0 / pipeline.source_fps)
    finally:
        csv_fh.close()
        pipeline.stop()
        if pipeline.shots:
            avg_vs = float(np.mean([
                s.vertex_score for s in pipeline.shots
                if s.vertex_score >= 0
            ])) if any(s.vertex_score >= 0 for s in pipeline.shots) else 0.0
            summary = {
                "total_shots": len(pipeline.shots),
                "avg_hold": round(float(np.mean([
                    s.hold_seconds for s in pipeline.shots])), 2),
                "avg_vertex_score": round(avg_vs, 1),
            }
            json_path = sess_path.replace(".csv", ".json")
            write_session_json(json_path, pipeline.shots, summary)
        _running = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def _create_app():
    """Factory — creates the FastAPI application with all routes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        global _running
        _running = False

    app = FastAPI(
        title="Vertex Engine",
        version="0.3.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- MJPEG feed ----------------------------------------------------------

    async def _mjpeg_generator():
        boundary = b"--frame\r\n"
        # Wait for pipeline to produce first frame (up to 10s)
        for _ in range(300):
            if _latest_frame:
                break
            await asyncio.sleep(0.033)

        while True:
            frame_bytes = _latest_frame
            if frame_bytes:
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame_bytes)}\r\n".encode()
                    + b"\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            if not _running and not _latest_frame:
                break
            await asyncio.sleep(0.033)  # ~30 fps

    @app.get("/api/feed")
    async def video_feed():
        return StreamingResponse(
            _mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # -- WebSocket live metrics ----------------------------------------------

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket):
        await websocket.accept()
        _ws_clients.append(websocket)
        try:
            prev_idx = 0
            while True:
                result = _latest_result
                if result is not None and result.frame_idx != prev_idx:
                    prev_idx = result.frame_idx
                    await websocket.send_text(_result_to_json(result))
                await asyncio.sleep(0.033)
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)

    # -- REST API ------------------------------------------------------------

    @app.get("/api/sessions")
    async def list_sessions():
        sess_dir = SESSIONS_DIR
        if not os.path.isdir(sess_dir):
            return JSONResponse([])
        files = sorted([
            f for f in os.listdir(sess_dir)
            if f.endswith(".json")
        ], reverse=True)
        sessions = []
        for f in files[:50]:
            path = os.path.join(sess_dir, f)
            try:
                with open(path) as fh:
                    data = json.load(fh)
                sessions.append({
                    "filename": f,
                    "timestamp": data.get("timestamp", ""),
                    "total_shots": data.get("summary", {}).get("total_shots", 0),
                    "avg_vertex_score": data.get("summary", {}).get(
                        "avg_vertex_score", 0),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return JSONResponse(sessions)

    @app.get("/api/sessions/{filename}")
    async def get_session(filename: str):
        # Validate filename to prevent path traversal
        if "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = os.path.join(SESSIONS_DIR, filename)
        if not os.path.isfile(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        with open(path) as fh:
            return JSONResponse(json.load(fh))

    @app.get("/api/status")
    async def status():
        return JSONResponse({
            "running": _running,
            "pipeline": _pipeline is not None,
            "shot_count": _pipeline.sm.shot_count if _pipeline and _pipeline.sm else 0,
        })

    @app.post("/api/start")
    async def start_pipeline(input_source: str = "0", output: str = SESSIONS_DIR):
        global _running
        if _running:
            return JSONResponse({"error": "pipeline already running"}, status_code=409)
        os.makedirs(output, exist_ok=True)
        t = threading.Thread(
            target=_pipeline_loop, args=(input_source, output), daemon=True)
        t.start()
        # Wait briefly for pipeline to initialize
        for _ in range(20):
            if _running:
                break
            time.sleep(0.1)
        return JSONResponse({"status": "started"})

    @app.post("/api/stop")
    async def stop_pipeline():
        global _running
        _running = False
        return JSONResponse({"status": "stopped"})

    @app.post("/api/reset")
    async def reset_session():
        if _pipeline:
            _pipeline.reset()
        return JSONResponse({"status": "reset"})

    return app


# Lazy app instance — created on first access
app = _create_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Start the Vertex web server."""
    import uvicorn

    p = argparse.ArgumentParser(description="Vertex Engine — web server")
    p.add_argument("--input", default="0", help="Input source (camera/video/image)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("-o", "--output", default=SESSIONS_DIR)
    args = p.parse_args()

    # Auto-start pipeline
    os.makedirs(args.output, exist_ok=True)
    t = threading.Thread(
        target=_pipeline_loop,
        args=(args.input, args.output),
        daemon=True,
    )
    t.start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
