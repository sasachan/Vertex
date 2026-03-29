import { useEffect, useRef, useState, useCallback } from "react";
import type { FrameEvent, BioMetrics, ShotRecord, ConnectionStatus } from "../types/vertex.ts";

interface StreamState {
  status: ConnectionStatus;
  state: string;
  hold: number;
  fps: number;
  shotCount: number;
  avgHold: number;
  bestHold: number;
  vertexScore: number;
  flags: string;
  bio: BioMetrics | null;
  shots: ShotRecord[];
  tremorRms: number;
  releaseConfidence: string;
  setupPostureScore: number;
  frameIdx: number;
  totalFrames: number;
}

const INITIAL: StreamState = {
  status: "disconnected",
  state: "IDLE",
  hold: 0,
  fps: 0,
  shotCount: 0,
  avgHold: 0,
  bestHold: 0,
  vertexScore: -1,
  flags: "",
  bio: null,
  shots: [],
  tremorRms: -1,
  releaseConfidence: "",
  setupPostureScore: -1,
  frameIdx: 0,
  totalFrames: 0,
};

/**
 * WebSocket hook — connects to the Vertex engine and streams live metrics.
 * Auto-reconnects on disconnect with exponential backoff.
 */
export function useVertexStream(url = "/ws/live"): StreamState {
  const [stream, setStream] = useState<StreamState>(INITIAL);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(1000);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = url.startsWith("/")
      ? `${protocol}//${window.location.host}${url}`
      : url;

    setStream((s) => ({ ...s, status: "connecting" }));
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      retryRef.current = 1000;
      setStream((s) => ({ ...s, status: "connected" }));
    };

    ws.onmessage = (ev) => {
      const data: FrameEvent = JSON.parse(ev.data as string);
      setStream((prev) => {
        const next: StreamState = {
          ...prev,
          state: data.state,
          hold: data.hold,
          fps: data.fps,
          shotCount: data.shot_count,
          avgHold: data.avg_hold,
          bestHold: data.best_hold,
          vertexScore: data.vertex_score,
          flags: data.flags,
          bio: data.bio ?? prev.bio,
          tremorRms: data.tremor_rms,
          releaseConfidence: data.release_confidence,
          setupPostureScore: data.setup_posture_score,
          frameIdx: data.frame_idx,
          totalFrames: data.total_frames,
        };
        if (data.shot) {
          next.shots = [...prev.shots, data.shot];
        }
        return next;
      });
    };

    ws.onclose = () => {
      setStream((s) => ({ ...s, status: "disconnected" }));
      // Exponential backoff
      const delay = Math.min(retryRef.current, 10000);
      retryRef.current = delay * 1.5;
      setTimeout(connect, delay);
    };

    ws.onerror = () => ws.close();
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return stream;
}
