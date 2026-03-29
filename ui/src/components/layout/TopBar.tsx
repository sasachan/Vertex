import { Wifi, WifiOff, Loader } from "lucide-react";
import type { ConnectionStatus } from "../../types/vertex.ts";
import { cn } from "../../lib/utils.ts";

interface TopBarProps {
  fps: number;
  status: ConnectionStatus;
}

export function TopBar({ fps, status }: TopBarProps) {
  return (
    <header className="flex items-center justify-between h-14 px-5 border-b border-vertex-border bg-vertex-surface">
      <div className="text-sm text-vertex-muted">
        TheSighter — Archery Biomechanics
      </div>
      <div className="flex items-center gap-4">
        {/* FPS */}
        <span
          className={cn(
            "text-xs font-mono",
            fps >= 25 ? "text-vertex-green" : "text-vertex-red",
          )}
        >
          {fps.toFixed(0)} FPS
        </span>

        {/* Connection status */}
        <div className="flex items-center gap-1.5">
          {status === "connected" && (
            <>
              <Wifi size={14} className="text-vertex-green" />
              <span className="text-xs text-vertex-green">Live</span>
            </>
          )}
          {status === "connecting" && (
            <>
              <Loader size={14} className="text-vertex-yellow animate-spin" />
              <span className="text-xs text-vertex-yellow">Connecting</span>
            </>
          )}
          {status === "disconnected" && (
            <>
              <WifiOff size={14} className="text-vertex-muted" />
              <span className="text-xs text-vertex-muted">Offline</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
