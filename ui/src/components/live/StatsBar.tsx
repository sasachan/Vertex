import { cn } from "../../lib/utils.ts";

interface StatsBarProps {
  shotCount: number;
  avgHold: number;
  bestHold: number;
  vertexScore: number;
}

export function StatsBar({
  shotCount,
  avgHold,
  bestHold,
  vertexScore,
}: StatsBarProps) {
  return (
    <div className="flex items-center justify-between px-5 py-2.5 bg-vertex-surface border-t border-vertex-border">
      <Stat label="Shots" value={shotCount.toString()} />
      <Stat label="Avg Hold" value={`${avgHold.toFixed(1)}s`} />
      <Stat label="Best" value={`${bestHold.toFixed(1)}s`} />
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-vertex-muted uppercase tracking-wider">
          Vertex Score
        </span>
        <span
          className={cn(
            "text-lg font-bold font-mono",
            vertexScore >= 70
              ? "text-vertex-green"
              : vertexScore >= 40
                ? "text-vertex-yellow"
                : vertexScore >= 0
                  ? "text-vertex-red"
                  : "text-vertex-muted",
          )}
        >
          {vertexScore >= 0 ? vertexScore.toFixed(0) : "—"}
        </span>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-vertex-muted uppercase tracking-wider">
        {label}
      </span>
      <span className="text-sm font-mono text-white">{value}</span>
    </div>
  );
}
