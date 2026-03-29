import { cn } from "../../lib/utils.ts";

interface StatePanelProps {
  state: string;
  hold: number;
}

const STATE_COLORS: Record<string, string> = {
  IDLE: "bg-vertex-muted",
  SETUP: "bg-blue-500",
  DRAW: "bg-vertex-yellow",
  ANCHOR: "bg-vertex-green",
  AIM: "bg-cyan-400",
  RELEASE: "bg-vertex-red",
  FOLLOW_THROUGH: "bg-purple-400",
};

function holdColor(hold: number): string {
  if (hold >= 2.0) return "bg-vertex-green";
  if (hold >= 1.0) return "bg-vertex-yellow";
  return "bg-vertex-red";
}

export function StatePanel({ state, hold }: StatePanelProps) {
  const dotColor = STATE_COLORS[state] ?? "bg-vertex-muted";
  const showTimer = state === "ANCHOR" || state === "AIM";
  const barWidth = showTimer ? Math.min((hold / 5.0) * 100, 100) : 0;

  return (
    <div className="space-y-3">
      {/* State indicator */}
      <div className="flex items-center gap-3">
        <span className={cn("w-3.5 h-3.5 rounded-full", dotColor)} />
        <span className="text-xl font-bold text-white tracking-wide">
          {state}
        </span>
      </div>

      {/* Hold timer */}
      {showTimer && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-vertex-muted">Hold</span>
            <span className="font-mono text-white">{hold.toFixed(1)}s</span>
          </div>
          <div className="h-2 bg-vertex-bg rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all duration-100", holdColor(hold))}
              style={{ width: `${barWidth}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
