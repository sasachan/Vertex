import type { ShotRecord } from "../../types/vertex.ts";
import { cn } from "../../lib/utils.ts";

interface ShotTableProps {
  shots: ShotRecord[];
}

export function ShotTable({ shots }: ShotTableProps) {
  if (shots.length === 0) {
    return (
      <div className="text-xs text-vertex-muted italic py-2 text-center">
        No shots recorded yet
      </div>
    );
  }

  return (
    <div className="overflow-y-auto max-h-48">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-vertex-surface">
          <tr className="text-vertex-muted border-b border-vertex-border">
            <th className="text-left py-1.5 px-2 font-medium">#</th>
            <th className="text-right py-1.5 px-2 font-medium">Hold</th>
            <th className="text-right py-1.5 px-2 font-medium">BSA</th>
            <th className="text-right py-1.5 px-2 font-medium">DEA</th>
            <th className="text-right py-1.5 px-2 font-medium">VS</th>
            <th className="text-left py-1.5 px-2 font-medium">Flag</th>
          </tr>
        </thead>
        <tbody>
          {shots.map((s) => (
            <tr
              key={s.shot_number}
              className={cn(
                "border-b border-vertex-border/50 hover:bg-white/3",
                !s.is_valid && "opacity-50",
              )}
            >
              <td className="py-1.5 px-2 text-vertex-muted">
                {s.shot_number}
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-white">
                {s.hold_seconds.toFixed(1)}s
              </td>
              <td className="py-1.5 px-2 text-right font-mono">
                {s.bow_shoulder_angle.toFixed(0)}°
              </td>
              <td className="py-1.5 px-2 text-right font-mono">
                {s.draw_elbow_angle.toFixed(0)}°
              </td>
              <td
                className={cn(
                  "py-1.5 px-2 text-right font-mono font-bold",
                  s.vertex_score >= 70
                    ? "text-vertex-green"
                    : s.vertex_score >= 40
                      ? "text-vertex-yellow"
                      : "text-vertex-red",
                )}
              >
                {s.vertex_score >= 0 ? s.vertex_score.toFixed(0) : "—"}
              </td>
              <td className="py-1.5 px-2">
                {s.is_snap_shot && (
                  <span className="text-vertex-yellow">⚡</span>
                )}
                {s.is_overtime && (
                  <span className="text-vertex-red">⏱</span>
                )}
                {s.is_valid && !s.is_snap_shot && !s.is_overtime && (
                  <span className="text-vertex-green">✓</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
