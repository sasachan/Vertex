import { cn } from "../../lib/utils.ts";
import type { CoachColor } from "../../types/vertex.ts";

/** Colored dot indicator. */
export function StatusDot({ color }: { color: CoachColor }) {
  return (
    <span
      className={cn(
        "inline-block w-2.5 h-2.5 rounded-full",
        color === "green" && "bg-vertex-green",
        color === "yellow" && "bg-vertex-yellow",
        color === "red" && "bg-vertex-red",
        color === "gray" && "bg-vertex-muted",
      )}
    />
  );
}

/** Metric row with label, value, and rating dot. */
export function MetricRow({
  label,
  value,
  unit = "",
  color = "gray",
}: {
  label: string;
  value: string | number;
  unit?: string;
  color?: CoachColor;
}) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-xs text-vertex-muted">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-white">
          {value}
          {unit && <span className="text-vertex-muted ml-0.5">{unit}</span>}
        </span>
        <StatusDot color={color} />
      </div>
    </div>
  );
}

/** Card wrapper. */
export function Card({
  title,
  children,
  className,
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "bg-vertex-surface border border-vertex-border rounded-xl",
        className,
      )}
    >
      {title && (
        <div className="px-4 py-2.5 border-b border-vertex-border">
          <h3 className="text-xs font-medium text-vertex-muted uppercase tracking-wider">
            {title}
          </h3>
        </div>
      )}
      <div className="px-4 py-3">{children}</div>
    </div>
  );
}
