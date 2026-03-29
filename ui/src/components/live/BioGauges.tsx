import type { BioMetrics } from "../../types/vertex.ts";
import { ratingColor } from "../../types/vertex.ts";
import { MetricRow } from "../shared/StatusDot.tsx";

interface BioGaugesProps {
  bio: BioMetrics | null;
}

export function BioGauges({ bio }: BioGaugesProps) {
  if (!bio) {
    return (
      <div className="text-xs text-vertex-muted italic py-2">
        Waiting for pose data...
      </div>
    );
  }

  return (
    <div className="space-y-0.5">
      <MetricRow
        label="Bow Shoulder"
        value={bio.bsa.toFixed(0)}
        unit="°"
        color={ratingColor(bio.bsa, 85, 100)}
      />
      <MetricRow
        label="Draw Elbow"
        value={bio.dea.toFixed(0)}
        unit="°"
        color={ratingColor(bio.dea, 130, 155)}
      />
      <MetricRow
        label="Shoulder Tilt"
        value={bio.shoulder_tilt.toFixed(1)}
        unit="°"
        color={bio.shoulder_tilt <= 3.0 ? "green" : bio.shoulder_tilt <= 5.0 ? "yellow" : "red"}
      />
      <MetricRow
        label="Torso Lean"
        value={bio.torso_lean.toFixed(1)}
        unit="°"
        color={bio.torso_lean <= 3.0 ? "green" : bio.torso_lean <= 5.0 ? "yellow" : "red"}
      />
      <MetricRow
        label="DFL Angle"
        value={bio.dfl_angle.toFixed(1)}
        unit="°"
        color={bio.dfl_angle < 5 ? "green" : bio.dfl_angle < 10 ? "yellow" : "red"}
      />
      <MetricRow
        label="Draw Length"
        value={bio.draw_length.toFixed(3)}
        color="gray"
      />
      <MetricRow
        label="Draw Arm"
        value={bio.draw_arm_angle.toFixed(0)}
        unit="°"
        color={bio.draw_arm_angle <= 25 ? "green" : "yellow"}
      />
    </div>
  );
}
