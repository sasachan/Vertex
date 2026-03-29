/** Vertex engine types — mirrors Python dataclasses. */

export interface BioMetrics {
  anchor_dist: number;
  hand_xy: [number, number];
  jaw_xy: [number, number];
  bsa: number;
  dea: number;
  shoulder_tilt: number;
  torso_lean: number;
  draw_length: number;
  dfl_angle: number;
  hip_mid: [number, number];
  draw_arm_angle: number;
}

export interface ShotRecord {
  shot_number: number;
  hold_seconds: number;
  anchor_distance_mean: number;
  anchor_distance_var: number;
  release_jump_mag: number;
  bow_shoulder_angle: number;
  draw_elbow_angle: number;
  shoulder_tilt_deg: number;
  torso_lean_deg: number;
  draw_length_norm: number;
  dfl_angle: number;
  sway_range_x: number;
  sway_range_y: number;
  sway_velocity: number;
  is_snap_shot: boolean;
  is_overtime: boolean;
  is_valid: boolean;
  vertex_score: number;
  flags: string;
  tremor_rms_wrist: number;
  release_confidence: string;
}

export interface FrameEvent {
  type: "frame";
  state: string;
  hold: number;
  fps: number;
  shot_count: number;
  avg_hold: number;
  best_hold: number;
  vertex_score: number;
  flags: string;
  tremor_rms: number;
  release_confidence: string;
  setup_posture_score: number;
  frame_idx: number;
  total_frames: number;
  bio?: BioMetrics;
  shot?: ShotRecord;
}

export interface SessionSummary {
  filename: string;
  timestamp: string;
  total_shots: number;
  avg_vertex_score: number;
}

export type ConnectionStatus = "connected" | "connecting" | "disconnected";

/** Coaching color — maps to Tailwind classes. */
export type CoachColor = "green" | "yellow" | "red" | "gray";

export function ratingColor(
  value: number,
  low: number,
  high: number,
): CoachColor {
  if (value >= low && value <= high) return "green";
  const margin = (high - low) * 0.15;
  if (value >= low - margin && value <= high + margin) return "yellow";
  return "red";
}
