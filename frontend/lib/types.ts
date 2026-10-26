// Mirrors backend/app/schemas/schemas.py. Keep in sync — these are the contract.

export interface Patient {
  id: number;
  name: string;
  mrn: string;
  created_at: string;
}

export interface VeinScan {
  id: number;
  patient_id: number;
  image_url: string;
  mask_url: string | null;
  /** Null when the selector refused. Never render a marker for a null point. */
  insertion_x: number | null;
  insertion_y: number | null;
  angle_deg: number | null;
  vein_width_px: number | null;
  confidence: number;
  reason: string | null;
  model_version: string;
  created_at: string;
  overlay_png_base64: string | null;
  vein_grade: string | null;
}

export interface SiteCheckin {
  id: number;
  patient_id: number;
  image_url: string;
  roi_bbox: number[] | null;
  risk_score: number;
  cannula_confidence: number | null;
  redness_level: number | null;
  redness_label: string | null;
  reddened_fraction: number | null;
  spread_hub_widths: number | null;
  skin_visible_fraction: number | null;
  baseline_reddened_fraction: number | null;
  change_since_baseline: number | null;
  overlay_png_base64: string | null;
  vip_score: number | null;
  risk_velocity: string | null;
  clinical_explanation: string[] | null;
  delta_from_previous: number;
  color_distance: number | null;
  alert_triggered: boolean;
  alert_reason: string | null;
  trend: "improving" | "stable" | "worsening" | null;
  registration_degraded: boolean;
  model_version: string;
  created_at: string;
}

export interface HistoryPoint {
  timestamp: string;
  risk_score: number;
  delta_from_previous: number;
  color_distance: number | null;
  alert_triggered: boolean;
  registration_degraded: boolean;
}

export interface PatientHistory {
  patient_id: number;
  n_checkins: number;
  points: HistoryPoint[];
  latest_trend: string | null;
}

export interface AlertMessage {
  patient_id: number;
  patient_name: string;
  risk_score: number;
  reason: string;
  trend: string;
  created_at: string;
}
