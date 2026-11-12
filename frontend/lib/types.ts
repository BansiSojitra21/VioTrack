export type ReviewStatus = "pending" | "needs_review" | "reviewed" | "false_positive";
export type JobStatus = "pending" | "processing" | "cancelling" | "cancelled" | "completed" | "failed";

export interface AnalysisConfig {
  conf: number;
  iou: number;
  imgsz?: number;
  track_buffer?: number;
  tracker_type?: "bytetrack" | "botsort";
  debug_mode?: boolean;
  frame_stride: number;
  target_fps?: number | null;
  max_frames: number;
  stop_line: number[][] | null;
  signal_timeline: { start: number; end: number; state: "red" | "green" | "yellow" }[] | null;
  lanes: { id: string; polygon: number[][]; direction_deg: number }[] | null;
  calibration: { meters_per_pixel: number | null; validated: boolean } | null;
  enable_red_light: boolean;
  enable_wrong_lane: boolean;
  enable_overspeed: boolean;
  enable_helmet: boolean;
  save_annotated: boolean;
  model_name: string | null;
}

export interface ModuleReport {
  name: string;
  status: string;
  message: string;
  count: number;
}

export interface JobSummary {
  sample?: boolean;
  vehicles_detected?: number;
  vehicles_visible_now?: number;
  vehicle_detections_total?: number;
  violations_total?: number;
  violations_confirmed?: number;
  processing_seconds?: number;
  frames_processed?: number;
  frame_stride?: number;
  vehicle_counts?: Record<string, number>;
  violation_counts?: Record<string, number>;
  model?: {
    name: string;
    architecture: string;
    weights: string;
    classes: string[];
    conf: number;
    iou: number;
    input_size: number;
    track_buffer?: number;
  };
  modules?: ModuleReport[];
  annotated_video?: boolean;
}

export interface JobRecord {
  id: string;
  video_id: string;
  status: JobStatus;
  progress: number;
  frames_processed: number;
  frames_total: number;
  error: string | null;
  config: AnalysisConfig;
  summary: JobSummary | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  annotated_ready: boolean;
  sample: boolean;
  video?: VideoRecord | null;
}

export interface VideoRecord {
  id: string;
  original_name: string;
  content_type?: string | null;
  size_bytes: number;
  duration_sec: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  storage: string;
  url: string;
  media_path: string;
  created_at: string;
  status: JobStatus;
  latest_job: JobRecord | null;
  sample: boolean;
}

export interface DetectionRecord {
  id: string;
  job_id: string;
  video_id: string;
  frame_number: number;
  frame_index?: number;
  timestamp_sec: number;
  timestamp?: number;
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: number[];
  tracking_id: number | null;
  track_id?: number | null;
  plate_number?: string | null;
  sample: boolean;
}

export interface ViolationEvidence {
  explanation: string;
  rule: string;
  needs_review: boolean;
  bbox?: number[];
  details?: Record<string, unknown>;
  plate_number?: string | null;
  image?: string | null;
  model?: string;
}

export interface ViolationRecord {
  id: string;
  job_id: string;
  video_id: string;
  video_name: string;
  tracking_id: number | null;
  vehicle_category: string;
  violation_type: string;
  timestamp_sec: number;
  frame_number: number;
  confidence: number;
  review_status: ReviewStatus;
  evidence: ViolationEvidence;
  created_at: string;
  sample: boolean;
}

export interface Summary {
  sample: boolean;
  videos_analyzed: number;
  videos_uploaded: number;
  vehicles_detected: number;
  vehicles_visible_now?: number;
  vehicle_detections_total?: number;
  vehicle_counts: Record<string, number>;
  violations_total: number;
  violations_confirmed?: number;
  violation_counts: Record<string, number>;
  red_light: number;
  wrong_lane: number;
  helmet: number;
  overspeed: number;
  average_speed_kmh: number | null;
  average_speed_status: "calibrated" | "unavailable";
  detection_accuracy: number | null;
  detection_accuracy_note: string;
  most_common_violation: string | null;
}

export interface TrafficSeries {
  sample: boolean;
  vehicles_by_minute: { minute: number; vehicles: number }[];
  violations_by_minute: { minute: number; violation_type: string; count: number }[];
  density: { t: number; detections: number }[];
  hourly: { hour: number; violations: number }[];
  daily: { date: string; violations: number }[];
  by_video: { video_id: string; video_name: string; violations: number }[];
}

export interface ModelInfo {
  sample: boolean;
  loaded: boolean;
  weights_on_disk: boolean;
  allow_weight_download?: boolean;
  name: string;
  version: string;
  architecture: string;
  task: string;
  dataset: string;
  dataset_note: string;
  classes: string[];
  num_classes: number;
  input_size: number;
  confidence_threshold: number;
  iou_threshold: number;
  device: string;
  weights: string;
  error: string | null;
  metrics: EvaluationMetrics | null;
  helmet: { configured: boolean; weights?: string | null; message: string };
  available_models: string[];
  custom_weights_help: string[];
}

export interface EvaluationMetrics {
  source: string;
  sample: boolean;
  precision?: number;
  recall?: number;
  f1?: number;
  map50?: number;
  map50_95?: number;
  confusion_matrix?: number[][];
  class_names?: string[];
  train_loss?: number[];
  val_loss?: number[];
}

export interface HistoryItem {
  job_id: string;
  video_id: string;
  video_name: string;
  finished_at: string | null;
  summary: JobSummary | null;
}

export interface HealthInfo {
  status: string;
  model_loaded: boolean;
  weights_on_disk: boolean;
  model_error: string | null;
  max_upload_mb: number;
  max_duration_sec: number;
}

export interface TrackPath {
  id: number;
  class_name: string;
  points: [number, number][];
  sample: boolean;
}

export interface AppPayload {
  source: "sample" | "live";
  notice: string | null;
  summary: Summary;
  traffic: TrafficSeries;
  videos: VideoRecord[];
  violations: ViolationRecord[];
  detections: DetectionRecord[];
  tracks: TrackPath[];
  model: ModelInfo;
  history: HistoryItem[];
  health: HealthInfo | null;
}
