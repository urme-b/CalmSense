// API Response Types

export interface PredictionResponse {
  prediction: number;
  class_name: string;
  probabilities: Record<string, number>;
  confidence: number;
  model_used: string;
  inference_time_ms: number;
  explanation?: Record<string, any>;
  timestamp: string;
}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[];
  total_samples: number;
  total_inference_time_ms: number;
  model_used: string;
}

export interface FeatureImportance {
  feature_name: string;
  importance: number;
  direction: 'positive' | 'negative';
}

export interface ExplanationResponse {
  prediction: number;
  class_name: string;
  probabilities: Record<string, number>;
  explanation_type: string;
  feature_importances: FeatureImportance[];
  clinical_interpretation?: ClinicalReport;
  visualization_base64?: string;
  model_used: string;
  computation_time_ms: number;
}

export interface ClinicalFinding {
  feature: string;
  value: number;
  unit: string;
  deviation: string;
  stress_implication: string;
  confidence: number;
  interpretation: string;
}

export interface ClinicalReport {
  stress_level: string;
  stress_score: number;
  findings: ClinicalFinding[];
  summary: string;
  recommendations: string[];
  disclaimer: string;
}

export interface ModelInfo {
  name: string;
  model_type: string;
  version: string;
  num_classes: number;
  input_dim?: number;
  num_parameters?: number;
  is_loaded: boolean;
  load_time_ms?: number;
  last_used?: string;
  metadata?: Record<string, any>;
}

export interface ModelListResponse {
  models: ModelInfo[];
  default_model: string;
  total_models: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: number;
  gpu_available: boolean;
  uptime_seconds: number;
  timestamp: string;
}

export interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  mcc: number;
  auc_roc: number;
  training_time_s?: number;
  inference_time_ms?: number;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface SignalData {
  ecg?: number[];
  eda?: number[];
  temp?: number[];
  acc_x?: number[];
  acc_y?: number[];
  acc_z?: number[];
  resp?: number[];
  emg?: number[];
  timestamps?: number[];
  sampling_rate: number;
  condition?: string;
  subject_id?: string;
}

export interface FeatureVector {
  values: number[];
  feature_names?: string[];
  timestamp?: string;
  subject_id?: string;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: string;
  data: Record<string, any>;
  timestamp: string;
}

export interface StreamingPrediction {
  prediction: number;
  class_name: string;
  confidence: number;
  probabilities: Record<string, number>;
  sequence_id?: string;
  timestamp: string;
}

// Chart Data Types
export interface ChartDataPoint {
  name: string;
  value: number;
  color?: string;
}

export interface TimeSeriesDataPoint {
  time: number;
  value: number;
  label?: string;
}

// Dashboard Types
export interface DashboardMetric {
  label: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down' | 'neutral';
  icon?: string;
}

export interface RecentPrediction {
  id: string;
  timestamp: string;
  prediction: string;
  confidence: number;
  model: string;
}

// Subject Data
export interface SubjectInfo {
  id: string;
  name: string;
  conditions: string[];
  total_samples: number;
}
