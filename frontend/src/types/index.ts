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
