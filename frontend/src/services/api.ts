import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  PredictionResponse,
  BatchPredictionResponse,
  ExplanationResponse,
  ClinicalReport,
  ModelInfo,
  ModelListResponse,
  HealthResponse,
  FeatureVector,
  WebSocketMessage,
  StreamingPrediction,
} from '../types';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      console.error('Unauthorized request');
    }
    return Promise.reject(error);
  }
);

export const predictFromFeatures = async (
  features: Record<string, number>,
  modelName?: string,
  returnProbabilities: boolean = true,
  includeExplanation: boolean = false
): Promise<PredictionResponse> => {
  const response: AxiosResponse<PredictionResponse> = await apiClient.post('/predict', {
    features: {
      values: Object.values(features),
      feature_names: Object.keys(features),
    },
    model_name: modelName,
    return_probabilities: returnProbabilities,
    include_explanation: includeExplanation,
  });
  return response.data;
};

export const batchPredict = async (
  samples: Record<string, number>[],
  modelName?: string
): Promise<BatchPredictionResponse> => {
  const response: AxiosResponse<BatchPredictionResponse> = await apiClient.post('/predict/batch', {
    samples: samples.map((s) => ({
      values: Object.values(s),
      feature_names: Object.keys(s),
    })),
    model_name: modelName,
    return_probabilities: true,
  });
  return response.data;
};

export const getPredictionClasses = async (): Promise<{ classes: string[]; num_classes: number }> => {
  const response = await apiClient.get('/predict/classes');
  return response.data;
};

export const getExplanation = async (
  features: Record<string, number>,
  explanationType: 'shap' | 'lime' | 'gradcam' | 'attention' | 'clinical' = 'shap',
  modelName?: string,
  numFeatures: number = 10,
  includeVisualization: boolean = false
): Promise<ExplanationResponse> => {
  const response: AxiosResponse<ExplanationResponse> = await apiClient.post('/explain', {
    features: {
      values: Object.values(features),
      feature_names: Object.keys(features),
    },
    explanation_type: explanationType,
    model_name: modelName,
    num_features: numFeatures,
    include_visualization: includeVisualization,
  });
  return response.data;
};

export const getClinicalInterpretation = async (
  features: Record<string, number>,
  prediction: number,
  probabilities: number[]
): Promise<ClinicalReport> => {
  const response: AxiosResponse<ClinicalReport> = await apiClient.post('/explain/clinical', {
    features: {
      values: Object.values(features),
      feature_names: Object.keys(features),
    },
    prediction,
    probabilities,
    include_recommendations: true,
  });
  return response.data;
};

export const getExplanationTypes = async (): Promise<any[]> => {
  const response = await apiClient.get('/explain/types');
  return response.data.types;
};

export const getModels = async (): Promise<ModelListResponse> => {
  const response: AxiosResponse<ModelListResponse> = await apiClient.get('/models');
  return response.data;
};

export const getModelInfo = async (modelName: string): Promise<ModelInfo> => {
  const response: AxiosResponse<ModelInfo> = await apiClient.get(`/models/${modelName}`);
  return response.data;
};

export const loadModel = async (
  modelName: string,
  forceReload: boolean = false
): Promise<{ success: boolean; message: string; load_time_ms: number }> => {
  const response = await apiClient.post('/models/load', {
    model_name: modelName,
    force_reload: forceReload,
  });
  return response.data;
};

export const unloadModel = async (modelName: string): Promise<{ success: boolean; message: string }> => {
  const response = await apiClient.post(`/models/${modelName}/unload`);
  return response.data;
};

export const setDefaultModel = async (modelName: string): Promise<{ success: boolean; message: string }> => {
  const response = await apiClient.post(`/models/default/${modelName}`);
  return response.data;
};

export const getModelStatistics = async (): Promise<any> => {
  const response = await apiClient.get('/models/statistics');
  return response.data;
};

export const getHealth = async (): Promise<HealthResponse> => {
  const response: AxiosResponse<HealthResponse> = await axios.get(`${API_BASE_URL}/health`);
  return response.data;
};

export const getDetailedHealth = async (): Promise<any> => {
  const response = await axios.get(`${API_BASE_URL}/health/detailed`);
  return response.data;
};

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private messageHandlers: Map<string, ((data: any) => void)[]> = new Map();

  constructor(private endpoint: string = '/ws/predict') {
    this.connect();
  }

  private connect(): void {
    const url = `${WS_BASE_URL}${this.endpoint}`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connected', { timestamp: new Date().toISOString() });
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.emit('disconnected', { timestamp: new Date().toISOString() });
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', { error, timestamp: new Date().toISOString() });
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.emit(message.type, message.data);
        this.emit('message', message);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting reconnect ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
      setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
    }
  }

  on(event: string, handler: (data: any) => void): void {
    if (!this.messageHandlers.has(event)) {
      this.messageHandlers.set(event, []);
    }
    this.messageHandlers.get(event)!.push(handler);
  }

  off(event: string, handler: (data: any) => void): void {
    const handlers = this.messageHandlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any): void {
    const handlers = this.messageHandlers.get(event);
    if (handlers) {
      handlers.forEach((handler) => handler(data));
    }
  }

  send(type: string, data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data, timestamp: new Date().toISOString() }));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  predict(features: number[], sequenceId?: string): void {
    this.send('predict', { features, sequence_id: sequenceId });
  }

  batchPredict(samples: number[][], batchId?: string): void {
    this.send('batch_predict', { samples, batch_id: batchId });
  }

  ping(): void {
    this.send('ping', {});
  }

  close(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export const parseCSV = async (file: File): Promise<Record<string, number>[]> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map((h) => h.trim());

        const data: Record<string, number>[] = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map((v) => parseFloat(v.trim()));
          const row: Record<string, number> = {};
          headers.forEach((header, index) => {
            row[header] = values[index];
          });
          data.push(row);
        }
        resolve(data);
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
};

export const downloadResults = (data: any, filename: string): void => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

// Default export
export default {
  predictFromFeatures,
  batchPredict,
  getPredictionClasses,
  getExplanation,
  getClinicalInterpretation,
  getExplanationTypes,
  getModels,
  getModelInfo,
  loadModel,
  unloadModel,
  setDefaultModel,
  getModelStatistics,
  getHealth,
  getDetailedHealth,
  WebSocketClient,
  parseCSV,
  downloadResults,
};
