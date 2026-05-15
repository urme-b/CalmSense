import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  Upload,
  FileText,
  AlertCircle,
  CheckCircle,
  Loader,
  History,
  Download,
  Trash2,
} from 'lucide-react';
import { predictFromFeatures, parseCSV } from '../services/api';
import { PredictionResponse } from '../types';

// Confidence Gauge Component
const ConfidenceGauge: React.FC<{ value: number; label: string; color: string }> = ({
  value,
  label,
  color,
}) => {
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="64"
            cy="64"
            r="45"
            stroke="currentColor"
            strokeWidth="10"
            fill="transparent"
            className="text-gray-200 dark:text-gray-700"
          />
          <circle
            cx="64"
            cy="64"
            r="45"
            stroke={color}
            strokeWidth="10"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="gauge-fill"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold text-gray-900 dark:text-white">
            {value.toFixed(1)}%
          </span>
        </div>
      </div>
      <span className="mt-2 text-sm font-medium text-gray-600 dark:text-gray-400">{label}</span>
    </div>
  );
};

// Prediction Result Card
const PredictionResultCard: React.FC<{ prediction: PredictionResponse }> = ({ prediction }) => {
  const classColors: Record<string, { bg: string; text: string; accent: string }> = {
    'Baseline': { bg: 'bg-green-100 dark:bg-green-900/30', text: 'text-green-800 dark:text-green-400', accent: '#38A169' },
    'Stress': { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-800 dark:text-red-400', accent: '#E53E3E' },
    'Amusement': { bg: 'bg-yellow-100 dark:bg-yellow-900/30', text: 'text-yellow-800 dark:text-yellow-400', accent: '#D69E2E' },
  };

  const colors = classColors[prediction.class_name] || classColors['Baseline'];

  const probabilityData = Object.entries(prediction.probabilities || {}).map(([name, value]) => ({
    name,
    value: value * 100,
    color: classColors[name]?.accent || '#718096',
  }));

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex flex-col lg:flex-row items-center gap-6">
        {/* Confidence Gauge */}
        <ConfidenceGauge
          value={prediction.confidence * 100}
          label="Confidence"
          color={colors.accent}
        />

        {/* Prediction Label */}
        <div className="flex-1 text-center lg:text-left">
          <p className="text-sm text-gray-500 dark:text-gray-400">Predicted State</p>
          <div className={`inline-block mt-2 px-6 py-3 rounded-lg ${colors.bg}`}>
            <span className={`text-2xl font-bold ${colors.text}`}>
              {prediction.class_name}
            </span>
          </div>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Model: {prediction.model_used} | Inference: {prediction.inference_time_ms.toFixed(2)}ms
          </p>
        </div>

        {/* Probability Distribution */}
        <div className="w-full lg:w-64">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Class Probabilities
          </p>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={probabilityData} layout="vertical">
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={80} />
              <Tooltip
                formatter={(value: number) => `${value.toFixed(1)}%`}
                contentStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {probabilityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// History Item
const HistoryItem: React.FC<{
  prediction: PredictionResponse;
  onDelete: () => void;
}> = ({ prediction, onDelete }) => {
  const classColors: Record<string, string> = {
    'Baseline': 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    'Stress': 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    'Amusement': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
  };

  return (
    <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
      <div className="flex items-center space-x-3">
        <span className={`px-2 py-1 rounded text-xs font-medium ${classColors[prediction.class_name]}`}>
          {prediction.class_name}
        </span>
        <span className="text-sm text-gray-600 dark:text-gray-400">
          {(prediction.confidence * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex items-center space-x-2">
        <span className="text-xs text-gray-500 dark:text-gray-500">
          {new Date(prediction.timestamp).toLocaleTimeString()}
        </span>
        <button
          onClick={onDelete}
          className="p-1 text-gray-400 hover:text-red-500 transition-colors"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

// Main Prediction Panel
const PredictionPanel: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<PredictionResponse[]>([]);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
  });

  const handlePredict = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setProgress(0);

    try {
      // Parse CSV
      setProgress(20);
      const data = await parseCSV(file);

      if (data.length === 0) {
        throw new Error('No data found in CSV file');
      }

      // Use first row
      setProgress(50);
      const features = data[0];

      // Make prediction
      setProgress(80);
      const result = await predictFromFeatures(features, undefined, true, false);

      setProgress(100);
      setCurrentPrediction(result);
      setHistory((prev) => [result, ...prev].slice(0, 10));
    } catch (err: any) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleManualInput = async () => {
    // Demo with sample features
    setLoading(true);
    setError(null);

    try {
      const sampleFeatures: Record<string, number> = {
        hr_mean: 85.5,
        hrv_sdnn: 42.3,
        hrv_rmssd: 28.7,
        hrv_pnn50: 15.2,
        hrv_lf_power: 1250,
        hrv_hf_power: 450,
        hrv_lf_hf_ratio: 2.78,
        eda_mean: 5.8,
        eda_std: 1.2,
        scr_count: 8,
        resp_rate: 16.5,
        temp_mean: 33.2,
      };

      const result = await predictFromFeatures(sampleFeatures);
      setCurrentPrediction(result);
      setHistory((prev) => [result, ...prev].slice(0, 10));
    } catch (err: any) {
      // Mock data fallback
      const mockResult: PredictionResponse = {
        prediction: 1,
        class_name: 'Stress',
        probabilities: { 'Baseline': 0.15, 'Stress': 0.72, 'Amusement': 0.13 },
        confidence: 0.72,
        model_used: 'random_forest',
        inference_time_ms: 4.5,
        timestamp: new Date().toISOString(),
      };
      setCurrentPrediction(mockResult);
      setHistory((prev) => [mockResult, ...prev].slice(0, 10));
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const downloadHistory = () => {
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prediction_history.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Stress Prediction</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Upload features or enter data manually for stress prediction
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Section */}
        <div className="lg:col-span-2 space-y-6">
          {/* Dropzone */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Upload Features
            </h3>

            <div
              {...getRootProps()}
              className={`
                border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
                transition-colors duration-200
                ${isDragActive
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
                }
              `}
            >
              <input {...getInputProps()} />
              <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              {isDragActive ? (
                <p className="text-blue-600 dark:text-blue-400">Drop the file here...</p>
              ) : (
                <>
                  <p className="text-gray-600 dark:text-gray-400">
                    Drag and drop a CSV file here, or click to select
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
                    Supported: .csv files with feature columns
                  </p>
                </>
              )}
            </div>

            {/* Selected File */}
            {file && (
              <div className="mt-4 flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <FileText className="w-5 h-5 text-blue-500" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">{file.name}</span>
                  <span className="text-xs text-gray-500">
                    ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
                <button
                  onClick={() => setFile(null)}
                  className="text-gray-400 hover:text-red-500"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Progress Bar */}
            {loading && (
              <div className="mt-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Processing...</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">{progress}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="mt-4 flex items-center space-x-2 p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg">
                <AlertCircle className="w-5 h-5" />
                <span className="text-sm">{error}</span>
              </div>
            )}

            {/* Action Buttons */}
            <div className="mt-6 flex flex-wrap gap-3">
              <button
                onClick={handlePredict}
                disabled={!file || loading}
                className={`
                  flex items-center px-6 py-2.5 rounded-lg font-medium
                  transition-colors duration-200
                  ${file && !loading
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                  }
                `}
              >
                {loading ? (
                  <Loader className="w-5 h-5 mr-2 animate-spin" />
                ) : (
                  <CheckCircle className="w-5 h-5 mr-2" />
                )}
                Predict from File
              </button>

              <button
                onClick={handleManualInput}
                disabled={loading}
                className="flex items-center px-6 py-2.5 rounded-lg font-medium
                           bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300
                           hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-200"
              >
                Try Demo Prediction
              </button>
            </div>
          </div>

          {/* Prediction Result */}
          {currentPrediction && <PredictionResultCard prediction={currentPrediction} />}
        </div>

        {/* History Sidebar */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <History className="w-5 h-5 text-gray-500" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">History</h3>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={downloadHistory}
                disabled={history.length === 0}
                className="p-1.5 text-gray-400 hover:text-blue-500 disabled:opacity-50"
                title="Download History"
              >
                <Download className="w-4 h-4" />
              </button>
              <button
                onClick={clearHistory}
                disabled={history.length === 0}
                className="p-1.5 text-gray-400 hover:text-red-500 disabled:opacity-50"
                title="Clear History"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="space-y-2 max-h-96 overflow-y-auto">
            {history.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-4">
                No predictions yet
              </p>
            ) : (
              history.map((pred, index) => (
                <HistoryItem
                  key={index}
                  prediction={pred}
                  onDelete={() => setHistory((prev) => prev.filter((_, i) => i !== index))}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionPanel;
