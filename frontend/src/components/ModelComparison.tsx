import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { Trophy, Clock, Layers, ArrowUpDown } from 'lucide-react';
import { ModelMetrics } from '../types';

// Sample model metrics
const modelMetrics: ModelMetrics[] = [
  {
    name: 'Random Forest',
    accuracy: 0.942,
    precision: 0.935,
    recall: 0.948,
    f1_score: 0.941,
    mcc: 0.912,
    auc_roc: 0.978,
    training_time_s: 12.5,
    inference_time_ms: 4.2,
  },
  {
    name: 'XGBoost',
    accuracy: 0.938,
    precision: 0.931,
    recall: 0.944,
    f1_score: 0.937,
    mcc: 0.906,
    auc_roc: 0.975,
    training_time_s: 8.3,
    inference_time_ms: 2.8,
  },
  {
    name: 'LightGBM',
    accuracy: 0.935,
    precision: 0.928,
    recall: 0.940,
    f1_score: 0.934,
    mcc: 0.901,
    auc_roc: 0.972,
    training_time_s: 5.2,
    inference_time_ms: 1.9,
  },
  {
    name: 'CatBoost',
    accuracy: 0.931,
    precision: 0.925,
    recall: 0.936,
    f1_score: 0.930,
    mcc: 0.895,
    auc_roc: 0.969,
    training_time_s: 15.8,
    inference_time_ms: 3.5,
  },
  {
    name: 'SVM (RBF)',
    accuracy: 0.912,
    precision: 0.905,
    recall: 0.918,
    f1_score: 0.911,
    mcc: 0.867,
    auc_roc: 0.958,
    training_time_s: 45.2,
    inference_time_ms: 8.7,
  },
  {
    name: 'CNN-LSTM',
    accuracy: 0.925,
    precision: 0.918,
    recall: 0.931,
    f1_score: 0.924,
    mcc: 0.886,
    auc_roc: 0.965,
    training_time_s: 180.5,
    inference_time_ms: 15.3,
  },
  {
    name: 'Transformer',
    accuracy: 0.928,
    precision: 0.921,
    recall: 0.934,
    f1_score: 0.927,
    mcc: 0.891,
    auc_roc: 0.968,
    training_time_s: 245.8,
    inference_time_ms: 22.1,
  },
];

// Confusion matrix data
const confusionMatrices: Record<string, number[][]> = {
  'Random Forest': [
    [95, 3, 2],
    [2, 94, 4],
    [3, 3, 94],
  ],
  'XGBoost': [
    [94, 4, 2],
    [3, 93, 4],
    [2, 4, 94],
  ],
  'LightGBM': [
    [93, 4, 3],
    [3, 92, 5],
    [3, 4, 93],
  ],
};

// Per-subject performance
const subjectPerformance = [
  { subject: 'S2', RF: 0.95, XGB: 0.94, LGBM: 0.93 },
  { subject: 'S3', RF: 0.92, XGB: 0.91, LGBM: 0.90 },
  { subject: 'S4', RF: 0.94, XGB: 0.93, LGBM: 0.92 },
  { subject: 'S5', RF: 0.96, XGB: 0.95, LGBM: 0.94 },
  { subject: 'S6', RF: 0.93, XGB: 0.92, LGBM: 0.91 },
  { subject: 'S7', RF: 0.91, XGB: 0.90, LGBM: 0.89 },
  { subject: 'S8', RF: 0.95, XGB: 0.94, LGBM: 0.93 },
  { subject: 'S9', RF: 0.94, XGB: 0.93, LGBM: 0.92 },
  { subject: 'S10', RF: 0.96, XGB: 0.95, LGBM: 0.94 },
  { subject: 'S11', RF: 0.93, XGB: 0.92, LGBM: 0.91 },
];

// Metrics table component
const MetricsTable: React.FC<{
  metrics: ModelMetrics[];
  sortKey: keyof ModelMetrics;
  onSort: (key: keyof ModelMetrics) => void;
}> = ({ metrics, sortKey, onSort }) => {
  const formatValue = (value: number | undefined, isTime: boolean = false) => {
    if (value === undefined) return '-';
    if (isTime) return value.toFixed(1);
    return (value * 100).toFixed(1) + '%';
  };

  const columns: { key: keyof ModelMetrics; label: string; isTime?: boolean }[] = [
    { key: 'name', label: 'Model' },
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'f1_score', label: 'F1 Score' },
    { key: 'mcc', label: 'MCC' },
    { key: 'auc_roc', label: 'AUC-ROC' },
    { key: 'inference_time_ms', label: 'Inference (ms)', isTime: true },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => onSort(col.key)}
                className="px-4 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <div className="flex items-center space-x-1">
                  <span>{col.label}</span>
                  {sortKey === col.key && <ArrowUpDown className="w-4 h-4" />}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((model, index) => (
            <tr
              key={model.name}
              className={`
                border-b border-gray-100 dark:border-gray-700
                ${index === 0 ? 'bg-green-50 dark:bg-green-900/20' : ''}
              `}
            >
              <td className="px-4 py-3">
                <div className="flex items-center space-x-2">
                  {index === 0 && <Trophy className="w-4 h-4 text-yellow-500" />}
                  <span className="font-medium text-gray-900 dark:text-white">{model.name}</span>
                </div>
              </td>
              {columns.slice(1).map((col) => (
                <td key={col.key} className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                  {formatValue(model[col.key] as number, col.isTime)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Main Component
const ModelComparison: React.FC = () => {
  const [sortKey, setSortKey] = useState<keyof ModelMetrics>('accuracy');
  const [selectedModel, setSelectedModel] = useState('Random Forest');

  const sortedMetrics = [...modelMetrics].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortKey === 'inference_time_ms' || sortKey === 'training_time_s'
        ? aVal - bVal
        : bVal - aVal;
    }
    return 0;
  });

  // Radar chart data
  const radarData = modelMetrics.slice(0, 4).map((model) => ({
    model: model.name,
    Accuracy: model.accuracy * 100,
    Precision: model.precision * 100,
    Recall: model.recall * 100,
    F1: model.f1_score * 100,
    MCC: model.mcc * 100,
    AUC: model.auc_roc * 100,
  }));

  // Bar chart data
  const barChartData = modelMetrics.map((m) => ({
    name: m.name.length > 10 ? m.name.substring(0, 10) + '...' : m.name,
    Accuracy: m.accuracy * 100,
    F1: m.f1_score * 100,
  }));

  // Selected confusion matrix
  const confMatrix = confusionMatrices[selectedModel] || confusionMatrices['Random Forest'];
  const classLabels = ['Baseline', 'Stress', 'Amusement'];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Comparison</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Compare performance metrics across ML and DL models
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <Trophy className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Best Model</p>
              <p className="text-lg font-bold text-gray-900 dark:text-white">Random Forest</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <Layers className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Best Accuracy</p>
              <p className="text-lg font-bold text-gray-900 dark:text-white">94.2%</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <Clock className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Fastest Inference</p>
              <p className="text-lg font-bold text-gray-900 dark:text-white">1.9ms (LightGBM)</p>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Performance Metrics (LOSO Cross-Validation)
        </h3>
        <MetricsTable metrics={sortedMetrics} sortKey={sortKey} onSort={setSortKey} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar Chart Comparison */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Accuracy & F1 Comparison
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barChartData} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis
                dataKey="name"
                angle={-45}
                textAnchor="end"
                height={80}
                tick={{ fontSize: 11 }}
              />
              <YAxis domain={[85, 100]} />
              <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="Accuracy" fill="#3182CE" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill="#38A169" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Confusion Matrix */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Confusion Matrix
            </h3>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-3 py-1.5 bg-gray-100 dark:bg-gray-700 border-0 rounded-lg text-sm"
            >
              {Object.keys(confusionMatrices).map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>

          <Plot
            data={[
              {
                z: confMatrix,
                x: classLabels,
                y: classLabels,
                type: 'heatmap',
                colorscale: 'Blues',
                showscale: true,
                hovertemplate: 'True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>',
              },
            ]}
            layout={{
              height: 280,
              margin: { l: 80, r: 40, t: 20, b: 60 },
              xaxis: { title: 'Predicted', side: 'bottom' },
              yaxis: { title: 'True', autorange: 'reversed' },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              annotations: confMatrix.flatMap((row, i) =>
                row.map((val, j) => ({
                  x: classLabels[j],
                  y: classLabels[i],
                  text: val.toString(),
                  font: { color: val > 50 ? 'white' : 'black' },
                  showarrow: false,
                }))
              ),
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Per-Subject Performance */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Per-Subject Performance (LOSO)
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={subjectPerformance} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="subject" />
            <YAxis domain={[0.85, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
            <Legend />
            <Bar dataKey="RF" name="Random Forest" fill="#3182CE" radius={[4, 4, 0, 0]} />
            <Bar dataKey="XGB" name="XGBoost" fill="#38A169" radius={[4, 4, 0, 0]} />
            <Bar dataKey="LGBM" name="LightGBM" fill="#D69E2E" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ModelComparison;
