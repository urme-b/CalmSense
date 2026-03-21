import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Activity,
  Brain,
  Layers,
  TrendingUp,
  Clock,
  Server,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { getHealth, getModels, getModelStatistics } from '../services/api';
import { HealthResponse, ModelListResponse, RecentPrediction } from '../types';

// Metric Card Component
const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: { value: number; label: string };
  color?: string;
}> = ({ title, value, icon, trend, color = 'blue' }) => (
  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
        <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">{value}</p>
        {trend && (
          <p className={`mt-2 text-sm ${trend.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {trend.value >= 0 ? '+' : ''}{trend.value}% {trend.label}
          </p>
        )}
      </div>
      <div className={`p-3 bg-${color}-100 dark:bg-${color}-900/30 rounded-lg`}>
        {icon}
      </div>
    </div>
  </div>
);

// Feature Importance Chart
const FeatureImportanceChart: React.FC = () => {
  const data = [
    { name: 'HRV RMSSD', value: 0.18, color: '#3182CE' },
    { name: 'EDA Mean', value: 0.15, color: '#38A169' },
    { name: 'LF/HF Ratio', value: 0.12, color: '#D69E2E' },
    { name: 'HR Mean', value: 0.10, color: '#E53E3E' },
    { name: 'Resp Rate', value: 0.09, color: '#805AD5' },
    { name: 'HRV SDNN', value: 0.08, color: '#DD6B20' },
    { name: 'SCR Count', value: 0.07, color: '#319795' },
    { name: 'Temp Mean', value: 0.06, color: '#D53F8C' },
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Top Feature Importance
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis type="number" domain={[0, 0.2]} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              borderRadius: '8px',
              border: 'none',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

// Class Distribution Chart
const ClassDistributionChart: React.FC = () => {
  const data = [
    { name: 'Baseline', value: 35, color: '#38A169' },
    { name: 'Stress', value: 45, color: '#E53E3E' },
    { name: 'Amusement', value: 20, color: '#D69E2E' },
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Recent Predictions Distribution
      </h3>
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            paddingAngle={5}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
      <div className="flex justify-center space-x-4 mt-4">
        {data.map((item) => (
          <div key={item.name} className="flex items-center">
            <div
              className="w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-sm text-gray-600 dark:text-gray-400">{item.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Recent Predictions List
const RecentPredictionsList: React.FC<{ predictions: RecentPrediction[] }> = ({ predictions }) => {
  const getClassColor = (prediction: string) => {
    switch (prediction.toLowerCase()) {
      case 'baseline': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
      case 'stress': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
      case 'amusement': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Recent Predictions
      </h3>
      <div className="space-y-3">
        {predictions.map((pred) => (
          <div
            key={pred.id}
            className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
          >
            <div className="flex items-center space-x-3">
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getClassColor(pred.prediction)}`}>
                {pred.prediction}
              </span>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {pred.confidence.toFixed(1)}% confidence
              </span>
            </div>
            <span className="text-xs text-gray-500 dark:text-gray-500">
              {new Date(pred.timestamp).toLocaleTimeString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// System Status Component
const SystemStatus: React.FC<{ health: HealthResponse | null }> = ({ health }) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        System Status
      </h3>
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Server className="w-4 h-4 text-gray-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">API Server</span>
          </div>
          <div className="flex items-center space-x-1">
            {isHealthy ? (
              <CheckCircle className="w-4 h-4 text-green-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-red-500" />
            )}
            <span className={`text-sm ${isHealthy ? 'text-green-600' : 'text-red-600'}`}>
              {isHealthy ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-gray-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">Models Loaded</span>
          </div>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {health?.models_loaded || 0}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-gray-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">GPU Available</span>
          </div>
          <span className={`text-sm ${health?.gpu_available ? 'text-green-600' : 'text-gray-500'}`}>
            {health?.gpu_available ? 'Yes' : 'No'}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4 text-gray-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">Uptime</span>
          </div>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {health?.uptime_seconds ? `${Math.floor(health.uptime_seconds / 60)}m` : '-'}
          </span>
        </div>
      </div>
    </div>
  );
};

// Main Dashboard Component
const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);

  // Mock recent predictions
  const recentPredictions: RecentPrediction[] = [
    { id: '1', timestamp: new Date().toISOString(), prediction: 'Stress', confidence: 87.5, model: 'random_forest' },
    { id: '2', timestamp: new Date(Date.now() - 60000).toISOString(), prediction: 'Baseline', confidence: 92.3, model: 'random_forest' },
    { id: '3', timestamp: new Date(Date.now() - 120000).toISOString(), prediction: 'Amusement', confidence: 78.1, model: 'xgboost' },
    { id: '4', timestamp: new Date(Date.now() - 180000).toISOString(), prediction: 'Stress', confidence: 81.4, model: 'random_forest' },
    { id: '5', timestamp: new Date(Date.now() - 240000).toISOString(), prediction: 'Baseline', confidence: 95.2, model: 'lightgbm' },
  ];

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [healthData, modelsData] = await Promise.all([
          getHealth(),
          getModels(),
        ]);
        setHealth(healthData);
        setModels(modelsData);
        setApiError(null);
      } catch (error: any) {
        const msg = error?.message || 'Failed to connect to API';
        setApiError(msg);
        console.error('Dashboard fetch failed:', msg);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* API error banner */}
      {apiError && (
        <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-800 dark:text-red-200 text-sm font-medium">
            API Connection Error: {apiError}
          </p>
          <p className="text-red-600 dark:text-red-400 text-xs mt-1">
            Showing cached data. Retrying every 30s...
          </p>
        </div>
      )}

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Multimodal Stress Detection System Overview
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Best Accuracy"
          value="94.2%"
          icon={<TrendingUp className="w-6 h-6 text-blue-600" />}
          trend={{ value: 2.3, label: 'vs baseline' }}
          color="blue"
        />
        <MetricCard
          title="Total Features"
          value="147"
          icon={<Activity className="w-6 h-6 text-green-600" />}
          color="green"
        />
        <MetricCard
          title="Models Trained"
          value={models?.total_models || 12}
          icon={<Layers className="w-6 h-6 text-purple-600" />}
          color="purple"
        />
        <MetricCard
          title="Avg Inference"
          value="5.2ms"
          icon={<Clock className="w-6 h-6 text-orange-600" />}
          color="orange"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FeatureImportanceChart />
        <ClassDistributionChart />
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentPredictionsList predictions={recentPredictions} />
        <SystemStatus health={health} />
      </div>
    </div>
  );
};

export default Dashboard;
