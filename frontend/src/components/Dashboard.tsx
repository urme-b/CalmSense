import React from 'react';
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
import { Activity, Brain, Layers, Award } from 'lucide-react';
import results from '../results.json';

const r = results as any;

// Static color classes (dynamic `bg-${color}-100` would be purged by Tailwind)
const CARD_COLORS: Record<string, string> = {
  blue: 'bg-blue-100 dark:bg-blue-900/30',
  green: 'bg-green-100 dark:bg-green-900/30',
  purple: 'bg-purple-100 dark:bg-purple-900/30',
  orange: 'bg-orange-100 dark:bg-orange-900/30',
};

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color?: string;
}> = ({ title, value, icon, color = 'blue' }) => (
  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
        <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">{value}</p>
      </div>
      <div className={`p-3 rounded-lg ${CARD_COLORS[color]}`}>{icon}</div>
    </div>
  </div>
);

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

const FeatureImportanceChart: React.FC = () => {
  const palette = ['#3182CE', '#38A169', '#D69E2E', '#E53E3E', '#805AD5', '#DD6B20', '#319795', '#D53F8C'];
  const data = (r.shap || []).slice(0, 8).map((s: any, i: number) => ({
    name: s.feature,
    value: s.mean_abs_shap,
    color: palette[i % palette.length],
  }));

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Top biomarkers (mean |SHAP|, binary model)
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical" margin={{ left: 110 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} />
          <Tooltip />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry: { color: string }, index: number) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

// The project's signature finding: subject-independent vs within-subject accuracy
const OptimismGapChart: React.FC = () => {
  const b = r.binary || {};
  const loso = b.loso_pooled_accuracy ?? b.loso_accuracy;
  const within = b.within_subject_accuracy;
  const data = [
    { name: 'LOSO\n(subject-independent)', value: loso, color: '#3182CE' },
    { name: 'Within-subject\n(5-fold)', value: within, color: '#E67E22' },
  ];
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">Optimism gap</h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        Within-subject validation inflates accuracy by {b.optimism_gap_pts} points
      </p>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip formatter={(v: number) => pct(v)} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const ModelComparisonList: React.FC = () => {
  const models = [...((r.binary || {}).models || [])].sort(
    (a: any, b: any) => b.accuracy_mean - a.accuracy_mean
  );
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Binary LOSO accuracy by model
      </h3>
      <div className="space-y-3">
        {models.map((m: any) => (
          <div key={m.model} className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400 w-40">{m.model}</span>
            <div className="flex-1 mx-3 bg-gray-100 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full"
                style={{ width: `${m.accuracy_mean * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white w-16 text-right">
              {pct(m.accuracy_mean)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const DatasetSummary: React.FC = () => {
  const b = r.binary || {};
  const m = r.multiclass || {};
  const rows = [
    ['Dataset', 'WESAD (chest, 15 subjects)'],
    ['Windows (binary)', b.n_windows],
    ['Features', b.n_features],
    ['Binary classes', (b.classes || []).join(', ')],
    ['3-class classes', (m.classes || []).join(', ')],
    ['Validation', 'Leave-One-Subject-Out'],
  ];
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Dataset & setup</h3>
      <div className="space-y-3">
        {rows.map(([k, v]) => (
          <div key={k as string} className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">{k}</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const Dashboard: React.FC = () => {
  const b = r.binary || {};
  const m = r.multiclass || {};

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Subject-independent stress detection — real LOSO results
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Binary LOSO accuracy" value={pct(b.loso_accuracy)} icon={<Activity className="w-6 h-6 text-blue-600" />} color="blue" />
        <MetricCard title="3-class LOSO accuracy" value={pct(m.loso_accuracy)} icon={<Brain className="w-6 h-6 text-green-600" />} color="green" />
        <MetricCard title="Features" value={b.n_features} icon={<Layers className="w-6 h-6 text-purple-600" />} color="purple" />
        <MetricCard title="Best model" value={b.best_model} icon={<Award className="w-6 h-6 text-orange-600" />} color="orange" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FeatureImportanceChart />
        <OptimismGapChart />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModelComparisonList />
        <DatasetSummary />
      </div>
    </div>
  );
};

export default Dashboard;
