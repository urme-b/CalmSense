import React, { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Trophy, AlertTriangle, Activity } from 'lucide-react';
import results from '../results.json';

type Task = 'binary' | 'multiclass';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

const ModelComparison: React.FC = () => {
  const [task, setTask] = useState<Task>('binary');
  const data = (results as any)[task];
  const models = [...data.models].sort(
    (a: any, b: any) => b.accuracy_mean - a.accuracy_mean
  );
  const best = models[0];
  const losoPooled = data.loso_pooled_accuracy ?? data.loso_accuracy;
  const gap = (data.within_subject_accuracy - losoPooled) * 100;
  const shap = (results as any).shap || [];

  const barData = models.map((m: any) => ({
    name: m.model,
    accuracy: +(m.accuracy_mean * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Comparison</h1>
          <p className="text-gray-500 dark:text-gray-400">
            Leave-One-Subject-Out cross-validation on WESAD (15 subjects). No data leakage.
          </p>
        </div>
        <select
          value={task}
          onChange={(e) => setTask(e.target.value as Task)}
          className="px-3 py-1.5 bg-gray-100 dark:bg-gray-700 border-0 rounded-lg text-sm"
        >
          <option value="binary">Binary (stress vs. non-stress)</option>
          <option value="multiclass">3-class (baseline/stress/amusement)</option>
        </select>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card icon={<Trophy className="w-6 h-6 text-green-600" />} label="Best model" value={best.model} />
        <Card
          icon={<Activity className="w-6 h-6 text-blue-600" />}
          label="LOSO accuracy"
          value={pct(best.accuracy_mean)}
        />
        <Card
          icon={<AlertTriangle className="w-6 h-6 text-orange-500" />}
          label="Within-subject optimism"
          value={`+${gap.toFixed(1)} pts`}
        />
      </div>

      {/* Optimism note */}
      <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl p-4 text-sm text-orange-800 dark:text-orange-200">
        The same model scores <strong>{pct(data.within_subject_accuracy)}</strong> under within-subject
        5-fold but only <strong>{pct(losoPooled)}</strong> when tested on unseen subjects — the
        gap that inflates many reported WESAD results.
      </div>

      {/* Generalization */}
      {(results as any).cross_dataset && (results as any).wrist && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 text-sm">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">Wrist-only is enough</p>
            <p className="text-gray-600 dark:text-gray-400">
              With the same model, Empatica E4 wrist signals reach{' '}
              <strong>{pct((results as any).wrist.same_model_rf.wrist)}</strong> vs{' '}
              {pct((results as any).wrist.same_model_rf.chest)} for the chest — a{' '}
              {(results as any).wrist.same_model_rf.drop_pts.toFixed(1)}-pt drop. No chest strap needed.
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 text-sm">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">It doesn't cross datasets</p>
            <p className="text-gray-600 dark:text-gray-400">
              Trained on WESAD, tested on PhysioNet Non-EEG, balanced accuracy falls to{' '}
              <strong>{pct((results as any).cross_dataset.wesad_to_noneeg.balanced_accuracy)}</strong> —
              near chance. Within-dataset success ≠ real-world generalization.
            </p>
          </div>
        </div>
      )}

      {/* Metrics table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Subject-independent performance ({data.n_windows} windows)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-300">
                <th className="px-4 py-2">Model</th>
                <th className="px-4 py-2">Accuracy</th>
                <th className="px-4 py-2">Macro-F1</th>
                <th className="px-4 py-2">Balanced acc.</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m: any, i: number) => (
                <tr
                  key={m.model}
                  className={`border-b border-gray-100 dark:border-gray-700 ${
                    i === 0 ? 'bg-green-50 dark:bg-green-900/20' : ''
                  }`}
                >
                  <td className="px-4 py-2 font-medium text-gray-900 dark:text-white flex items-center gap-2">
                    {i === 0 && <Trophy className="w-4 h-4 text-yellow-500" />}
                    {m.model}
                  </td>
                  <td className="px-4 py-2">{pct(m.accuracy_mean)}</td>
                  <td className="px-4 py-2">{pct(m.f1_macro_mean)}</td>
                  <td className="px-4 py-2">{pct(m.balanced_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Accuracy bar chart */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">LOSO accuracy</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="name" angle={-30} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} unit="%" />
              <Tooltip formatter={(v) => `${v}%`} />
              <Legend />
              <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
                {barData.map((_: any, i: number) => (
                  <Cell key={i} fill={i === 0 ? '#38A169' : '#3182CE'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Top SHAP biomarkers */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Top biomarkers (mean |SHAP|)
          </h3>
          {shap.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={shap} layout="vertical" margin={{ left: 40, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="mean_abs_shap" name="mean |SHAP|" fill="#805AD5" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-gray-500">Run the experiment to populate SHAP values.</p>
          )}
        </div>
      </div>
    </div>
  );
};

const Card: React.FC<{ icon: React.ReactNode; label: string; value: string }> = ({
  icon,
  label,
  value,
}) => (
  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
    <div className="flex items-center space-x-3">
      <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">{icon}</div>
      <div>
        <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
        <p className="text-lg font-bold text-gray-900 dark:text-white">{value}</p>
      </div>
    </div>
  </div>
);

export default ModelComparison;
