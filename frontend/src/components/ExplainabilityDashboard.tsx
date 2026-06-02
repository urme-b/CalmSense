import React from 'react';
import Plot from 'react-plotly.js';
import { FileSearch, Info, AlertTriangle } from 'lucide-react';
import results from '../results.json';

const prettify = (f: string) => f.replace(/_/g, ' ');

// Real global feature importance: mean |SHAP| from the trained Random Forest
const shap: { feature: string; mean_abs_shap: number }[] = (results as any).shap;
const importance = shap
  .slice()
  .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
  .slice(0, 12)
  .map((d) => ({ feature: prettify(d.feature), value: d.mean_abs_shap }));
const maxImportance = Math.max(...importance.map((d) => d.value));

const ExplainabilityDashboard: React.FC = () => {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <FileSearch className="w-6 h-6" /> Explainability
        </h1>
        <p className="text-gray-500 dark:text-gray-400">
          Global feature importance (mean |SHAP|) from the trained Random Forest, over the
          held-out predictions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Importance bars */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Top features (mean |SHAP|)
          </h3>
          <div className="space-y-2">
            {importance.map((item) => (
              <div key={item.feature}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300">{item.feature}</span>
                  <span className="font-mono text-gray-500">{item.value.toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-100 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-indigo-500 h-2 rounded-full"
                    style={{ width: `${(item.value / maxImportance) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Ranking chart */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Importance ranking
          </h3>
          <Plot
            data={[
              {
                x: importance.map((d) => d.value),
                y: importance.map((d) => d.feature),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#6366F1' },
                hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
              },
            ]}
            layout={{
              height: 360,
              margin: { l: 150, r: 20, t: 10, b: 40 },
              xaxis: { title: 'mean |SHAP|' },
              yaxis: { autorange: 'reversed' },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Honest interpretation */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          What the model relies on
        </h3>
        <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300">
          <p>
            The strongest single contributor is <strong>{importance[0].feature}</strong>, a motion
            feature. Autonomic biomarkers follow: heart-rate-interval features (HRV MedianNN /
            MeanNN) and electrodermal activity (EDA SCR / SCL), which track the sympathetic arousal
            expected under acute stress (Task Force, 1996).
          </p>
          <div className="flex items-start gap-2 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5 shrink-0" />
            <p className="text-yellow-800 dark:text-yellow-300">
              <strong>Motion confound:</strong> the top feature is accelerometer-based. The stress
              condition (public speaking) involves more movement than seated baseline, so motion may
              partly stand in for the task rather than for physiology. The ablation study quantifies
              how much accuracy survives without motion features.
            </p>
          </div>
          <div className="flex items-start gap-2 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <Info className="w-5 h-5 text-blue-500 mt-0.5 shrink-0" />
            <p className="text-blue-800 dark:text-blue-300">
              <strong>Disclaimer:</strong> research demonstration only, not a medical device.
              Reference ranges follow Task Force (1996) HRV guidelines.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExplainabilityDashboard;
