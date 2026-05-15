import React, { useState } from 'react';
import Plot from 'react-plotly.js';
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
  FileSearch,
  AlertTriangle,
  CheckCircle,
  Info,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { ClinicalFinding } from '../types';

// Sample SHAP data
const shapData = [
  { feature: 'HRV RMSSD', importance: -0.18, direction: 'negative' },
  { feature: 'EDA Mean', importance: 0.15, direction: 'positive' },
  { feature: 'LF/HF Ratio', importance: 0.12, direction: 'positive' },
  { feature: 'HR Mean', importance: 0.10, direction: 'positive' },
  { feature: 'Resp Rate', importance: 0.08, direction: 'positive' },
  { feature: 'HRV SDNN', importance: -0.07, direction: 'negative' },
  { feature: 'SCR Count', importance: 0.06, direction: 'positive' },
  { feature: 'Temp Mean', importance: -0.05, direction: 'negative' },
  { feature: 'HRV pNN50', importance: -0.04, direction: 'negative' },
  { feature: 'BVP Amplitude', importance: -0.03, direction: 'negative' },
];

// Sample clinical findings
const clinicalFindings: ClinicalFinding[] = [
  {
    feature: 'HR Mean',
    value: 95.0,
    unit: 'bpm',
    deviation: 'high',
    stress_implication: 'Elevated - consistent with stress response',
    confidence: 0.85,
    interpretation: 'Heart rate above normal range (60-100 bpm), indicating sympathetic activation',
  },
  {
    feature: 'HRV RMSSD',
    value: 18.0,
    unit: 'ms',
    deviation: 'low',
    stress_implication: 'Reduced - consistent with stress response',
    confidence: 0.92,
    interpretation: 'RMSSD below normal range (20-75 ms), indicating reduced parasympathetic activity',
  },
  {
    feature: 'LF/HF Ratio',
    value: 3.8,
    unit: 'ratio',
    deviation: 'high',
    stress_implication: 'Elevated - sympathetic dominance',
    confidence: 0.88,
    interpretation: 'LF/HF ratio above normal (0.5-2.0), suggesting sympathovagal imbalance',
  },
  {
    feature: 'EDA Mean',
    value: 8.5,
    unit: 'μS',
    deviation: 'high',
    stress_implication: 'Elevated - increased sympathetic arousal',
    confidence: 0.78,
    interpretation: 'Skin conductance above baseline, consistent with stress-induced sweating',
  },
  {
    feature: 'Resp Rate',
    value: 22.0,
    unit: 'breaths/min',
    deviation: 'high',
    stress_implication: 'Elevated - may indicate stress',
    confidence: 0.72,
    interpretation: 'Respiratory rate slightly elevated (normal: 12-20), common during stress',
  },
];

// Feature Importance Bar Component
const FeatureImportanceBar: React.FC<{
  feature: string;
  importance: number;
  maxImportance: number;
}> = ({ feature, importance, maxImportance }) => {
  const isPositive = importance > 0;
  const width = (Math.abs(importance) / maxImportance) * 100;

  return (
    <div className="flex items-center space-x-3 py-2">
      <span className="w-28 text-sm text-gray-600 dark:text-gray-400 truncate">{feature}</span>
      <div className="flex-1 flex items-center">
        <div className="w-1/2 flex justify-end">
          {!isPositive && (
            <div
              className="h-5 bg-red-500 rounded-l"
              style={{ width: `${width}%` }}
            />
          )}
        </div>
        <div className="w-px h-6 bg-gray-300 dark:bg-gray-600" />
        <div className="w-1/2">
          {isPositive && (
            <div
              className="h-5 bg-green-500 rounded-r"
              style={{ width: `${width}%` }}
            />
          )}
        </div>
      </div>
      <span className="w-16 text-sm text-right font-mono">
        {importance > 0 ? '+' : ''}{importance.toFixed(3)}
      </span>
    </div>
  );
};

// Clinical Finding Card
const ClinicalFindingCard: React.FC<{
  finding: ClinicalFinding;
  expanded: boolean;
  onToggle: () => void;
}> = ({ finding, expanded, onToggle }) => {
  const getDeviationStyle = (deviation: string) => {
    switch (deviation) {
      case 'high':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
      case 'low':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
      case 'critical_high':
      case 'critical_low':
        return 'bg-red-200 text-red-900 dark:bg-red-800/30 dark:text-red-300';
      default:
        return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
    }
  };

  const getDeviationIcon = (deviation: string) => {
    switch (deviation) {
      case 'high':
      case 'low':
        return <AlertTriangle className="w-4 h-4" />;
      case 'critical_high':
      case 'critical_low':
        return <AlertTriangle className="w-4 h-4 text-red-600" />;
      default:
        return <CheckCircle className="w-4 h-4" />;
    }
  };

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700/50 hover:bg-gray-100 dark:hover:bg-gray-700"
      >
        <div className="flex items-center space-x-3">
          <span className={`flex items-center space-x-1 px-2 py-1 rounded text-xs font-medium ${getDeviationStyle(finding.deviation)}`}>
            {getDeviationIcon(finding.deviation)}
            <span className="uppercase">{finding.deviation}</span>
          </span>
          <span className="font-medium text-gray-900 dark:text-white">{finding.feature}</span>
        </div>
        <div className="flex items-center space-x-4">
          <span className="text-sm font-mono text-gray-600 dark:text-gray-400">
            {finding.value} {finding.unit}
          </span>
          {expanded ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </div>
      </button>

      {expanded && (
        <div className="p-4 bg-white dark:bg-gray-800 space-y-3">
          <div className="flex items-start space-x-2">
            <Info className="w-4 h-4 text-blue-500 mt-0.5" />
            <p className="text-sm text-gray-600 dark:text-gray-400">{finding.interpretation}</p>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Stress Implication:</span>
            <span className="text-gray-700 dark:text-gray-300">{finding.stress_implication}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Confidence:</span>
            <div className="flex items-center space-x-2">
              <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full"
                  style={{ width: `${finding.confidence * 100}%` }}
                />
              </div>
              <span className="text-gray-700 dark:text-gray-300">
                {(finding.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Main Component
const ExplainabilityDashboard: React.FC = () => {
  const [selectedMethod, setSelectedMethod] = useState<'shap' | 'lime'>('shap');
  const [expandedFindings, setExpandedFindings] = useState<Set<number>>(new Set([0, 1]));

  const maxImportance = Math.max(...shapData.map((d) => Math.abs(d.importance)));

  const toggleFinding = (index: number) => {
    setExpandedFindings((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  // Waterfall chart data
  const waterfallData = shapData.map((d, i) => ({
    x: [d.importance],
    y: [d.feature],
    type: 'bar',
    orientation: 'h',
    marker: {
      color: d.importance > 0 ? '#38A169' : '#E53E3E',
    },
    hovertemplate: `${d.feature}: ${d.importance > 0 ? '+' : ''}${d.importance.toFixed(3)}<extra></extra>`,
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Explainability</h1>
          <p className="text-gray-500 dark:text-gray-400">
            Understand model predictions with feature importance and clinical insights
          </p>
        </div>

        {/* Method Toggle */}
        <div className="flex items-center bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
          {['shap', 'lime'].map((method) => (
            <button
              key={method}
              onClick={() => setSelectedMethod(method as 'shap' | 'lime')}
              className={`
                px-4 py-2 rounded-md text-sm font-medium transition-colors
                ${selectedMethod === method
                  ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }
              `}
            >
              {method.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Importance */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            {selectedMethod === 'shap' ? 'SHAP' : 'LIME'} Feature Importance
          </h3>

          <div className="mb-4 flex items-center justify-center space-x-6 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500 rounded" />
              <span className="text-gray-600 dark:text-gray-400">Contradicts Stress</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-green-500 rounded" />
              <span className="text-gray-600 dark:text-gray-400">Supports Stress</span>
            </div>
          </div>

          <div className="space-y-1">
            {shapData.map((item) => (
              <FeatureImportanceBar
                key={item.feature}
                feature={item.feature}
                importance={item.importance}
                maxImportance={maxImportance}
              />
            ))}
          </div>
        </div>

        {/* Waterfall Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Feature Contribution Waterfall
          </h3>

          <Plot
            data={[
              {
                x: shapData.map((d) => d.importance),
                y: shapData.map((d) => d.feature),
                type: 'bar',
                orientation: 'h',
                marker: {
                  color: shapData.map((d) => (d.importance > 0 ? '#38A169' : '#E53E3E')),
                },
                hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
              },
            ]}
            layout={{
              height: 350,
              margin: { l: 100, r: 20, t: 20, b: 40 },
              xaxis: {
                title: 'SHAP Value',
                zeroline: true,
                zerolinecolor: '#718096',
              },
              yaxis: {
                autorange: 'reversed',
              },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Clinical Interpretation */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Clinical Interpretation
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Physiologically-grounded analysis of biomarkers
            </p>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1.5 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
            <AlertTriangle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
            <span className="text-sm font-medium text-yellow-800 dark:text-yellow-400">
              Stress Level: Elevated
            </span>
          </div>
        </div>

        {/* Summary */}
        <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <p className="text-gray-700 dark:text-gray-300">
            <strong>Summary:</strong> Multiple physiological indicators suggest elevated stress.
            The reduced HRV RMSSD (18.0 ms) and elevated LF/HF ratio (3.8) indicate sympathetic
            dominance and decreased parasympathetic activity, consistent with acute stress response.
            Elevated skin conductance (8.5 μS) further supports increased sympathetic arousal.
          </p>
        </div>

        {/* Findings */}
        <div className="space-y-3">
          {clinicalFindings.map((finding, index) => (
            <ClinicalFindingCard
              key={finding.feature}
              finding={finding}
              expanded={expandedFindings.has(index)}
              onToggle={() => toggleFinding(index)}
            />
          ))}
        </div>

        {/* Disclaimer */}
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <div className="flex items-start space-x-2">
            <Info className="w-5 h-5 text-blue-500 mt-0.5" />
            <p className="text-sm text-blue-800 dark:text-blue-300">
              <strong>Disclaimer:</strong> This interpretation is for research purposes only
              and should not be used for clinical diagnosis. Always consult a healthcare
              professional for medical advice. Reference ranges based on Task Force (1996) guidelines.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExplainabilityDashboard;
