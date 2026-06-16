import React from 'react';
import Plot from 'react-plotly.js';
import { Gauge, Target, AlertTriangle, TrendingDown, Info } from 'lucide-react';
import results from '../results.json';
import { Calibration } from '../types';

const fmt = (v: number) => v.toFixed(3);
const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;

const TRANSPARENT = 'rgba(0,0,0,0)';
const AXIS_FONT = '#6B7280'; // gray-500: legible on both light and dark cards
const COLORS = {
  within: '#E67E22',
  loso: '#3182CE',
  recal: '#38A169',
  diagonal: '#9CA3AF',
};

const CalibrationPanel: React.FC = () => {
  const cal = (results as any).calibration as Calibration | undefined;
  if (!cal) return null;

  const { loso, within_subject, recalibrated_isotonic, recalibrated_sigmoid, decision_curve } = cal;
  const dc = decision_curve;

  const rows = [
    { key: 'Within-subject', s: within_subject },
    { key: 'LOSO', s: loso },
    { key: 'LOSO + isotonic', s: recalibrated_isotonic },
    { key: 'LOSO + sigmoid', s: recalibrated_sigmoid },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Gauge className="w-6 h-6" /> Calibration
        </h1>
        <p className="text-gray-500 dark:text-gray-400">
          Are the confidence scores trustworthy? Expected calibration error (ECE) under
          leave-one-subject-out, the optimism it hides, and a leak-free recalibration fix
          ({cal.n_windows} windows, {cal.n_bins} bins).
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card icon={<Target className="w-6 h-6 text-blue-600" />} label="LOSO ECE" value={fmt(loso.ece)} />
        <Card
          icon={<AlertTriangle className="w-6 h-6 text-orange-500" />}
          label="Calibration optimism (ECE)"
          value={signed(cal.calibration_optimism_gap_ece)}
        />
        <Card
          icon={<TrendingDown className="w-6 h-6 text-green-600" />}
          label="Recalibration cuts ECE by"
          value={signed(cal.recalibration_reduction_ece)}
        />
      </div>

      {/* Optimism note */}
      <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl p-4 text-sm text-orange-800 dark:text-orange-200">
        Within-subject validation reports an ECE of <strong>{fmt(within_subject.ece)}</strong>, but on
        unseen subjects the same model is off by <strong>{fmt(loso.ece)}</strong> — a calibration
        optimism gap of <strong>{signed(cal.calibration_optimism_gap_ece)}</strong>. A leak-free
        isotonic recalibration brings LOSO ECE down to <strong>{fmt(recalibrated_isotonic.ece)}</strong>.
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reliability diagram */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Reliability diagram</h3>
          <Plot
            data={[
              {
                x: [0, 1],
                y: [0, 1],
                type: 'scatter',
                mode: 'lines',
                name: 'Perfectly calibrated',
                line: { color: COLORS.diagonal, dash: 'dot' },
              },
              {
                x: within_subject.reliability.map((r) => r.confidence),
                y: within_subject.reliability.map((r) => r.accuracy),
                type: 'scatter',
                mode: 'lines+markers',
                name: `Within-subject (ECE ${fmt(within_subject.ece)})`,
                line: { color: COLORS.within },
                marker: { color: COLORS.within },
              },
              {
                x: loso.reliability.map((r) => r.confidence),
                y: loso.reliability.map((r) => r.accuracy),
                type: 'scatter',
                mode: 'lines+markers',
                name: `LOSO (ECE ${fmt(loso.ece)})`,
                line: { color: COLORS.loso },
                marker: { color: COLORS.loso },
              },
              {
                x: recalibrated_isotonic.reliability.map((r) => r.confidence),
                y: recalibrated_isotonic.reliability.map((r) => r.accuracy),
                type: 'scatter',
                mode: 'lines+markers',
                name: `LOSO recalibrated (ECE ${fmt(recalibrated_isotonic.ece)})`,
                line: { color: COLORS.recal },
                marker: { color: COLORS.recal },
              },
            ]}
            layout={{
              height: 380,
              margin: { l: 50, r: 20, t: 10, b: 50 },
              xaxis: { title: 'Confidence', range: [0, 1] },
              yaxis: { title: 'Accuracy', range: [0, 1] },
              legend: { x: 0.02, y: 0.98, bgcolor: TRANSPARENT, font: { size: 10 } },
              paper_bgcolor: TRANSPARENT,
              plot_bgcolor: TRANSPARENT,
              font: { color: AXIS_FONT },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>

        {/* ECE bar chart */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Expected calibration error
          </h3>
          <Plot
            data={[
              {
                x: ['Within-subject', 'LOSO', 'LOSO recalibrated'],
                y: [within_subject.ece, loso.ece, recalibrated_isotonic.ece],
                type: 'bar',
                marker: { color: [COLORS.within, COLORS.loso, COLORS.recal] },
                text: [within_subject.ece, loso.ece, recalibrated_isotonic.ece].map(fmt),
                textposition: 'outside',
                hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
              },
            ]}
            layout={{
              height: 380,
              margin: { l: 50, r: 20, t: 20, b: 50 },
              yaxis: { title: 'ECE', rangemode: 'tozero' },
              paper_bgcolor: TRANSPARENT,
              plot_bgcolor: TRANSPARENT,
              font: { color: AXIS_FONT },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Decision-curve analysis */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
          Decision-curve analysis
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Clinical net benefit across alert thresholds, versus alerting everyone or no one.
        </p>
        <Plot
          data={[
            {
              x: dc.thresholds,
              y: dc.net_benefit_uncalibrated,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Uncalibrated',
              line: { color: COLORS.loso },
              marker: { color: COLORS.loso },
            },
            {
              x: dc.thresholds,
              y: dc.net_benefit_recalibrated,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Recalibrated',
              line: { color: COLORS.recal },
              marker: { color: COLORS.recal },
            },
            {
              x: dc.thresholds,
              y: dc.treat_all,
              type: 'scatter',
              mode: 'lines',
              name: 'Alert everyone',
              line: { color: COLORS.diagonal, dash: 'dash' },
            },
            {
              x: dc.thresholds,
              y: dc.thresholds.map(() => 0),
              type: 'scatter',
              mode: 'lines',
              name: 'Alert no one',
              line: { color: '#6B7280', dash: 'dot' },
            },
          ]}
          layout={{
            height: 360,
            margin: { l: 60, r: 20, t: 10, b: 50 },
            xaxis: { title: 'Alert threshold' },
            yaxis: { title: 'Net benefit' },
            legend: { orientation: 'h', y: -0.2, font: { size: 11 } },
            paper_bgcolor: TRANSPARENT,
            plot_bgcolor: TRANSPARENT,
            font: { color: AXIS_FONT },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      {/* Metrics table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Calibration metrics by evaluation
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-300">
                <th className="px-4 py-2">Evaluation</th>
                <th className="px-4 py-2">ECE</th>
                <th className="px-4 py-2">MCE</th>
                <th className="px-4 py-2">Brier</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key} className="border-b border-gray-100 dark:border-gray-700">
                  <td className="px-4 py-2 font-medium text-gray-900 dark:text-white">{row.key}</td>
                  <td className="px-4 py-2">{fmt(row.s.ece)}</td>
                  <td className="px-4 py-2">{fmt(row.s.mce)}</td>
                  <td className="px-4 py-2">{fmt(row.s.brier)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-sm">
        <Info className="w-5 h-5 text-blue-500 mt-0.5 shrink-0" />
        <p className="text-blue-800 dark:text-blue-300">
          Recalibration uses a calibrator fit only on out-of-fold training probabilities, so the
          held-out subject is never seen — no leakage. Research demonstration only, not a medical
          device.
        </p>
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

export default CalibrationPanel;
