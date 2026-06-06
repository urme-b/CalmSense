import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';
import realSignals from '../signals.json';

// Real WESAD chest signals (baseline -> stress -> amusement), downsampled for display
const subjects = Object.keys(realSignals);

const PANELS = [
  { key: 'ecg', title: 'ECG (mV)', series: [{ y: 'ecg', name: 'ECG', color: '#E53E3E' }] },
  { key: 'eda', title: 'EDA (μS)', series: [{ y: 'eda', name: 'EDA', color: '#3182CE' }] },
  { key: 'temp', title: 'Temp (°C)', series: [{ y: 'temp', name: 'Temperature', color: '#38A169' }] },
  {
    key: 'acc',
    title: 'ACC (g)',
    series: [
      { y: 'accX', name: 'ACC X', color: '#805AD5' },
      { y: 'accY', name: 'ACC Y', color: '#DD6B20' },
      { y: 'accZ', name: 'ACC Z', color: '#319795' },
    ],
  },
] as const;

const CONDITION_COLORS: Record<string, string> = {
  Baseline: 'rgba(56, 161, 105, 0.10)',
  Stress: 'rgba(229, 62, 62, 0.10)',
  Amusement: 'rgba(214, 158, 46, 0.10)',
};
const LABEL_COLORS: Record<string, string> = {
  Baseline: '#38A169',
  Stress: '#E53E3E',
  Amusement: '#D69E2E',
};

const SignalExplorer: React.FC = () => {
  const [selectedSubject, setSelectedSubject] = useState(subjects[0]);
  const [visibleSignals, setVisibleSignals] = useState({ ecg: true, eda: true, temp: true, acc: false });
  const signalData = useMemo(() => (realSignals as any)[selectedSubject], [selectedSubject]);
  const duration = signalData.time[signalData.time.length - 1];
  const [xRange, setXRange] = useState<[number, number]>([0, Math.round(duration)]);

  // Contiguous condition segments straight from the labelled samples
  const segments = useMemo(() => {
    const conds: string[] = signalData.conditions;
    const time: number[] = signalData.time;
    const segs: { name: string; x0: number; x1: number }[] = [];
    let start = 0;
    for (let i = 1; i <= conds.length; i++) {
      if (i === conds.length || conds[i] !== conds[start]) {
        segs.push({ name: conds[start], x0: time[start], x1: time[Math.min(i, time.length - 1)] });
        start = i;
      }
    }
    return segs;
  }, [signalData]);

  const toggleSignal = (signal: keyof typeof visibleSignals) =>
    setVisibleSignals((prev) => ({ ...prev, [signal]: !prev[signal] }));

  const handleZoomIn = () => {
    const range = xRange[1] - xRange[0];
    const center = (xRange[0] + xRange[1]) / 2;
    setXRange([center - range / 4, center + range / 4]);
  };
  const handleZoomOut = () => {
    const range = xRange[1] - xRange[0];
    const center = (xRange[0] + xRange[1]) / 2;
    const newRange = Math.min(range * 2, duration);
    setXRange([Math.max(0, center - newRange / 2), Math.min(duration, center + newRange / 2)]);
  };
  const handleReset = () => setXRange([0, Math.round(duration)]);

  const visiblePanels = PANELS.filter((p) => (visibleSignals as any)[p.key]);

  const buildTraces = () => {
    const traces: any[] = [];
    visiblePanels.forEach((panel, i) => {
      const axis = i === 0 ? '' : String(i + 1);
      panel.series.forEach((s) =>
        traces.push({
          x: signalData.time,
          y: signalData[s.y],
          type: 'scatter',
          mode: 'lines',
          name: s.name,
          line: { color: s.color, width: 1 },
          xaxis: 'x',
          yaxis: `y${axis}`,
        })
      );
    });
    return traces;
  };

  const layout: any = {
    title: { text: `Signal Explorer — Subject ${selectedSubject}`, font: { size: 18 } },
    showlegend: true,
    legend: { orientation: 'h', y: -0.12 },
    xaxis: { title: 'Time (s)', range: xRange, showgrid: true, gridcolor: 'rgba(0,0,0,0.1)' },
    height: 600,
    margin: { t: 70, b: 70, l: 60, r: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    shapes: segments.map((s) => ({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: s.x0,
      x1: s.x1,
      y0: 0,
      y1: 1,
      fillcolor: CONDITION_COLORS[s.name] || 'rgba(0,0,0,0.04)',
      line: { width: 0 },
    })),
    annotations: segments.map((s) => ({
      x: (s.x0 + s.x1) / 2,
      y: 1.04,
      xref: 'x',
      yref: 'paper',
      text: s.name,
      showarrow: false,
      font: { color: LABEL_COLORS[s.name] || '#666' },
    })),
  };

  // One stacked y-axis per visible signal, evenly split with a small gap
  const n = visiblePanels.length || 1;
  const slice = 1 / n;
  const gap = n > 1 ? 0.06 : 0;
  visiblePanels.forEach((panel, i) => {
    const axisKey = i === 0 ? 'yaxis' : `yaxis${i + 1}`;
    const top = 1 - i * slice;
    const bottom = Math.max(0, 1 - (i + 1) * slice + gap);
    layout[axisKey] = {
      title: panel.title,
      domain: [bottom, top],
      showgrid: true,
      gridcolor: 'rgba(0,0,0,0.1)',
    };
  });

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Signal Explorer</h1>
          <p className="text-gray-500 dark:text-gray-400">Real WESAD chest signals across conditions</p>
        </div>
        <select
          value={selectedSubject}
          onChange={(e) => setSelectedSubject(e.target.value)}
          className="px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
        >
          {subjects.map((subject) => (
            <option key={subject} value={subject}>
              Subject {subject}
            </option>
          ))}
        </select>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Signals:</span>
            {Object.entries(visibleSignals).map(([signal, visible]) => (
              <label key={signal} className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={visible}
                  onChange={() => toggleSignal(signal as keyof typeof visibleSignals)}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <span className="text-sm text-gray-600 dark:text-gray-400 uppercase">{signal}</span>
              </label>
            ))}
          </div>
          <div className="flex items-center space-x-2 ml-auto">
            <button onClick={handleZoomIn} className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600" title="Zoom In">
              <ZoomIn className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <button onClick={handleZoomOut} className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600" title="Zoom Out">
              <ZoomOut className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <button onClick={handleReset} className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600" title="Reset View">
              <RefreshCw className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <Plot
          data={buildTraces()}
          layout={layout}
          config={{ responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
          style={{ width: '100%' }}
          onRelayout={(e: any) => {
            if (e['xaxis.range[0]'] !== undefined) {
              setXRange([e['xaxis.range[0]'], e['xaxis.range[1]']]);
            }
          }}
        />
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">Condition Legend</h3>
        <div className="flex flex-wrap gap-4">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-4 bg-green-200 rounded"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">Baseline (Rest)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-6 h-4 bg-red-200 rounded"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">Stress (TSST)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-6 h-4 bg-yellow-200 rounded"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">Amusement (Fun Videos)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignalExplorer;
