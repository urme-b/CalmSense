import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ZoomIn, ZoomOut, Move, RefreshCw, Download } from 'lucide-react';
import realSignals from '../signals.json';

// Real WESAD chest signals (baseline -> stress -> amusement), downsampled for display
const subjects = Object.keys(realSignals);

const SignalExplorer: React.FC = () => {
  const [selectedSubject, setSelectedSubject] = useState(subjects[0]);
  const [visibleSignals, setVisibleSignals] = useState({
    ecg: true,
    eda: true,
    temp: true,
    acc: false,
  });
  const [xRange, setXRange] = useState<[number, number]>([0, 60]);

  const signalData = useMemo(
    () => (realSignals as any)[selectedSubject],
    [selectedSubject]
  );

  const toggleSignal = (signal: keyof typeof visibleSignals) => {
    setVisibleSignals((prev) => ({ ...prev, [signal]: !prev[signal] }));
  };

  const handleZoomIn = () => {
    const range = xRange[1] - xRange[0];
    const center = (xRange[0] + xRange[1]) / 2;
    const newRange = range * 0.5;
    setXRange([center - newRange / 2, center + newRange / 2]);
  };

  const handleZoomOut = () => {
    const range = xRange[1] - xRange[0];
    const center = (xRange[0] + xRange[1]) / 2;
    const newRange = Math.min(range * 2, 300);
    setXRange([Math.max(0, center - newRange / 2), Math.min(300, center + newRange / 2)]);
  };

  const handleReset = () => {
    setXRange([0, 60]);
  };

  // Create subplot data
  const createSubplots = () => {
    const traces: any[] = [];
    const visibleCount = Object.values(visibleSignals).filter(Boolean).length;
    let currentRow = 1;

    // Condition background
    const conditionColors: Record<string, string> = {
      'Baseline': 'rgba(56, 161, 105, 0.1)',
      'Stress': 'rgba(229, 62, 62, 0.1)',
      'Amusement': 'rgba(214, 158, 46, 0.1)',
    };

    if (visibleSignals.ecg) {
      traces.push({
        x: signalData.time,
        y: signalData.ecg,
        type: 'scatter',
        mode: 'lines',
        name: 'ECG',
        line: { color: '#E53E3E', width: 1 },
        xaxis: 'x',
        yaxis: `y${currentRow === 1 ? '' : currentRow}`,
      });
      currentRow++;
    }

    if (visibleSignals.eda) {
      traces.push({
        x: signalData.time,
        y: signalData.eda,
        type: 'scatter',
        mode: 'lines',
        name: 'EDA',
        line: { color: '#3182CE', width: 1.5 },
        xaxis: 'x',
        yaxis: `y${currentRow === 1 ? '' : currentRow}`,
      });
      currentRow++;
    }

    if (visibleSignals.temp) {
      traces.push({
        x: signalData.time,
        y: signalData.temp,
        type: 'scatter',
        mode: 'lines',
        name: 'Temperature',
        line: { color: '#38A169', width: 1.5 },
        xaxis: 'x',
        yaxis: `y${currentRow === 1 ? '' : currentRow}`,
      });
      currentRow++;
    }

    if (visibleSignals.acc) {
      traces.push(
        {
          x: signalData.time,
          y: signalData.accX,
          type: 'scatter',
          mode: 'lines',
          name: 'ACC X',
          line: { color: '#805AD5', width: 1 },
          xaxis: 'x',
          yaxis: `y${currentRow === 1 ? '' : currentRow}`,
        },
        {
          x: signalData.time,
          y: signalData.accY,
          type: 'scatter',
          mode: 'lines',
          name: 'ACC Y',
          line: { color: '#DD6B20', width: 1 },
          xaxis: 'x',
          yaxis: `y${currentRow === 1 ? '' : currentRow}`,
        },
        {
          x: signalData.time,
          y: signalData.accZ,
          type: 'scatter',
          mode: 'lines',
          name: 'ACC Z',
          line: { color: '#319795', width: 1 },
          xaxis: 'x',
          yaxis: `y${currentRow === 1 ? '' : currentRow}`,
        }
      );
    }

    return traces;
  };

  const layout: any = {
    title: {
      text: `Signal Explorer - Subject ${selectedSubject}`,
      font: { size: 18 },
    },
    showlegend: true,
    legend: { orientation: 'h', y: -0.15 },
    xaxis: {
      title: 'Time (s)',
      range: xRange,
      showgrid: true,
      gridcolor: 'rgba(0, 0, 0, 0.1)',
    },
    yaxis: {
      title: 'ECG (mV)',
      showgrid: true,
      gridcolor: 'rgba(0, 0, 0, 0.1)',
      domain: visibleSignals.ecg ? [0.7, 1] : [0, 0],
    },
    yaxis2: {
      title: 'EDA (μS)',
      showgrid: true,
      gridcolor: 'rgba(0, 0, 0, 0.1)',
      domain: visibleSignals.eda ? [0.35, 0.65] : [0, 0],
    },
    yaxis3: {
      title: 'Temp (°C)',
      showgrid: true,
      gridcolor: 'rgba(0, 0, 0, 0.1)',
      domain: visibleSignals.temp ? [0, 0.3] : [0, 0],
    },
    height: 600,
    margin: { t: 60, b: 80, l: 60, r: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    shapes: [
      // Baseline region
      {
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: 0,
        x1: 100,
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(56, 161, 105, 0.1)',
        line: { width: 0 },
      },
      // Stress region
      {
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: 100,
        x1: 200,
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(229, 62, 62, 0.1)',
        line: { width: 0 },
      },
      // Amusement region
      {
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: 200,
        x1: 300,
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(214, 158, 46, 0.1)',
        line: { width: 0 },
      },
    ],
    annotations: [
      { x: 50, y: 1.05, xref: 'x', yref: 'paper', text: 'Baseline', showarrow: false, font: { color: '#38A169' } },
      { x: 150, y: 1.05, xref: 'x', yref: 'paper', text: 'Stress', showarrow: false, font: { color: '#E53E3E' } },
      { x: 250, y: 1.05, xref: 'x', yref: 'paper', text: 'Amusement', showarrow: false, font: { color: '#D69E2E' } },
    ],
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Signal Explorer</h1>
          <p className="text-gray-500 dark:text-gray-400">
            Visualize and explore physiological signals
          </p>
        </div>

        {/* Subject Selector */}
        <select
          value={selectedSubject}
          onChange={(e) => setSelectedSubject(e.target.value)}
          className="px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg
                     text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
        >
          {subjects.map((subject) => (
            <option key={subject} value={subject}>
              Subject {subject}
            </option>
          ))}
        </select>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Signal Toggles */}
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

          {/* Zoom Controls */}
          <div className="flex items-center space-x-2 ml-auto">
            <button
              onClick={handleZoomIn}
              className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
              title="Zoom In"
            >
              <ZoomIn className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
              title="Zoom Out"
            >
              <ZoomOut className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <button
              onClick={handleReset}
              className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
              title="Reset View"
            >
              <RefreshCw className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <button
              className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
              title="Download"
            >
              <Download className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
          </div>
        </div>
      </div>

      {/* Signal Plot */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <Plot
          data={createSubplots()}
          layout={layout}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          }}
          style={{ width: '100%' }}
          onRelayout={(e: any) => {
            if (e['xaxis.range[0]'] !== undefined) {
              setXRange([e['xaxis.range[0]'], e['xaxis.range[1]']]);
            }
          }}
        />
      </div>

      {/* Legend */}
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
