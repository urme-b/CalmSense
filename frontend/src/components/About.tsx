import React from 'react';
import {
  Heart,
  Github,
  BookOpen,
  Database,
  Cpu,
  Shield,
  Users,
  ExternalLink,
} from 'lucide-react';

const About: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in max-w-4xl mx-auto">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center space-x-3 mb-4">
          <Heart className="w-12 h-12 text-red-500" />
          <h1 className="text-4xl font-bold text-gradient">CalmSense</h1>
        </div>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          Multimodal Stress Detection from Physiological Signals
        </p>
        <p className="mt-2 text-gray-500 dark:text-gray-500">
          Version 1.0.0
        </p>
      </div>

      {/* Badges */}
      <div className="flex flex-wrap justify-center gap-2">
        {['Python 3.11', 'PyTorch', 'FastAPI', 'React', 'WESAD Dataset'].map((badge) => (
          <span
            key={badge}
            className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400 rounded-full text-sm font-medium"
          >
            {badge}
          </span>
        ))}
      </div>

      {/* Description */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          About the Project
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          CalmSense is a comprehensive stress detection system that analyzes physiological signals
          to identify stress states in real-time. Using the WESAD dataset, we developed a complete
          machine learning pipeline that includes signal preprocessing, feature extraction,
          model training with cross-validation, and deployment via a FastAPI backend with
          React dashboard.
        </p>
        <p className="mt-4 text-gray-600 dark:text-gray-400 leading-relaxed">
          Every result uses <strong>Leave-One-Subject-Out</strong> cross-validation, so models are always tested
          on people they never trained on. The best model (Random Forest) reaches <strong>91.2% accuracy</strong> for
          binary stress detection and 65.0% for three-class (baseline/stress/amusement), with SHAP-based and
          clinically-grounded interpretation built on HRV standards.
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center space-x-3 mb-3">
            <Database className="w-6 h-6 text-blue-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Dataset</h3>
          </div>
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li>• WESAD: Wearable Stress and Affect Detection</li>
            <li>• 15 subjects with multimodal signals</li>
            <li>• ECG, EDA, EMG, TEMP, RESP, ACC</li>
            <li>• Three conditions: Baseline, Stress, Amusement</li>
          </ul>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center space-x-3 mb-3">
            <Cpu className="w-6 h-6 text-green-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Models</h3>
          </div>
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li>• Classical ML: Logistic Regression, Random Forest, XGBoost, LightGBM</li>
            <li>• Deep Learning: residual 1D-CNN on raw signals</li>
            <li>• Leakage-free per-fold imputation and scaling</li>
            <li>• Leave-One-Subject-Out cross-validation</li>
          </ul>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center space-x-3 mb-3">
            <BookOpen className="w-6 h-6 text-purple-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Features</h3>
          </div>
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li>• 60+ physiological features extracted</li>
            <li>• HRV time/frequency/nonlinear analysis</li>
            <li>• EDA phasic/tonic decomposition</li>
            <li>• 60s windows, 50% overlap</li>
          </ul>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center space-x-3 mb-3">
            <Shield className="w-6 h-6 text-orange-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Explainability</h3>
          </div>
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li>• SHAP values for global/local importance</li>
            <li>• Top-biomarker contributions per prediction</li>
            <li>• Optimism-gap analysis (LOSO vs within-subject)</li>
            <li>• Clinical interpretation (Task Force 1996)</li>
          </ul>
        </div>
      </div>

      {/* Architecture */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          System Architecture
        </h2>
        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <pre className="text-gray-700 dark:text-gray-300">
{`┌─────────────────────────────────────────────────────────────┐
│                    CalmSense Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  WESAD   │───▶│  Signal  │───▶│ Feature  │              │
│  │ Dataset  │    │Processing│    │Extraction│              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                       │                      │
│                                       ▼                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  React   │◀───│ FastAPI  │◀───│  ML/DL   │              │
│  │Dashboard │    │ Backend  │    │  Models  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │               │                                      │
│       ▼               ▼                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │Prediction│    │WebSocket │    │  SHAP/   │              │
│  │   UI     │    │Streaming │    │  LIME    │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘`}
          </pre>
        </div>
      </div>

      {/* Results */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Key Results
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Model</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">LOSO Accuracy</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Macro-F1</th>
              </tr>
            </thead>
            <tbody>
              {[
                { model: 'Random Forest', acc: '91.2%', f1: '89.8%' },
                { model: 'XGBoost', acc: '88.8%', f1: '85.8%' },
                { model: 'LightGBM', acc: '87.5%', f1: '83.8%' },
                { model: 'Logistic Regression', acc: '86.8%', f1: '85.3%' },
                { model: '1D-CNN', acc: '64.7%', f1: '39.3%' },
              ].map((row, i) => (
                <tr key={row.model} className={`border-b border-gray-100 dark:border-gray-700 ${i === 0 ? 'bg-green-50 dark:bg-green-900/20' : ''}`}>
                  <td className="px-4 py-2 text-gray-900 dark:text-white font-medium">{row.model}</td>
                  <td className="px-4 py-2 text-gray-600 dark:text-gray-400">{row.acc}</td>
                  <td className="px-4 py-2 text-gray-600 dark:text-gray-400">{row.f1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* References */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          References
        </h2>
        <ul className="space-y-3 text-gray-600 dark:text-gray-400">
          <li className="flex items-start space-x-2">
            <span className="text-blue-500 mt-1">•</span>
            <span>
              Schmidt, P., et al. (2018). "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection."
              <em> ICMI 2018</em>.
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-blue-500 mt-1">•</span>
            <span>
              Task Force of ESC and NASPE (1996). "Heart rate variability: Standards of measurement, physiological interpretation, and clinical use."
              <em> Circulation</em>.
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-blue-500 mt-1">•</span>
            <span>
              Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions."
              <em> NeurIPS 2017</em>.
            </span>
          </li>
        </ul>
      </div>

      {/* Links */}
      <div className="flex flex-wrap justify-center gap-4">
        <a
          href="https://github.com/urme-b/CalmSense"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center space-x-2 px-4 py-2 bg-gray-900 dark:bg-gray-700 text-white rounded-lg hover:bg-gray-800 dark:hover:bg-gray-600 transition-colors"
        >
          <Github className="w-5 h-5" />
          <span>GitHub Repository</span>
          <ExternalLink className="w-4 h-4" />
        </a>
        <a
          href="https://github.com/urme-b/CalmSense/blob/main/api/main.py"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <BookOpen className="w-5 h-5" />
          <span>API Documentation</span>
          <ExternalLink className="w-4 h-4" />
        </a>
      </div>

      {/* Footer */}
      <div className="text-center text-gray-500 dark:text-gray-500 text-sm">
        <p>© 2024 CalmSense Project. Licensed under MIT License.</p>
        <p className="mt-1">Built with React, TypeScript, Tailwind CSS, FastAPI, and PyTorch.</p>
      </div>
    </div>
  );
};

export default About;
