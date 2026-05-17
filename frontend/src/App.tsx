import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Activity,
  Brain,
  FileSearch,
  BarChart3,
  Info,
  Sun,
  Moon,
  Menu,
  X,
  Heart
} from 'lucide-react';

// Components
import Dashboard from './components/Dashboard';
import SignalExplorer from './components/SignalExplorer';
import PredictionPanel from './components/PredictionPanel';
import ExplainabilityDashboard from './components/ExplainabilityDashboard';
import ModelComparison from './components/ModelComparison';
import About from './components/About';
import ErrorBoundary from './components/ErrorBoundary';

// Navigation items
const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/signals', icon: Activity, label: 'Signal Explorer' },
  { path: '/predict', icon: Brain, label: 'Prediction' },
  { path: '/explain', icon: FileSearch, label: 'Explainability' },
  { path: '/models', icon: BarChart3, label: 'Model Comparison' },
  { path: '/about', icon: Info, label: 'About' },
];

// Sidebar component
const Sidebar: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  darkMode: boolean;
  toggleDarkMode: () => void;
}> = ({ isOpen, onClose, darkMode, toggleDarkMode }) => {
  const location = useLocation();

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed top-0 left-0 z-30 h-full w-64
          bg-primary-700 dark:bg-gray-900
          transform transition-transform duration-300 ease-in-out
          lg:translate-x-0 lg:static
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        {/* Logo */}
        <div className="flex items-center justify-between p-4 border-b border-primary-600 dark:border-gray-700">
          <div className="flex items-center space-x-2">
            <Heart className="w-8 h-8 text-red-400" />
            <span className="text-xl font-bold text-white">CalmSense</span>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden text-white hover:text-gray-300"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;

            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={onClose}
                className={`
                  flex items-center space-x-3 px-4 py-3 rounded-lg
                  transition-colors duration-200
                  ${isActive
                    ? 'bg-accent-500 text-white'
                    : 'text-gray-300 hover:bg-primary-600 dark:hover:bg-gray-800'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Dark mode toggle */}
        <div className="absolute bottom-4 left-4 right-4">
          <button
            onClick={toggleDarkMode}
            className="flex items-center justify-center w-full px-4 py-3
                       bg-primary-600 dark:bg-gray-800 rounded-lg
                       text-white hover:bg-primary-500 dark:hover:bg-gray-700
                       transition-colors duration-200"
          >
            {darkMode ? (
              <>
                <Sun className="w-5 h-5 mr-2" />
                <span>Light Mode</span>
              </>
            ) : (
              <>
                <Moon className="w-5 h-5 mr-2" />
                <span>Dark Mode</span>
              </>
            )}
          </button>
        </div>
      </aside>
    </>
  );
};

// Header component
const Header: React.FC<{ onMenuClick: () => void }> = ({ onMenuClick }) => {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between px-4 py-3">
        <button
          onClick={onMenuClick}
          className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <Menu className="w-6 h-6 text-gray-600 dark:text-gray-300" />
        </button>

        <div className="flex items-center space-x-4">
          <div className="hidden sm:flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
            <span className={`w-2 h-2 ${process.env.REACT_APP_API_URL ? 'bg-green-500' : 'bg-amber-500'} rounded-full animate-pulse`}></span>
            <span>{process.env.REACT_APP_API_URL ? 'API Connected' : 'Demo Mode'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};

// Main App component
const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  return (
    <Router basename={process.env.PUBLIC_URL}>
      <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${darkMode ? 'dark' : ''}`}>
        <div className="flex h-screen overflow-hidden">
          {/* Sidebar */}
          <Sidebar
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
            darkMode={darkMode}
            toggleDarkMode={toggleDarkMode}
          />

          {/* Main content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <Header onMenuClick={() => setSidebarOpen(true)} />

            <main className="flex-1 overflow-y-auto p-4 lg:p-6">
              <Routes>
                <Route path="/" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
                <Route path="/signals" element={<ErrorBoundary><SignalExplorer /></ErrorBoundary>} />
                <Route path="/predict" element={<ErrorBoundary><PredictionPanel /></ErrorBoundary>} />
                <Route path="/explain" element={<ErrorBoundary><ExplainabilityDashboard /></ErrorBoundary>} />
                <Route path="/models" element={<ErrorBoundary><ModelComparison /></ErrorBoundary>} />
                <Route path="/about" element={<ErrorBoundary><About /></ErrorBoundary>} />
              </Routes>
            </main>

            {/* Footer */}
            <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-4 py-3">
              <div className="text-center text-sm text-gray-500 dark:text-gray-400">
                CalmSense v1.0.0 | Multimodal Stress Detection System
              </div>
            </footer>
          </div>
        </div>
      </div>
    </Router>
  );
};

export default App;
