# CalmSense React Dashboard

This directory will contain the React-based dashboard for CalmSense.

## Planned Features

- Real-time signal visualization
- Stress level monitoring
- SHAP explanation plots
- Model performance metrics
- Confusion matrix display
- Historical data charts

## Setup (To Be Implemented)

```bash
# Create React app
npx create-react-app calmsense-dashboard --template typescript

# Or with Vite (recommended)
npm create vite@latest calmsense-dashboard -- --template react-ts

# Install dependencies
cd calmsense-dashboard
npm install

# Additional packages to install
npm install axios recharts @tanstack/react-query tailwindcss
npm install @headlessui/react @heroicons/react
npm install plotly.js react-plotly.js
```

## Directory Structure (Planned)

```
frontend/
├── src/
│   ├── components/
│   │   ├── SignalPlot.tsx
│   │   ├── StressGauge.tsx
│   │   ├── SHAPWaterfall.tsx
│   │   ├── ConfusionMatrix.tsx
│   │   └── MetricsCard.tsx
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Analysis.tsx
│   │   └── Settings.tsx
│   ├── hooks/
│   │   └── useStressAPI.ts
│   ├── services/
│   │   └── api.ts
│   └── App.tsx
├── public/
├── package.json
└── README.md
```

## API Integration

The dashboard will communicate with the FastAPI backend:

- `GET /health` - Check API status
- `POST /predict/features` - Get stress prediction
- `POST /explain` - Get SHAP explanation
- `POST /batch/predict` - Batch predictions
