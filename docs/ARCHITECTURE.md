# Architecture

CalmSense is a pipeline, not a monolith: raw wearable signals flow through cleaning, windowing, feature
extraction, a leakage-free benchmark, and calibration, then the trained model is exported for in-browser
inference. Each stage is a small module with one job.

## Pipeline stages → modules

| Stage | What happens | Code |
| ----- | ------------ | ---- |
| 1. Ingest | Load raw WESAD pickles (chest + wrist) and Non-EEG records | src/data/loader.py, src/datasets/non_eeg.py |
| 2. Preprocess | Butterworth filtering, ECG R-peak detection + ectopic correction, EDA tonic/phasic decomposition | src/preprocessing/{filters,ecg_processor,eda_processor}.py |
| 3. Window | Segment into 60 s / 50%-overlap windows; keep windows ≥90% one condition | src/dataset.py (chest), src/dataset_wrist.py (wrist), shared window_label() |
| 4. Features | 58 features/window: HRV (time/freq/nonlinear), EDA, temperature, respiration, motion | src/features/feature_pipeline.py + per-modality extractors |
| 5. Benchmark | Leakage-free LOSO; in-fold impute/scale/balance; LR/RF/XGBoost/LightGBM + 1D-CNN | scripts/run_experiment.py, src/models/ml/classifiers.py, src/models/dl/cnn_1d.py |
| 6. Calibration | ECE/MCE/Brier, decision-curve net benefit, leak-free recalibration, few-shot personalization | src/calibration.py, scripts/{calibration,personalize}.py |
| 7. Analysis | Optimism gap, ablation, wrist-vs-chest, cross-dataset, SHAP, stats | scripts/{ablation,wrist,cross_dataset,stats,tuning}.py |
| 8. Export & serve | ONNX export (browser parity), FastAPI service, React dashboard | scripts/export_onnx.py, src/portable.py, api/, frontend/ |

## Cross-cutting

- **Config**, frozen dataclasses (sampling rates, filter params, valid subjects): src/config.py.
- **Logging**, structured logs via LoggerMixin: src/logging_config.py.
- **Synthetic data**, src/synthetic.py reproduces the full pipeline without the real dataset for
  make demo/CI. It is intentionally near-separable, so calibration/optimism numbers from it are not
  meaningful (see the module docstring).
- **Reproducibility**, everything is seeded (SEED = 42). make reproduce regenerates results/ and
  outputs/figures/ from WESAD; results/README.md records provenance.

## Data flow at a glance

```
raw WESAD ─▶ preprocess ─▶ window ─▶ features ─▶ LOSO benchmark ─▶ metrics.json
                                              └─▶ calibration / personalization ─▶ calibration.json
                                              └─▶ SHAP / ablation / wrist / cross-dataset
trained model ─▶ ONNX export ─▶ React dashboard (browser)  ·  FastAPI service (optional)
```
