<div align="center">

# CalmSense

**Honest, leakage-free stress detection from wearable physiology.**

[![CI](https://github.com/urme-b/CalmSense/actions/workflows/ci.yml/badge.svg)](https://github.com/urme-b/CalmSense/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://urme-b.github.io/CalmSense/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[**Live demo**](https://urme-b.github.io/CalmSense/) · [**Paper**](PAPER.md) · [**Results**](results/)

</div>

Most WESAD papers report 95–99% accuracy. Much of it is an artefact of how it's measured. CalmSense
detects stress from wearable signals (ECG, EDA, temperature, respiration, motion) and — more
importantly — **strips away three layers of evaluation optimism** to show what honestly survives
when models are tested on people they never trained on.

The contribution isn't a new architecture; it's a careful, fully reproducible accounting of what
subject-independent wearable stress detection actually delivers.

---

## Results

All numbers use strict **Leave-One-Subject-Out (LOSO)** cross-validation on WESAD (15 subjects) —
imputation, scaling, and class balancing are fit *inside each fold*, so no subject crosses the
train/test boundary.

**Binary — stress vs. non-stress** (869 windows, 58 features)

| Model | LOSO Accuracy | Macro-F1 |
|-------|:-------------:|:--------:|
| **Random Forest** | **0.913** | **0.898** |
| XGBoost | 0.903 | 0.873 |
| Logistic Regression | 0.902 | 0.883 |
| LightGBM | 0.894 | 0.860 |
| 1D-CNN (raw signals) | 0.718 | 0.648 |

The four feature models are **statistically indistinguishable** (Friedman p=0.81), so we report the
family rather than crown a winner. Three-class (baseline/stress/amusement) tops out at **0.67** — the
honest ceiling for this task.

## The three layers of optimism

| 1 · Subject leakage | 2 · Motion confound | 3 · Dataset shift |
|:---:|:---:|:---:|
| ![gap](outputs/figures/multiclass_optimism_gap.png) | ![ablation](outputs/figures/ablation.png) | ![cross](outputs/figures/cross_dataset.png) |
| Within-subject CV inflates 3-class accuracy **0.67 → 0.79** | Remove motion entirely and accuracy holds (**0.913 → 0.901**) — it's physiology, not movement | Train on WESAD, test on PhysioNet Non-EEG → **near chance (0.50–0.57)** |

**Plus a practical win:** a wrist-only model (Empatica E4) reaches **0.893 vs 0.913** for the chest —
a ~2-point drop, no chest strap needed.

<div align="center">
<img src="outputs/figures/chest_vs_wrist.png" width="45%"> <img src="outputs/figures/shap_beeswarm.png" width="45%">
</div>

The most informative biomarkers (mean |SHAP|) are a motion descriptor, heart-rate level
(`HRV_MedianNN`/`HRV_MeanNN`), skin-conductance responses, and respiration rate — consistent with the
physiology of acute stress.

## Quick start

```bash
git clone https://github.com/urme-b/CalmSense.git && cd CalmSense
python -m venv .venv && source .venv/bin/activate
pip install -e .
# macOS only, for xgboost/lightgbm: brew install libomp

# Download WESAD into data/raw/WESAD/ first — see data/raw/README.md
make reproduce          # regenerate every result, figure, model + dashboard data
make api                # serve the model at http://localhost:8000/docs
make test               # 19 tests
```

No data needed to *use* it — the trained model is committed, and the [live demo](https://urme-b.github.io/CalmSense/)
runs it entirely in your browser (ONNX).

## How it works

```
raw WESAD signals → filter · R-peak/EDA processing → 60s windows (50% overlap, ≥90% pure)
   → 58 HRV / EDA / temperature / respiration / motion features
   → leakage-free LOSO benchmark (impute + scale fit per fold)
   → metrics · confusion matrices · per-subject scores · SHAP · optimism gap
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Stress prediction + class probabilities from a feature vector |
| `POST` | `/explain` | Prediction plus top SHAP feature contributions |
| `GET`  | `/model`   | Model classes and expected feature names |
| `GET`  | `/health`  | Liveness and whether a model is loaded |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"HRV_MeanNN": 650, "HRV_RMSSD": 18, "EDA_SCL_mean": 6.0}}'
```

## Project structure

```
src/preprocessing/   ECG / EDA / respiration filtering, R-peak detection
src/features/        HRV, EDA, temperature, respiration, accelerometer extractors
src/dataset.py       raw signals → feature matrix + raw CNN tensors (cached)
src/models/          classical classifiers (LR/RF/XGB/LGBM) + residual 1D-CNN
src/portable.py      shared feature space for cross-dataset transfer
scripts/             run_experiment · ablation · wrist · cross_dataset · stats · export_onnx
api/                 FastAPI prediction service
frontend/            React dashboard (runs the model in-browser via ONNX)
```

## Dataset

**WESAD** (Schmidt et al., ICMI 2018) — 15 subjects, chest (RespiBAN) + wrist (Empatica E4), four
conditions. Download from the
[UCI repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
into `data/raw/WESAD/`; it is not redistributed here. Cross-dataset transfer uses the public
[PhysioNet Non-EEG](https://physionet.org/content/noneeg/1.0.0/) dataset.

## Limitations

- **15 subjects, lab-induced (TSST) stress** — per-subject LOSO accuracy ranges 0.71–1.00; no claim to
  real-world or chronic stress.
- **No hyperparameter tuning** — fixed, sensible defaults; the goal is an honest baseline, not a
  leaderboard score.
- **Deep learning underperforms** at this data scale; reported rather than omitted.

Full methodology, statistics, and references are in [`PAPER.md`](PAPER.md).

## Citation

If you use this work, please cite via [`CITATION.cff`](CITATION.cff). Licensed under [MIT](LICENSE).
