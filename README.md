# CalmSense

Stress detection from wearable physiology (ECG, electrodermal activity, temperature,
respiration, motion), evaluated honestly. Every result uses Leave-One-Subject-Out (LOSO)
cross-validation, so models are always tested on people they were not trained on. The goal
is not a new architecture but a reproducible account of what subject-independent stress
detection actually delivers, in contrast to the inflated numbers common on this benchmark.

- Live demo: https://urme-b.github.io/CalmSense/ (the trained model runs in the browser via ONNX)
- Full methodology, statistics, and figures: [PAPER.md](PAPER.md)

## Pipeline

1. Filter the raw signals and detect ECG R-peaks; decompose EDA into tonic and phasic components.
2. Segment into 60-second windows with 50% overlap, keeping a window only if at least 90% of its
   samples share one condition.
3. Extract 58 features per window: 30 HRV (time, frequency, nonlinear), 15 EDA, 5 temperature,
   3 respiration, 5 motion.
4. Evaluate with LOSO. Imputation, scaling, and class balancing are fit inside each fold, so no
   subject crosses the train/test boundary.

## Results

Binary stress vs. non-stress, LOSO over 15 subjects (869 windows, 58 features):

| Model               | Accuracy | Macro-F1 |
| ------------------- | -------- | -------- |
| Random Forest       | 0.913    | 0.898    |
| XGBoost             | 0.903    | 0.873    |
| Logistic Regression | 0.902    | 0.883    |
| LightGBM            | 0.894    | 0.860    |
| 1D-CNN (raw)        | 0.718    | 0.648    |

Random Forest leads with a 95% CI of [0.86, 0.96], but the four feature-based models are
statistically indistinguishable (Friedman p = 0.81). The three-class task
(baseline/stress/amusement) tops out at 0.67.

Secondary findings:

- Within-subject cross-validation inflates three-class accuracy from 0.67 to 0.79.
- Removing all motion features changes binary accuracy by about one point (0.913 to 0.901);
  motion alone reaches 0.885, so the signal is physiological, not just movement.
- A wrist-only model (Empatica E4) reaches 0.893, ~2 points behind the chest sensor.
- Trained on WESAD and tested on the PhysioNet Non-EEG dataset, balanced accuracy falls to
  0.50–0.57 (near chance), showing within-dataset accuracy does not imply generalization.

Figures for each finding are in `outputs/figures/`.

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/urme-b/CalmSense.git
cd CalmSense
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

On macOS, XGBoost and LightGBM need OpenMP: `brew install libomp`.

## Usage

The trained model is committed, so the API and tests run without the dataset.

```bash
make api        # serve the model at http://localhost:8000/docs
make test       # run the test suite (19 tests)
make reproduce  # regenerate all results and figures (requires the dataset; see Data)
```

Single prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"HRV_MeanNN": 650, "HRV_RMSSD": 18, "EDA_SCL_mean": 6.0}}'
```

Unknown or empty feature payloads return HTTP 422; missing known features are imputed.

## Configuration

No environment variables are required for local use. Optional settings:

| Variable                 | Used by         | Default                                          | Purpose                                     |
| ------------------------ | --------------- | ------------------------------------------------ | ------------------------------------------- |
| `CALMSENSE_CORS_ORIGINS` | API             | `http://localhost:3000,https://urme-b.github.io` | Comma-separated allowed origins             |
| `REACT_APP_API_URL`      | dashboard build | unset (model runs in-browser)                    | Point the dashboard at a hosted API instead |

See [.env.example](.env.example).

## API

| Method | Endpoint   | Description                                         |
| ------ | ---------- | -------------------------------------------------- |
| POST   | `/predict` | Class prediction and probabilities from features   |
| POST   | `/explain` | Prediction plus the top SHAP feature contributions |
| GET    | `/model`   | Model classes and expected feature names           |
| GET    | `/health`  | Liveness and whether a model is loaded             |

## Project layout

```
src/preprocessing/   Signal filtering, R-peak detection, EDA decomposition
src/features/        HRV, EDA, temperature, respiration, accelerometer extractors
src/dataset.py       Raw signals to windowed feature matrix and CNN tensors
src/datasets/        PhysioNet Non-EEG loader for cross-dataset transfer
src/models/          Classical classifiers (LR, RF, XGBoost, LightGBM) and a 1D-CNN
src/portable.py      Shared feature space for cross-dataset transfer
scripts/             run_experiment, ablation, wrist, cross_dataset, stats, export_onnx
api/                 FastAPI prediction service
frontend/            React dashboard (see frontend/README.md)
tests/               Unit and methodology tests
```

## Data

WESAD and PhysioNet Non-EEG are public datasets and are not redistributed here. Download WESAD
into `data/raw/WESAD/` before running `make reproduce`; see [data/raw/README.md](data/raw/README.md)
for instructions and the expected layout.

## Development

```bash
make lint     # ruff check
make format   # ruff format and autofix
make test     # pytest
```

CI runs linting and the test suite on Python 3.9, 3.10, and 3.11. Keep changes covered by tests
and formatted with ruff.

## Limitations

- 15 subjects and lab-induced (TSST) stress; per-subject accuracy ranges 0.71–1.00. No claim is
  made about real-world or chronic stress.
- Hyperparameters are fixed defaults, not tuned.
- The 1D-CNN underperforms at this data scale and is reported as a baseline rather than omitted.

## License

MIT; see [LICENSE](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff).
