# CalmSense

[![CI](https://github.com/urme-b/CalmSense/actions/workflows/ci.yml/badge.svg)](https://github.com/urme-b/CalmSense/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

Detecting stress from wearable sensors, and measuring how much of the field's reported accuracy —
and confidence — actually holds up on people the model has never seen.

Most published results on the WESAD benchmark claim 95–99% accuracy. CalmSense shows that much of
that vanishes under honest, subject-independent testing, traces exactly where it goes, and adds the
part accuracy hides: whether the model's probabilities are calibrated enough to trigger an alert.

[Live demo](https://urme-b.github.io/CalmSense/) · [Run in Colab](https://colab.research.google.com/github/urme-b/CalmSense/blob/main/notebooks/CalmSense.ipynb) · [Paper](PAPER.md)

## Why it matters

A stress model is only useful if it works on a new person — and an alerting system is only safe if a
"0.8 chance of stress" really means 0.8. CalmSense evaluates everything with Leave-One-Subject-Out
validation (train on 14 people, test on the 15th) and quantifies four ways the usual numbers get
inflated: subject leakage, motion confounds, dataset shift, and **calibration** — the same
within-subject testing that inflates accuracy also makes the model look better calibrated than it is.
The result is a realistic picture of what wearable stress detection can and cannot do today.

## Results

Binary stress detection on held-out subjects (15 participants, 58 physiological features):

| Model               | Accuracy | F1    |
| ------------------- | -------- | ----- |
| Random Forest       | 0.913    | 0.898 |
| XGBoost             | 0.903    | 0.873 |
| Logistic Regression | 0.902    | 0.883 |
| LightGBM            | 0.894    | 0.860 |
| 1D-CNN              | 0.718    | 0.648 |

![Model comparison](outputs/figures/binary_model_comparison.png)

The four feature-based models land within noise of one another. The real contribution is that the
number is honest: it was measured on people the model never trained on. These headline numbers use
fixed, sensible hyperparameters (not per-fold tuned); `scripts/tuning.py` runs a separate leak-free
nested-CV sweep (inner grouped search, outer LOSO) as a robustness check — its output regenerates on
WESAD and is not committed.

## Key findings

| Subject leakage | Motion is not the trick |
| :---: | :---: |
| ![Optimism gap](outputs/figures/multiclass_optimism_gap.png) | ![Ablation](outputs/figures/ablation.png) |
| Testing within the same people inflates three-class accuracy from 0.67 to 0.79. | Remove every motion feature and accuracy barely moves (0.913 to 0.901). The signal is physiological. |
| **A wrist band is enough** | **It does not transfer across datasets** |
| ![Chest vs wrist](outputs/figures/chest_vs_wrist.png) | ![Cross-dataset](outputs/figures/cross_dataset.png) |
| A wrist-only model reaches 0.893, about two points behind a research-grade chest strap. | Train on WESAD, test on a second dataset, and accuracy collapses to near chance — one confounded transfer pair, illustrative not conclusive. Within-dataset success is not generalization. |

### Accuracy isn't the whole story: calibration

An alerting system acts on the probability, not the label. The same within-subject evaluation that
inflates accuracy also makes the model look better calibrated than it is on a new person. CalmSense
implements a full subject-independent calibration analysis — ECE, MCE, Brier, reliability diagrams,
and decision-curve net benefit — tests the optimism gap for significance (paired Wilcoxon on
per-subject Brier), and applies a **leakage-free recalibration**: an isotonic map fit only on
out-of-fold *training* subjects, never the test subject. A **few-shot personalization** variant fits a
per-subject calibrator from a short labeled enrollment, drawn only from non-overlapping windows the
evaluation never sees.

> **Status:** these calibration and personalization numbers are computed on WESAD, which is not
> redistributed, so they are **not committed** to the repo — the [paper](PAPER.md) §4.7–4.8 tables are
> placeholders until you run them. Generate them with `make reproduce` (needs the WESAD download);
> `make demo` exercises the identical pipeline end-to-end on synthetic data (where, by design, stress
> is near-separable and the gap is not meaningful — see `data/raw/README.md`).

## What the model relies on

![Feature importance](outputs/figures/shap_beeswarm.png)

Heart-rate level, skin-conductance responses, and movement carry the signal, matching the physiology
expected under acute stress.

## How it works

1. **Clean the signals.** Per-channel Butterworth filtering, R-peak detection with NeuroKit2, and
   tonic/phasic decomposition of the electrodermal signal, with ectopic-beat correction on the
   heart-rate series.
2. **Window.** Segment into 60-second windows at 50% overlap, keeping a window only when at least
   90% of its samples share one condition.
3. **Extract features.** 58 features per window across five groups: heart-rate variability (time,
   frequency, nonlinear), electrodermal activity, temperature, respiration, and motion.
4. **Evaluate honestly.** 15-fold Leave-One-Subject-Out, with median imputation, standardization,
   and class balancing all fit inside each fold so nothing leaks from test to train.
5. **Check calibration.** Measure ECE/MCE/Brier on the pooled out-of-fold probabilities, compare to
   the within-subject baseline, and recalibrate inside each fold without touching the test subject.
6. **Serve.** The trained model is exported to run directly in the browser, so the live demo needs
   no backend.

## Tech stack

| Area | Tools |
| --- | --- |
| Modelling | scikit-learn, XGBoost, LightGBM, PyTorch |
| Signal processing | NeuroKit2, SciPy |
| Interpretability | SHAP |
| Service | FastAPI |
| Dashboard | React, TypeScript, Plotly |
| Tooling | Docker, GitHub Actions, ruff, pytest |

## Reproduce

```bash
pip install -e .
make demo         # full pipeline (features → LOSO → calibration → personalization) on synthetic data
make data         # PhysioNet Non-EEG for the cross-dataset transfer (downloads on run)
make reproduce    # regenerate every WESAD number, figure, and model — requires the WESAD download
```

From a clean clone, **`make demo` is what runs offline** — it exercises the entire pipeline on a
seeded synthetic generator, like the [Colab notebook](notebooks/CalmSense.ipynb). The headline WESAD
results in [`results/`](results/) are a **committed snapshot**; regenerating them needs the ~2 GB WESAD
dataset (see [data/raw/README.md](data/raw/README.md)), which is not redistributed. Everything is seeded.

## Limitations
- 15 subjects and lab-induced stress; every subject-level result is preliminary and underpowered, with
  wide confidence intervals. No real-world or clinical claim.
- Secondary analyses (ablation, calibration, personalization) are exploratory, not multiplicity-corrected.
- The 1D-CNN is a small-scale baseline, not a fair test of deep learning (no pretraining/transfer).
- Cross-corpus generalization rests on a single confounded transfer pair; ≥3 matched corpora are needed.

Methodology, statistics, and references are in the [paper](PAPER.md).

## Future scope

- [ ] A third corpus (SWELL / AffectiveROAD) for leave-one-dataset-out generalization
- [ ] Real-world, non-lab stress data beyond the 15-subject benchmark
- [ ] Real-time streaming inference from a live wearable

## Ethics & data use

Physiological signals are sensitive personal data, and CalmSense is a research benchmark, not a product:
it should support people, not surveil them. We endorse **data minimization** (collect and keep only what
an analysis needs) and a **no-surveillance** stance — stress inference must not be used to monitor or
penalize individuals without informed consent. Honest, subject-independent evaluation is itself part of
this: a model validated only within-subject overstates its reliability for the people it is meant to help.

The **code is MIT-licensed** ([LICENSE](LICENSE)), but the **datasets carry their own terms** — WESAD is
research-only under its provider's agreement and PhysioNet Non-EEG under the PhysioNet license; neither is
redistributed here (see [data/raw/README.md](data/raw/README.md)).

## Citation

If you use CalmSense, please cite it via [CITATION.cff](CITATION.cff). A versioned DOI is minted by
Zenodo on each tagged GitHub release. Once you enable Zenodo for the repo and publish the `v0.1.0`
release, paste the minted DOI into `CITATION.cff` and add the badge below to the top of this README:

<!-- [![DOI](https://zenodo.org/badge/1152348155.svg)](https://zenodo.org/badge/latestdoi/1152348155) -->

## License

MIT — [LICENSE](LICENSE).
