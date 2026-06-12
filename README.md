# CalmSense

Detecting stress from wearable sensors, and measuring how much of the field's reported accuracy
actually holds up on people the model has never seen.

Most published results on the WESAD benchmark claim 95–99% accuracy. CalmSense shows that much of
that vanishes under honest, subject-independent testing, and traces exactly where it goes.

[Live demo](https://urme-b.github.io/CalmSense/) · [Paper](PAPER.md)

## Why it matters

A stress model is only useful if it works on a new person. CalmSense evaluates everything with
Leave-One-Subject-Out validation (train on 14 people, test on the 15th) and quantifies three ways the
usual numbers get inflated: subject leakage, motion confounds, and dataset shift. The result is a
realistic picture of what wearable stress detection can and cannot do today.

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
number is honest: it was measured on people the model never trained on.

## Key findings

| Subject leakage | Motion is not the trick |
| :---: | :---: |
| ![Optimism gap](outputs/figures/multiclass_optimism_gap.png) | ![Ablation](outputs/figures/ablation.png) |
| Testing within the same people inflates three-class accuracy from 0.67 to 0.79. | Remove every motion feature and accuracy barely moves (0.913 to 0.901). The signal is physiological. |
| **A wrist band is enough** | **It does not transfer across datasets** |
| ![Chest vs wrist](outputs/figures/chest_vs_wrist.png) | ![Cross-dataset](outputs/figures/cross_dataset.png) |
| A wrist-only model reaches 0.893, about two points behind a research-grade chest strap. | Train on WESAD, test on a second dataset, and accuracy collapses to near chance. Within-dataset success is not generalization. |

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
5. **Serve.** The trained model is exported to run directly in the browser, so the live demo needs
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

## Honest limitations

- 15 subjects and lab-induced stress; per-subject accuracy ranges from 0.71 to 1.00.
- Hyperparameters are sensible defaults, not tuned.
- The deep model underperforms at this data scale and is kept only as a baseline.

Methodology, statistics, and references are in the [paper](PAPER.md).

## Future scope

- [ ] Validate across more datasets to test true cross-corpus generalization
- [ ] Nested hyperparameter tuning instead of fixed defaults
- [ ] Personalization to close the within- vs. cross-subject gap
- [ ] Real-world, non-lab stress data beyond the 15-subject benchmark
- [ ] Real-time streaming inference from a live wearable

## License

MIT — [LICENSE](LICENSE). Citation: [CITATION.cff](CITATION.cff).
