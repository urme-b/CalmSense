# CalmSense

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
number is honest: it was measured on people the model never trained on.

## Key findings

| Subject leakage | Motion is not the trick |
| :---: | :---: |
| ![Optimism gap](outputs/figures/multiclass_optimism_gap.png) | ![Ablation](outputs/figures/ablation.png) |
| Testing within the same people inflates three-class accuracy from 0.67 to 0.79. | Remove every motion feature and accuracy barely moves (0.913 to 0.901). The signal is physiological. |
| **A wrist band is enough** | **It does not transfer across datasets** |
| ![Chest vs wrist](outputs/figures/chest_vs_wrist.png) | ![Cross-dataset](outputs/figures/cross_dataset.png) |
| A wrist-only model reaches 0.893, about two points behind a research-grade chest strap. | Train on WESAD, test on a second dataset, and accuracy collapses to near chance. Within-dataset success is not generalization. |

### Accuracy isn't the whole story: calibration

An alerting system acts on the probability, not the label. The same within-subject evaluation that
inflates accuracy also makes the model look better calibrated than it is on a new person, and that
miscalibration costs real net benefit at deployment thresholds. CalmSense measures this (ECE, MCE,
Brier, reliability diagrams, decision-curve analysis), tests the gap for significance (paired Wilcoxon
on per-subject Brier), and applies a leakage-free recalibration that recovers most of the gap — fit
only on held-out training subjects, never on the test subject. A **few-shot personalization** step then
closes the rest: a short labeled enrollment from the target subject beats global recalibration with no
retraining. All models are tuned by nested CV (inner grouped search, outer LOSO), so the comparison is
fair. See [paper §4.7–4.8](PAPER.md). Regenerate with `make reproduce`, or try it with no data via `make demo`.

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
make demo         # full calibration pipeline on synthetic data — no download
make data         # PhysioNet Non-EEG (WESAD: see data/raw/README.md)
make reproduce    # regenerate every number, figure, model, and the dashboard
```

`make demo` and the [Colab notebook](notebooks/CalmSense.ipynb) run the entire pipeline on a built-in
synthetic generator, so nothing is gated on the 2 GB WESAD download. Everything is seeded.

## Limitations
- 15 subjects and lab-induced stress; per-subject accuracy ranges from 0.71 to 1.00.
- The deep model underperforms at this data scale and is kept only as a baseline.
- Cross-corpus generalization rests on a single transfer pair; a third dataset is needed.

Methodology, statistics, and references are in the [paper](PAPER.md).

## Future scope

- [ ] A third corpus (SWELL / AffectiveROAD) for leave-one-dataset-out generalization
- [ ] Real-world, non-lab stress data beyond the 15-subject benchmark
- [ ] Real-time streaming inference from a live wearable

## License

MIT — [LICENSE](LICENSE). Citation: [CITATION.cff](CITATION.cff).
