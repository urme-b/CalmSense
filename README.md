# CalmSense

> Wearable stress detection scored on people it has never seen, so the accuracy you read is the accuracy you get.

[Live demo](https://urme-b.github.io/CalmSense/) · [Colab](https://colab.research.google.com/github/urme-b/CalmSense/blob/main/notebooks/CalmSense.ipynb) · [Paper](PAPER.md)

## What this is

- Detects `stress` vs `baseline` from wearable signals: ECG, EDA (skin conductance), temperature, respiration, motion.
- Scored Leave-One-Subject-Out (LOSO): train on 14 people, test on the 15th, rotate.
- Shows where the usual high numbers come from: subject leakage, motion, dataset shift, calibration.
- Runs in the browser (ONNX, no backend). `make demo` runs the full pipeline offline on synthetic signals.

## Results

Binary (`baseline` vs `stress`), 15 subjects, LOSO, mean over held-out subjects.

<table width="100%">
<tr><th align="left">Model</th><th align="left">Accuracy</th><th align="left">F1 (macro)</th></tr>
<tr><td>Random Forest</td><td>0.913</td><td>0.898</td></tr>
<tr><td>XGBoost</td><td>0.903</td><td>0.873</td></tr>
<tr><td>Logistic Regression</td><td>0.902</td><td>0.883</td></tr>
<tr><td>LightGBM</td><td>0.894</td><td>0.860</td></tr>
<tr><td>1D-CNN (raw signal)</td><td>0.718</td><td>0.648</td></tr>
</table>

- The 4 feature models are a statistical tie (Friedman p = 0.81). RF 95% CI: [0.860, 0.960].

Key findings, one per check:

<table width="100%">
<tr><th align="left">Check</th><th align="left">Question</th><th align="left">Result</th></tr>
<tr><td>Subject leakage</td><td>Does same-person testing inflate scores?</td><td>3-class 0.66 to 0.79 (+13 pts)</td></tr>
<tr><td>Motion confound</td><td>Is it just movement?</td><td>Drop all motion: 0.913 to 0.901</td></tr>
<tr><td>Wrist vs chest</td><td>Is a cheap sensor enough?</td><td>0.893 vs 0.913 (2 pts lower)</td></tr>
<tr><td>Dataset shift</td><td>Does it transfer to another dataset?</td><td>Near chance (0.50 balanced)</td></tr>
<tr><td>Calibration</td><td>Are the probabilities trustworthy?</td><td>ECE 0.070; isotonic map to 0.025</td></tr>
<tr><td>Personalization</td><td>Does a short enrollment help?</td><td>20 windows: ECE 0.146 to 0.069</td></tr>
</table>

## Models

<table width="100%">
<tr><th align="left">Model</th><th align="left">Type</th><th align="left">Key settings</th></tr>
<tr><td>Logistic Regression</td><td>Linear</td><td>C=1.0, L2, class-balanced</td></tr>
<tr><td>Random Forest</td><td>Bagged trees</td><td>200 trees, depth 10, class-balanced</td></tr>
<tr><td>XGBoost</td><td>Boosted trees</td><td>200 trees, depth 7, lr 0.1</td></tr>
<tr><td>LightGBM</td><td>Boosted trees</td><td>200 trees, 50 leaves, lr 0.1</td></tr>
<tr><td>1D-CNN</td><td>Deep net on raw signal</td><td>Residual blocks, AdamW, early stopping</td></tr>
</table>

- Every model runs inside `impute (median) to scale to classifier`, fit per fold, seeded.

## Features (58)

<table width="100%">
<tr><th align="left">Group</th><th align="left">Count</th><th align="left">Examples</th></tr>
<tr><td>HRV time domain</td><td>12</td><td>MeanNN, SDNN, RMSSD, pNN50</td></tr>
<tr><td>HRV frequency</td><td>8</td><td>LF/HF power, LF/HF ratio</td></tr>
<tr><td>HRV nonlinear</td><td>10</td><td>SampEn, DFA, SD1/SD2, CSI</td></tr>
<tr><td>EDA (skin conductance)</td><td>15</td><td>SCL level, SCR count, SCR amplitude</td></tr>
<tr><td>Temperature + respiration</td><td>8</td><td>temp slope, respiration rate</td></tr>
<tr><td>Accelerometer (motion)</td><td>5</td><td>magnitude mean, std, energy</td></tr>
</table>

## Graphs & charts

<table>
<tr>
<td align="center"><img src="outputs/figures/binary_model_comparison.png" width="260" alt="Model comparison"><br>Model comparison (LOSO)</td>
<td align="center"><img src="outputs/figures/binary_optimism_gap.png" width="260" alt="Optimism gap"><br>Optimism gap (leakage)</td>
<td align="center"><img src="outputs/figures/ablation.png" width="260" alt="Ablation"><br>Feature ablation</td>
</tr>
<tr>
<td align="center"><img src="outputs/figures/chest_vs_wrist.png" width="260" alt="Wrist vs chest"><br>Wrist vs chest</td>
<td align="center"><img src="outputs/figures/cross_dataset.png" width="260" alt="Cross-dataset"><br>Cross-dataset transfer</td>
<td align="center"><img src="outputs/figures/calibration_reliability.png" width="260" alt="Reliability"><br>Calibration reliability</td>
</tr>
<tr>
<td align="center"><img src="outputs/figures/personalization.png" width="260" alt="Personalization"><br>Few-shot personalization</td>
<td align="center"><img src="outputs/figures/shap_beeswarm.png" width="260" alt="SHAP"><br>Top features (SHAP)</td>
<td align="center"><img src="outputs/figures/binary_confusion.png" width="260" alt="Confusion"><br>Confusion matrix</td>
</tr>
</table>

## Tech stack

<table width="100%">
<tr><th align="left">Area</th><th align="left">Tools</th></tr>
<tr><td>Modelling</td><td>scikit-learn, XGBoost, LightGBM, PyTorch</td></tr>
<tr><td>Signal processing</td><td>NeuroKit2, SciPy</td></tr>
<tr><td>Explainability</td><td>SHAP</td></tr>
<tr><td>Dashboard</td><td>React, TypeScript, ONNX Runtime Web</td></tr>
<tr><td>Tooling</td><td>GitHub Actions, ruff, mypy, pytest</td></tr>
</table>

## Limitations

- 15 subjects, lab-induced stress. Underpowered, wide CIs. No clinical claim.
- Ablation, calibration, and personalization are exploratory, not multiplicity-corrected.
- The 1D-CNN is a small baseline, not a fair test of deep learning.
- Cross-dataset uses one confounded pair. Illustrative, not conclusive.

## Future work

- A third corpus (SWELL / AffectiveROAD) for leave-one-dataset-out generalization.
- Real-world, non-lab stress data beyond the 15-subject benchmark.
- Real-time streaming inference from a live wearable.

## Ethics & data use

- Physiological signals are sensitive personal data.
- This is a research benchmark, not a product.
- Data minimization: collect and keep only what an analysis needs.
- No surveillance: do not monitor or penalize people without informed consent.
- Datasets keep their own licenses and are not redistributed here.

## License

[MIT License](LICENSE)
