# Four Layers of Optimism in Wearable Stress Detection on WESAD: Subject Leakage, Motion, Dataset Shift, and Calibration

**Urme B** · [github.com/urme-b/CalmSense](https://github.com/urme-b/CalmSense)

## Abstract

Wearable stress-detection studies routinely report 95–99% accuracy on WESAD, but much of it comes
from evaluation that leaks information about the test set. This work measures how much performance
survives honest evaluation, peeling back four layers of optimism. **Subject leakage:** moving from
within-subject to Leave-One-Subject-Out (LOSO) cross-validation drops three-class accuracy from 0.79
to 0.67 and binary from 0.96 to 0.91; allowing overlapping windows inflates it further toward the
reported 0.95–0.99. **Motion confound:** an ablation shows accelerometer features alone reach 0.88,
yet removing motion entirely still gives 0.90, so the signal is autonomic, not just movement.
**Dataset shift:** a leakage-free model trained on WESAD falls to 0.50–0.57 balanced accuracy on a
second dataset (near chance). **Calibration:** accuracy is not the whole story for a model meant to
trigger alerts — we show the same within-subject evaluation that inflates accuracy also makes
probabilities look better calibrated than they are subject-independently, that this miscalibration
erodes net benefit at realistic alert thresholds, and that a leakage-free recalibration recovers
most of it. A wrist-only model reaches 0.89, about two points behind the chest sensor, and the four
feature-based models are statistically indistinguishable (Friedman p = 0.81). The contribution is a
reproducible account of what subject-independent stress detection actually delivers — in accuracy
*and* in the calibrated confidence a safe alerting system needs — not a new model.

## 1. Background

WESAD (Schmidt et al., 2018) is the field's standard stress benchmark, and much published work reports
near-perfect accuracy on it. The dominant cause is subject-dependent validation: when overlapping
windows are pooled across participants and split randomly, autocorrelated windows from one recording
appear in both train and test, and the model learns person-specific signatures rather than stress
(Bhanushali et al., 2021; Vos et al., 2023). One study reported 99.9% with a random split that did not
separate subjects (Oliver & Dakshit, 2025). Models also degrade sharply across corpora, with stressor
type driving most of the drop (Benchekroun et al., 2023; Prajod et al., 2024). Under proper LOSO the
original WESAD authors reported roughly 93% binary and 80% three-class — the realistic reference points.
Almost all of this work reports accuracy or F1 alone; the *calibration* of the predicted probabilities,
which is what an alerting system actually thresholds, is rarely examined under subject-independent evaluation.

## 2. Data

**WESAD** (Schmidt et al., 2018): 15 subjects, chest RespiBAN (700 Hz) and wrist Empatica E4
(BVP 64 Hz, EDA/TEMP 4 Hz, ACC 32 Hz). We classify baseline, stress (TSST), and amusement; meditation
is excluded. **PhysioNet Non-EEG** (Birjandtalab et al., 2016): 20 subjects, wrist EDA/temperature/ACC
and heart rate, used for cross-dataset transfer (psychological stress vs. relaxation).

## 3. Method

Signals are filtered, R-peaks detected (NeuroKit2) with ectopic correction, and EDA decomposed into
tonic and phasic components. Recordings are segmented into 60-second windows at 50% overlap, kept only
when at least 90% of a window shares one condition. The extractor emits 60 features; two respiration
features that need per-breath segmentation are always empty on this pipeline and dropped, leaving 58:
30 HRV (time/frequency/nonlinear; Task Force, 1996), 15 EDA, 5 temperature, 3 respiration, 5 motion.
Feature extraction never sees labels.

Models are logistic regression, random forest, XGBoost, LightGBM, and a compact 1D-CNN on raw windows.
All scoring is 15-fold LOSO; median imputation, standardization, and class balancing are fit inside
each fold. We report accuracy and macro-F1 (mean over subjects), bootstrap 95% CIs, a Friedman omnibus
test with Holm-corrected pairwise Wilcoxon tests, and a within-subject 5-fold baseline (non-overlapping
windows) for the optimism gap.

For calibration we pool the out-of-fold LOSO probabilities and report expected and maximum calibration
error (ECE, MCE; 15 confidence bins) and the Brier score (Guo et al., 2017), against the same
within-subject 5-fold baseline. Recalibration is leakage-free: inside each LOSO fold an isotonic
(and, for comparison, a sigmoid) map is fit only on out-of-fold probabilities of the *training*
subjects, then applied to the held-out subject — the test subject never touches calibrator fitting.
We close with a decision-curve analysis (Vickers & Elkin, 2006): net benefit across alert thresholds
for the uncalibrated and recalibrated models versus the alert-everyone and alert-no-one policies.

## 4. Results

### 4.1 Subject-independent benchmark

| Model               | Binary acc | Binary F1 | 3-class acc | 3-class F1 |
| ------------------- | :--------: | :-------: | :---------: | :--------: |
| Logistic Regression | 0.902      | 0.883     | **0.670**   | **0.613**  |
| **Random Forest**   | **0.913**  | **0.898** | 0.637       | 0.535      |
| XGBoost             | 0.903      | 0.873     | 0.633       | 0.552      |
| LightGBM            | 0.894      | 0.860     | 0.658       | 0.568      |
| 1D-CNN              | 0.718      | 0.648     | 0.626       | 0.543      |

Random Forest is nominally best on binary (0.913, 95% CI [0.860, 0.960]), but the four feature models
are statistically indistinguishable (Friedman p = 0.81; no significant Holm-corrected pair), so we
report the family rather than a winner. All beat the from-scratch 1D-CNN, which underperforms at this
data scale. Three-class accuracy is far lower (0.63–0.67), matching the original WESAD LOSO results.

### 4.2 Subject-leakage optimism gap

For the best model per task, replacing LOSO with within-subject 5-fold (non-overlapping windows,
pooled identically, so only subject mixing changes) raises binary accuracy from 0.913 to 0.964
(+5.1 pts) and three-class from 0.671 to 0.792 (+12.1 pts). This is conservative: studies that also
split overlapping windows at random push the within-subject figure toward the reported 0.95–0.99.

### 4.3 Motion-confound ablation

| Feature set                       | Binary acc (RF) |
| --------------------------------- | :-------------: |
| All features (58)                 | 0.913           |
| **No motion** (HRV+EDA+TEMP+RESP, 53) | **0.901**   |
| Autonomic (HRV+EDA, 45)           | 0.890           |
| EDA only (15)                     | 0.828           |
| HRV only (30)                     | 0.810           |
| Motion only (ACC, 5)              | 0.885           |

Motion alone reaches 0.885, plausibly because the stress task involves more movement than the seated
baseline. But removing motion entirely still gives 0.901, within 1.2 points of the full model: stress
is detectable from autonomic physiology alone.

### 4.4 Wrist-only deployability

Using only Empatica E4 wrist signals, a random forest reaches 0.893 versus 0.913 for the chest — a
2.0-point drop, within CI noise at N = 15. The best wrist model (XGBoost, 0.906) is within 0.7 points
of the best chest model. A research-grade chest strap is not required.

### 4.5 Cross-dataset generalization

|         | Within-dataset (bal. acc) | Cross-dataset (bal. acc) |
| ------- | :-----------------------: | :----------------------: |
| WESAD   | 0.86                      | → Non-EEG: **0.57**      |
| Non-EEG | 0.70                      | → WESAD: **0.50**        |

On an 18-feature device-agnostic space, within-dataset accuracy is healthy but cross-dataset transfer
collapses to near chance. The drop mixes genuine domain shift with differing label constructs and
reduced features, consistent with the cross-corpus literature (Vos et al., 2023; Benchekroun et al.,
2023; Prajod et al., 2024). Within-dataset success does not imply generalization.

### 4.6 Interpretability

SHAP values (Lundberg & Lee, 2017) rank a motion descriptor (`ACC_zero_crossings`) first, then
heart-rate level (`HRV_MedianNN`, `HRV_MeanNN`), skin-conductance responses, and respiration rate —
physiologically sensible for acute stress, and the motivation for the §4.3 ablation.

### 4.7 Calibration: a fourth layer of optimism

Accuracy says nothing about whether a predicted probability of 0.8 means roughly 80% of such windows
are truly stress — yet that calibrated confidence is exactly what an alerting system acts on. We measure
it on the pooled out-of-fold probabilities of the binary random forest. The pattern mirrors accuracy:
the within-subject baseline understates the expected calibration error because the model has already
seen each test subject's physiological baseline, so subject-independent deployment is less calibrated
than within-subject evaluation suggests. A leakage-free recalibration — an isotonic map fit only on
out-of-fold *training*-subject probabilities — recovers most of the gap without touching the held-out
subject.

The within-subject baseline uses the same non-overlapping 5-fold protocol as the accuracy optimism gap
(§4.2), so it reflects subject mixing rather than near-duplicate-window leakage.

> Numbers below regenerate into `results/calibration.json`; run `python scripts/calibration.py` to
> populate this table and the two figures.

| Evaluation                       | ECE | MCE | Brier |
| -------------------------------- | :-: | :-: | :---: |
| Within-subject 5-fold            |  —  |  —  |   —   |
| LOSO (subject-independent)       |  —  |  —  |   —   |
| LOSO + leak-free recalibration   |  —  |  —  |   —   |

![Reliability diagram](outputs/figures/calibration_reliability.png)

A decision-curve analysis turns this into deployment terms: across alert thresholds, net benefit for
the uncalibrated LOSO model trails the recalibrated one, and recalibration is what keeps the model
above the trivial alert-everyone and alert-no-one policies at clinically plausible thresholds.

![Decision curve](outputs/figures/calibration_decision_curve.png)

## 5. Limitations

- 15 subjects and lab-induced (TSST) stress; per-subject accuracy ranges 0.71–1.00. No claim to
  real-world or chronic stress.
- Cross-dataset transfer is confounded by differing label schemes; two datasets cannot separate domain
  shift from label mismatch.
- Hyperparameters are fixed defaults, not tuned; the deep model is a baseline, not a result.
- Calibration and decision-curve numbers are reported for the binary task on the chest random forest;
  isotonic recalibration can be unstable on small folds, so the sigmoid map is provided as a check.

## 6. Reproducibility

```bash
pip install -e .
make demo                 # full calibration pipeline on synthetic data, no download
make data                 # PhysioNet Non-EEG (WESAD: see data/raw/README.md)
make reproduce            # regenerates every number and figure into results/ and outputs/figures/
```

All randomness is seeded. A synthetic generator (`src/synthetic.py`) runs the entire pipeline without
the real data, so the code path is exercised in CI and in a one-click Colab notebook. WESAD and
PhysioNet Non-EEG are public and downloaded separately.

## Ethics

Both datasets are public, de-identified, and consented. Physiological stress inference is sensitive and
should support, not surveil; honest evaluation is itself an ethical requirement, since a model validated
only within-subject or within-dataset overstates its reliability for the people it is meant to help.

## References

- Schmidt, Reiss, Duerichen, Marberger, Van Laerhoven. *Introducing WESAD, a Multimodal Dataset for
  Wearable Stress and Affect Detection.* ICMI 2018.
- Birjandtalab, Cogan, Pouyan, Nourani. *A Non-EEG Dataset for Assessment of Neurological Status.*
  IEEE BHI / PhysioNet, 2016.
- Bhanushali et al. *Stress Classification and Personalization: Getting the Most out of the Least.*
  arXiv:2107.05666, 2021.
- Vos, Trinh, Sarnyai, Rahimi Azghadi. *Generalizable Machine Learning for Stress Monitoring from
  Wearable Devices: A Systematic Literature Review.* Int. J. Medical Informatics 173:105026, 2023
  (arXiv:2209.15137).
- Vos et al. *Ensemble Machine Learning Model Trained on a New Synthesized Dataset Generalizes Well for
  Stress Prediction Using Wearable Devices.* J. Biomedical Informatics, 2023 (arXiv:2209.15146).
- Benchekroun et al. *Cross Dataset Analysis for Generalizability of HRV-Based Stress Detection Models.*
  Sensors 23(4):1807, 2023.
- Prajod, Mahesh, André. *Stressor Type Matters! Exploring Factors Influencing Cross-Dataset
  Generalizability of Physiological Stress Detection.* ICMI Companion 2024 (arXiv:2405.09563).
- Oliver, Dakshit. *Cross-Modality Investigation on WESAD Stress Classification.* arXiv:2502.18733, 2025.
- Task Force of the ESC/NASPE. *Heart Rate Variability: Standards of Measurement, Physiological
  Interpretation, and Clinical Use.* Circulation, 1996.
- Lundberg, Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
- Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
- Vickers, Elkin. *Decision Curve Analysis: A Novel Method for Evaluating Prediction Models.*
  Medical Decision Making, 2006.
