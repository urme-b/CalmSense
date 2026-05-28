# Subject-Independent Stress Detection from Wearable Physiology: A Leakage-Free Benchmark on WESAD

**Urme B** · [github.com/urme-b/CalmSense](https://github.com/urme-b/CalmSense)

## Abstract

Wearable stress-detection studies frequently report accuracies above 95%, but many of these
numbers come from evaluation protocols that let windows from the same person appear in both
training and test sets. Under such within-subject validation a model can recognise *who* a
window belongs to rather than *whether* the person is stressed, and the reported accuracy does
not transfer to new users. CalmSense re-examines stress detection on the WESAD dataset under a
strict Leave-One-Subject-Out (LOSO) protocol, where every test subject is entirely unseen during
training. We extract 60+ physiologically grounded biomarkers from chest-worn ECG, electrodermal
activity, temperature, respiration, and accelerometer signals, and benchmark four classical
models (logistic regression, random forest, XGBoost, LightGBM) against a residual 1D-CNN trained
on the raw windows. We report results for both binary (stress vs. non-stress) and three-class
(baseline / stress / amusement) tasks, quantify the optimism introduced by within-subject
validation, and interpret the models with SHAP. The contribution is not a new architecture but a
careful, fully reproducible, leakage-free baseline together with an honest account of what
subject-independent stress detection actually achieves.

## 1. Introduction

Affective computing on wearables promises continuous, unobtrusive stress monitoring. The WESAD
dataset (Schmidt et al., 2018) has become a standard benchmark, and a large body of work reports
near-perfect accuracy on it. A recurring methodological problem undermines many of these results:
**data leakage through subject-dependent validation.** When 60-second windows are pooled across
all subjects and split randomly (or by k-fold), windows from a single recording — which are
highly autocorrelated and carry person-specific physiological signatures — land in both the
training and test folds. The classifier can then exploit individual identity, and accuracy is
inflated relative to the realistic deployment setting where the model meets a *new* user.

This project takes the opposite stance. Every number we report comes from Leave-One-Subject-Out
cross-validation: the model is trained on 14 subjects and tested on the 15th, repeated for all
folds. We make three contributions:

1. A clean, end-to-end, reproducible pipeline from raw WESAD signals to LOSO metrics, runnable
   with a single command.
2. A like-for-like comparison of classical feature-based models and a raw-signal 1D-CNN under
   identical, leakage-free conditions.
3. A direct measurement of the **optimism gap** — the difference between within-subject k-fold
   accuracy and subject-independent LOSO accuracy — on the same features and model, making the
   cost of the common shortcut explicit.

## 2. Dataset

WESAD (Schmidt et al., 2018) contains synchronised physiological recordings from 15 healthy
participants (subjects S2–S17; S1 and S12 are excluded by the dataset authors) wearing a
chest-mounted RespiBAN and a wrist-worn Empatica E4. Participants moved through four conditions:
a neutral **baseline**, **stress** induced by the Trier Social Stress Test (public speaking plus
mental arithmetic under evaluation), **amusement** from funny video clips, and a guided
**meditation** recovery. We use the chest signals, sampled at 700 Hz: ECG, electrodermal
activity (EDA), body temperature, respiration, and three-axis acceleration. Following common
practice we classify the baseline, stress, and amusement conditions; meditation is excluded from
the labelled tasks.

## 3. Signal processing and features

Each subject's continuous recording is processed once, then segmented into 60-second windows with
50% overlap. A window is assigned a label only if at least 90% of its samples share a single
condition; mixed-condition windows at transitions are discarded.

- **ECG → HRV.** The ECG is band-pass filtered (0.5–40 Hz, 4th-order Butterworth); R-peaks are
  detected with NeuroKit2; RR intervals are formed and ectopic beats (>20% deviation from the
  local median) are removed before HRV computation. We extract time-domain (e.g. RMSSD, SDNN,
  pNN50), frequency-domain (LF, HF, LF/HF via Welch PSD on the interpolated tachogram), and
  nonlinear (sample entropy, Poincaré SD1/SD2, CSI/CVI) features following Task Force (1996)
  conventions.
- **EDA.** Low-pass filtered and median-cleaned, then decomposed into tonic (skin conductance
  level) and phasic (skin conductance response) components. We extract SCL statistics and SCR
  count, rate, amplitude, and rise-time features.
- **Temperature, respiration, accelerometer.** Filtered and summarised by statistical and
  rate-based descriptors (slope, variability, breathing rate, motion magnitude/energy).

This yields 60+ features per window. Feature extraction never sees the labels, and no
normalisation is applied before cross-validation splitting.

## 4. Models and evaluation

**Classical.** Logistic regression, random forest, XGBoost, and LightGBM operate on the feature
matrix. Within every LOSO fold we fit a median imputer and a standard scaler **on the training
subjects only** and apply them to the held-out subject, so no test-set statistics leak into
preprocessing. Class imbalance is handled with balanced class weights (balanced sample weights for
XGBoost).

**Deep learning.** A compact residual 1D-CNN consumes the five raw chest channels resampled to a
fixed length per window. Per-channel standardisation statistics are estimated on the training fold
only; training uses AdamW, cosine-annealed learning rate, class-weighted cross-entropy, and early
stopping on an internal validation split.

**Protocol.** All headline numbers use Leave-One-Subject-Out cross-validation (15 folds). We
report accuracy and macro-F1 averaged across folds, balanced accuracy on pooled predictions,
per-subject performance, and normalised confusion matrices. For the best model we additionally run
within-subject 5-fold cross-validation to quantify the optimism gap.

## 5. Results

<!-- RESULTS:BEGIN -->
**Binary (stress vs. non-stress, 869 windows).** Random forest achieves the best subject-independent
performance at **0.912 ± 0.095** accuracy (macro-F1 0.898, balanced accuracy 0.901). XGBoost (0.888),
LightGBM (0.875) and logistic regression (0.868) follow closely. The 1D-CNN reaches only 0.647 with a
balanced accuracy of 0.50 — i.e. it essentially predicts the majority class.

**Three-class (baseline / stress / amusement, 1032 windows).** Accuracy drops sharply once amusement is
added: LightGBM leads at **0.650** (macro-F1 0.566), with logistic regression close behind on macro-F1
(0.588). The CNN again collapses toward chance-level balanced accuracy (0.33).

| Model | Binary acc | Binary F1 | 3-class acc | 3-class F1 |
|-------|:----------:|:---------:|:-----------:|:----------:|
| Logistic Regression | 0.868 | 0.853 | 0.647 | 0.588 |
| Random Forest | **0.912** | **0.898** | 0.629 | 0.519 |
| XGBoost | 0.888 | 0.858 | 0.627 | 0.545 |
| LightGBM | 0.875 | 0.838 | **0.650** | 0.566 |
| 1D-CNN (raw) | 0.647 | 0.393 | 0.545 | 0.235 |

*Table 1. Leave-One-Subject-Out results (15 folds). Best per column in bold.*

**The optimism gap.** On the same best model and features, replacing LOSO with within-subject 5-fold
cross-validation raises accuracy from 0.912 to 0.978 on the binary task (+6.6 points) and from 0.650 to
0.987 on the three-class task (**+33.7 points**). The three-class figure is the crux: a protocol that
leaks subject identity turns a 65%-accurate model into an apparent 99%-accurate one. Reported "success"
on WESAD is, in large part, this artefact.

**Two observations.** First, classical feature-based models clearly outperform the from-scratch 1D-CNN
under LOSO; with ~1000 windows the CNN cannot learn subject-invariant structure and defaults to the
majority class. Engineered HRV/EDA/motion features encode the relevant physiology far more
data-efficiently. Second, per-subject accuracy ranges from 0.73 to 1.00 (binary), underscoring that a
single pooled number hides large between-person variation — which is exactly why the protocol matters.
<!-- RESULTS:END -->

## 6. Interpretability

We compute SHAP values for the best tree-based model on the binary task. The ranking of mean
absolute SHAP contributions identifies which biomarkers drive stress predictions, and these are
cross-checked against published physiological reference ranges (e.g. reduced RMSSD and elevated
LF/HF under stress; rising skin conductance level). The top contributors and the SHAP summary
plot are saved to `results/shap_top_features.csv` and `outputs/figures/shap_beeswarm.png`.

## 7. Discussion and limitations

- **Sample size.** WESAD has 15 subjects; per-subject accuracy varies substantially, and LOSO
  means are reported with their across-subject standard deviation rather than as a single point.
- **Lab-induced stress.** The TSST elicits acute stress in a controlled setting; generalisation to
  real-world, chronic, or low-grade stress is not established here.
- **Chest sensor.** We use the RespiBAN chest signals, which give cleaner ECG than the wrist E4;
  a wrist-only model would be more deployable but is expected to perform worse.
- **No deep tuning.** Models use sensible fixed hyperparameters rather than exhaustive search; the
  goal is an honest, reproducible baseline, not a leaderboard score.

## 8. Reproducibility

The full pipeline is one command:

```bash
pip install -e .
python scripts/run_experiment.py        # downloads/uses WESAD, writes results/ and outputs/figures/
```

All randomness is seeded. Metrics are written to `results/metrics.json` and per-model CSVs; every
figure in the paper is regenerated by the script.

## Ethics

WESAD is a public, de-identified research dataset collected with participant consent. Stress
detection from physiology is sensitive: such models should support, not surveil, and should never
be used for employment, insurance, or other high-stakes decisions about individuals. The
subject-independence emphasised here is also an honesty requirement — deploying a model validated
only within-subject would overstate its reliability for new users.

## References

- P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, K. Van Laerhoven. *Introducing WESAD, a
  Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and
  Electrophysiology. *Heart Rate Variability: Standards of Measurement, Physiological
  Interpretation, and Clinical Use.* Circulation, 1996.
- S. M. Lundberg, S.-I. Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
