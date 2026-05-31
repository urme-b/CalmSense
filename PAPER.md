# How Far Does Wearable Stress Detection Generalize? Three Layers of Optimism on WESAD

**Urme B** · [github.com/urme-b/CalmSense](https://github.com/urme-b/CalmSense)

## Abstract

Wearable stress-detection studies routinely report accuracies of 95–99%, yet these numbers are
frequently produced by evaluation protocols that leak information about the test set. CalmSense
peels back three layers of this optimism on the WESAD dataset and quantifies how much performance
survives each one. **(1) Subject leakage:** moving from within-subject k-fold to strict
Leave-One-Subject-Out (LOSO) cross-validation drops three-class accuracy from 0.99 to 0.65 — the
inflation that recurs across the literature. **(2) Motion confound:** a feature-group ablation shows
that accelerometer features alone reach 0.84, but removing motion entirely still yields 0.90 — so
autonomic physiology alone is sufficient for strong performance, though motion features are
independently informative. **(3) Dataset shift:** even a leakage-free, subject-independent model does
not transfer across these corpora — in this WESAD↔Non-EEG pairing (different stressors, devices, and
label schemes, on a reduced shared 18-feature space), balanced accuracy collapses to 0.50–0.57, near
chance. We also show that a wrist-only model using only consumer-wearable signals (Empatica E4: EDA,
BVP, temperature, accelerometer) reaches 0.89–0.91 binary accuracy, within ~2 points of the
research-grade chest sensor (0.912). The contribution is not a new architecture but a careful,
fully reproducible accounting of what subject-independent wearable stress detection actually
delivers — and a warning that within-dataset success does not imply real-world generalization.

## 1. Introduction

Affective computing on wearables promises continuous, unobtrusive stress monitoring, with obvious
relevance to mental-health care. WESAD (Schmidt et al., 2018) is the field's standard benchmark, and
a large body of work reports near-perfect accuracy on it. Much of that accuracy is an artefact of
evaluation. The clearest failure mode is **subject-dependent validation**: when 60-second windows
are pooled across all participants and split randomly (or by k-fold), highly autocorrelated windows
from one recording land in both train and test, and the model exploits person-specific physiological
signatures rather than stress itself (Bhanushali et al., 2021; Vos et al., 2022). A recent example
obtained 99.9% on WESAD with a random 85:15 split that did not separate subjects (Oliver & Dakshit,
2025).

This paper takes the opposite stance and reports only what survives honest evaluation. Our
contributions:

1. A clean, reproducible, leakage-free **LOSO benchmark** on WESAD (classical models and a 1D-CNN),
   with paired significance tests and bootstrap confidence intervals.
2. A direct measurement of the **subject-leakage optimism gap** — within-subject k-fold vs LOSO,
   measured per task on the best model for that task.
3. A **motion-confound ablation** that separates genuine autonomic physiology from accelerometer
   artefacts correlated with the task protocol — a confound single-split benchmarks never control.
4. A **wrist-only, deployment-realistic model** using only Empatica E4 signals, quantifying the cost
   of dropping the chest sensor.
5. A **cross-dataset generalization** study (WESAD ↔ PhysioNet Non-EEG) on a shared feature space,
   characterizing the generalization ceiling rather than hiding it.

## 2. Related Work

**WESAD and subject-independent evaluation.** WESAD (Schmidt et al., 2018) comprises 15 subjects
recorded with a chest RespiBAN and a wrist Empatica E4 across baseline, amusement, TSST stress, and
meditation. Under LOSO the original authors reported up to ~93% for binary stress-vs-non-stress and
~80% for the three-class problem; these remain the realistic reference points. Subsequent work frames
strong LOSO results explicitly as estimates of performance on an unseen participant, in contrast to
within-subject splits that overfit (Bhanushali et al., 2021).

**The leakage problem.** A persistent issue is that many studies report 95–99.9% accuracy using
random k-fold over windows without separating subjects; because overlapping windows from one person
appear in both train and test, the reported accuracy is optimistically biased (Vos et al., 2022;
Oliver & Dakshit, 2025). Subject-dependent accuracy near 95% commonly falls to ~67% under
subject-independent evaluation (Li et al., 2024, as surveyed by Vos et al., 2022). Systematic reviews
explicitly recommend subject-disjoint validation.

**Cross-dataset generalization.** Models that perform well within a corpus degrade sharply across
corpora, often below 50% F1 when trained on one dataset and tested on another (Vos et al., 2023;
Benchekroun et al., 2023). The dominant driver of the drop is the stressor/elicitation type rather
than device or population (Prajod et al., 2024), and random forests tend to transfer most stably
(Benchekroun et al., 2023). Our cross-dataset results are consistent with all three findings.

## 3. Datasets

**WESAD** (Schmidt et al., 2018): 15 healthy subjects (S2–S17), chest RespiBAN + wrist Empatica E4,
four conditions. We classify baseline, stress (TSST), and amusement; meditation is excluded. Chest
signals are sampled at 700 Hz; wrist E4 at BVP 64 Hz, EDA/TEMP 4 Hz, ACC 32 Hz.

**PhysioNet Non-EEG** (Birjandtalab et al., 2016): 20 subjects, wrist EDA/temperature/3-axis ACC
(8 Hz) and heart rate (1 Hz), with annotated relaxation and physical/cognitive/emotional stress
blocks. For cross-dataset transfer we use psychological stress (cognitive + emotional) vs relaxation,
excluding the motion-heavy physical block.

## 4. Signal processing and features

Each recording is processed once, then segmented into 60-second windows with 50% overlap; a window
keeps its label only if ≥90% of its samples share one condition (transition windows are dropped).
From the chest signals we extract 60+ features: HRV time/frequency/nonlinear (RMSSD, SDNN, LF/HF,
sample entropy, Poincaré SD1/SD2; Task Force, 1996) from Pan–Tompkins/NeuroKit2 R-peaks after ectopic
correction; EDA tonic (SCL) and phasic (SCR) features from a decomposition; and temperature,
respiration, and accelerometer descriptors. Feature extraction never sees the labels, and no
normalization precedes the cross-validation split. For wrist and cross-dataset experiments we compute
matched feature sets from the corresponding lower-rate signals (HRV from BVP peaks for the wrist; an
18-feature device-agnostic EDA+TEMP+ACC+HR set shared between WESAD and Non-EEG).

## 5. Models and evaluation

**Models.** Logistic regression, random forest, XGBoost, and LightGBM on the feature matrix; a compact
residual 1D-CNN on the raw windows. Within every LOSO fold, a median imputer and standard scaler are
fit **on the training subjects only**; class imbalance is handled with balanced weights inside each
fold. **Protocol.** All headline numbers use LOSO (15 folds for WESAD). We report accuracy and
macro-F1 (mean ± across-subject SD), balanced accuracy on pooled predictions, bootstrap 95% CIs, and
Wilcoxon signed-rank tests across the per-subject scores. For the best model we also run within-subject
5-fold CV to measure the optimism gap.

## 6. Results

### 6.1 Subject-independent benchmark

| Model | Binary acc | Binary F1 | 3-class acc | 3-class F1 |
|-------|:----------:|:---------:|:-----------:|:----------:|
| Logistic Regression | 0.868 | 0.853 | 0.647 | 0.588 |
| **Random Forest** | **0.912** | **0.898** | 0.629 | 0.519 |
| XGBoost | 0.888 | 0.858 | 0.627 | 0.545 |
| LightGBM | 0.875 | 0.838 | 0.650 | 0.566 |
| 1D-CNN (raw) | 0.647 | 0.393 | 0.545 | 0.235 |

*Table 1. LOSO results (15 folds). Bold marks the best binary model.* On the binary task Random Forest
achieves **0.912 (95% CI [0.863, 0.956])** and outperforms the other feature-based models on
per-subject accuracy (Wilcoxon signed-rank: vs logistic regression p=0.033, vs XGBoost p=0.043, vs
LightGBM p=0.041); these p-values are marginal and uncorrected (none survive Bonferroni correction),
so the ranking among the feature models is suggestive rather than decisive. The feature-based models do,
however, clearly beat the from-scratch 1D-CNN, which with ~1000 windows collapses toward the majority
class (balanced accuracy 0.50). On the three-class task the feature models are statistically
indistinguishable (top accuracies 0.629–0.650 with across-subject SD ≈ 0.16–0.21, no significant
pairwise differences), and overall accuracy is markedly lower — consistent with the original WESAD LOSO
numbers. Random Forest is the best binary model but not the best three-class model.

### 6.2 The subject-leakage optimism gap

For the best model on each task (random forest for binary, LightGBM for three-class), replacing LOSO
with within-subject 5-fold CV inflates accuracy from 0.912 to 0.978 on binary (+6.6 points) and from
0.650 to **0.987 on three-class (+33.7 points)**. The three-class figure is the crux: a protocol that
leaks subject identity turns a 65%-accurate model into an apparent 99%-accurate one — exactly the
inflation that recurs in the WESAD literature.

### 6.3 Motion-confound ablation

| Feature set | Binary acc (RF) |
|-------------|:---------------:|
| All features (54) | 0.912 |
| **No motion** (HRV+EDA+TEMP+RESP, 50) | **0.903** |
| Autonomic (HRV+EDA, 45) | 0.884 |
| HRV only (30) | 0.813 |
| EDA only (15) | 0.828 |
| Motion only (ACC, 4) | 0.844 |

*Table 2. Feature-group ablation.* Accelerometer features alone reach 0.844 — motion is genuinely
informative, plausibly because the stress condition (public speaking/arithmetic while standing) differs
in movement from seated baseline, which would also explain why the top SHAP feature is a motion
descriptor (we do not measure posture/activity directly, so this is a plausible rather than confirmed
mechanism). **Crucially, removing motion entirely still yields 0.903, within 0.9 points of the full
model.** Stress is therefore detectable from autonomic physiology alone; the headline result does not
depend on motion, even though motion is independently predictive.

### 6.4 Wrist-only deployability

Using only consumer-wearable Empatica E4 signals (BVP→HRV, EDA, temperature, accelerometer), the
wrist matches the chest closely. For a like-for-like comparison (random forest on both), wrist accuracy
is **0.891 vs the chest's 0.912 — a 2.1-point drop** (within the 95% CI noise at N=15). The best wrist
model (XGBoost, 0.908) comes within 0.4 points of the best chest model, though the two "best" models
differ. Either way, subject-independent stress detection does not require a research-grade chest sensor;
a wrist wearable suffices, which matters for any real deployment.

### 6.5 Cross-dataset generalization

| | Within-dataset LOSO (bal. acc) | Cross-dataset (bal. acc) |
|---|:---:|:---:|
| WESAD | 0.86 | → Non-EEG: **0.57** |
| Non-EEG | 0.70 | → WESAD: **0.50 (chance)** |

*Table 3. Shared 18-feature space; Random Forest.* While within-dataset subject-independent accuracy
is healthy (0.70–0.86 balanced), **cross-dataset transfer collapses to near chance** (0.50–0.57) in
this pairing. Two caveats matter for interpretation: the label constructs differ (WESAD
baseline-vs-TSST; Non-EEG relaxation-vs-cognitive/emotional), and the shared space is reduced to 18
device-agnostic features (no HRV, since Non-EEG provides only 1 Hz heart rate). The collapse therefore
reflects a combination of genuine domain shift (different stressors, devices, populations) and
imperfect label/feature harmonization — we cannot cleanly separate the two with two datasets. With that
caveat, the result is consistent with the cross-corpus literature, where transfer commonly falls below
50% F1 and stressor type dominates the drop (Vos et al., 2023; Benchekroun et al., 2023; Prajod et al.,
2024). It is the third and largest layer of optimism: even honest within-dataset evaluation can
overstate generalization to a new dataset.

## 7. Interpretability

SHAP values for the best tree model rank the most influential features as a motion descriptor
(`ACC_zero_crossings`), followed by heart-rate level (`HRV_MedianNN`, `HRV_MeanNN`) and
skin-conductance responses (`EDA_SCR_*`) — physiologically sensible for acute stress (vagal withdrawal
raises heart rate; sympathetic arousal drives EDA). The prominence of the motion feature is exactly
what motivated the ablation in §6.3, which shows the physiological signal stands on its own once motion
is removed.

## 8. From acute stress to mental-health-crisis early warning

This section is forward-looking motivation, not a tested result: this work uses only acute,
lab-induced stress in 15 healthy subjects, and makes no clinical or longitudinal claim. With that
stated, the connection is plausible. Acute stress detection is a building block, not an endpoint, for
crisis prediction; the same wrist-measurable autonomic signatures that mark an acute stress response —
vagal withdrawal and reduced HRV, elevated electrodermal activity, distal skin cooling — are
hypothesized to relate to the sustained autonomic dysregulation associated with some mental-health
crises, a link that would need clinical longitudinal data to establish. What this work *does* establish
are prerequisites for any such early-warning system: a benchmark inflated by within-subject leakage
will not survive deployment on a new individual, and a model that does not transfer across datasets
will not transfer to a new clinic, device, or population. Honest, subject-independent, wrist-only,
cross-dataset-aware evaluation is therefore the methodological groundwork on which forecasting
mental-health crises from continuous wearable data would have to be built.

## 9. Discussion and limitations

- **Small N, lab-induced stress.** WESAD has 15 subjects and uses the TSST; per-subject LOSO accuracy
  ranges 0.73–1.00. Generalization to real-world or chronic stress is not established.
- **Cross-dataset asymmetry.** Non-EEG→WESAD transfer is at chance while WESAD→Non-EEG is slightly
  above; the shared 18-feature space and differing stressors/label schemes limit what transfer can
  achieve. We report this as-is rather than tuning it away.
- **No exhaustive tuning.** Models use fixed, sensible hyperparameters; the goal is an honest baseline,
  not a leaderboard score.
- **Deep learning underperforms** at this data scale; we report it rather than omit it.

## 10. Reproducibility

```bash
pip install -e .
python scripts/run_experiment.py     # LOSO benchmark, optimism gap, SHAP, model
python scripts/ablation.py           # motion-confound ablation
python scripts/wrist.py              # wrist-only model
python scripts/stats.py              # significance tests + CIs
python scripts/cross_dataset.py      # WESAD <-> Non-EEG transfer
```

All randomness is seeded; every number and figure in this paper is regenerated by these scripts into
`results/` and `outputs/figures/`. WESAD downloads on first run; the Non-EEG dataset is public on
PhysioNet.

## Ethics

WESAD and Non-EEG are public, de-identified research datasets collected with consent. Stress and
mental-health inference from physiology is sensitive; such models should support, not surveil, and
must never drive employment, insurance, or other high-stakes decisions about individuals. The
emphasis on honest evaluation here is itself an ethical requirement: deploying a model validated only
within-subject or within-dataset would overstate its reliability for the very people it is meant to
help.

## References

- P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, K. Van Laerhoven. *Introducing WESAD, a Multimodal
  Dataset for Wearable Stress and Affect Detection.* ICMI 2018.
- J. Birjandtalab, D. Cogan, M. B. Pouyan, M. Nourani. *A Non-EEG Dataset for Assessment of
  Neurological Status.* IEEE BHI / PhysioNet, 2016.
- S. Bhanushali et al. *Stress Classification and Personalization: Getting the Most out of the Least.*
  arXiv:2107.05666, 2021.
- G. Vos, K. Trinh, Z. Sarnyai, M. Rahimi Azghadi. *Generalizable Machine Learning for Stress
  Monitoring from Wearable Devices: A Systematic Literature Review.* Int. J. Medical Informatics, 2023
  (arXiv:2209.15137).
- G. Vos et al. *Ensemble Machine Learning Trained on a Synthesized Dataset Generalizes Well for Stress
  Prediction.* J. Biomedical Informatics, 2023 (arXiv:2209.15146).
- M. Benchekroun et al. *Cross-Dataset Analysis for Generalizability of HRV-Based Stress Detection
  Models.* Sensors 23(4):1807, 2023.
- P. Prajod, B. Mahesh, E. André. *Stressor Type Matters! Cross-Dataset Generalizability of
  Physiological Stress Detection.* ICMI Companion, 2024 (arXiv:2405.09563).
- J. Oliver, S. Dakshit. *Cross-Modality Investigation on WESAD Stress Classification.* arXiv:2502.18733,
  2025.
- Task Force of the ESC/NASPE. *Heart Rate Variability: Standards of Measurement, Physiological
  Interpretation, and Clinical Use.* Circulation, 1996.
- S. M. Lundberg, S.-I. Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
