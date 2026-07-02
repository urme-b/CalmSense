# WESAD dataset

WESAD (Wearable Stress and Affect Detection) is a public multimodal dataset of 15 subjects
recorded with a chest (RespiBAN) and wrist (Empatica E4) device across baseline, stress,
amusement, and meditation conditions. It is not redistributed with this repository.

**Two distinct datasets, two distinct roles.** WESAD is the **primary** dataset for the
Leave-One-Subject-Out benchmark and every headline result. The **PhysioNet Non-EEG** dataset
(Birjandtalab et al.) is used **only** for the cross-dataset transfer experiment
(`scripts/cross_dataset.py`); it is a separate corpus (read via `wfdb`, whereas WESAD ships as `.pkl`).

Reference: Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and
Affect Detection," ICMI 2018.

## Download

WESAD (the primary dataset, needed for `make experiment`/`make reproduce`) is behind a one-time
research agreement; the cross-dataset PhysioNet Non-EEG set downloads directly.

```bash
make wesad   # WESAD (~2 GB) -> data/raw/WESAD          (primary; required for reproduce)
make data    # PhysioNet Non-EEG -> data/external/noneeg (cross-dataset transfer only)
```

Prefer to fetch WESAD manually? Request it from the UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection),
agree to the research-only terms, and extract it here:

```bash
cd data/raw
unzip WESAD.zip   # produces WESAD/S2 ... WESAD/S17
```

No download at all? make demo runs the full calibration pipeline on synthetic data.

## Expected layout

```
data/raw/WESAD/
  S2/
    S2.pkl          # signals and labels
    S2_readme.txt
    S2_quest.csv
  S3/
  ...
  S17/
```

Subjects are S2 to S17; S1 and S12 do not exist in the dataset.

## Data format

Each S*.pkl is a pickle (load with encoding="latin1") containing:

- signal.chest at 700 Hz: ACC, ECG, EMG, EDA, Temp, Resp
- signal.wrist: ACC (32 Hz), BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz)
- label at 700 Hz

> Security: unpickling runs arbitrary code. Only load .pkl files you downloaded from the official
> WESAD source or generated yourself, never a .pkl from an untrusted third party.

## Label encoding

| Label | Condition  | Used for                  |
| ----- | ---------- | ------------------------- |
| 1     | Baseline   | binary, three-class       |
| 2     | Stress     | binary, three-class       |
| 3     | Amusement  | three-class               |
| 4     | Meditation | excluded                  |
| 0, 5 to 7| Other      | excluded                  |

src/data/loader.py reads this format directly; make reproduce then builds the feature
matrix and runs the benchmark.

## Citation

```bibtex
@inproceedings{schmidt2018wesad,
  title     = {Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection},
  author    = {Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  booktitle = {Proceedings of the 20th ACM International Conference on Multimodal Interaction},
  pages     = {400--408},
  year      = {2018}
}
```
