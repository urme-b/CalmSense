# WESAD dataset

WESAD (Wearable Stress and Affect Detection) is a public multimodal dataset of 15 subjects
recorded with a chest (RespiBAN) and wrist (Empatica E4) device across baseline, stress,
amusement, and meditation conditions. It is not redistributed with this repository.

Reference: Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and
Affect Detection," ICMI 2018.

## Download

1. Request the dataset from the UCI Machine Learning Repository:
   https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
2. Agree to the research-only terms and download the archive (~4 GB).
3. Extract it into this directory so the layout matches the structure below.

```bash
cd data/raw
unzip WESAD.zip   # produces WESAD/S2 ... WESAD/S17
```

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

Subjects are S2–S17; S1 and S12 do not exist in the dataset.

## Data format

Each `S*.pkl` is a pickle (load with `encoding="latin1"`) containing:

- `signal.chest` at 700 Hz: `ACC`, `ECG`, `EMG`, `EDA`, `Temp`, `Resp`
- `signal.wrist`: `ACC` (32 Hz), `BVP` (64 Hz), `EDA` (4 Hz), `TEMP` (4 Hz)
- `label` at 700 Hz

## Label encoding

| Label | Condition  | Used for                  |
| ----- | ---------- | ------------------------- |
| 1     | Baseline   | binary, three-class       |
| 2     | Stress     | binary, three-class       |
| 3     | Amusement  | three-class               |
| 4     | Meditation | excluded                  |
| 0, 5–7| Other      | excluded                  |

`src/data/loader.py` reads this format directly; `make reproduce` then builds the feature
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
