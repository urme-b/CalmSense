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

## Verify integrity

WESAD ships no official checksum, so the SHA-256 values below are computed from the official
Uni-Siegen distribution as a reference. After extracting, verify your copy — save this block as
`data/raw/WESAD/SHA256SUMS` and run `cd data/raw/WESAD && shasum -a 256 -c SHA256SUMS` (a mismatch
means re-download from the official source):

```
36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23  S2/S2.pkl
5c8bd4a82af029c082e610bca28a011fca2ae3b23e14a18458ebb5990be4015e  S3/S3.pkl
0f0740a79388723360ff12b4f47c465665ea7827d1399b18ac43908daac17900  S4/S4.pkl
74bd187e3a9c1ca4259af52d04974c8e7ff7dc49ceea7e269f499ca98fe6d8ec  S5/S5.pkl
8aa9bf57b69f4fe5bce06c550230857627c3f05befa2f787151646bb29ee8f62  S6/S6.pkl
9cb62705ae7f53dca327a9a00a6f9fdabf5128d449174ab37594658e912cb6d8  S7/S7.pkl
dac1141dac11d56b3641be982f45da63f05e9d74154f59e6ea0cdcf47fc72710  S8/S8.pkl
24dc004e201bd541f092989443f0a29ebf89e4a227a80bb6b6d1987255039544  S9/S9.pkl
41da29c68366f33650f3d41a6be78107bf6942929c3bb0ef46238078ddddee9f  S10/S10.pkl
f39557a8d660b10154936f51debf2926aea7ebb9b26a168858f59502f914d8f7  S11/S11.pkl
772fb490f19b279e49367271e009fc10d3a3ca1e3456df0d68b9063a73992066  S13/S13.pkl
e7bd33c57538319a25c6d53e6a9fb6c1abd12800cfc64bb63275d89de8d2fd60  S14/S14.pkl
1ea573bc6b45ba79fb134f9460d691b86176f60dce23420dc514c28017d4049c  S15/S15.pkl
f65cf40cada75c3e9f5813276d7dcc90359c3b06dec41d68656c0a6e61dbc575  S16/S16.pkl
3315796a75227d54d7b0056736f671484fd2fb85afffa65818fd76aeff2920fa  S17/S17.pkl
```

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
