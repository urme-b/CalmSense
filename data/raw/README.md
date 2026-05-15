# WESAD Dataset Download Instructions

## Dataset Overview

**WESAD (Wearable Stress and Affect Detection)** is a publicly available multimodal dataset for stress detection research.

- **Paper**: Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection" (ICMI 2018)
- **Size**: ~4 GB
- **Subjects**: 15 participants (S2-S17, no S1 or S12)
- **Duration**: ~2 hours per subject

## Download Steps

### 1. Access the Dataset

Visit the UCI Machine Learning Repository:
**https://archive.ics.uci.edu/ml/datasets/WESAD**

Or direct download link:
**https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx**

### 2. Request Access

The dataset requires agreeing to terms of use:
1. Click "Download" on the UCI page
2. Fill out the data request form
3. Agree to use data for research purposes only
4. You will receive a download link via email

### 3. Extract the Data

```bash
# Navigate to raw data directory
cd data/raw/

# Extract the downloaded archive
unzip WESAD.zip

# Verify structure
ls WESAD/
# Should show: S2/ S3/ S4/ ... S17/
```

### 4. Expected Directory Structure

After extraction, your `data/raw/` should look like:

```
data/raw/
├── README.md (this file)
├── .gitkeep
└── WESAD/
    ├── S2/
    │   ├── S2.pkl           # Main data file
    │   ├── S2_readme.txt    # Subject-specific notes
    │   └── S2_quest.csv     # Questionnaire responses
    ├── S3/
    │   ├── S3.pkl
    │   ├── S3_readme.txt
    │   └── S3_quest.csv
    ├── ...
    └── S17/
        ├── S17.pkl
        ├── S17_readme.txt
        └── S17_quest.csv
```

## Data Format

Each subject's `.pkl` file contains a dictionary with:

```python
import pickle

with open('data/raw/WESAD/S2/S2.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Keys:
# 'subject': Subject ID
# 'signal': Dictionary of signals
#   - 'chest': RespiBAN chest device (700 Hz)
#       - 'ACC': Acceleration (700 Hz, 3-axis)
#       - 'ECG': Electrocardiogram (700 Hz)
#       - 'EMG': Electromyogram (700 Hz)
#       - 'EDA': Electrodermal Activity (700 Hz)
#       - 'Temp': Temperature (700 Hz)
#       - 'Resp': Respiration (700 Hz)
#   - 'wrist': Empatica E4 wristband
#       - 'ACC': Acceleration (32 Hz, 3-axis)
#       - 'BVP': Blood Volume Pulse (64 Hz)
#       - 'EDA': Electrodermal Activity (4 Hz)
#       - 'TEMP': Temperature (4 Hz)
# 'label': Ground truth labels (700 Hz)
```

## Label Encoding

| Label | Condition | Description |
|-------|-----------|-------------|
| 0 | Not defined | Transient/unlabeled |
| 1 | Baseline | Neutral reading |
| 2 | Stress | TSST (Trier Social Stress Test) |
| 3 | Amusement | Funny video clips |
| 4 | Meditation | Guided meditation |
| 5, 6, 7 | Other | Ignore for classification |

**For binary classification**: Use label 1 (baseline) vs label 2 (stress)
**For 3-class classification**: Use labels 1, 2, 3 (baseline, stress, amusement)

## Verification Script

Run this script to verify your download:

```python
import os
import pickle

WESAD_PATH = "data/raw/WESAD"
EXPECTED_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                     'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

def verify_wesad():
    if not os.path.exists(WESAD_PATH):
        print("ERROR: WESAD directory not found!")
        return False

    missing = []
    for subj in EXPECTED_SUBJECTS:
        pkl_path = os.path.join(WESAD_PATH, subj, f"{subj}.pkl")
        if not os.path.exists(pkl_path):
            missing.append(subj)

    if missing:
        print(f"ERROR: Missing subjects: {missing}")
        return False

    # Verify one subject's structure
    with open(os.path.join(WESAD_PATH, 'S2', 'S2.pkl'), 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    required_keys = ['subject', 'signal', 'label']
    for key in required_keys:
        if key not in data:
            print(f"ERROR: Missing key '{key}' in data")
            return False

    print("SUCCESS: WESAD dataset verified!")
    print(f"  - Subjects found: {len(EXPECTED_SUBJECTS)}")
    print(f"  - Chest signals: {list(data['signal']['chest'].keys())}")
    print(f"  - Wrist signals: {list(data['signal']['wrist'].keys())}")
    return True

if __name__ == "__main__":
    verify_wesad()
```

## Citation

If you use the WESAD dataset, please cite:

```bibtex
@inproceedings{schmidt2018introducing,
  title={Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marber{\textsuperscript{g}}, Claus and Van Laerhoven, Kristof},
  booktitle={Proceedings of the 20th ACM International Conference on Multimodal Interaction},
  pages={400--408},
  year={2018}
}
```

## Troubleshooting

### Pickle Loading Issues
If you encounter encoding issues:
```python
# Use latin1 encoding for Python 3
data = pickle.load(f, encoding='latin1')
```

### Memory Issues
Each subject's data is ~250MB. Load one at a time:
```python
# Don't load all subjects at once!
for subj in subjects:
    data = load_subject(subj)
    features = extract_features(data)
    save_features(features, subj)
    del data  # Free memory
```

### Missing Subjects
Subjects S1 and S12 do not exist in the dataset (excluded due to data quality issues).
