"""PhysioNet Non-EEG dataset -> shared feature space for cross-dataset transfer.

20 subjects, wrist EDA/TEMP/ACC (8 Hz) + HR (1 Hz), with .atr annotations marking
relaxation and physical/cognitive/emotional stress blocks. We use psychological
stress (cognitive + emotional) vs relaxation, excluding the motion-heavy physical block.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DATA_DIR
from ..portable import OVERLAP, WINDOW_SEC, portable_features

DATA_DIR = (
    EXTERNAL_DATA_DIR / "noneeg" / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
)
ACC_FS = 8
HR_FS = 1
STRESS = {"CognitiveStress", "EmotionalStress"}
RELAX = {"Relax"}


def _segments(record: str):
    import wfdb

    ann = wfdb.rdann(record, "atr")
    bounds = list(ann.sample) + [None]
    for i, note in enumerate(ann.aux_note):
        yield int(ann.sample[i]), bounds[i + 1], note


def build(subjects: Optional[list] = None) -> pd.DataFrame:
    import wfdb

    subjects = subjects or [f"Subject{i}" for i in range(1, 21)]
    win = int(WINDOW_SEC * ACC_FS)
    step = int(win * (1 - OVERLAP))
    rows = []

    for sid in subjects:
        acc_rec = str(DATA_DIR / f"{sid}_AccTempEDA")
        hr_rec = str(DATA_DIR / f"{sid}_SpO2HR")
        if not Path(acc_rec + ".hea").exists():
            continue
        sig = wfdb.rdrecord(acc_rec).p_signal  # ax, ay, az, temp, EDA @ 8 Hz
        hr = wfdb.rdrecord(hr_rec).p_signal[:, 1]  # hr @ 1 Hz
        acc_mag = np.sqrt(np.sum(sig[:, 0:3] ** 2, axis=1))
        temp, eda = sig[:, 3], sig[:, 4]

        for s0, s1, note in _segments(acc_rec):
            s1 = s1 if s1 is not None else len(eda)
            if note in STRESS:
                label = 1
            elif note in RELAX:
                label = 0
            else:
                continue  # skip physical stress
            for w0 in range(s0, s1 - win + 1, step):
                w1 = w0 + win
                hr_win = hr[w0 // ACC_FS : w1 // ACC_FS]
                row = portable_features(eda[w0:w1], temp[w0:w1], acc_mag[w0:w1], hr_win)
                row["subject"] = sid
                row["label"] = label
                rows.append(row)

    return pd.DataFrame(rows)
