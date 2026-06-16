"""Synthetic WESAD-format data so the full pipeline runs without the real dataset.

NeuroKit2 simulators generate physiologically plausible chest and wrist signals.
Stress blocks carry higher heart rate, more skin-conductance responses, faster
breathing, and more motion, so the models learn a real (if easy) signal. This is
for smoke tests and demos only — never for reported results.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

CHEST_FS = 700
WRIST = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}

CONDITIONS = {"baseline": 1, "stress": 2, "amusement": 3}
HR = {"baseline": 74, "stress": 86, "amusement": 80}
SCR = {"baseline": 2, "stress": 4, "amusement": 3}
EDA_LEVEL = {"baseline": 4.5, "stress": 6.5, "amusement": 5.2}
RESP_RATE = {"baseline": 15, "stress": 19, "amusement": 16}
MOTION = {"baseline": 0.03, "stress": 0.07, "amusement": 0.05}
TEMP_BASE = {"baseline": 33.1, "stress": 33.5, "amusement": 33.3}
NOISE = 0.06


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, x)


def _fixlen(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x)
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)), mode="edge")


def _chest_block(cond: str, seconds: int, rng: np.random.RandomState, seed: int) -> Dict:
    import neurokit2 as nk

    n = seconds * CHEST_FS
    hr = HR[cond] + rng.uniform(-6, 6)
    ecg = _fixlen(
        nk.ecg_simulate(
            duration=seconds, sampling_rate=CHEST_FS, heart_rate=hr, noise=NOISE, random_state=seed
        ),
        n,
    )
    eda = EDA_LEVEL[cond] + _fixlen(
        nk.eda_simulate(
            duration=seconds,
            sampling_rate=CHEST_FS,
            scr_number=SCR[cond],
            noise=NOISE,
            random_state=seed,
        ),
        n,
    )
    resp = _fixlen(
        nk.rsp_simulate(
            duration=seconds,
            sampling_rate=CHEST_FS,
            respiratory_rate=RESP_RATE[cond],
            random_state=seed,
        ),
        n,
    )
    temp = TEMP_BASE[cond] + rng.normal(0, 0.02, n).cumsum() / CHEST_FS
    acc = rng.normal(0, MOTION[cond], (n, 3)) + np.array([0.0, 0.0, 1.0])
    emg = rng.normal(0, 0.01, n)

    return {
        "ECG": ecg.reshape(-1, 1),
        "EDA": eda.reshape(-1, 1),
        "Temp": temp.reshape(-1, 1),
        "Resp": resp.reshape(-1, 1),
        "EMG": emg.reshape(-1, 1),
        "ACC": acc,
    }


def _wrist_block(
    cond: str, seconds: float, chest: Dict, rng: np.random.RandomState, seed: int
) -> Dict:
    import neurokit2 as nk

    bvp = nk.ppg_simulate(
        duration=int(seconds), sampling_rate=WRIST["BVP"], heart_rate=HR[cond], random_state=seed
    )
    eda_n = int(seconds * WRIST["EDA"])
    acc_n = int(seconds * WRIST["ACC"])
    return {
        "BVP": bvp.reshape(-1, 1),
        "EDA": _resample(chest["EDA"].ravel(), eda_n).reshape(-1, 1),
        "TEMP": _resample(chest["Temp"].ravel(), eda_n).reshape(-1, 1),
        "ACC": rng.normal(0, MOTION[cond], (acc_n, 3)) + np.array([0.0, 0.0, 1.0]),
    }


def _subject(seed: int, block_sec: float) -> Dict:
    rng = np.random.RandomState(seed)
    block_sec = int(block_sec)
    order = ["baseline", "stress", "amusement", "baseline", "stress"]

    chest_parts: Dict[str, list] = {k: [] for k in ["ECG", "EDA", "Temp", "Resp", "EMG", "ACC"]}
    wrist_parts: Dict[str, list] = {k: [] for k in WRIST}
    labels = []

    for i, cond in enumerate(order):
        chest = _chest_block(cond, block_sec, rng, seed + i)
        wrist = _wrist_block(cond, block_sec, chest, rng, seed + i)
        for k in chest_parts:
            chest_parts[k].append(chest[k])
        for k in wrist_parts:
            wrist_parts[k].append(wrist[k])
        labels.append(np.full(len(chest["ECG"]), CONDITIONS[cond]))

    return {
        "signal": {
            "chest": {k: np.concatenate(v) for k, v in chest_parts.items()},
            "wrist": {k: np.concatenate(v) for k, v in wrist_parts.items()},
        },
        "label": np.concatenate(labels),
    }


def write_dataset(
    out_dir: Path, n_subjects: int = 4, block_sec: float = 120.0, seed: int = 0
) -> Path:
    """Write S2..S(n+1) pickles in WESAD layout under out_dir/WESAD."""
    root = Path(out_dir) / "WESAD"
    for i in range(n_subjects):
        sid = f"S{i + 2}"
        data = _subject(seed + i * 100, block_sec)
        subj_dir = root / sid
        subj_dir.mkdir(parents=True, exist_ok=True)
        with open(subj_dir / f"{sid}.pkl", "wb") as f:
            pickle.dump(data, f, protocol=4)
    return root


def features(n_subjects: int = 4, block_sec: float = 120.0, seed: int = 0) -> Tuple:
    """Build a small feature matrix from freshly generated synthetic subjects."""
    import tempfile

    from .dataset import WindowedDataset

    tmp = Path(tempfile.mkdtemp(prefix="calmsense_synth_"))
    root = write_dataset(tmp, n_subjects=n_subjects, block_sec=block_sec, seed=seed)

    ds = WindowedDataset(data_path=root)
    return ds.build(subjects=ds.loader.subjects, cache=False)
