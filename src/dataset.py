"""Window WESAD chest signals into features and raw CNN tensors."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import resample

from .config import FEATURE_PARAMS, FS, PROCESSED_DATA_DIR, VALID_SUBJECTS
from .data.loader import WESADLoader
from .features.feature_pipeline import FeatureExtractionPipeline
from .logging_config import LoggerMixin
from .preprocessing.ecg_processor import ECGProcessor
from .preprocessing.eda_processor import EDAProcessor
from .preprocessing.filters import SignalProcessor

# Conditions kept for classification
CONDITION_LABELS = {1: "baseline", 2: "stress", 3: "amusement"}
CNN_CHANNELS = ["ECG", "EDA", "Temp", "Resp", "ACC"]


def window_label(labels: np.ndarray, purity: float) -> Optional[int]:
    """Dominant condition of a window, or None if out-of-set or below `purity`."""
    values, counts = np.unique(labels, return_counts=True)
    dominant = values[counts.argmax()]
    if dominant not in CONDITION_LABELS:
        return None
    if counts.max() / counts.sum() < purity:
        return None
    return int(dominant)


class WindowedDataset(LoggerMixin):
    """Build the WESAD chest feature matrix and raw CNN tensors, subject by subject.

    For each subject the raw signals are filtered and processed (ECG R-peaks + ectopic
    correction, EDA tonic/phasic + SCR peaks, temperature, respiration, ACC magnitude),
    segmented into overlapping windows, and kept only when at least ``purity`` of a
    window's samples share one in-set condition. Each kept window yields a feature row
    and a fixed-length raw tensor for the 1D-CNN. Processing is strictly per-subject, so
    no cross-subject statistic ever leaks into feature construction.
    """

    def __init__(
        self,
        window_sec: float = FEATURE_PARAMS.WINDOW_SIZE_SEC,
        overlap: float = FEATURE_PARAMS.WINDOW_OVERLAP,
        purity: float = 0.9,
        cnn_length: int = 1024,
        fs: float = FS.CHEST,
        data_path: Optional[Union[str, Path]] = None,
    ):
        self.fs = fs
        self.window_samples = int(window_sec * fs)
        self.step = int(self.window_samples * (1 - overlap))
        self.purity = purity
        self.cnn_length = cnn_length

        self.loader = WESADLoader(data_path=data_path)
        self.ecg = ECGProcessor(sampling_rate=fs)
        self.eda = EDAProcessor(sampling_rate=fs)
        self.sig = SignalProcessor(fs=fs)
        # All chest modalities share 700 Hz
        self.features = FeatureExtractionPipeline(chest_fs=fs, wrist_eda_fs=fs, wrist_acc_fs=fs)

    def _window_label(self, labels: np.ndarray) -> Optional[int]:
        return window_label(labels, self.purity)

    def _process_subject(self, subject_id: str) -> Tuple[List[Dict], List[np.ndarray], List[int]]:
        data = self.loader.load_subject(subject_id)
        chest = data["chest"]
        labels = np.asarray(data["label"]).flatten()

        ecg_filt = self.ecg.bandpass_filter(chest["ECG"].flatten())
        r_peaks = self.ecg.detect_r_peaks(ecg_filt)

        eda_filt = self.eda.remove_artifacts(self.eda.lowpass_filter(chest["EDA"].flatten()))
        tonic, phasic = self.eda.decompose_eda(eda_filt)
        _, scr_features = self.eda.detect_scr_peaks(phasic)
        scr_idx = np.array([s["peak_idx"] for s in scr_features])

        temp_filt = self.sig.process_temperature(chest["Temp"].flatten())
        resp_filt = self.sig.process_respiration(chest["Resp"].flatten())
        acc_mag = np.sqrt(np.sum(np.asarray(chest["ACC"]) ** 2, axis=1))

        n = min(len(ecg_filt), len(labels))
        windows, raws, ys = [], [], []

        for start in range(0, n - self.window_samples + 1, self.step):
            end = start + self.window_samples
            label = self._window_label(labels[start:end])
            if label is None:
                continue

            mask = (r_peaks >= start) & (r_peaks < end)
            rr = self.ecg.extract_rr_intervals(r_peaks[mask], unit="ms")
            _, valid = self.ecg.remove_ectopic_beats(rr)
            rr_clean = self.ecg.interpolate_artifacts(rr, valid)

            scr_in = [s for s, idx in zip(scr_features, scr_idx) if start <= idx < end]

            window = {
                "rr_intervals": rr_clean,
                "eda_tonic": tonic[start:end],
                "eda_phasic": phasic[start:end],
                "eda_raw": eda_filt[start:end],
                "scr_peaks": scr_in,
                "temperature": temp_filt[start:end],
                "respiration": resp_filt[start:end],
                "accelerometer": {"magnitude": acc_mag[start:end]},
                "subject_id": subject_id,
                "label": label,
            }
            windows.append(window)
            raws.append(
                self._raw_tensor(
                    ecg_filt[start:end],
                    eda_filt[start:end],
                    temp_filt[start:end],
                    resp_filt[start:end],
                    acc_mag[start:end],
                )
            )
            ys.append(label)

        self.logger.info(f"{subject_id}: {len(ys)} windows")
        return windows, raws, ys

    def _raw_tensor(self, *channels: np.ndarray) -> np.ndarray:
        stacked = [resample(np.asarray(c, dtype=np.float32), self.cnn_length) for c in channels]
        return np.stack(stacked).astype(np.float32)

    def build(
        self, subjects: Optional[List[str]] = None, cache: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        subjects = subjects or self.loader.subjects or VALID_SUBJECTS

        all_windows, all_raw, all_y = [], [], []
        for subject_id in subjects:
            windows, raws, ys = self._process_subject(subject_id)
            all_windows.extend(windows)
            all_raw.extend(raws)
            all_y.extend(ys)

        features_df = self.features.extract_all_features(all_windows, show_progress=True)
        features_df["label_name"] = features_df["label"].map(CONDITION_LABELS)
        x_raw = np.stack(all_raw) if all_raw else np.empty((0, len(CNN_CHANNELS), self.cnn_length))

        if cache:
            self._save(features_df, x_raw)
        return features_df, x_raw, np.asarray(all_y)

    def _save(self, features_df: pd.DataFrame, x_raw: np.ndarray) -> None:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(PROCESSED_DATA_DIR / "features.parquet", index=False)
        np.savez_compressed(
            PROCESSED_DATA_DIR / "raw_windows.npz",
            x=x_raw,
            subject=features_df["subject_id"].to_numpy(),
            label=features_df["label"].to_numpy(),
        )
        self.logger.info(f"Cached dataset to {PROCESSED_DATA_DIR}")


def load_cached() -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    feat_path = PROCESSED_DATA_DIR / "features.parquet"
    raw_path = PROCESSED_DATA_DIR / "raw_windows.npz"
    if not feat_path.exists() or not raw_path.exists():
        return None
    features_df = pd.read_parquet(feat_path)
    x_raw = np.load(raw_path)["x"]
    return features_df, x_raw
