"""Window WESAD wrist (Empatica E4) signals into features for a deployable model."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FS, PROCESSED_DATA_DIR
from .data.loader import WESADLoader
from .dataset import CONDITION_LABELS, window_label
from .features.feature_pipeline import FeatureExtractionPipeline
from .logging_config import LoggerMixin
from .preprocessing.ecg_processor import ECGProcessor
from .preprocessing.eda_processor import EDAProcessor


class WristDataset(LoggerMixin):
    def __init__(self, window_sec: float = 60.0, overlap: float = 0.5, purity: float = 0.9):
        self.window_sec = window_sec
        self.overlap = overlap
        self.purity = purity
        self.label_fs = FS.CHEST  # labels sampled at 700 Hz
        self.loader = WESADLoader()
        self.eda = EDAProcessor(sampling_rate=FS.WRIST_EDA)
        self.rr = ECGProcessor(sampling_rate=FS.WRIST_BVP)  # reuse RR/ectopic helpers
        self.features = FeatureExtractionPipeline(
            chest_fs=FS.CHEST, wrist_eda_fs=FS.WRIST_EDA, wrist_acc_fs=FS.WRIST_ACC
        )

    def _window_label(self, labels: np.ndarray) -> Optional[int]:
        return window_label(labels, self.purity)

    def _bvp_peaks(self, bvp: np.ndarray) -> np.ndarray:
        import neurokit2 as nk

        clean = nk.ppg_clean(bvp, sampling_rate=int(FS.WRIST_BVP))
        info = nk.ppg_findpeaks(clean, sampling_rate=int(FS.WRIST_BVP))
        return np.asarray(info["PPG_Peaks"])

    def _process_subject(self, subject_id: str) -> Tuple[List[dict], List[int]]:
        data = self.loader.load_subject(subject_id)
        wrist = data["wrist"]
        labels = np.asarray(data["label"]).flatten()

        bvp = np.asarray(wrist["BVP"]).flatten()
        eda = np.asarray(wrist["EDA"]).flatten()
        temp = np.asarray(wrist["TEMP"]).flatten()
        acc = np.asarray(wrist["ACC"])
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))

        peaks = self._bvp_peaks(bvp)
        eda_filt = self.eda.lowpass_filter(eda)
        tonic, phasic = self.eda.decompose_eda(eda_filt)
        _, scr_features = self.eda.detect_scr_peaks(phasic)
        scr_idx = np.array([s["peak_idx"] for s in scr_features])

        step = self.window_sec * (1 - self.overlap)
        duration = len(labels) / self.label_fs
        windows, ys = [], []
        t = 0.0
        while t + self.window_sec <= duration:
            lab = self._window_label(
                labels[int(t * self.label_fs) : int((t + self.window_sec) * self.label_fs)]
            )
            if lab is not None:
                b0, b1 = int(t * FS.WRIST_BVP), int((t + self.window_sec) * FS.WRIST_BVP)
                in_win = (peaks >= b0) & (peaks < b1)
                rr = self.rr.extract_rr_intervals(peaks[in_win], unit="ms")
                _, valid = self.rr.remove_ectopic_beats(rr)
                rr_clean = self.rr.interpolate_artifacts(rr, valid)

                e0, e1 = int(t * FS.WRIST_EDA), int((t + self.window_sec) * FS.WRIST_EDA)
                a0, a1 = int(t * FS.WRIST_ACC), int((t + self.window_sec) * FS.WRIST_ACC)
                scr_in = [s for s, idx in zip(scr_features, scr_idx) if e0 <= idx < e1]

                windows.append(
                    {
                        "rr_intervals": rr_clean,
                        "eda_tonic": tonic[e0:e1],
                        "eda_phasic": phasic[e0:e1],
                        "eda_raw": eda_filt[e0:e1],
                        "scr_peaks": scr_in,
                        "temperature": temp[e0:e1],
                        "respiration": None,
                        "accelerometer": {"magnitude": acc_mag[a0:a1]},
                        "subject_id": subject_id,
                        "label": lab,
                    }
                )
                ys.append(lab)
            t += step

        self.logger.info(f"{subject_id} (wrist): {len(ys)} windows")
        return windows, ys

    def build(self, subjects: Optional[List[str]] = None, cache: bool = True) -> pd.DataFrame:
        subjects = subjects or self.loader.subjects
        all_windows = []
        for s in subjects:
            windows, _ = self._process_subject(s)
            all_windows.extend(windows)
        df = self.features.extract_all_features(all_windows, show_progress=False)
        df["label_name"] = df["label"].map(CONDITION_LABELS)
        if cache:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(PROCESSED_DATA_DIR / "features_wrist.parquet", index=False)
            self.logger.info(f"Cached wrist features ({len(df)} windows)")
        return df


def load_wrist() -> Optional[pd.DataFrame]:
    path = Path(PROCESSED_DATA_DIR) / "features_wrist.parquet"
    return pd.read_parquet(path) if path.exists() else None
