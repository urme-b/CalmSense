from typing import Dict, Tuple, Union

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from ..config import FS, FILTER_PARAMS
from ..logging_config import LoggerMixin


class SignalProcessor(LoggerMixin):
    def __init__(self, fs: float = FS.CHEST):
        self.fs = fs
        self.logger.debug(f"SignalProcessor initialized with fs={fs} Hz")

    def butterworth_filter(
        self,
        data: np.ndarray,
        cutoff: Union[float, Tuple[float, float]],
        order: int = 4,
        btype: str = "low",
    ) -> np.ndarray:
        nyq = 0.5 * self.fs

        if isinstance(cutoff, tuple):
            normalized_cutoff = (cutoff[0] / nyq, cutoff[1] / nyq)
            if normalized_cutoff[0] >= 1.0 or normalized_cutoff[1] >= 1.0:
                raise ValueError(
                    f"Cutoff frequencies {cutoff} exceed Nyquist frequency {nyq} Hz"
                )
        else:
            normalized_cutoff = cutoff / nyq
            if normalized_cutoff >= 1.0:
                raise ValueError(
                    f"Cutoff frequency {cutoff} exceeds Nyquist frequency {nyq} Hz"
                )

        if len(data) == 0:
            return data

        b, a = signal.butter(order, normalized_cutoff, btype=btype)
        return signal.filtfilt(b, a, data)

    def notch_filter(
        self,
        data: np.ndarray,
        freq: float = FILTER_PARAMS.ECG_NOTCH_FREQ,
        q: float = FILTER_PARAMS.NOTCH_Q_FACTOR,
    ) -> np.ndarray:
        nyq = 0.5 * self.fs
        w0 = freq / nyq

        if w0 >= 1.0:
            self.logger.warning(
                f"Notch frequency {freq} Hz exceeds Nyquist, skipping filter"
            )
            return data

        b, a = signal.iirnotch(w0, q)
        return signal.filtfilt(b, a, data)

    def process_ecg(self, ecg: np.ndarray) -> np.ndarray:
        ecg = np.asarray(ecg).flatten()

        filtered = self.butterworth_filter(
            ecg,
            cutoff=(FILTER_PARAMS.ECG_BANDPASS_LOW, FILTER_PARAMS.ECG_BANDPASS_HIGH),
            order=FILTER_PARAMS.ECG_FILTER_ORDER,
            btype="band",
        )

        filtered = self.notch_filter(filtered, freq=FILTER_PARAMS.ECG_NOTCH_FREQ)

        return filtered

    def process_eda(self, eda: np.ndarray) -> np.ndarray:
        eda = np.asarray(eda).flatten()

        filtered = self.butterworth_filter(
            eda,
            cutoff=FILTER_PARAMS.EDA_LOWPASS,
            order=FILTER_PARAMS.EDA_FILTER_ORDER,
            btype="low",
        )

        filtered = median_filter(filtered, size=FILTER_PARAMS.EDA_MEDIAN_SIZE)

        return filtered

    def process_emg(self, emg: np.ndarray) -> np.ndarray:
        emg = np.asarray(emg).flatten()

        filtered = self.butterworth_filter(
            emg,
            cutoff=(FILTER_PARAMS.EMG_BANDPASS_LOW, FILTER_PARAMS.EMG_BANDPASS_HIGH),
            order=FILTER_PARAMS.EMG_FILTER_ORDER,
            btype="band",
        )

        filtered = self.notch_filter(filtered, freq=FILTER_PARAMS.EMG_NOTCH_FREQ)

        return filtered

    def process_respiration(self, resp: np.ndarray) -> np.ndarray:
        resp = np.asarray(resp).flatten()

        return self.butterworth_filter(
            resp,
            cutoff=(FILTER_PARAMS.RESP_BANDPASS_LOW, FILTER_PARAMS.RESP_BANDPASS_HIGH),
            order=FILTER_PARAMS.RESP_FILTER_ORDER,
            btype="band",
        )

    def process_temperature(self, temp: np.ndarray) -> np.ndarray:
        temp = np.asarray(temp).flatten()

        return self.butterworth_filter(
            temp,
            cutoff=FILTER_PARAMS.TEMP_LOWPASS,
            order=FILTER_PARAMS.TEMP_FILTER_ORDER,
            btype="low",
        )

    def normalize_zscore(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        mean = np.mean(data)
        std = np.std(data)

        if std < 1e-10:
            self.logger.warning("Near-zero standard deviation, returning zeros")
            return np.zeros_like(data)

        return (data - mean) / std

    def normalize_minmax(
        self, data: np.ndarray, feature_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        data = np.asarray(data)
        min_val, max_val = feature_range
        data_min, data_max = np.min(data), np.max(data)

        data_range = data_max - data_min
        if data_range < 1e-10:
            self.logger.warning("Near-zero data range, returning constant")
            return np.full_like(data, (min_val + max_val) / 2)

        return (data - data_min) / data_range * (max_val - min_val) + min_val

    def segment_signal(
        self, data: np.ndarray, window_size: int, overlap: float = 0.5
    ) -> np.ndarray:
        data = np.asarray(data).flatten()

        if window_size > len(data):
            raise ValueError(
                f"Window size {window_size} exceeds signal length {len(data)}"
            )

        if not 0 <= overlap < 1:
            raise ValueError(f"Overlap must be in [0, 1), got {overlap}")

        step = int(window_size * (1 - overlap))
        n_windows = (len(data) - window_size) // step + 1

        segments = np.zeros((n_windows, window_size))
        for i in range(n_windows):
            start = i * step
            segments[i] = data[start : start + window_size]

        self.logger.debug(
            f"Created {n_windows} segments of size {window_size} "
            f"with {overlap * 100:.0f}% overlap"
        )

        return segments

    def process_all(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        processed = {}

        signal_processors = {
            "ECG": self.process_ecg,
            "EDA": self.process_eda,
            "EMG": self.process_emg,
            "Resp": self.process_respiration,
            "Temp": self.process_temperature,
        }

        for name, data in signals.items():
            if name in signal_processors:
                processed[name] = signal_processors[name](data)
                self.logger.debug(f"Processed {name} signal")
            elif name == "ACC":
                processed[name] = np.asarray(data)
                self.logger.debug(f"Passed through {name} signal")
            else:
                self.logger.warning(f"Unknown signal type: {name}, skipping")

        return processed
