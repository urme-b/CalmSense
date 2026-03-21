from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..config import FS, FEATURE_PARAMS
from ..logging_config import LoggerMixin


class SignalWindower(LoggerMixin):
    def __init__(
        self,
        window_size_sec: float = FEATURE_PARAMS.WINDOW_SIZE_SEC,
        overlap: float = FEATURE_PARAMS.WINDOW_OVERLAP,
        sampling_rate: float = FS.CHEST,
    ):
        if not 0 <= overlap < 1:
            raise ValueError(f"Overlap must be in [0, 1), got {overlap}")

        self.window_size_sec = window_size_sec
        self.overlap = overlap
        self.sampling_rate = sampling_rate

        self.window_size_samples = int(window_size_sec * sampling_rate)
        self.step_samples = int(self.window_size_samples * (1 - overlap))

        self.logger.info(
            f"SignalWindower initialized: {window_size_sec}s windows, "
            f"{overlap * 100:.0f}% overlap, {self.window_size_samples} samples/window"
        )

    def create_windows(
        self,
        signal: np.ndarray,
        labels: Optional[np.ndarray] = None,
        label_strategy: str = "majority",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        signal = np.asarray(signal)

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False

        n_samples, n_channels = signal.shape

        if self.window_size_samples > n_samples:
            self.logger.warning(
                f"Window size ({self.window_size_samples}) exceeds signal length ({n_samples})"
            )
            return np.array([]), None

        n_windows = (n_samples - self.window_size_samples) // self.step_samples + 1

        windows = np.zeros((n_windows, self.window_size_samples, n_channels))

        window_starts = []
        window_ends = []

        for i in range(n_windows):
            start = i * self.step_samples
            end = start + self.window_size_samples
            windows[i] = signal[start:end]
            window_starts.append(start)
            window_ends.append(end)

        if squeeze_output:
            windows = windows.squeeze(axis=-1)

        window_labels = None
        if labels is not None:
            labels = np.asarray(labels).flatten()
            if len(labels) != n_samples:
                raise ValueError(
                    f"Labels length ({len(labels)}) doesn't match signal ({n_samples})"
                )

            window_labels = self.assign_window_labels(
                labels, window_starts, window_ends, strategy=label_strategy
            )

        self.logger.debug(
            f"Created {n_windows} windows from signal of length {n_samples}"
        )

        return windows, window_labels

    def assign_window_labels(
        self,
        labels: np.ndarray,
        window_starts: List[int],
        window_ends: List[int],
        strategy: str = "majority",
    ) -> np.ndarray:
        n_windows = len(window_starts)
        window_labels = np.zeros(n_windows, dtype=np.int32)

        for i, (start, end) in enumerate(zip(window_starts, window_ends)):
            window_label_segment = labels[start:end]

            if strategy == "majority":
                unique, counts = np.unique(window_label_segment, return_counts=True)
                window_labels[i] = unique[np.argmax(counts)]

            elif strategy == "first":
                window_labels[i] = window_label_segment[0]

            elif strategy == "last":
                window_labels[i] = window_label_segment[-1]

            elif strategy == "all_same":
                if np.all(window_label_segment == window_label_segment[0]):
                    window_labels[i] = window_label_segment[0]
                else:
                    window_labels[i] = -1  # mixed labels

            else:
                raise ValueError(f"Unknown label strategy: {strategy}")

        return window_labels

    def validate_window(
        self,
        window: np.ndarray,
        min_valid_ratio: float = 0.9,
        max_nan_ratio: float = 0.1,
        min_r_peaks: int = 40,
    ) -> Dict[str, Union[bool, float]]:
        window = np.asarray(window).flatten()

        results = {
            "is_valid": True,
            "nan_ratio": 0.0,
            "flatline_ratio": 0.0,
            "outlier_ratio": 0.0,
        }

        nan_mask = ~np.isfinite(window)
        results["nan_ratio"] = float(np.mean(nan_mask))

        if results["nan_ratio"] > max_nan_ratio:
            results["is_valid"] = False
            return results

        valid_window = window[~nan_mask]
        if len(valid_window) == 0:
            results["is_valid"] = False
            return results

        diff = np.diff(valid_window)
        flatline_mask = np.abs(diff) < FEATURE_PARAMS.EPSILON
        results["flatline_ratio"] = float(np.mean(flatline_mask))

        if results["flatline_ratio"] > 0.9:
            results["is_valid"] = False

        mean = np.mean(valid_window)
        std = np.std(valid_window)
        if std > FEATURE_PARAMS.EPSILON:
            z_scores = np.abs((valid_window - mean) / std)
            outlier_mask = z_scores > 5
            results["outlier_ratio"] = float(np.mean(outlier_mask))

            if results["outlier_ratio"] > 0.05:
                results["is_valid"] = False

        return results

    def validate_ecg_window(
        self, window: np.ndarray, min_r_peaks: int = 40
    ) -> Dict[str, Union[bool, float, int]]:
        from scipy import signal as scipy_signal

        results = self.validate_window(window)
        results["r_peak_count"] = 0

        if not results["is_valid"]:
            return results

        window = np.asarray(window).flatten()

        min_distance = int(0.3 * self.sampling_rate)
        prominence = np.std(window) * 0.5

        try:
            peaks, _ = scipy_signal.find_peaks(
                window, distance=min_distance, prominence=prominence
            )
            results["r_peak_count"] = len(peaks)
        except Exception:
            results["r_peak_count"] = 0

        if results["r_peak_count"] < min_r_peaks:
            results["is_valid"] = False
            self.logger.debug(
                f"ECG window rejected: {results['r_peak_count']} R-peaks "
                f"(minimum {min_r_peaks})"
            )

        return results

    def create_windows_multimodal(
        self,
        signals: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
        reference_signal: str = "ECG",
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], np.ndarray]:
        if reference_signal not in signals:
            raise ValueError(f"Reference signal '{reference_signal}' not in signals")

        ref_signal = signals[reference_signal]
        ref_windows, window_labels = self.create_windows(ref_signal, labels)
        n_windows = len(ref_windows)

        self.logger.info(f"Creating {n_windows} multimodal windows")

        all_windows = {reference_signal: ref_windows}
        valid_mask = np.ones(n_windows, dtype=bool)

        for i in range(n_windows):
            validation = self.validate_window(ref_windows[i])
            if not validation["is_valid"]:
                valid_mask[i] = False

        for name, sig in signals.items():
            if name == reference_signal:
                continue

            signal_duration = len(ref_signal) / self.sampling_rate
            signal_fs = len(sig) / signal_duration

            signal_windower = SignalWindower(
                window_size_sec=self.window_size_sec,
                overlap=self.overlap,
                sampling_rate=signal_fs,
            )

            sig_windows, _ = signal_windower.create_windows(sig)

            if len(sig_windows) != n_windows:
                self.logger.warning(
                    f"Window count mismatch for {name}: {len(sig_windows)} vs {n_windows}"
                )
                if len(sig_windows) > n_windows:
                    sig_windows = sig_windows[:n_windows]
                else:
                    original_len = len(sig_windows)
                    pad_shape = (n_windows - original_len,) + sig_windows.shape[1:]
                    sig_windows = np.concatenate([sig_windows, np.zeros(pad_shape)])
                    valid_mask[original_len:] = False

            all_windows[name] = sig_windows

            for i in range(min(len(sig_windows), n_windows)):
                validation = self.validate_window(sig_windows[i])
                if not validation["is_valid"]:
                    valid_mask[i] = False

        # Filter invalid windows
        n_valid = int(np.sum(valid_mask))
        if n_valid < n_windows:
            self.logger.info(
                f"Filtering {n_windows - n_valid} invalid windows "
                f"({n_valid}/{n_windows} kept)"
            )
            for name in all_windows:
                all_windows[name] = all_windows[name][valid_mask]
            if window_labels is not None:
                window_labels = window_labels[valid_mask]
            valid_mask = np.ones(n_valid, dtype=bool)

        self.logger.info(f"Multimodal windowing complete: {n_valid} valid windows")

        return all_windows, window_labels, valid_mask

    def filter_by_label(
        self, windows: np.ndarray, labels: np.ndarray, valid_labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isin(labels, valid_labels)
        filtered_windows = windows[mask]
        filtered_labels = labels[mask]

        self.logger.debug(
            f"Filtered {len(labels)} windows to {len(filtered_labels)} "
            f"with labels {valid_labels}"
        )

        return filtered_windows, filtered_labels

    def get_window_info(self) -> Dict:
        return {
            "window_size_sec": self.window_size_sec,
            "window_size_samples": self.window_size_samples,
            "overlap": self.overlap,
            "step_samples": self.step_samples,
            "sampling_rate": self.sampling_rate,
        }
