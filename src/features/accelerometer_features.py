from typing import Dict, Optional

import numpy as np
from scipy import signal as scipy_signal

from ..logging_config import LoggerMixin


class AccelerometerFeatureExtractor(LoggerMixin):
    def __init__(self, sampling_rate: float = 32.0):
        self.sampling_rate = sampling_rate
        self.logger.debug(
            f"AccelerometerFeatureExtractor initialized, fs={sampling_rate} Hz"
        )

    def _validate_signal(self, signal: np.ndarray) -> Optional[np.ndarray]:
        if signal is None:
            return None

        signal = np.asarray(signal).flatten()
        signal = signal[np.isfinite(signal)]

        if len(signal) < 10:
            self.logger.warning("Accelerometer signal too short")
            return None

        return signal

    def compute_magnitude(
        self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray
    ) -> np.ndarray:
        return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    def extract_all(
        self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray
    ) -> Dict[str, float]:
        acc_x = self._validate_signal(acc_x)
        acc_y = self._validate_signal(acc_y)
        acc_z = self._validate_signal(acc_z)

        if acc_x is None or acc_y is None or acc_z is None:
            self.logger.warning("Invalid accelerometer data")
            return {
                "ACC_magnitude": np.nan,
                "ACC_std": np.nan,
                "ACC_zero_crossings": np.nan,
                "ACC_energy": np.nan,
                "ACC_peak_freq": np.nan,
            }

        min_len = min(len(acc_x), len(acc_y), len(acc_z))
        magnitude = self.compute_magnitude(
            acc_x[:min_len], acc_y[:min_len], acc_z[:min_len]
        )
        return self.extract_from_magnitude(magnitude)

    def extract_from_magnitude(self, magnitude: np.ndarray) -> Dict[str, float]:
        features = {
            "ACC_magnitude": np.nan,
            "ACC_std": np.nan,
            "ACC_zero_crossings": np.nan,
            "ACC_energy": np.nan,
            "ACC_peak_freq": np.nan,
        }

        magnitude = self._validate_signal(magnitude)
        if magnitude is None:
            return features

        try:
            features["ACC_magnitude"] = float(np.mean(magnitude))
            features["ACC_std"] = float(np.std(magnitude))

            mag_centered = magnitude - np.mean(magnitude)
            zero_crossings = np.sum(np.diff(np.sign(mag_centered)) != 0)
            duration = len(magnitude) / self.sampling_rate
            features["ACC_zero_crossings"] = (
                float(zero_crossings / duration) if duration > 0 else 0.0
            )

            n_samples = len(magnitude)
            features["ACC_energy"] = (
                float(np.sum(magnitude**2) / n_samples) if n_samples > 0 else 0.0
            )

            if len(magnitude) > 64:
                freqs, psd = scipy_signal.welch(
                    mag_centered,
                    fs=self.sampling_rate,
                    nperseg=min(64, len(magnitude) // 2),
                )
                mask = (freqs >= 0.1) & (freqs <= 10.0)
                if np.any(mask):
                    mov_freqs = freqs[mask]
                    mov_psd = psd[mask]
                    features["ACC_peak_freq"] = float(mov_freqs[np.argmax(mov_psd)])

        except Exception as e:
            self.logger.error(f"Accelerometer feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "ACC_magnitude": "Mean vector magnitude (g or m/s²)",
            "ACC_std": "Magnitude standard deviation",
            "ACC_zero_crossings": "Zero-crossing rate (Hz)",
            "ACC_energy": "Mean squared magnitude (energy)",
            "ACC_peak_freq": "Dominant frequency in movement band (Hz)",
        }
