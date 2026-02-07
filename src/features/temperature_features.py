from typing import Dict, Optional

import numpy as np
from scipy import stats

from ..logging_config import LoggerMixin


class TemperatureFeatureExtractor(LoggerMixin):
    def __init__(self, sampling_rate: float = 4.0):
        self.sampling_rate = sampling_rate
        self.logger.debug(
            f"TemperatureFeatureExtractor initialized, fs={sampling_rate} Hz"
        )

    def _validate_signal(self, signal: np.ndarray) -> Optional[np.ndarray]:
        if signal is None:
            return None

        signal = np.asarray(signal).flatten()
        signal = signal[np.isfinite(signal)]

        if len(signal) < 10:
            self.logger.warning("Temperature signal too short")
            return None

        if np.mean(signal) < 15 or np.mean(signal) > 45:
            self.logger.warning(
                f"Temperature out of physiological range: mean={np.mean(signal):.1f}"
            )

        return signal

    def extract_all(self, temp: np.ndarray) -> Dict[str, float]:
        features = {
            "TEMP_mean": np.nan,
            "TEMP_std": np.nan,
            "TEMP_slope": np.nan,
            "TEMP_min": np.nan,
            "TEMP_max": np.nan,
        }

        temp = self._validate_signal(temp)
        if temp is None:
            self.logger.warning("Invalid temperature signal, returning NaN features")
            return features

        try:
            features["TEMP_mean"] = float(np.mean(temp))
            features["TEMP_std"] = float(np.std(temp))
            features["TEMP_min"] = float(np.min(temp))
            features["TEMP_max"] = float(np.max(temp))

            x = np.arange(len(temp)) / self.sampling_rate
            if len(x) > 1:
                slope, _, _, _, _ = stats.linregress(x, temp)
                features["TEMP_slope"] = float(slope)

            self.logger.debug(
                f"Extracted 5 temperature features, mean={features['TEMP_mean']:.2f}°C"
            )

        except Exception as e:
            self.logger.error(f"Temperature feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "TEMP_mean": "Mean skin temperature (°C)",
            "TEMP_std": "Temperature standard deviation (°C)",
            "TEMP_slope": "Temperature linear trend (°C/s)",
            "TEMP_min": "Minimum temperature (°C)",
            "TEMP_max": "Maximum temperature (°C)",
        }
