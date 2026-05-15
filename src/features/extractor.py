from typing import Dict

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats

from ..config import FS, FEATURE_PARAMS
from ..logging_config import LoggerMixin


class FeatureExtractor(LoggerMixin):
    def __init__(self, fs: float = FS.CHEST):
        self.fs = fs
        self.logger.debug(f"FeatureExtractor initialized with fs={fs} Hz")

    def extract_time_domain(self, data: np.ndarray) -> Dict[str, float]:
        data = np.asarray(data).flatten()
        features: Dict[str, float] = {}

        if len(data) == 0:
            return features

        # Strip NaN
        nan_mask = np.isfinite(data)
        if not np.all(nan_mask):
            nan_ratio = 1.0 - np.mean(nan_mask)
            self.logger.debug(f"Input has {nan_ratio:.1%} NaN/Inf, filtering")
            data = data[nan_mask]
            if len(data) == 0:
                return features

        features["mean"] = float(np.mean(data))
        features["std"] = float(np.std(data))
        features["var"] = float(np.var(data))
        features["min"] = float(np.min(data))
        features["max"] = float(np.max(data))
        features["range"] = float(np.ptp(data))
        features["median"] = float(np.median(data))

        features["skewness"] = float(stats.skew(data))
        features["kurtosis"] = float(stats.kurtosis(data))

        features["rms"] = float(np.sqrt(np.mean(data**2)))
        features["energy"] = float(np.sum(data**2))

        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        features["zcr"] = float(zero_crossings / len(data))

        features["p25"] = float(np.percentile(data, 25))
        features["p75"] = float(np.percentile(data, 75))
        features["iqr"] = features["p75"] - features["p25"]

        return features

    def extract_frequency_domain(self, data: np.ndarray) -> Dict[str, float]:
        data = np.asarray(data).flatten()
        features: Dict[str, float] = {}

        nperseg = min(FEATURE_PARAMS.PSD_NPERSEG, len(data))
        freqs, psd = scipy_signal.welch(data, fs=self.fs, nperseg=nperseg)

        psd = np.maximum(psd, 0)

        features["total_power"] = float(np.trapz(psd, freqs))

        psd_sum = np.sum(psd) + FEATURE_PARAMS.EPSILON
        features["spectral_mean"] = float(np.sum(freqs * psd) / psd_sum)

        spectral_var = np.sum((freqs - features["spectral_mean"]) ** 2 * psd) / psd_sum
        features["spectral_std"] = float(np.sqrt(max(spectral_var, 0)))

        psd_norm = psd / psd_sum
        features["spectral_entropy"] = float(
            stats.entropy(psd_norm + FEATURE_PARAMS.EPSILON)
        )

        features["dominant_freq"] = float(freqs[np.argmax(psd)])
        features["max_psd"] = float(np.max(psd))

        # HRV-style band powers
        vlf_mask = (freqs >= FEATURE_PARAMS.HRV_VLF_LOW) & (
            freqs < FEATURE_PARAMS.HRV_VLF_HIGH
        )
        lf_mask = (freqs >= FEATURE_PARAMS.HRV_LF_LOW) & (
            freqs < FEATURE_PARAMS.HRV_LF_HIGH
        )
        hf_mask = (freqs >= FEATURE_PARAMS.HRV_HF_LOW) & (
            freqs < FEATURE_PARAMS.HRV_HF_HIGH
        )

        features["vlf_power"] = float(
            np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
        )
        features["lf_power"] = float(
            np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        )
        features["hf_power"] = float(
            np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        )

        # LF/HF ratio (autonomic balance)
        features["lf_hf_ratio"] = float(
            features["lf_power"] / (features["hf_power"] + FEATURE_PARAMS.EPSILON)
        )

        return features

    def extract_nonlinear(self, data: np.ndarray) -> Dict[str, float]:
        data = np.asarray(data).flatten()
        features: Dict[str, float] = {}

        features["sample_entropy"] = self._sample_entropy(
            data, m=FEATURE_PARAMS.SAMPLE_ENTROPY_M, r=FEATURE_PARAMS.SAMPLE_ENTROPY_R
        )

        # Hjorth parameters
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)

        var_data = np.var(data) + FEATURE_PARAMS.EPSILON
        var_diff1 = np.var(diff1) + FEATURE_PARAMS.EPSILON
        var_diff2 = np.var(diff2) + FEATURE_PARAMS.EPSILON

        features["hjorth_activity"] = float(var_data)
        features["hjorth_mobility"] = float(np.sqrt(var_diff1 / var_data))
        features["hjorth_complexity"] = float(
            np.sqrt(var_diff2 / var_diff1) / features["hjorth_mobility"]
        )

        return features

    def _sample_entropy(
        self,
        data: np.ndarray,
        m: int = FEATURE_PARAMS.SAMPLE_ENTROPY_M,
        r: float = FEATURE_PARAMS.SAMPLE_ENTROPY_R,
    ) -> float:
        n = len(data)
        if n < m + 2:
            return 0.0

        r_val = r * np.std(data)

        def _count_matches(template_length: int) -> int:
            count = 0
            for i in range(n - template_length):
                for j in range(i + 1, n - template_length):
                    template_i = data[i : i + template_length]
                    template_j = data[j : j + template_length]
                    if np.max(np.abs(template_i - template_j)) < r_val:
                        count += 1
            return count

        a = _count_matches(m + 1)
        b = _count_matches(m)

        if b == 0:
            return 0.0

        return float(
            -np.log((a + FEATURE_PARAMS.EPSILON) / (b + FEATURE_PARAMS.EPSILON))
        )

    def extract_all(self, data: np.ndarray, prefix: str = "") -> Dict[str, float]:
        features: Dict[str, float] = {}

        time_features = self.extract_time_domain(data)
        for k, v in time_features.items():
            features[f"{prefix}{k}"] = v

        freq_features = self.extract_frequency_domain(data)
        for k, v in freq_features.items():
            features[f"{prefix}{k}"] = v

        nonlinear_features = self.extract_nonlinear(data)
        for k, v in nonlinear_features.items():
            features[f"{prefix}{k}"] = v

        self.logger.debug(f"Extracted {len(features)} features with prefix '{prefix}'")

        return features

    def extract_from_segments(
        self, segments: np.ndarray, prefix: str = ""
    ) -> np.ndarray:
        first_features = self.extract_all(segments[0], prefix=prefix)
        feature_names = list(first_features.keys())
        n_features = len(feature_names)

        n_segments = len(segments)
        features_array = np.zeros((n_segments, n_features))

        features_array[0] = list(first_features.values())

        for i in range(1, n_segments):
            segment_features = self.extract_all(segments[i], prefix=prefix)
            features_array[i] = [segment_features[name] for name in feature_names]

        self.logger.info(
            f"Extracted features from {n_segments} segments: "
            f"({n_segments}, {n_features}) array"
        )

        return features_array

    def get_feature_names(self, prefix: str = "") -> list:
        dummy_data = np.random.randn(1000)
        features = self.extract_all(dummy_data, prefix=prefix)
        return list(features.keys())
