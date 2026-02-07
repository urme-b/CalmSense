from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from ..logging_config import LoggerMixin


class EDAFeatureExtractor(LoggerMixin):
    def __init__(self, sampling_rate: float = 4.0):
        self.sampling_rate = sampling_rate
        self.logger.debug(f"EDAFeatureExtractor initialized, fs={sampling_rate} Hz")

    def _validate_signal(self, signal: np.ndarray) -> Optional[np.ndarray]:
        if signal is None:
            return None

        signal = np.asarray(signal).flatten()
        signal = signal[np.isfinite(signal)]

        if len(signal) < 10:
            self.logger.warning("EDA signal too short")
            return None

        return signal

    def extract_tonic_features(self, scl: np.ndarray) -> Dict[str, float]:
        features = {
            "SCL_mean": np.nan,
            "SCL_std": np.nan,
            "SCL_slope": np.nan,
            "SCL_min": np.nan,
            "SCL_max": np.nan,
        }

        scl = self._validate_signal(scl)
        if scl is None:
            return features

        try:
            features["SCL_mean"] = float(np.mean(scl))
            features["SCL_std"] = float(np.std(scl))
            features["SCL_min"] = float(np.min(scl))
            features["SCL_max"] = float(np.max(scl))

            x = np.arange(len(scl)) / self.sampling_rate
            if len(x) > 1:
                slope, _, _, _, _ = stats.linregress(x, scl)
                features["SCL_slope"] = float(slope)

        except Exception as e:
            self.logger.warning(f"Tonic feature extraction failed: {e}")

        return features

    def extract_phasic_features(
        self, scr_peaks: List[Dict], signal_duration: float
    ) -> Dict[str, float]:
        features = {
            "SCR_count": np.nan,
            "SCR_rate": np.nan,
            "SCR_amplitude_mean": np.nan,
            "SCR_amplitude_max": np.nan,
            "SCR_rise_time_mean": np.nan,
            "SCR_recovery_time_mean": np.nan,
            "SCR_AUC": np.nan,
        }

        if scr_peaks is None or len(scr_peaks) == 0:
            features["SCR_count"] = 0.0
            features["SCR_rate"] = 0.0
            features["SCR_amplitude_mean"] = 0.0
            features["SCR_amplitude_max"] = 0.0
            features["SCR_rise_time_mean"] = 0.0
            features["SCR_recovery_time_mean"] = 0.0
            features["SCR_AUC"] = 0.0
            return features

        try:
            n_scr = len(scr_peaks)
            features["SCR_count"] = float(n_scr)

            duration_min = signal_duration / 60.0
            features["SCR_rate"] = (
                float(n_scr / duration_min) if duration_min > 0 else 0.0
            )

            amplitudes = [
                scr.get("amplitude", 0) for scr in scr_peaks if "amplitude" in scr
            ]
            if len(amplitudes) > 0:
                features["SCR_amplitude_mean"] = float(np.mean(amplitudes))
                features["SCR_amplitude_max"] = float(np.max(amplitudes))

            rise_times = [
                scr.get("rise_time", 0) for scr in scr_peaks if "rise_time" in scr
            ]
            if len(rise_times) > 0:
                features["SCR_rise_time_mean"] = float(np.mean(rise_times))

            recovery_times = [
                scr.get("recovery_time", 0)
                for scr in scr_peaks
                if "recovery_time" in scr and scr.get("recovery_time") is not None
            ]
            if len(recovery_times) > 0:
                features["SCR_recovery_time_mean"] = float(np.mean(recovery_times))

            auc_total = 0.0
            for scr in scr_peaks:
                amp = scr.get("amplitude", 0)
                rise = scr.get("rise_time", 0)
                recovery = scr.get("recovery_time")
                if recovery is None:
                    recovery = rise * 2
                auc_total += 0.5 * amp * (rise + recovery)
            features["SCR_AUC"] = float(auc_total)

        except Exception as e:
            self.logger.warning(f"Phasic feature extraction failed: {e}")

        return features

    def extract_statistical_features(self, eda: np.ndarray) -> Dict[str, float]:
        features = {
            "EDA_mean": np.nan,
            "EDA_range": np.nan,
            "EDA_kurtosis": np.nan,
        }

        eda = self._validate_signal(eda)
        if eda is None:
            return features

        try:
            features["EDA_mean"] = float(np.mean(eda))
            features["EDA_range"] = float(np.ptp(eda))
            features["EDA_kurtosis"] = float(stats.kurtosis(eda))
        except Exception as e:
            self.logger.warning(f"Statistical feature extraction failed: {e}")

        return features

    def extract_all(
        self,
        eda_decomposed: Dict,
        scr_peaks: Optional[List[Dict]] = None,
        raw_eda: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        features = {}

        tonic = eda_decomposed.get("tonic") if eda_decomposed else None
        if tonic is not None:
            tonic_features = self.extract_tonic_features(tonic)
            features.update(tonic_features)
            signal_duration = len(tonic) / self.sampling_rate
        else:
            features.update(
                {
                    "SCL_mean": np.nan,
                    "SCL_std": np.nan,
                    "SCL_slope": np.nan,
                    "SCL_min": np.nan,
                    "SCL_max": np.nan,
                }
            )
            signal_duration = 60.0

        phasic_features = self.extract_phasic_features(scr_peaks, signal_duration)
        features.update(phasic_features)

        if raw_eda is None and eda_decomposed:
            tonic = eda_decomposed.get("tonic")
            phasic = eda_decomposed.get("phasic")
            if tonic is not None and phasic is not None:
                raw_eda = tonic + phasic

        stat_features = self.extract_statistical_features(raw_eda)
        features.update(stat_features)

        if np.isfinite(features.get("SCL_mean", np.nan)):
            self.logger.debug(
                f"Extracted 15 EDA features, "
                f"SCL_mean={features['SCL_mean']:.2f}µS, "
                f"SCR_count={features.get('SCR_count', 0)}"
            )
        else:
            self.logger.debug("Extracted 15 EDA features")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "SCL_mean": "Mean skin conductance level (µS)",
            "SCL_std": "Standard deviation of SCL (µS)",
            "SCL_slope": "Linear trend of SCL (µS/s)",
            "SCL_min": "Minimum SCL (µS)",
            "SCL_max": "Maximum SCL (µS)",
            "SCR_count": "Number of SCR events",
            "SCR_rate": "SCR events per minute",
            "SCR_amplitude_mean": "Mean SCR amplitude (µS)",
            "SCR_amplitude_max": "Maximum SCR amplitude (µS)",
            "SCR_rise_time_mean": "Mean SCR rise time (s)",
            "SCR_recovery_time_mean": "Mean SCR half-recovery time (s)",
            "SCR_AUC": "Total area under SCR curves (µS·s)",
            "EDA_mean": "Overall EDA mean (µS)",
            "EDA_range": "EDA dynamic range (µS)",
            "EDA_kurtosis": "EDA distribution kurtosis",
        }
