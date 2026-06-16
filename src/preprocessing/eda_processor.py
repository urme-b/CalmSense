from typing import Dict, List, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from ..config import FILTER_PARAMS, FS
from ..logging_config import LoggerMixin


class EDAProcessor(LoggerMixin):
    def __init__(self, sampling_rate: float = FS.WRIST_EDA):
        self.sampling_rate = sampling_rate
        self.logger.info(f"EDAProcessor initialized with fs={sampling_rate} Hz")

    def lowpass_filter(
        self,
        eda: np.ndarray,
        cutoff: float = FILTER_PARAMS.EDA_LOWPASS,
        order: int = FILTER_PARAMS.EDA_FILTER_ORDER,
    ) -> np.ndarray:
        eda = np.asarray(eda).flatten()

        nyq = 0.5 * self.sampling_rate
        cutoff_norm = cutoff / nyq

        if cutoff_norm >= 1.0:
            self.logger.warning(
                f"Cutoff {cutoff} Hz exceeds Nyquist {nyq} Hz, returning unfiltered"
            )
            return eda

        b, a = signal.butter(order, cutoff_norm, btype="low")
        filtered = signal.filtfilt(b, a, eda)

        self.logger.debug(f"Applied low-pass filter: {cutoff} Hz, order={order}")
        return filtered

    def remove_artifacts(
        self, eda: np.ndarray, median_kernel: int = FILTER_PARAMS.EDA_MEDIAN_SIZE
    ) -> np.ndarray:
        eda = np.asarray(eda).flatten()
        filtered = median_filter(eda, size=median_kernel)
        self.logger.debug(f"Applied median filter: kernel={median_kernel}")
        return filtered

    def decompose_eda(
        self, eda: np.ndarray, method: str = "highpass"
    ) -> Tuple[np.ndarray, np.ndarray]:
        eda = np.asarray(eda).flatten()

        if method == "cvxeda":
            return self._decompose_cvxeda(eda)
        elif method == "median":
            return self._decompose_median(eda)
        else:
            return self._decompose_highpass(eda)

    def _decompose_highpass(
        self, eda: np.ndarray, cutoff: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * self.sampling_rate
        cutoff_norm = cutoff / nyq

        if cutoff_norm >= 1.0:
            self.logger.warning("Cutoff too high for Nyquist, returning original as tonic")
            return eda.copy(), np.zeros_like(eda)

        b, a = signal.butter(2, cutoff_norm, btype="low")
        tonic = signal.filtfilt(b, a, eda)
        phasic = eda - tonic

        self.logger.debug("Decomposed EDA using high-pass method")
        return tonic, phasic

    def _decompose_median(
        self, eda: np.ndarray, window_sec: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        window_samples = int(window_sec * self.sampling_rate)
        if window_samples % 2 == 0:
            window_samples += 1
        window_samples = max(3, window_samples)

        tonic = median_filter(eda, size=window_samples)
        phasic = eda - tonic

        self.logger.debug(f"Decomposed EDA using median filter ({window_samples} samples)")
        return tonic, phasic

    def _decompose_cvxeda(self, eda: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import neurokit2 as nk

            decomposed = nk.eda_phasic(eda, sampling_rate=int(self.sampling_rate), method="cvxeda")
            tonic = decomposed["EDA_Tonic"].values
            phasic = decomposed["EDA_Phasic"].values
            self.logger.debug("Decomposed EDA using cvxEDA (neurokit2)")
            return tonic, phasic
        except ImportError:
            self.logger.debug("neurokit2 not available, using median decomposition")
            return self._decompose_median(eda)
        except Exception as e:
            self.logger.warning(f"cvxEDA failed: {e}, using median decomposition")
            return self._decompose_median(eda)

    def detect_scr_peaks(
        self,
        phasic: np.ndarray,
        min_amplitude: float = 0.01,
        min_rise_time: float = 0.5,
        max_rise_time: float = 4.0,
    ) -> Tuple[np.ndarray, List[Dict]]:
        phasic = np.asarray(phasic).flatten()

        min_rise_samples = int(min_rise_time * self.sampling_rate)
        max_rise_samples = int(max_rise_time * self.sampling_rate)
        min_distance = max(1, min_rise_samples)

        peaks, properties = signal.find_peaks(
            phasic,
            height=min_amplitude,
            distance=min_distance,
            prominence=min_amplitude / 2,
        )

        scr_features = []
        valid_peaks = []

        for i, peak_idx in enumerate(peaks):
            search_start = max(0, peak_idx - max_rise_samples)
            onset_region = phasic[search_start:peak_idx]

            if len(onset_region) == 0:
                continue

            onset_local_idx = np.argmin(onset_region)
            onset_idx = search_start + onset_local_idx

            rise_samples = peak_idx - onset_idx
            rise_time = rise_samples / self.sampling_rate

            if rise_time < min_rise_time or rise_time > max_rise_time:
                continue

            amplitude = phasic[peak_idx] - phasic[onset_idx]

            if amplitude < min_amplitude:
                continue

            # Half-recovery point
            recovery_target = phasic[peak_idx] - amplitude / 2
            search_end = min(len(phasic), peak_idx + max_rise_samples * 2)
            recovery_region = phasic[peak_idx:search_end]

            recovery_idx = None
            for j, val in enumerate(recovery_region):
                if val <= recovery_target:
                    recovery_idx = peak_idx + j
                    break

            recovery_time = None
            if recovery_idx is not None:
                recovery_time = (recovery_idx - peak_idx) / self.sampling_rate

            valid_peaks.append(peak_idx)
            scr_features.append(
                {
                    "peak_idx": peak_idx,
                    "onset_idx": onset_idx,
                    "amplitude": float(amplitude),
                    "rise_time": float(rise_time),
                    "recovery_time": float(recovery_time) if recovery_time else None,
                    "peak_time": float(peak_idx / self.sampling_rate),
                }
            )

        self.logger.debug(f"Detected {len(valid_peaks)} SCR peaks")
        return np.array(valid_peaks), scr_features
