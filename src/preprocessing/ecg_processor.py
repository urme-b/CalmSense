from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from ..config import FS, FILTER_PARAMS, FEATURE_PARAMS
from ..logging_config import LoggerMixin


class ECGProcessor(LoggerMixin):
    def __init__(self, sampling_rate: float = FS.CHEST):
        self.sampling_rate = sampling_rate
        self.logger.info(f"ECGProcessor initialized with fs={sampling_rate} Hz")

    def bandpass_filter(
        self,
        ecg: np.ndarray,
        low: float = FILTER_PARAMS.ECG_BANDPASS_LOW,
        high: float = FILTER_PARAMS.ECG_BANDPASS_HIGH,
        order: int = FILTER_PARAMS.ECG_FILTER_ORDER,
    ) -> np.ndarray:
        ecg = np.asarray(ecg).flatten()

        nyq = 0.5 * self.sampling_rate
        low_norm = low / nyq
        high_norm = high / nyq

        if low_norm >= 1.0 or high_norm >= 1.0:
            self.logger.warning(
                f"Cutoff frequencies ({low}, {high}) exceed Nyquist ({nyq} Hz)"
            )
            return ecg

        b, a = signal.butter(order, [low_norm, high_norm], btype="band")
        filtered = signal.filtfilt(b, a, ecg)

        self.logger.debug(f"Applied bandpass filter: {low}-{high} Hz, order={order}")
        return filtered

    def detect_r_peaks(
        self, ecg: np.ndarray, method: str = "pantompkins"
    ) -> np.ndarray:
        ecg = np.asarray(ecg).flatten()
        duration_sec = len(ecg) / self.sampling_rate

        try:
            import neurokit2 as nk

            _, info = nk.ecg_peaks(ecg, sampling_rate=int(self.sampling_rate))
            r_peaks = np.array(info["ECG_R_Peaks"])
        except ImportError:
            self.logger.debug("neurokit2 not available, using Pan-Tompkins")
            r_peaks = self._pan_tompkins(ecg)

        # Validate peak count
        n_peaks = len(r_peaks)
        expected_min = int(0.5 * duration_sec)
        expected_max = int(3.5 * duration_sec)

        if n_peaks < expected_min:
            self.logger.warning(
                f"Suspiciously few R-peaks: {n_peaks} in {duration_sec:.1f}s "
                f"(expected >={expected_min})"
            )
        elif n_peaks > expected_max:
            self.logger.warning(
                f"Suspiciously many R-peaks: {n_peaks} in {duration_sec:.1f}s "
                f"(expected <={expected_max})"
            )

        self.logger.debug(f"Detected {n_peaks} R-peaks in {duration_sec:.1f}s")
        return r_peaks

    def _pan_tompkins(self, ecg: np.ndarray) -> np.ndarray:
        diff_ecg = np.diff(ecg)
        squared = diff_ecg**2

        # 150ms integration window
        window_size = int(0.150 * self.sampling_rate)
        integrated = np.convolve(
            squared, np.ones(window_size) / window_size, mode="same"
        )

        init_samples = int(2 * self.sampling_rate)
        threshold = 0.5 * np.max(integrated[: min(init_samples, len(integrated))])

        # 300 BPM max
        min_rr = int(0.2 * self.sampling_rate)

        r_peaks = []
        search_start = 0

        while search_start < len(integrated) - min_rr:
            search_window = integrated[
                search_start : search_start + int(self.sampling_rate)
            ]

            if len(search_window) == 0:
                break

            peaks, _ = signal.find_peaks(
                search_window, height=threshold, distance=min_rr
            )

            if len(peaks) > 0:
                peak_idx = search_start + peaks[0]

                # Refine peak
                refine_window = int(0.05 * self.sampling_rate)
                start = max(0, peak_idx - refine_window)
                end = min(len(ecg), peak_idx + refine_window)

                refined_peak = start + np.argmax(ecg[start:end])
                r_peaks.append(refined_peak)

                # Adaptive threshold
                threshold = 0.5 * (threshold + 0.25 * integrated[peak_idx])

                search_start = refined_peak + min_rr
            else:
                threshold *= 0.8
                search_start += int(0.5 * self.sampling_rate)

        return np.array(r_peaks)

    def extract_rr_intervals(self, r_peaks: np.ndarray, unit: str = "ms") -> np.ndarray:
        r_peaks = np.asarray(r_peaks).flatten()

        if len(r_peaks) < 2:
            self.logger.warning("Less than 2 R-peaks, cannot compute RR intervals")
            return np.array([])

        rr_samples = np.diff(r_peaks)

        if unit == "ms":
            rr_intervals = (rr_samples / self.sampling_rate) * 1000
        elif unit == "s":
            rr_intervals = rr_samples / self.sampling_rate
        else:
            rr_intervals = rr_samples.astype(float)

        self.logger.debug(
            f"Extracted {len(rr_intervals)} RR intervals, "
            f"mean={np.mean(rr_intervals):.1f} {unit}"
        )
        return rr_intervals

    def remove_ectopic_beats(
        self, rr_intervals: np.ndarray, threshold: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        rr = np.asarray(rr_intervals).flatten()

        if len(rr) < 3:
            return rr, np.ones(len(rr), dtype=bool)

        window = 5
        valid_mask = np.ones(len(rr), dtype=bool)

        for i in range(len(rr)):
            start = max(0, i - window // 2)
            end = min(len(rr), i + window // 2 + 1)

            local_median = np.median(rr[start:end])

            if local_median > 0:
                relative_diff = abs(rr[i] - local_median) / local_median
                if relative_diff > threshold:
                    valid_mask[i] = False

        n_ectopic = np.sum(~valid_mask)
        if n_ectopic > 0:
            self.logger.debug(
                f"Marked {n_ectopic} ectopic beats ({100 * n_ectopic / len(rr):.1f}%)"
            )

        return rr[valid_mask], valid_mask

    def interpolate_artifacts(
        self,
        rr_intervals: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        method: str = "cubic",
    ) -> np.ndarray:
        rr = np.asarray(rr_intervals).flatten()

        if valid_mask is None:
            return rr

        valid_mask = np.asarray(valid_mask).flatten()

        if len(valid_mask) != len(rr):
            raise ValueError(
                f"Mask length {len(valid_mask)} doesn't match RR length {len(rr)}"
            )

        if np.all(valid_mask):
            return rr
        if not np.any(valid_mask):
            self.logger.warning("All intervals marked as invalid, returning mean")
            return np.full_like(rr, np.mean(rr))

        x_valid = np.where(valid_mask)[0]
        x_all = np.arange(len(rr))

        interpolator = interp1d(
            x_valid,
            rr[valid_mask],
            kind=method,
            fill_value="extrapolate",
            bounds_error=False,
        )

        interpolated = interpolator(x_all)

        n_interpolated = np.sum(~valid_mask)
        self.logger.debug(f"Interpolated {n_interpolated} RR intervals using {method}")

        return interpolated

    def compute_signal_quality(self, ecg: np.ndarray) -> Dict[str, float]:
        ecg = np.asarray(ecg).flatten()

        quality = {}

        # SNR estimation
        nyq = 0.5 * self.sampling_rate

        try:
            signal_low = min(5 / nyq, 0.99)
            signal_high = min(15 / nyq, 0.99)
            b, a = signal.butter(2, [signal_low, signal_high], btype="band")
            ecg_signal = signal.filtfilt(b, a, ecg)
            signal_power = np.var(ecg_signal)

            noise_low = min(50 / nyq, 0.99)
            if noise_low < 0.99:
                b, a = signal.butter(2, noise_low, btype="high")
                ecg_noise = signal.filtfilt(b, a, ecg)
                noise_power = np.var(ecg_noise) + FEATURE_PARAMS.EPSILON
            else:
                noise_power = FEATURE_PARAMS.EPSILON

            quality["snr_db"] = float(10 * np.log10(signal_power / noise_power))
        except Exception as e:
            self.logger.warning(f"SNR computation failed: {e}")
            quality["snr_db"] = 0.0

        from scipy.stats import kurtosis

        quality["kurtosis"] = float(kurtosis(ecg))

        # Baseline stability
        try:
            baseline_cutoff = min(0.5 / nyq, 0.99)
            b, a = signal.butter(2, baseline_cutoff, btype="low")
            baseline = signal.filtfilt(b, a, ecg)
            quality["baseline_variance"] = float(np.var(baseline))
        except Exception:
            quality["baseline_variance"] = float(np.var(ecg))

        try:
            r_peaks = self.detect_r_peaks(self.bandpass_filter(ecg))
            rr = self.extract_rr_intervals(r_peaks)
            _, valid_mask = self.remove_ectopic_beats(rr)
            quality["valid_beat_ratio"] = float(np.mean(valid_mask))
        except Exception:
            quality["valid_beat_ratio"] = 0.0

        self.logger.debug(f"ECG quality metrics: SNR={quality['snr_db']:.1f} dB")
        return quality

    def process(self, ecg: np.ndarray) -> Dict[str, np.ndarray]:
        self.logger.info(f"Processing ECG signal ({len(ecg)} samples)")

        filtered = self.bandpass_filter(ecg)
        r_peaks = self.detect_r_peaks(filtered)
        rr_ms = self.extract_rr_intervals(r_peaks, unit="ms")

        rr_clean, valid_mask = self.remove_ectopic_beats(rr_ms)
        rr_interpolated = self.interpolate_artifacts(rr_ms, valid_mask)
        quality = self.compute_signal_quality(ecg)

        results = {
            "filtered_ecg": filtered,
            "r_peaks": r_peaks,
            "rr_intervals_ms": rr_ms,
            "rr_clean": rr_clean,
            "rr_interpolated": rr_interpolated,
            "valid_beat_mask": valid_mask,
            "quality": quality,
        }

        if len(rr_ms) > 0:
            self.logger.info(
                f"ECG processed: {len(r_peaks)} R-peaks, mean HR={60000 / np.mean(rr_ms):.1f} BPM"
            )
        else:
            self.logger.info("ECG processed")

        return results
