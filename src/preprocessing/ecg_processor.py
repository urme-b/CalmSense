from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from ..config import FILTER_PARAMS, FS
from ..logging_config import LoggerMixin


class ECGProcessor(LoggerMixin):
    """ECG processing for HRV analysis: bandpass filtering, R-peak detection, and
    RR-interval extraction with ectopic-beat correction.

    R-peaks are detected with NeuroKit2 when available, falling back to a built-in
    Pan-Tompkins detector. All methods take 1-D arrays sampled at ``sampling_rate`` Hz.
    """

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
            self.logger.warning(f"Cutoff frequencies ({low}, {high}) exceed Nyquist ({nyq} Hz)")
            return ecg

        b, a = signal.butter(order, [low_norm, high_norm], btype="band")
        filtered = signal.filtfilt(b, a, ecg)

        self.logger.debug(f"Applied bandpass filter: {low}-{high} Hz, order={order}")
        return filtered

    def detect_r_peaks(self, ecg: np.ndarray, method: str = "pantompkins") -> np.ndarray:
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
        integrated = np.convolve(squared, np.ones(window_size) / window_size, mode="same")

        init_samples = int(2 * self.sampling_rate)
        threshold = 0.5 * np.max(integrated[: min(init_samples, len(integrated))])

        # 300 BPM max
        min_rr = int(0.2 * self.sampling_rate)

        r_peaks = []
        search_start = 0

        while search_start < len(integrated) - min_rr:
            search_window = integrated[search_start : search_start + int(self.sampling_rate)]

            if len(search_window) == 0:
                break

            peaks, _ = signal.find_peaks(search_window, height=threshold, distance=min_rr)

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
        """Convert R-peak sample indices to successive RR intervals.

        Args:
            r_peaks: R-peak sample indices (ascending).
            unit: Output unit, one of ``"ms"``, ``"s"``, or ``"samples"``.

        Returns:
            RR intervals in the requested unit; empty array if fewer than two peaks.
        """
        if unit not in ("ms", "s", "samples"):
            raise ValueError(f"unit must be 'ms', 's', or 'samples', got {unit!r}")

        r_peaks = np.asarray(r_peaks).flatten()

        if len(r_peaks) < 2:
            self.logger.warning("Less than 2 R-peaks, cannot compute RR intervals")
            return np.array([])

        rr_samples = np.diff(r_peaks)

        if unit == "ms":
            rr_intervals = (rr_samples / self.sampling_rate) * 1000
        elif unit == "s":
            rr_intervals = rr_samples / self.sampling_rate
        else:  # "samples"
            rr_intervals = rr_samples.astype(float)

        self.logger.debug(
            f"Extracted {len(rr_intervals)} RR intervals, mean={np.mean(rr_intervals):.1f} {unit}"
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
            raise ValueError(f"Mask length {len(valid_mask)} doesn't match RR length {len(rr)}")

        if np.all(valid_mask):
            return rr
        if not np.any(valid_mask):
            self.logger.warning("All intervals marked as invalid, returning mean")
            return np.full_like(rr, np.mean(rr))

        x_valid = np.where(valid_mask)[0]
        x_all = np.arange(len(rr))

        # A single valid beat cannot be interpolated; fall back to that value
        if len(x_valid) == 1:
            return np.full_like(rr, float(rr[valid_mask][0]))

        # Drop to an order the valid points can support (cubic needs >=4, quadratic >=3)
        if method == "cubic" and len(x_valid) < 4:
            method = "quadratic" if len(x_valid) >= 3 else "linear"

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
