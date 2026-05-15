from typing import Dict, Tuple

import numpy as np
from scipy import signal

from ..config import FS, FILTER_PARAMS, FEATURE_PARAMS
from ..logging_config import LoggerMixin


class RespiratoryProcessor(LoggerMixin):
    def __init__(self, sampling_rate: float = FS.CHEST):
        self.sampling_rate = sampling_rate
        self.logger.info(f"RespiratoryProcessor initialized with fs={sampling_rate} Hz")

    def bandpass_filter(
        self,
        resp: np.ndarray,
        low: float = FILTER_PARAMS.RESP_BANDPASS_LOW,
        high: float = FILTER_PARAMS.RESP_BANDPASS_HIGH,
        order: int = FILTER_PARAMS.RESP_FILTER_ORDER,
    ) -> np.ndarray:
        resp = np.asarray(resp).flatten()

        nyq = 0.5 * self.sampling_rate
        low_norm = low / nyq
        high_norm = high / nyq

        if low_norm >= 1.0 or high_norm >= 1.0:
            self.logger.warning(
                f"Cutoff frequencies exceed Nyquist ({nyq} Hz), returning unfiltered"
            )
            return resp

        # Lower order for stability
        effective_order = order
        if low_norm < 0.01 or high_norm < 0.01:
            effective_order = min(2, order)
            self.logger.debug(
                f"Reducing filter order to {effective_order} for low frequencies"
            )

        b, a = signal.butter(effective_order, [low_norm, high_norm], btype="band")
        filtered = signal.filtfilt(b, a, resp)

        self.logger.debug(
            f"Applied bandpass filter: {low}-{high} Hz (order={effective_order})"
        )
        return filtered

    def detect_breaths(
        self,
        resp: np.ndarray,
        min_breath_duration: float = 1.5,
        max_breath_duration: float = 10.0,
        method: str = "zero_crossing",
    ) -> Dict[str, np.ndarray]:
        resp = np.asarray(resp).flatten()

        min_distance = int(min_breath_duration * self.sampling_rate)
        max_distance = int(max_breath_duration * self.sampling_rate)

        resp_centered = resp - np.mean(resp)

        sign_changes = np.diff(np.sign(resp_centered))
        pos_crossings = np.where(sign_changes > 0)[0]  # exhale to inhale
        neg_crossings = np.where(sign_changes < 0)[0]  # inhale to exhale
        all_crossings = np.sort(np.concatenate([pos_crossings, neg_crossings]))

        if len(all_crossings) > 1:
            valid_crossings = [all_crossings[0]]
            for i in range(1, len(all_crossings)):
                if all_crossings[i] - valid_crossings[-1] >= min_distance // 2:
                    valid_crossings.append(all_crossings[i])
            all_crossings = np.array(valid_crossings)

        breath_intervals_zc = np.array([])
        if len(pos_crossings) > 1:
            valid_pos = [pos_crossings[0]]
            for i in range(1, len(pos_crossings)):
                if pos_crossings[i] - valid_pos[-1] >= min_distance:
                    valid_pos.append(pos_crossings[i])
            valid_pos = np.array(valid_pos)
            if len(valid_pos) > 1:
                breath_intervals_zc = np.diff(valid_pos) / self.sampling_rate

        peaks, peak_props = signal.find_peaks(
            resp, distance=min_distance, prominence=np.std(resp) * 0.3
        )

        troughs, trough_props = signal.find_peaks(
            -resp, distance=min_distance, prominence=np.std(resp) * 0.3
        )

        valid_peaks = [peaks[0]] if len(peaks) > 0 else []
        for i in range(1, len(peaks)):
            if min_distance <= peaks[i] - valid_peaks[-1] <= max_distance:
                valid_peaks.append(peaks[i])

        valid_troughs = [troughs[0]] if len(troughs) > 0 else []
        for i in range(1, len(troughs)):
            if min_distance <= troughs[i] - valid_troughs[-1] <= max_distance:
                valid_troughs.append(troughs[i])

        peaks = np.array(valid_peaks)
        troughs = np.array(valid_troughs)

        if len(breath_intervals_zc) > 0:
            breath_intervals = breath_intervals_zc
        elif len(peaks) > 1:
            breath_intervals = np.diff(peaks) / self.sampling_rate
        else:
            breath_intervals = np.array([])

        self.logger.debug(
            f"Detected {len(all_crossings)} zero crossings, "
            f"{len(peaks)} inhale peaks, {len(troughs)} exhale troughs"
        )

        return {
            "peaks": peaks,
            "troughs": troughs,
            "breath_intervals": breath_intervals,
            "zero_crossings": all_crossings,
        }

    def compute_breathing_rate(
        self, breath_data: Dict[str, np.ndarray], method: str = "interval"
    ) -> Dict[str, float]:
        features = {}

        breath_intervals = breath_data.get("breath_intervals", np.array([]))

        if len(breath_intervals) > 0:
            mean_interval = np.mean(breath_intervals)
            features["breathing_rate_bpm"] = (
                float(60.0 / mean_interval) if mean_interval > 0 else 0.0
            )

            features["breath_interval_mean"] = float(mean_interval)
            features["breath_interval_std"] = float(np.std(breath_intervals))
            features["breath_interval_cv"] = (
                float(features["breath_interval_std"] / mean_interval)
                if mean_interval > 0
                else 0.0
            )

            if len(breath_intervals) > 1:
                diff_sq = np.diff(breath_intervals) ** 2
                features["breath_rmssd"] = float(np.sqrt(np.mean(diff_sq)))
            else:
                features["breath_rmssd"] = 0.0
        else:
            features["breathing_rate_bpm"] = 0.0
            features["breath_interval_mean"] = 0.0
            features["breath_interval_std"] = 0.0
            features["breath_interval_cv"] = 0.0
            features["breath_rmssd"] = 0.0

        self.logger.debug(f"Breathing rate: {features['breathing_rate_bpm']:.1f} BPM")
        return features

    def compute_spectral_breathing_rate(
        self, resp: np.ndarray, freq_range: Tuple[float, float] = (0.1, 0.5)
    ) -> float:
        resp = np.asarray(resp).flatten()

        # Normalize large signals
        resp_std = np.std(resp)
        if resp_std > 1e6:
            resp = resp / resp_std
            self.logger.debug(
                f"Normalized signal with std={resp_std:.2e} for spectral computation"
            )

        resp = np.clip(resp, -1e6, 1e6)

        nperseg = min(int(30 * self.sampling_rate), len(resp))
        freqs, psd = signal.welch(resp, fs=self.sampling_rate, nperseg=nperseg)

        band_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not np.any(band_mask):
            return 0.0

        band_freqs = freqs[band_mask]
        band_psd = psd[band_mask]

        peak_idx = np.argmax(band_psd)
        peak_freq = band_freqs[peak_idx]

        breathing_rate = peak_freq * 60.0

        self.logger.debug(f"Spectral breathing rate: {breathing_rate:.1f} BPM")
        return float(breathing_rate)

    def compute_amplitude_features(
        self, resp: np.ndarray, breath_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        resp = np.asarray(resp).flatten()
        peaks = breath_data.get("peaks", np.array([]))
        troughs = breath_data.get("troughs", np.array([]))

        features = {}

        if len(peaks) > 0:
            peak_values = resp[peaks]
            features["inspiration_depth_mean"] = float(np.mean(peak_values))
            features["inspiration_depth_std"] = float(np.std(peak_values))
        else:
            features["inspiration_depth_mean"] = 0.0
            features["inspiration_depth_std"] = 0.0

        if len(troughs) > 0:
            trough_values = resp[troughs]
            features["expiration_depth_mean"] = float(np.mean(trough_values))
            features["expiration_depth_std"] = float(np.std(trough_values))
        else:
            features["expiration_depth_mean"] = 0.0
            features["expiration_depth_std"] = 0.0

        amplitudes = []
        for i, peak_idx in enumerate(peaks):
            troughs_before = troughs[troughs < peak_idx]
            troughs_after = troughs[troughs > peak_idx]

            if len(troughs_before) > 0 and len(troughs_after) > 0:
                trough_before = troughs_before[-1]
                trough_after = troughs_after[0]
                amp = resp[peak_idx] - 0.5 * (resp[trough_before] + resp[trough_after])
                amplitudes.append(amp)

        if len(amplitudes) > 0:
            features["breath_amplitude_mean"] = float(np.mean(amplitudes))
            features["breath_amplitude_std"] = float(np.std(amplitudes))
        else:
            features["breath_amplitude_mean"] = float(np.std(resp))
            features["breath_amplitude_std"] = 0.0

        return features

    def compute_timing_features(
        self, resp: np.ndarray, breath_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        peaks = breath_data.get("peaks", np.array([]))
        troughs = breath_data.get("troughs", np.array([]))

        features = {}

        inspiration_times = []
        expiration_times = []

        for peak_idx in peaks:
            troughs_before = troughs[troughs < peak_idx]
            if len(troughs_before) > 0:
                insp_start = troughs_before[-1]
                insp_time = (peak_idx - insp_start) / self.sampling_rate
                inspiration_times.append(insp_time)

            troughs_after = troughs[troughs > peak_idx]
            if len(troughs_after) > 0:
                exp_end = troughs_after[0]
                exp_time = (exp_end - peak_idx) / self.sampling_rate
                expiration_times.append(exp_time)

        if len(inspiration_times) > 0:
            features["inspiration_time_mean"] = float(np.mean(inspiration_times))
            features["inspiration_time_std"] = float(np.std(inspiration_times))
        else:
            features["inspiration_time_mean"] = 0.0
            features["inspiration_time_std"] = 0.0

        if len(expiration_times) > 0:
            features["expiration_time_mean"] = float(np.mean(expiration_times))
            features["expiration_time_std"] = float(np.std(expiration_times))
        else:
            features["expiration_time_mean"] = 0.0
            features["expiration_time_std"] = 0.0

        # I:E ratio
        if features["expiration_time_mean"] > 0:
            features["ie_ratio"] = float(
                features["inspiration_time_mean"] / features["expiration_time_mean"]
            )
        else:
            features["ie_ratio"] = 0.0

        return features

    def compute_signal_quality(self, resp: np.ndarray) -> Dict[str, float]:
        resp = np.asarray(resp).flatten()

        resp_std = np.std(resp)
        if resp_std > 1e6:
            resp = resp / resp_std
            self.logger.debug(
                f"Normalized signal with std={resp_std:.2e} for quality computation"
            )

        resp = np.clip(resp, -1e6, 1e6)

        quality = {}

        # Periodicity check
        autocorr = np.correlate(resp, resp, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / autocorr[0]

        min_lag = int(1.5 * self.sampling_rate)
        max_lag = int(10 * self.sampling_rate)
        search_region = autocorr[min_lag : min(max_lag, len(autocorr))]

        if len(search_region) > 0:
            peaks, _ = signal.find_peaks(search_region)
            if len(peaks) > 0:
                quality["periodicity"] = float(search_region[peaks[0]])
            else:
                quality["periodicity"] = 0.0
        else:
            quality["periodicity"] = 0.0

        spectral_br = self.compute_spectral_breathing_rate(resp)
        if spectral_br > 0:
            br_freq = spectral_br / 60.0
            nyq = 0.5 * self.sampling_rate

            try:
                sig_low = max(0.01, br_freq - 0.05) / nyq
                sig_high = min(0.99, (br_freq + 0.05) / nyq)
                b, a = signal.butter(2, [sig_low, sig_high], btype="band")
                sig_component = signal.filtfilt(b, a, resp)
                signal_power = np.var(sig_component)

                noise_cutoff = min(0.99, max(br_freq + 0.1, 0.5) / nyq)
                b, a = signal.butter(2, noise_cutoff, btype="high")
                noise_component = signal.filtfilt(b, a, resp)
                noise_power = np.var(noise_component) + FEATURE_PARAMS.EPSILON

                quality["snr_db"] = float(10 * np.log10(signal_power / noise_power))
            except Exception:
                quality["snr_db"] = 0.0
        else:
            quality["snr_db"] = 0.0

        quality["quality_score"] = float(
            np.clip(quality["periodicity"] * (1 + quality["snr_db"] / 20) / 2, 0, 1)
        )

        self.logger.debug(
            f"Respiratory quality: periodicity={quality['periodicity']:.2f}"
        )
        return quality

    def process(self, resp: np.ndarray) -> Dict[str, np.ndarray]:
        self.logger.info(f"Processing respiratory signal ({len(resp)} samples)")

        filtered = self.bandpass_filter(resp)
        breath_data = self.detect_breaths(filtered)
        rate_features = self.compute_breathing_rate(breath_data)

        rate_features["breathing_rate_spectral"] = self.compute_spectral_breathing_rate(
            filtered
        )

        amplitude_features = self.compute_amplitude_features(filtered, breath_data)
        timing_features = self.compute_timing_features(filtered, breath_data)

        all_features = {}
        all_features.update(rate_features)
        all_features.update(amplitude_features)
        all_features.update(timing_features)

        quality = self.compute_signal_quality(resp)

        results = {
            "filtered_resp": filtered,
            "breaths": breath_data,
            "features": all_features,
            "quality": quality,
        }

        self.logger.info(
            f"Respiratory processed: {all_features['breathing_rate_bpm']:.1f} BPM, "
            f"{len(breath_data['peaks'])} breaths detected"
        )

        return results
