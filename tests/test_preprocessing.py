import numpy as np
import pytest

from src.preprocessing import (
    SignalProcessor,
    ECGProcessor,
    EDAProcessor,
    RespiratoryProcessor,
    SignalWindower,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_ecg():

    np.random.seed(42)
    fs = 700  # Hz
    duration = 10  # seconds
    t = np.linspace(0, duration, fs * duration)

    # Create ECG-like signal with
    ecg = np.zeros_like(t)
    heart_rate = 72  # bpm
    rr_interval = 60 / heart_rate  # seconds

    # Add R-peaks at regular
    for peak_time in np.arange(0, duration, rr_interval):
        peak_idx = int(peak_time * fs)
        if peak_idx < len(ecg):
            # QRS complex approximation
            window = 50  # samples
            if peak_idx + window < len(ecg):
                ecg[peak_idx : peak_idx + window] = np.sin(
                    np.linspace(0, np.pi, window)
                )

    # Add baseline and noise
    ecg += 0.1 * np.sin(2 * np.pi * 0.15 * t)  # Baseline wander
    ecg += 0.05 * np.sin(2 * np.pi * 50 * t)  # 50 Hz noise
    ecg += 0.02 * np.random.randn(len(t))  # Random noise

    return ecg, fs


@pytest.fixture
def sample_eda():

    np.random.seed(42)
    fs = 4  # Hz (typical wrist EDA)
    duration = 60  # seconds
    n_samples = fs * duration

    # Tonic component (slow drift)
    t = np.linspace(0, duration, n_samples)
    tonic = 5 + 0.5 * np.sin(2 * np.pi * 0.01 * t)

    # Phasic component (SCRs)
    phasic = np.zeros(n_samples)
    scr_times = [10, 25, 40, 55]  # seconds
    for scr_time in scr_times:
        idx = int(scr_time * fs)
        if idx < n_samples - 20:
            # SCR shape: fast rise,
            rise = np.linspace(0, 0.5, 4)
            decay = 0.5 * np.exp(-np.linspace(0, 2, 16))
            phasic[idx : idx + 4] = rise
            phasic[idx + 4 : idx + 20] = decay

    eda = tonic + phasic + 0.01 * np.random.randn(n_samples)
    return eda, fs


@pytest.fixture
def sample_resp():

    np.random.seed(42)
    fs = 700  # Hz
    duration = 30  # seconds
    t = np.linspace(0, duration, fs * duration)

    # Breathing at ~15 breaths/min
    breathing_freq = 0.25
    resp = np.sin(2 * np.pi * breathing_freq * t)

    # Add some variability
    resp += 0.1 * np.sin(2 * np.pi * 0.05 * t)
    resp += 0.02 * np.random.randn(len(t))

    return resp, fs


@pytest.fixture
def sample_labels():

    # 60 seconds at 700
    n_samples = 42000
    labels = np.zeros(n_samples, dtype=int)

    # First 20 seconds: baseline
    labels[:14000] = 1

    # Middle 20 seconds: stress
    labels[14000:28000] = 2

    # Last 20 seconds: amusement
    labels[28000:] = 3

    return labels


# ============================================================================
# SignalProcessor Tests
# ============================================================================


class TestSignalProcessor:
    @pytest.fixture
    def processor(self):

        return SignalProcessor(fs=700.0)

    def test_butterworth_lowpass(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        filtered = processor.butterworth_filter(ecg, cutoff=40.0, order=4, btype="low")

        assert len(filtered) == len(ecg)
        assert not np.isnan(filtered).any()
        # Filtered signal should have
        assert np.std(np.diff(filtered)) < np.std(np.diff(ecg))

    def test_butterworth_bandpass(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        filtered = processor.butterworth_filter(
            ecg, cutoff=(0.5, 40.0), order=4, btype="band"
        )

        assert len(filtered) == len(ecg)
        assert not np.isnan(filtered).any()

    def test_notch_filter(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        filtered = processor.notch_filter(ecg, freq=50.0)

        assert len(filtered) == len(ecg)
        assert not np.isnan(filtered).any()

    def test_process_ecg(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        processed = processor.process_ecg(ecg)

        assert len(processed) == len(ecg)
        assert not np.isnan(processed).any()

    def test_process_eda(self, processor, sample_eda):

        eda, _ = sample_eda
        # Upsample to chest rate
        eda_upsampled = np.repeat(eda, 175)  # 4 Hz to 700

        processed = processor.process_eda(eda_upsampled)

        assert len(processed) == len(eda_upsampled)
        assert not np.isnan(processed).any()

    def test_normalize_zscore(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        normalized = processor.normalize_zscore(ecg)

        assert np.abs(np.mean(normalized)) < 1e-10
        assert np.abs(np.std(normalized) - 1.0) < 1e-10

    def test_normalize_minmax(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        normalized = processor.normalize_minmax(ecg, feature_range=(0, 1))

        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1

    def test_segment_signal(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        window_size = 700  # 1 second
        overlap = 0.5

        segments = processor.segment_signal(ecg, window_size, overlap)

        expected_step = int(window_size * (1 - overlap))
        expected_n_windows = (len(ecg) - window_size) // expected_step + 1

        assert segments.shape[0] == expected_n_windows
        assert segments.shape[1] == window_size

    def test_process_all(self, processor):

        np.random.seed(42)
        signals = {
            "ECG": np.random.randn(7000),
            "EDA": np.abs(np.random.randn(7000)),
            "EMG": np.random.randn(7000),
            "Resp": np.random.randn(7000),
            "Temp": 36.5 + 0.1 * np.random.randn(7000),
        }

        processed = processor.process_all(signals)

        assert set(processed.keys()) == set(signals.keys())
        for key in processed:
            assert len(processed[key]) == len(signals[key])
            assert not np.isnan(processed[key]).any()


# ============================================================================
# ECGProcessor Tests
# ============================================================================


class TestECGProcessor:
    @pytest.fixture
    def processor(self):

        return ECGProcessor(sampling_rate=700)

    def test_bandpass_filter(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        filtered = processor.bandpass_filter(ecg)

        assert len(filtered) == len(ecg)
        assert not np.isnan(filtered).any()

    def test_detect_r_peaks(self, processor, sample_ecg):

        ecg, fs = sample_ecg
        filtered = processor.bandpass_filter(ecg)
        r_peaks = processor.detect_r_peaks(filtered)

        # Should detect approximately correct
        # 10 seconds at 72
        assert len(r_peaks) > 5
        assert len(r_peaks) < 20

        # R-peaks should be positive
        assert all(r >= 0 for r in r_peaks)
        assert all(r < len(ecg) for r in r_peaks)

    def test_extract_rr_intervals(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        filtered = processor.bandpass_filter(ecg)
        r_peaks = processor.detect_r_peaks(filtered)
        rr_ms = processor.extract_rr_intervals(r_peaks, unit="ms")

        assert len(rr_ms) == len(r_peaks) - 1

        # RR intervals should be
        assert all(rr > 200 for rr in rr_ms)
        assert all(rr < 2500 for rr in rr_ms)

    def test_remove_ectopic_beats(self, processor):

        # Create RR intervals with
        rr = np.array([800, 820, 810, 400, 815, 790, 805])  # 400 is ectopic
        clean_rr, valid_mask = processor.remove_ectopic_beats(rr, threshold=0.2)

        # The outlier should be
        assert len(clean_rr) < len(rr)
        assert not valid_mask[3]  # The 400ms interval

    def test_interpolate_artifacts(self, processor):

        rr = np.array([800, 820, 810, 400, 815, 790, 805])
        _, valid_mask = processor.remove_ectopic_beats(rr, threshold=0.2)
        interpolated = processor.interpolate_artifacts(rr, valid_mask)

        assert len(interpolated) == len(rr)
        assert not np.isnan(interpolated).any()
        # Interpolated value should be
        assert 700 < interpolated[3] < 900

    def test_compute_signal_quality(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        quality = processor.compute_signal_quality(ecg)

        assert "snr_db" in quality
        assert "kurtosis" in quality
        assert "baseline_variance" in quality
        assert "valid_beat_ratio" in quality

    def test_full_process(self, processor, sample_ecg):

        ecg, _ = sample_ecg
        results = processor.process(ecg)

        assert "filtered_ecg" in results
        assert "r_peaks" in results
        assert "rr_intervals_ms" in results
        assert "rr_interpolated" in results
        assert "quality" in results

        assert len(results["filtered_ecg"]) == len(ecg)


# ============================================================================
# EDAProcessor Tests
# ============================================================================


class TestEDAProcessor:
    @pytest.fixture
    def processor(self):

        return EDAProcessor(sampling_rate=4)

    def test_lowpass_filter(self, processor, sample_eda):

        eda, _ = sample_eda
        filtered = processor.lowpass_filter(eda)

        assert len(filtered) == len(eda)
        assert not np.isnan(filtered).any()

    def test_decompose_eda(self, processor, sample_eda):

        eda, _ = sample_eda
        tonic, phasic = processor.decompose_eda(eda, method="highpass")

        assert len(tonic) == len(eda)
        assert len(phasic) == len(eda)
        assert not np.isnan(tonic).any()
        assert not np.isnan(phasic).any()

    def test_detect_scr_peaks(self, processor, sample_eda):

        eda, _ = sample_eda
        _, phasic = processor.decompose_eda(eda, method="highpass")
        scr_peaks, scr_features = processor.detect_scr_peaks(phasic)

        # We added 4 SCRs
        assert len(scr_peaks) >= 1  # Should detect at least

        if len(scr_features) > 0:
            assert "amplitude" in scr_features[0]
            assert "rise_time" in scr_features[0]

    def test_compute_scr_features(self, processor, sample_eda):

        eda, _ = sample_eda
        _, phasic = processor.decompose_eda(eda)
        _, scr_list = processor.detect_scr_peaks(phasic)

        features = processor.compute_scr_features(scr_list, signal_duration=60)

        assert "scr_count" in features
        assert "scr_rate" in features
        assert "scr_amplitude_mean" in features

    def test_compute_tonic_features(self, processor, sample_eda):

        eda, _ = sample_eda
        tonic, _ = processor.decompose_eda(eda)
        features = processor.compute_tonic_features(tonic)

        assert "scl_mean" in features
        assert "scl_std" in features
        assert "scl_slope" in features

    def test_compute_signal_quality(self, processor, sample_eda):

        eda, _ = sample_eda
        quality = processor.compute_signal_quality(eda)

        assert "valid_range_ratio" in quality
        assert "flatline_ratio" in quality
        assert "quality_score" in quality
        assert 0 <= quality["quality_score"] <= 1

    def test_full_process(self, processor, sample_eda):

        eda, _ = sample_eda
        results = processor.process(eda)

        assert "filtered_eda" in results
        assert "tonic" in results
        assert "phasic" in results
        assert "scr_peaks" in results
        assert "aggregate_features" in results
        assert "quality" in results


# ============================================================================
# RespiratoryProcessor Tests
# ============================================================================


class TestRespiratoryProcessor:
    @pytest.fixture
    def processor(self):

        return RespiratoryProcessor(sampling_rate=700)

    def test_bandpass_filter(self, processor, sample_resp):

        resp, _ = sample_resp
        filtered = processor.bandpass_filter(resp)

        assert len(filtered) == len(resp)
        assert not np.isnan(filtered).any()

    def test_detect_breaths(self, processor, sample_resp):

        resp, _ = sample_resp
        filtered = processor.bandpass_filter(resp)
        breath_data = processor.detect_breaths(filtered)

        assert "peaks" in breath_data
        assert "troughs" in breath_data
        assert "breath_intervals" in breath_data

        # Should detect approximately 7-8
        assert len(breath_data["peaks"]) > 3
        assert len(breath_data["peaks"]) < 15

    def test_compute_breathing_rate(self, processor, sample_resp):

        resp, _ = sample_resp
        filtered = processor.bandpass_filter(resp)
        breath_data = processor.detect_breaths(filtered)
        rate_features = processor.compute_breathing_rate(breath_data)

        assert "breathing_rate_bpm" in rate_features
        assert "breath_interval_mean" in rate_features
        assert "breath_rmssd" in rate_features

        # Should be approximately 15
        assert 10 < rate_features["breathing_rate_bpm"] < 25

    def test_compute_spectral_breathing_rate(self, processor, sample_resp):

        resp, _ = sample_resp
        filtered = processor.bandpass_filter(resp)
        br_spectral = processor.compute_spectral_breathing_rate(filtered)

        # Should be approximately 15
        assert 10 < br_spectral < 25

    def test_compute_amplitude_features(self, processor, sample_resp):

        resp, _ = sample_resp
        filtered = processor.bandpass_filter(resp)
        breath_data = processor.detect_breaths(filtered)
        amplitude_features = processor.compute_amplitude_features(filtered, breath_data)

        assert "breath_amplitude_mean" in amplitude_features
        assert "inspiration_depth_mean" in amplitude_features

    def test_full_process(self, processor, sample_resp):

        resp, _ = sample_resp
        results = processor.process(resp)

        assert "filtered_resp" in results
        assert "breaths" in results
        assert "features" in results
        assert "quality" in results

        assert "breathing_rate_bpm" in results["features"]


# ============================================================================
# SignalWindower Tests
# ============================================================================


class TestSignalWindower:
    @pytest.fixture
    def windower(self):

        return SignalWindower(window_size_sec=10, overlap=0.5, sampling_rate=700)

    def test_create_windows(self, windower, sample_ecg, sample_labels):

        ecg, _ = sample_ecg
        # Extend signal to match
        ecg_extended = np.tile(ecg, 6)[: len(sample_labels)]

        windows, labels = windower.create_windows(ecg_extended, sample_labels)

        # With 10s windows, 50%
        assert windows.shape[1] == windower.window_size_samples
        assert len(labels) == len(windows)

    def test_create_windows_no_labels(self, windower, sample_ecg):

        ecg, _ = sample_ecg
        windows, labels = windower.create_windows(ecg, labels=None)

        assert windows.shape[1] == windower.window_size_samples
        assert labels is None

    def test_label_assignment_majority(self, windower, sample_labels):

        signal = np.random.randn(len(sample_labels))
        windows, labels = windower.create_windows(
            signal, sample_labels, label_strategy="majority"
        )

        # All labels should be
        unique_labels = np.unique(labels)
        assert all(label in [1, 2, 3] for label in unique_labels)

    def test_label_assignment_all_same(self, windower):

        # Create signal and labels
        signal = np.random.randn(21000)  # 30 seconds at 700
        labels = np.concatenate(
            [
                np.ones(7000, dtype=int),
                np.ones(7000, dtype=int) * 2,
                np.ones(7000, dtype=int) * 3,
            ]
        )

        windows, window_labels = windower.create_windows(
            signal, labels, label_strategy="all_same"
        )

        # Some windows should have
        assert -1 in window_labels or len(np.unique(window_labels)) >= 2

    def test_validate_window(self, windower):

        # Valid window
        valid_window = np.random.randn(7000)
        result = windower.validate_window(valid_window)
        assert result["is_valid"] is True

        # Window with NaNs
        nan_window = valid_window.copy()
        nan_window[:1000] = np.nan
        result = windower.validate_window(nan_window)
        assert result["nan_ratio"] > 0

        # Flatline window
        flat_window = np.ones(7000)
        result = windower.validate_window(flat_window)
        assert result["flatline_ratio"] > 0.9

    def test_filter_by_label(self, windower, sample_labels):

        signal = np.random.randn(len(sample_labels))
        windows, labels = windower.create_windows(signal, sample_labels)

        # Keep only stress (2)
        filtered_windows, filtered_labels = windower.filter_by_label(
            windows, labels, valid_labels=[1, 2]
        )

        assert all(label in [1, 2] for label in filtered_labels)
        assert len(filtered_windows) <= len(windows)

    def test_get_window_info(self, windower):

        info = windower.get_window_info()

        assert info["window_size_sec"] == 10
        assert info["overlap"] == 0.5
        assert info["sampling_rate"] == 700


# ============================================================================
# Integration Tests
# ============================================================================


class TestPreprocessingIntegration:
    def test_ecg_to_hrv_pipeline(self, sample_ecg):

        ecg, fs = sample_ecg
        processor = ECGProcessor(sampling_rate=fs)

        # Full pipeline
        results = processor.process(ecg)

        # Should have RR intervals
        assert len(results["rr_intervals_ms"]) > 0

        # Mean HR should be
        mean_rr = np.mean(results["rr_intervals_ms"])
        mean_hr = 60000 / mean_rr  # Convert to BPM
        assert 40 < mean_hr < 150

    def test_eda_to_scr_pipeline(self, sample_eda):

        eda, fs = sample_eda
        processor = EDAProcessor(sampling_rate=fs)

        # Full pipeline
        results = processor.process(eda)

        # Should have tonic and
        assert np.mean(results["tonic"]) > 0  # Tonic is positive

        # Aggregate features should be
        assert results["aggregate_features"]["scl_mean"] > 0

    def test_multimodal_windowing(self, sample_ecg, sample_resp, sample_labels):

        ecg, _ = sample_ecg
        resp, _ = sample_resp

        # Match lengths with labels
        min_len = len(sample_labels)
        ecg_extended = np.tile(ecg, 10)[:min_len]
        resp_extended = np.tile(resp, 2)[:min_len]

        windower = SignalWindower(window_size_sec=10, overlap=0.5, sampling_rate=700)

        signals = {"ECG": ecg_extended, "Resp": resp_extended}

        windows_dict, labels, valid_mask = windower.create_windows_multimodal(
            signals, sample_labels
        )

        assert "ECG" in windows_dict
        assert "Resp" in windows_dict
        assert len(labels) == len(valid_mask)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_short_signal(self):

        processor = ECGProcessor(sampling_rate=700)
        short_signal = np.random.randn(100)  # Very short

        # Should handle gracefully
        filtered = processor.bandpass_filter(short_signal)
        assert len(filtered) == len(short_signal)

    def test_constant_signal(self):

        processor = EDAProcessor(sampling_rate=4)
        flat_signal = np.ones(240)  # 1 minute

        results = processor.process(flat_signal)

        # Should complete without error
        assert "filtered_eda" in results
        # Quality should reflect the
        assert results["quality"]["flatline_ratio"] > 0.9

    def test_nan_handling(self):

        windower = SignalWindower(window_size_sec=1, sampling_rate=100)

        signal_with_nan = np.random.randn(1000)
        signal_with_nan[100:200] = np.nan

        validation = windower.validate_window(signal_with_nan)
        assert validation["nan_ratio"] > 0

    def test_invalid_overlap(self):

        with pytest.raises(ValueError):
            SignalWindower(overlap=1.5)

        with pytest.raises(ValueError):
            SignalWindower(overlap=-0.1)
