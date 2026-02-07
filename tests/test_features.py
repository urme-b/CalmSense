import numpy as np
import pytest

from src.features import FeatureExtractor
from src.features import (
    HRVTimeDomainExtractor,
    HRVFrequencyDomainExtractor,
    HRVNonlinearExtractor,
    EDAFeatureExtractor,
    TemperatureFeatureExtractor,
    RespirationFeatureExtractor,
    AccelerometerFeatureExtractor,
    SignalImageEncoder,
    FeatureExtractionPipeline,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_rr_intervals():

    np.random.seed(42)
    # Normal heart rate ~75
    mean_rr = 800
    # Add HRV (typical SDNN
    rr = mean_rr + np.random.normal(0, 50, 100)
    # Keep physiological range
    rr = np.clip(rr, 500, 1200)
    return rr


@pytest.fixture
def sample_rr_intervals_long():

    np.random.seed(42)
    mean_rr = 800
    rr = mean_rr + np.random.normal(0, 50, 300)
    rr = np.clip(rr, 500, 1200)
    return rr


@pytest.fixture
def sample_eda_decomposed():

    np.random.seed(42)
    n_samples = 240  # 1 minute at 4
    t = np.linspace(0, 60, n_samples)

    # Tonic component
    tonic = 5.0 + 0.3 * np.sin(2 * np.pi * 0.01 * t)

    # Phasic component (some SCRs)
    phasic = np.zeros(n_samples)
    scr_times = [15, 30, 45]
    for scr_t in scr_times:
        idx = int(scr_t * 4)
        if idx < n_samples - 10:
            phasic[idx : idx + 4] = np.linspace(0, 0.5, 4)
            phasic[idx + 4 : idx + 10] = 0.5 * np.exp(-np.linspace(0, 2, 6))

    return {
        "tonic": tonic,
        "phasic": phasic,
    }


@pytest.fixture
def sample_scr_peaks():

    return [
        {"amplitude": 0.5, "rise_time": 1.0, "recovery_time": 2.0, "peak_idx": 60},
        {"amplitude": 0.3, "rise_time": 0.8, "recovery_time": 1.5, "peak_idx": 120},
        {"amplitude": 0.4, "rise_time": 1.2, "recovery_time": 2.5, "peak_idx": 180},
    ]


@pytest.fixture
def sample_temperature():

    np.random.seed(42)
    n_samples = 240
    # Skin temperature around 33°C
    temp = 33.0 + 0.5 * np.linspace(0, 1, n_samples) + 0.1 * np.random.randn(n_samples)
    return temp


@pytest.fixture
def sample_respiration():

    np.random.seed(42)
    fs = 700
    duration = 60
    t = np.linspace(0, duration, fs * duration)

    # Breathing at ~15 BPM
    resp = np.sin(2 * np.pi * 0.25 * t)
    resp += 0.05 * np.random.randn(len(t))
    return resp


@pytest.fixture
def sample_accelerometer():

    np.random.seed(42)
    n_samples = 1000

    # Mostly stationary with gravity
    acc_x = 0.05 * np.random.randn(n_samples)
    acc_y = 0.05 * np.random.randn(n_samples)
    acc_z = 1.0 + 0.05 * np.random.randn(n_samples)

    return {"x": acc_x, "y": acc_y, "z": acc_z}


# ============================================================================
# Original FeatureExtractor Tests
# ============================================================================


class TestFeatureExtractor:
    @pytest.fixture
    def extractor(self):

        return FeatureExtractor(fs=700.0)

    @pytest.fixture
    def sample_segment(self):

        np.random.seed(42)
        t = np.linspace(0, 1, 700)  # 1 second at 700
        # Simulated signal with known
        signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
        signal += 0.1 * np.random.randn(len(t))
        return signal

    def test_extract_time_domain(self, extractor, sample_segment):

        features = extractor.extract_time_domain(sample_segment)

        # Check all expected features
        expected_features = [
            "mean",
            "std",
            "var",
            "min",
            "max",
            "range",
            "median",
            "skewness",
            "kurtosis",
            "rms",
            "energy",
            "zcr",
            "p25",
            "p75",
            "iqr",
        ]
        for feat in expected_features:
            assert feat in features
            assert not np.isnan(features[feat])

    def test_extract_frequency_domain(self, extractor, sample_segment):

        features = extractor.extract_frequency_domain(sample_segment)

        expected_features = [
            "total_power",
            "spectral_mean",
            "spectral_std",
            "spectral_entropy",
            "dominant_freq",
            "max_psd",
            "vlf_power",
            "lf_power",
            "hf_power",
            "lf_hf_ratio",
        ]
        for feat in expected_features:
            assert feat in features
            assert not np.isnan(features[feat])

    def test_extract_nonlinear(self, extractor, sample_segment):

        features = extractor.extract_nonlinear(sample_segment)

        expected_features = [
            "sample_entropy",
            "hjorth_activity",
            "hjorth_mobility",
            "hjorth_complexity",
        ]
        for feat in expected_features:
            assert feat in features
            assert not np.isnan(features[feat])

    def test_extract_all(self, extractor, sample_segment):

        features = extractor.extract_all(sample_segment, prefix="ecg_")

        # Check prefix is applied
        assert all(k.startswith("ecg_") for k in features.keys())

        # Check feature count (should
        assert len(features) >= 25

        # Check no NaN values
        for k, v in features.items():
            assert not np.isnan(v), f"NaN found in feature {k}"


# ============================================================================
# HRV Time-Domain Tests
# ============================================================================


class TestHRVTimeDomain:
    @pytest.fixture
    def extractor(self):
        return HRVTimeDomainExtractor(min_rr_count=10)

    def test_extract_all_returns_12_features(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        assert len(features) == 12

    def test_mean_nn(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        assert 600 < features["MeanNN"] < 1000  # Physiological range

    def test_sdnn(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        assert features["SDNN"] > 0
        # SDNN should be close
        assert abs(features["SDNN"] - np.std(sample_rr_intervals, ddof=1)) < 1

    def test_rmssd(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        assert features["RMSSD"] > 0

    def test_pnn50(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        assert 0 <= features["pNN50"] <= 100

    def test_pnn20(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        assert 0 <= features["pNN20"] <= 100
        # pNN20 >= pNN50 always
        assert features["pNN20"] >= features["pNN50"]

    def test_cvnn(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        # CVNN = SDNN /
        expected = features["SDNN"] / features["MeanNN"]
        assert abs(features["CVNN"] - expected) < 0.001

    def test_hrvti(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        assert features["HRVTI"] > 0

    def test_invalid_input_returns_nan(self, extractor):

        # Too short
        features = extractor.extract_all(np.array([800, 810]))
        assert np.isnan(features["MeanNN"])

        # Empty
        features = extractor.extract_all(np.array([]))
        assert np.isnan(features["SDNN"])

    def test_constant_signal(self, extractor):

        constant_rr = np.ones(50) * 800
        features = extractor.extract_all(constant_rr)
        assert features["SDNN"] == 0 or np.isclose(features["SDNN"], 0, atol=1e-10)


# ============================================================================
# HRV Frequency-Domain Tests
# ============================================================================


class TestHRVFrequencyDomain:
    @pytest.fixture
    def extractor(self):
        return HRVFrequencyDomainExtractor(min_rr_count=30)

    def test_extract_all_returns_8_features(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        assert len(features) == 8

    def test_band_powers_non_negative(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        for key in ["VLF_power", "LF_power", "HF_power", "Total_power"]:
            if np.isfinite(features[key]):
                assert features[key] >= 0

    def test_normalized_powers_sum_to_100(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        if np.isfinite(features["LFn"]) and np.isfinite(features["HFn"]):
            assert abs(features["LFn"] + features["HFn"] - 100) < 1

    def test_lf_hf_ratio(self, extractor, sample_rr_intervals):

        features = extractor.extract_all(sample_rr_intervals)
        if np.isfinite(features["LF_HF_ratio"]):
            assert features["LF_HF_ratio"] > 0

    def test_psd_computation(self, extractor, sample_rr_intervals):

        freqs_welch, psd_welch = extractor.compute_psd(
            sample_rr_intervals, method="welch"
        )
        assert len(freqs_welch) > 0
        assert len(psd_welch) > 0
        assert np.all(psd_welch >= 0)

    def test_invalid_input(self, extractor):

        features = extractor.extract_all(np.array([800] * 5))  # Too short
        assert np.isnan(features["LF_power"])


# ============================================================================
# HRV Nonlinear Tests
# ============================================================================


class TestHRVNonlinear:
    @pytest.fixture
    def extractor(self):
        return HRVNonlinearExtractor(min_rr_count=50)

    def test_extract_all_returns_10_features(self, extractor, sample_rr_intervals_long):

        features = extractor.extract_all(sample_rr_intervals_long)
        assert len(features) == 10

    def test_sample_entropy(self, extractor, sample_rr_intervals_long):

        features = extractor.extract_all(sample_rr_intervals_long)
        # SampEn typically 0-2.5 for
        if np.isfinite(features["SampEn"]):
            assert 0 < features["SampEn"] < 5

    def test_poincare_features(self, extractor, sample_rr_intervals_long):

        features = extractor.extract_all(sample_rr_intervals_long)
        # SD1 and SD2 should
        assert features["SD1"] > 0
        assert features["SD2"] > 0
        # CSI = SD2/SD1
        if np.isfinite(features["CSI"]):
            assert abs(features["CSI"] - features["SD2"] / features["SD1"]) < 0.001

    def test_dfa(self, extractor, sample_rr_intervals_long):

        features = extractor.extract_all(sample_rr_intervals_long)
        # DFA alpha typically 0.5-1.5
        if np.isfinite(features["DFA_alpha1"]):
            assert 0 < features["DFA_alpha1"] < 3

    def test_invalid_input(self, extractor):

        features = extractor.extract_all(np.array([800] * 10))  # Too short
        assert np.isnan(features["SampEn"])


# ============================================================================
# EDA Features Tests
# ============================================================================


class TestEDAFeatures:
    @pytest.fixture
    def extractor(self):
        return EDAFeatureExtractor(sampling_rate=4)

    def test_extract_all_returns_15_features(
        self, extractor, sample_eda_decomposed, sample_scr_peaks
    ):

        features = extractor.extract_all(sample_eda_decomposed, sample_scr_peaks)
        assert len(features) == 15

    def test_tonic_features(self, extractor, sample_eda_decomposed):

        tonic_features = extractor.extract_tonic_features(
            sample_eda_decomposed["tonic"]
        )
        assert tonic_features["SCL_mean"] > 0
        assert np.isfinite(tonic_features["SCL_slope"])

    def test_phasic_features(self, extractor, sample_scr_peaks):

        phasic_features = extractor.extract_phasic_features(sample_scr_peaks, 60.0)
        assert phasic_features["SCR_count"] == 3
        assert phasic_features["SCR_rate"] == 3.0  # 3 per minute

    def test_no_scr(self, extractor, sample_eda_decomposed):

        features = extractor.extract_all(sample_eda_decomposed, scr_peaks=[])
        assert features["SCR_count"] == 0
        assert features["SCR_amplitude_mean"] == 0


# ============================================================================
# Temperature Features Tests
# ============================================================================


class TestTemperatureFeatures:
    @pytest.fixture
    def extractor(self):
        return TemperatureFeatureExtractor(sampling_rate=4)

    def test_extract_all_returns_5_features(self, extractor, sample_temperature):

        features = extractor.extract_all(sample_temperature)
        assert len(features) == 5

    def test_temperature_range(self, extractor, sample_temperature):

        features = extractor.extract_all(sample_temperature)
        assert 25 < features["TEMP_mean"] < 40

    def test_slope(self, extractor, sample_temperature):

        features = extractor.extract_all(sample_temperature)
        assert np.isfinite(features["TEMP_slope"])


# ============================================================================
# Respiration Features Tests
# ============================================================================


class TestRespirationFeatures:
    @pytest.fixture
    def extractor(self):
        return RespirationFeatureExtractor(sampling_rate=700)

    def test_extract_all_returns_5_features(self, extractor, sample_respiration):

        features = extractor.extract_all(sample_respiration)
        assert len(features) == 5

    def test_breathing_rate(self, extractor, sample_respiration):

        features = extractor.extract_all(sample_respiration)
        # Should be close to
        if np.isfinite(features["RESP_rate"]):
            assert 5 < features["RESP_rate"] < 30


# ============================================================================
# Accelerometer Features Tests
# ============================================================================


class TestAccelerometerFeatures:
    @pytest.fixture
    def extractor(self):
        return AccelerometerFeatureExtractor(sampling_rate=32)

    def test_extract_all_returns_5_features(self, extractor, sample_accelerometer):

        features = extractor.extract_all(
            sample_accelerometer["x"],
            sample_accelerometer["y"],
            sample_accelerometer["z"],
        )
        assert len(features) == 5

    def test_magnitude_near_1g(self, extractor, sample_accelerometer):

        features = extractor.extract_all(
            sample_accelerometer["x"],
            sample_accelerometer["y"],
            sample_accelerometer["z"],
        )
        # Should be close to
        assert 0.9 < features["ACC_magnitude"] < 1.1


# ============================================================================
# Signal Image Encoder Tests
# ============================================================================


class TestSignalImageEncoder:
    @pytest.fixture
    def encoder(self):
        return SignalImageEncoder(image_size=64)

    @pytest.fixture
    def sample_signal(self):
        np.random.seed(42)
        return np.sin(np.linspace(0, 4 * np.pi, 500)) + 0.1 * np.random.randn(500)

    def test_gasf_shape(self, encoder, sample_signal):

        gasf = encoder.encode_gasf(sample_signal)
        assert gasf.shape == (64, 64)

    def test_gasf_range(self, encoder, sample_signal):

        gasf = encoder.encode_gasf(sample_signal)
        assert np.min(gasf) >= -1.001
        assert np.max(gasf) <= 1.001

    def test_gadf_shape(self, encoder, sample_signal):

        gadf = encoder.encode_gadf(sample_signal)
        assert gadf.shape == (64, 64)

    def test_mtf_shape(self, encoder, sample_signal):

        mtf = encoder.encode_mtf(sample_signal)
        assert mtf.shape == (64, 64)

    def test_rgb_shape(self, encoder, sample_signal):

        rgb = encoder.encode_rgb(sample_signal)
        assert rgb.shape == (64, 64, 3)

    def test_rgb_range(self, encoder, sample_signal):

        rgb = encoder.encode_rgb(sample_signal)
        assert np.min(rgb) >= 0
        assert np.max(rgb) <= 1

    def test_batch_encode(self, encoder, sample_signal):

        signals = np.stack([sample_signal, sample_signal * 0.5, sample_signal * 2])
        batch = encoder.batch_encode(signals, method="gasf")
        assert batch.shape == (3, 64, 64)


# ============================================================================
# Feature Pipeline Tests
# ============================================================================


class TestFeatureExtractionPipeline:
    @pytest.fixture
    def pipeline(self):
        return FeatureExtractionPipeline()

    def test_feature_count_is_60_plus(self, pipeline):

        count = pipeline.get_feature_count()
        assert count >= 60

    def test_extract_window_features(
        self,
        pipeline,
        sample_rr_intervals,
        sample_eda_decomposed,
        sample_scr_peaks,
        sample_temperature,
        sample_respiration,
        sample_accelerometer,
    ):

        window_data = {
            "rr_intervals": sample_rr_intervals,
            "eda_tonic": sample_eda_decomposed["tonic"],
            "eda_phasic": sample_eda_decomposed["phasic"],
            "scr_peaks": sample_scr_peaks,
            "temperature": sample_temperature,
            "respiration": sample_respiration,
            "accelerometer": sample_accelerometer,
        }

        features = pipeline.extract_window_features(window_data)
        assert len(features) >= 60

    def test_feature_names(self, pipeline):

        names = pipeline.get_feature_names()
        assert len(names) >= 60
        # Check expected prefixes
        hrv_features = [n for n in names if n.startswith("HRV_")]
        eda_features = [n for n in names if n.startswith("EDA_")]
        assert len(hrv_features) >= 25  # Time + Freq +
        assert len(eda_features) >= 10

    def test_feature_descriptions(self, pipeline):

        descriptions = pipeline.get_feature_descriptions()
        assert len(descriptions) >= 60
        # Each feature should have
        for name, desc in descriptions.items():
            assert len(desc) > 0

    def test_feature_groups(self, pipeline):

        groups = pipeline.get_feature_groups()
        assert "HRV Time-Domain" in groups
        assert "HRV Frequency-Domain" in groups
        assert "HRV Nonlinear" in groups
        assert "EDA" in groups

    def test_missing_data_handling(self, pipeline):

        # Only RR intervals provided
        window_data = {
            "rr_intervals": np.array([800, 810, 795] * 20),
        }
        features = pipeline.extract_window_features(window_data)
        # Should still return all
        assert len(features) >= 60
        # HRV features should be
        assert np.isfinite(features["HRV_MeanNN"])
        # EDA features should be
        assert np.isnan(features["EDA_SCL_mean"])


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_empty_rr_intervals(self):

        extractor = HRVTimeDomainExtractor()
        features = extractor.extract_all(np.array([]))
        assert all(np.isnan(v) for v in features.values())

    def test_constant_signal_entropy(self):

        extractor = HRVNonlinearExtractor()
        constant_rr = np.ones(100) * 800
        features = extractor.extract_all(constant_rr)
        # SampEn should be 0
        assert features["SampEn"] == 0 or np.isnan(features["SampEn"])

    def test_nan_in_input(self):

        extractor = HRVTimeDomainExtractor()
        rr = np.array([800, np.nan, 810, 795, np.nan] * 20)
        features = extractor.extract_all(rr)
        # Should handle NaN gracefully
        assert np.isfinite(features["MeanNN"])

    def test_short_signal_image_encoding(self):

        encoder = SignalImageEncoder(image_size=64)
        short_signal = np.array([1, 2, 3])
        gasf = encoder.encode_gasf(short_signal)
        # Should still return correct
        assert gasf.shape == (64, 64)
