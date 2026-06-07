"""Feature extractors produce correct values on known signals."""

import numpy as np

from src.features.hrv_time_domain import HRVTimeDomainExtractor
from src.preprocessing.ecg_processor import ECGProcessor


def test_rmssd_constant_rr_is_zero():
    rr = np.full(60, 800.0)  # constant heartbeat
    features = HRVTimeDomainExtractor().extract_all(rr)
    assert features["RMSSD"] == 0.0
    assert abs(features["MeanNN"] - 800.0) < 1e-6


def test_hrv_matches_known_sequence():
    rr = np.tile([800.0, 820.0], 30)  # alternating RR, 60 beats
    f = HRVTimeDomainExtractor().extract_all(rr)
    assert abs(f["MeanNN"] - 810.0) < 1e-6
    assert abs(f["RMSSD"] - 20.0) < 1e-6  # every successive diff is 20 ms
    assert abs(f["SDNN"] - np.std(rr, ddof=1)) < 1e-6


def test_too_few_rr_returns_nan():
    # below the minimum RR count HRV is undefined
    features = HRVTimeDomainExtractor().extract_all(np.array([800.0, 810.0]))
    assert np.isnan(features["SDNN"])


def test_rpeaks_recover_known_rate():
    fs = 700
    t = np.arange(0, 30, 1 / fs)
    # 1 Hz synthetic beats -> ~60 BPM
    ecg = np.sin(2 * np.pi * 1.0 * t) ** 21
    peaks = ECGProcessor(sampling_rate=fs).detect_r_peaks(ecg)
    rate = len(peaks) / 30
    assert 0.8 < rate < 1.2
