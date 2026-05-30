"""The shared cross-dataset feature space is consistent and robust."""

import numpy as np

from src.portable import portable_features


def test_portable_features_keys_and_finite():
    rng = np.random.RandomState(0)
    feats = portable_features(
        eda=rng.rand(240) + 5,
        temp=rng.rand(240) + 33,
        acc_mag=rng.rand(1920) + 1,
        hr=rng.rand(70) * 10 + 70,
    )
    assert len(feats) == 18
    assert any(k.startswith("EDA_") for k in feats)
    assert any(k.startswith("HR_") for k in feats)
    assert np.isfinite(feats["EDA_mean"]) and np.isfinite(feats["HR_std"])


def test_portable_features_handle_empty():
    feats = portable_features(
        eda=np.array([]), temp=np.array([1.0]), acc_mag=np.array([]), hr=np.array([])
    )
    assert len(feats) == 18
    assert np.isnan(feats["EDA_mean"])


def test_same_columns_both_datasets():
    rng = np.random.RandomState(1)
    a = portable_features(rng.rand(240), rng.rand(240), rng.rand(1920), rng.rand(70))
    b = portable_features(rng.rand(480), rng.rand(480), rng.rand(480), rng.rand(60))
    assert set(a) == set(b)  # device-agnostic, identical feature names
