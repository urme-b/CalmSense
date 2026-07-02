"""Reproducibility: set_seed makes the RNGs and CNN training deterministic."""

import random

import numpy as np
import pytest

from src.utils import set_seed


def test_set_seed_makes_rngs_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand(3).tolist())
    set_seed(123)
    second = (random.random(), np.random.rand(3).tolist())
    assert first == second


def test_cnn_training_is_deterministic():
    pytest.importorskip("torch")
    from src.models.dl.cnn_1d import CNN1DClassifier

    rng = np.random.RandomState(0)
    n, channels, length = 24, 3, 256
    X = rng.randn(n, channels, length).astype("float32")
    y = np.array([0, 1] * (n // 2))

    def train_and_predict():
        set_seed(7)
        clf = CNN1DClassifier(in_channels=channels, max_epochs=2, batch_size=8, random_state=7)
        clf.fit(X, y)
        return clf.predict_proba(X)

    assert np.allclose(train_and_predict(), train_and_predict(), atol=1e-6), (
        "CNN training is not reproducible with a fixed seed"
    )
