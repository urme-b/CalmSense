"""The 1D-CNN trains and predicts with the right shape."""

import numpy as np

from src.models.dl.cnn_1d import CNN1DClassifier


def test_cnn_learns_separable_signal():
    rng = np.random.RandomState(0)
    n, c, length = 80, 5, 256
    X = rng.randn(n, c, length).astype("float32")
    y = (X[:, 0].mean(axis=1) > 0).astype(int)  # learnable rule
    X[y == 1] += 1.0

    model = CNN1DClassifier(in_channels=c, max_epochs=15, random_state=0)
    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (model.predict(X) == y).mean() > 0.7
