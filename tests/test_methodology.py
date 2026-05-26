"""Guards against the data-leakage traps that invalidate WESAD results."""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from src.dataset import WindowedDataset


def test_loso_never_shares_subjects():
    groups = np.repeat(np.arange(10), 20)
    X = np.random.RandomState(0).randn(len(groups), 5)
    y = np.random.RandomState(1).randint(0, 2, len(groups))

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        assert len(np.unique(groups[test_idx])) == 1


def test_window_label_rejects_mixed_segments():
    ds = WindowedDataset.__new__(WindowedDataset)
    ds.purity = 0.9
    pure = np.full(100, 2)
    mixed = np.concatenate([np.full(50, 1), np.full(50, 2)])
    excluded = np.full(100, 4)  # meditation

    assert ds._window_label(pure) == 2
    assert ds._window_label(mixed) is None
    assert ds._window_label(excluded) is None
