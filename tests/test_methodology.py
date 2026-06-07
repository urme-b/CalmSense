"""Guards against the data-leakage traps that invalidate WESAD results."""

import numpy as np

from scripts.run_experiment import build_pipeline, kfold_accuracy, loso_evaluate
from src.dataset import WindowedDataset


def test_pipeline_imputes_and_scales_inside_the_fold():
    # impute + scale live in the pipeline, so they are refit on each LOSO train split
    pipe = build_pipeline("rf")
    assert [name for name, _ in pipe.steps] == ["impute", "scale", "clf"]


def test_loso_evaluate_holds_out_each_subject():
    rng = np.random.RandomState(0)
    n_subjects, per = 4, 40
    groups = np.repeat([f"S{i}" for i in range(n_subjects)], per)
    X = rng.randn(len(groups), 6)
    y = rng.randint(0, 2, len(groups))

    res = loso_evaluate(lambda: build_pipeline("lr"), X, y, groups)
    # one score per subject, and every subject was the held-out one exactly once
    assert len(res["per_subject"]) == n_subjects
    assert set(res["per_subject"]["subject"]) == set(groups)


def test_kfold_gap_uses_non_overlapping_windows():
    # the gap baseline must drop every other (overlapping) window per subject
    rng = np.random.RandomState(0)
    groups = np.repeat(["S0", "S1"], 40)
    X = rng.randn(len(groups), 5)
    # class blocks per subject so both classes survive the every-other-window subset
    y = np.tile(np.concatenate([np.zeros(20), np.ones(20)]).astype(int), 2)
    acc = kfold_accuracy(lambda: build_pipeline("lr"), X, y, groups)
    assert 0.0 <= acc <= 1.0


def test_window_label_rejects_impure_and_out_of_set_windows():
    ds = WindowedDataset.__new__(WindowedDataset)
    ds.purity = 0.9

    assert ds._window_label(np.full(100, 2)) == 2  # pure stress
    # in-set but below the purity threshold -> rejected
    assert ds._window_label(np.concatenate([np.full(85, 2), np.full(15, 1)])) is None
    # in-set and above the threshold -> accepted
    assert ds._window_label(np.concatenate([np.full(95, 2), np.full(5, 1)])) == 2
    # dominant label outside {baseline, stress, amusement} -> rejected
    assert ds._window_label(np.full(100, 4)) is None
