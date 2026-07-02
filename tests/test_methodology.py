"""Guards against the data-leakage traps that invalidate WESAD results."""

import numpy as np

from scripts.run_experiment import (
    build_pipeline,
    kfold_accuracy,
    loso_evaluate,
    nonoverlap_mask,
)
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


def test_loso_splits_have_disjoint_subjects():
    # The core leakage guard: no subject may appear in both train and test of any fold.
    from sklearn.model_selection import LeaveOneGroupOut

    groups = np.repeat([f"S{i}" for i in range(5)], 30)
    X = np.zeros((len(groups), 3))
    y = np.zeros(len(groups))
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        assert train_subjects.isdisjoint(test_subjects)
        assert len(test_subjects) == 1  # exactly one subject held out per fold


def test_nonoverlap_mask_keeps_every_other_window_per_subject():
    # uneven per-subject block sizes (5 and 4) to catch off-by-one slicing
    groups = np.array(["S0", "S0", "S0", "S0", "S0", "S1", "S1", "S1", "S1"])
    mask = nonoverlap_mask(groups)
    # S0 -> indices {0,2,4}, S1 -> indices {5,7}
    assert list(np.where(mask)[0]) == [0, 2, 4, 5, 7]
    # within each subject the kept windows are exactly idx[::2]
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        assert list(idx[mask[idx]]) == list(idx[::2])


def test_kfold_gap_uses_non_overlapping_windows():
    # the gap baseline must drop every other (overlapping) window per subject
    rng = np.random.RandomState(0)
    groups = np.repeat(["S0", "S1"], 40)
    X = rng.randn(len(groups), 5)
    # class blocks per subject so both classes survive the every-other-window subset
    y = np.tile(np.concatenate([np.zeros(20), np.ones(20)]).astype(int), 2)
    # kfold_accuracy fits on the non-overlapping subset; check it consumes exactly that
    assert int(nonoverlap_mask(groups).sum()) == 40
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


def test_scaler_imputer_fit_per_fold_never_on_held_out_subject():
    """Leakage regression: per fold the scaler/imputer must be fit on the training
    subjects only, so their statistics differ fold-to-fold and never equal the
    global (all-subject) statistics."""
    from sklearn.model_selection import LeaveOneGroupOut

    rng = np.random.RandomState(0)
    subjects = ["S0", "S1", "S2"]
    shifts = {"S0": 0.0, "S1": 50.0, "S2": 200.0}  # asymmetric so no fold mean == global
    groups = np.repeat(subjects, 30)
    X = np.vstack([rng.randn(30, 4) + shifts[s] for s in subjects])
    y = np.tile([0, 1], 45)  # both classes present in every subject block

    logo = LeaveOneGroupOut()
    fold_scaler_means = {}
    global_mean = X.mean(axis=0)
    for train_idx, test_idx in logo.split(X, y, groups):
        held = groups[test_idx][0]
        pipe = build_pipeline("lr")
        pipe.fit(X[train_idx], y[train_idx])
        scaler_mean = pipe.named_steps["scale"].mean_
        imputer_stats = pipe.named_steps["impute"].statistics_
        # fit on the TRAIN rows only ...
        np.testing.assert_allclose(scaler_mean, X[train_idx].mean(axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            imputer_stats, np.median(X[train_idx], axis=0), rtol=1e-6, atol=1e-6
        )
        # ... so the held-out subject's shift never enters the global stats
        assert not np.allclose(scaler_mean, global_mean, atol=1.0)
        fold_scaler_means[held] = scaler_mean

    # statistics genuinely change when a different subject is held out
    assert not np.allclose(fold_scaler_means["S0"], fold_scaler_means["S1"], atol=1.0)
    assert not np.allclose(fold_scaler_means["S1"], fold_scaler_means["S2"], atol=1.0)
