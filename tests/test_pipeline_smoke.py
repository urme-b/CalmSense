"""The full pipeline runs end-to-end on synthetic data, no WESAD download needed."""

import pytest

from scripts import calibration
from scripts.run_experiment import build_pipeline, loso_evaluate, prepare_task
from src.synthetic import features


@pytest.fixture(scope="module")
def synth():
    df, x_raw, _ = features(n_subjects=4, block_sec=100, seed=3)
    return df, x_raw


def test_pipeline_builds_features(synth):
    df, _ = synth
    assert len(df) > 0
    assert df["subject_id"].nunique() == 4
    feat = [c for c in df.columns if c not in ("subject_id", "window_id", "label", "label_name")]
    assert len(feat) >= 50


def test_loso_runs_end_to_end(synth):
    df, x_raw = synth
    X, y, groups, _, _ = prepare_task(df, x_raw, [1, 2])
    res = loso_evaluate(lambda: build_pipeline("lr"), X, y, groups)
    assert 0.0 <= res["accuracy_mean"] <= 1.0
    assert len(res["per_subject"]) == 4


def test_calibration_outputs_are_valid(synth):
    df, x_raw = synth
    X, y, groups, _, _ = prepare_task(df, x_raw, [1, 2])
    out = calibration.compute(X, y, groups, model="lr", n_bins=10)
    for key in ("loso", "within_subject", "recalibrated_isotonic"):
        assert 0.0 <= out[key]["ece"] <= 1.0
        assert 0.0 <= out[key]["brier"] <= 2.0
    dc = out["decision_curve"]
    assert len(dc["thresholds"]) == len(dc["net_benefit_uncalibrated"])
