"""Calibration metrics for classifier probabilities (Guo et al., 2017)."""

from typing import Dict, List, Union

import numpy as np

Array = Union[np.ndarray, list]


def _confidence_correct(y: Array, proba: Array):
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y)
    if proba.ndim == 1:
        conf = np.where(proba >= 0.5, proba, 1.0 - proba)
        pred = (proba >= 0.5).astype(int)
    else:
        conf = proba.max(axis=1)
        pred = proba.argmax(axis=1)
    return conf, (pred == y).astype(float)


def _bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)


def reliability_curve(y: Array, proba: Array, n_bins: int = 15) -> List[Dict[str, float]]:
    conf, correct = _confidence_correct(y, proba)
    idx = _bin_index(conf, n_bins)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "confidence": float(conf[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows


def expected_calibration_error(y: Array, proba: Array, n_bins: int = 15) -> float:
    conf, correct = _confidence_correct(y, proba)
    idx = _bin_index(conf, n_bins)
    n = len(conf)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def maximum_calibration_error(y: Array, proba: Array, n_bins: int = 15) -> float:
    conf, correct = _confidence_correct(y, proba)
    idx = _bin_index(conf, n_bins)
    gaps = [
        abs(correct[idx == b].mean() - conf[idx == b].mean())
        for b in range(n_bins)
        if (idx == b).any()
    ]
    return float(max(gaps)) if gaps else 0.0


def brier_score(y: Array, proba: Array) -> float:
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y)
    if proba.ndim == 1:
        return float(np.mean((proba - y) ** 2))
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def summary(y: Array, proba: Array, n_bins: int = 15) -> Dict[str, object]:
    return {
        "ece": expected_calibration_error(y, proba, n_bins),
        "mce": maximum_calibration_error(y, proba, n_bins),
        "brier": brier_score(y, proba),
        "reliability": reliability_curve(y, proba, n_bins),
    }


def net_benefit(y: Array, p_pos: Array, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit at each probability threshold."""
    y = np.asarray(y)
    p_pos = np.asarray(p_pos, dtype=float)
    n = len(y)
    out = []
    for pt in thresholds:
        flagged = p_pos >= pt
        tp = np.sum(flagged & (y == 1))
        fp = np.sum(flagged & (y == 0))
        out.append(tp / n - (fp / n) * (pt / (1.0 - pt)))
    return np.array(out)
