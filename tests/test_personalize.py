"""Enrollment split is leak-free and keeps both classes."""

import numpy as np

from scripts.personalize import _sample_k, _stratified_split


def test_split_is_disjoint_and_covers_all():
    rng = np.random.RandomState(0)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ev, pool = _stratified_split(y, 0.5, rng)
    assert set(ev).isdisjoint(pool)
    assert sorted([*ev, *pool]) == list(range(len(y)))


def test_split_keeps_both_classes_each_side():
    rng = np.random.RandomState(1)
    y = np.array([0] * 6 + [1] * 6)
    ev, pool = _stratified_split(y, 0.5, rng)
    assert {0, 1} <= set(y[ev]) and {0, 1} <= set(y[pool])


def test_sample_k_balances_classes():
    rng = np.random.RandomState(2)
    y_pool = np.array([0] * 10 + [1] * 10)
    pick = _sample_k(y_pool, 6, rng)
    assert {0, 1} <= set(y_pool[pick])
    assert len(pick) <= 6
