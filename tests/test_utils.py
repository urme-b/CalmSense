"""Provenance stamping and effect-size helpers."""

from src.utils import paired_effect_size, provenance


def test_provenance_has_sha_and_timestamp():
    p = provenance()
    assert set(p) == {"git_sha", "generated_at"}
    assert isinstance(p["git_sha"], str) and len(p["git_sha"]) >= 7
    assert "T" in p["generated_at"]  # ISO-8601


def test_paired_effect_size_matches_hand_calc():
    a = [0.9, 0.8, 0.95, 0.7]
    b = [0.85, 0.78, 0.90, 0.72]
    es = paired_effect_size(a, b)
    assert es["n"] == 4
    # mean diff 0.025 over sd of diffs -> positive, small-sample g < d
    assert es["cohens_d"] > 0
    assert abs(es["hedges_g"]) < abs(es["cohens_d"])


def test_paired_effect_size_zero_when_identical():
    es = paired_effect_size([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert es["cohens_d"] == 0.0 and es["hedges_g"] == 0.0
