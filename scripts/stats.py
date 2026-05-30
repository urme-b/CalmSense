"""Paired significance tests and bootstrap CIs across the per-subject LOSO scores."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon

from scripts.run_experiment import (
    CLASSIFIERS,
    CLF_NAMES,
    RESULTS_DIR,
    build_pipeline,
    load_cached,
    loso_evaluate,
    prepare_task,
)

SEED = 42


def per_subject_acc(res) -> dict:
    df = res["per_subject"]
    return dict(zip(df["subject"], df["accuracy"]))


def bootstrap_ci(values, n=10000, seed=SEED):
    rng = np.random.RandomState(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run():
    features_df, x_raw = load_cached()
    X, y, groups, feature_cols, _ = prepare_task(features_df, x_raw, [1, 2])

    scores = {}
    for key in CLASSIFIERS:
        scores[key] = per_subject_acc(loso_evaluate(lambda k=key: build_pipeline(k), X, y, groups))

    best = max(CLASSIFIERS, key=lambda k: np.mean(list(scores[k].values())))
    subjects = sorted(scores[best])
    best_vals = np.array([scores[best][s] for s in subjects])

    lo, hi = bootstrap_ci(best_vals)
    out = {
        "best_model": CLF_NAMES[best],
        "accuracy_mean": float(best_vals.mean()),
        "ci95": [lo, hi],
        "paired_tests_vs_best": {},
    }
    print(f"Best: {CLF_NAMES[best]}  acc={best_vals.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("Wilcoxon signed-rank (best vs each, per-subject accuracy):")
    for key in CLASSIFIERS:
        if key == best:
            continue
        other = np.array([scores[key][s] for s in subjects])
        # Identical paired scores break wilcoxon
        if np.allclose(best_vals, other):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(best_vals, other)
        out["paired_tests_vs_best"][CLF_NAMES[key]] = {
            "delta_mean": float(best_vals.mean() - other.mean()),
            "p_value": float(p),
        }
        print(f"  vs {CLF_NAMES[key]:20s} Δ={best_vals.mean() - other.mean():+.3f}  p={p:.3f}")

    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'stats.json'}")


if __name__ == "__main__":
    run()
