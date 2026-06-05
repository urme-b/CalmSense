"""Omnibus and corrected pairwise significance tests across per-subject LOSO scores."""

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

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


def holm_bonferroni(pairs):
    """Holm-Bonferroni step-down correction over a list of (key, raw_p)."""
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    corrected, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min((m - rank) * p, 1.0))
        corrected[key] = running
    return corrected


def run():
    cached = load_cached()
    if cached is None:
        raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
    features_df, x_raw = cached
    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])

    scores = {}
    for key in CLASSIFIERS:
        scores[key] = per_subject_acc(loso_evaluate(lambda k=key: build_pipeline(k), X, y, groups))

    subjects = sorted(set.intersection(*[set(scores[k]) for k in CLASSIFIERS]))
    vecs = {k: np.array([scores[k][s] for s in subjects]) for k in CLASSIFIERS}

    # Omnibus: are the models different at all?
    chi2, omnibus_p = friedmanchisquare(*[vecs[k] for k in CLASSIFIERS])

    # All pairwise Wilcoxon with Holm correction (no winner pre-selection)
    raw = {}
    for a, b in combinations(CLASSIFIERS, 2):
        if np.allclose(vecs[a], vecs[b]):
            raw[(a, b)] = 1.0
        else:
            raw[(a, b)] = float(wilcoxon(vecs[a], vecs[b]).pvalue)
    corrected = holm_bonferroni(list(raw.items()))

    best = max(CLASSIFIERS, key=lambda k: vecs[k].mean())
    lo, hi = bootstrap_ci(vecs[best])

    out = {
        "best_model": CLF_NAMES[best],
        "best_accuracy_mean": float(vecs[best].mean()),
        "best_ci95": [lo, hi],
        "omnibus_friedman": {"chi2": float(chi2), "p_value": float(omnibus_p)},
        "pairwise_holm": {
            f"{CLF_NAMES[a]} vs {CLF_NAMES[b]}": {
                "delta_mean": float(vecs[a].mean() - vecs[b].mean()),
                "p_raw": raw[(a, b)],
                "p_holm": corrected[(a, b)],
            }
            for a, b in combinations(CLASSIFIERS, 2)
        },
    }

    print(f"Best: {CLF_NAMES[best]}  acc={vecs[best].mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"Friedman omnibus: chi2={chi2:.2f}  p={omnibus_p:.3f}")
    print("Pairwise Wilcoxon (Holm-corrected):")
    for a, b in combinations(CLASSIFIERS, 2):
        d = out["pairwise_holm"][f"{CLF_NAMES[a]} vs {CLF_NAMES[b]}"]
        print(
            f"  {CLF_NAMES[a]:20s} vs {CLF_NAMES[b]:20s} Δ={d['delta_mean']:+.3f}  p_holm={d['p_holm']:.3f}"
        )

    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'stats.json'}")


if __name__ == "__main__":
    run()
