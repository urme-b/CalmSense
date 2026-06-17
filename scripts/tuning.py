"""Nested hyperparameter tuning: inner grouped CV selects params, outer LOSO scores.
The test subject never touches tuning, so the numbers stay leak-free. Binary task."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

from scripts.run_experiment import (
    CLASSIFIERS,
    CLF_NAMES,
    FIGURES_DIR,
    RESULTS_DIR,
    build_pipeline,
    load_cached,
    prepare_task,
)

GRIDS = {
    "lr": {"clf__C": [0.1, 1.0, 10.0]},
    "rf": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [1, 5],
    },
    "xgb": {
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1],
        "clf__scale_pos_weight": [1, 3],
    },
    "lgbm": {
        "clf__num_leaves": [31, 63],
        "clf__learning_rate": [0.05, 0.1],
        "clf__n_estimators": [200, 400],
    },
}


def tune_model(key, X, y, groups, inner_splits=3):
    logo = LeaveOneGroupOut()
    rows, chosen = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        gtr = groups[train_idx]
        k = min(inner_splits, len(np.unique(gtr)))
        search = GridSearchCV(
            build_pipeline(key),
            GRIDS[key],
            cv=GroupKFold(n_splits=k),
            scoring="balanced_accuracy",
        )
        search.fit(X[train_idx], y[train_idx], groups=gtr)
        pred = search.predict(X[test_idx])
        rows.append(
            {
                "subject": groups[test_idx][0],
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(y[test_idx], pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            }
        )
        chosen.append(tuple(sorted(search.best_params_.items())))
    df = pd.DataFrame(rows)
    mode_params = dict(Counter(chosen).most_common(1)[0][0])
    return df, mode_params


def compute(X, y, groups, inner_splits=3):
    out = {}
    for key in CLASSIFIERS:
        df, params = tune_model(key, X, y, groups, inner_splits)
        out[CLF_NAMES[key]] = {
            "accuracy_mean": float(df["accuracy"].mean()),
            "accuracy_std": float(df["accuracy"].std()),
            "f1_macro_mean": float(df["f1_macro"].mean()),
            "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
            "best_params": {k.replace("clf__", ""): v for k, v in params.items()},
        }
    return out


def _defaults():
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        models = json.load(f).get("binary", {}).get("models", [])
    return {m["model"]: m["accuracy_mean"] for m in models}


def _plot(tuned, defaults, path):
    names = list(tuned)
    x = np.arange(len(names))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, [defaults.get(n, 0) for n in names], 0.4, label="default", color="#95a5a6")
    plt.bar(
        x + 0.2, [tuned[n]["accuracy_mean"] for n in names], 0.4, label="tuned", color="#3498db"
    )
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("LOSO accuracy")
    plt.ylim(0, 1)
    plt.title("Default vs nested-CV tuned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(synthetic=False, inner_splits=3):
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if synthetic:
        from src.synthetic import features

        print("Using synthetic data (demo only).")
        features_df, x_raw, _ = features(n_subjects=6, block_sec=150, seed=42)
    else:
        cached = load_cached()
        if cached is None:
            raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
        features_df, x_raw = cached

    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])
    tuned = compute(X, y, groups, inner_splits)
    defaults = _defaults()

    with open(RESULTS_DIR / "tuning.json", "w") as f:
        json.dump(tuned, f, indent=2)
    if defaults:
        _plot(tuned, defaults, FIGURES_DIR / "tuning.png")

    print(f"\n{'Model':20s} {'default':>8s} {'tuned':>8s}")
    for name, r in tuned.items():
        d = defaults.get(name)
        print(f"{name:20s} {d if d is None else f'{d:.3f}':>8} {r['accuracy_mean']:>8.3f}")
    print(f"\nWrote {RESULTS_DIR / 'tuning.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--inner-splits", type=int, default=3)
    args = parser.parse_args()
    run(synthetic=args.synthetic, inner_splits=args.inner_splits)
