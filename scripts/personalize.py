"""Few-shot per-subject recalibration. A short labeled enrollment from the target
subject should beat global recalibration at no extra modeling cost. Leak-free: per
subject we keep only non-overlapping windows, then hold out a fixed evaluation half
and draw enrollment from the other half, so enrollment never overlaps an eval window."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from scripts.calibration import _apply_calibrator, _fit_calibrator, _pos_proba
from scripts.run_experiment import (
    RESULTS_DIR,
    _fit_params,
    build_pipeline,
    load_cached,
    prepare_task,
)
from src import calibration as cal
from src.config import FIGURES_DIR

SEED = 42
K_VALUES = [5, 10, 20]
METHOD = "isotonic"


def _stratified_split(y, frac, rng):
    ev, pool = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_eval = max(1, int(round(len(idx) * frac)))
        ev.extend(idx[:n_eval])
        pool.extend(idx[n_eval:])
    return np.array(ev), np.array(pool)


def _sample_k(y_pool, k, rng):
    per = max(1, k // len(np.unique(y_pool)))
    picks = []
    for c in np.unique(y_pool):
        idx = np.where(y_pool == c)[0]
        rng.shuffle(idx)
        picks.extend(idx[: min(per, len(idx))])
    return np.array(picks)


def _global_calibrator(factory, Xtr, ytr, gtr, method):
    if len(np.unique(gtr)) < 2:
        return None
    oof = np.zeros(len(ytr))
    for itr, ical in GroupKFold(n_splits=min(5, len(np.unique(gtr)))).split(Xtr, ytr, gtr):
        p = factory()
        p.fit(Xtr[itr], ytr[itr], **_fit_params(p, ytr[itr]))
        oof[ical] = _pos_proba(p, Xtr[ical])
    return _fit_calibrator(oof, ytr, method)


def _metrics(y, p_pos):
    proba = np.column_stack([1.0 - p_pos, p_pos])
    return cal.expected_calibration_error(y, proba), cal.brier_score(y, p_pos)


def compute(X, y, groups, model="rf", k_values=K_VALUES):
    factory = lambda: build_pipeline(model)  # noqa: E731
    logo = LeaveOneGroupOut()
    rng = np.random.RandomState(SEED)
    acc = {"uncalibrated": [], "global": [], **{k: [] for k in k_values}}

    for train_idx, test_idx in logo.split(X, y, groups):
        Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
        base = factory()
        base.fit(Xtr, ytr, **_fit_params(base, ytr))
        raw = _pos_proba(base, X[test_idx])
        y_s = y[test_idx]
        # Windows overlap 50%, so adjacent ones share half their signal. Keep every
        # other window for this subject so enrollment can never overlap an eval window.
        nov = np.zeros(len(y_s), dtype=bool)
        nov[::2] = True
        raw, y_s = raw[nov], y_s[nov]
        if len(np.unique(y_s)) < 2:
            continue

        ev, pool = _stratified_split(y_s, 0.5, rng)
        raw_ev, y_ev = raw[ev], y_s[ev]
        glob = _global_calibrator(factory, Xtr, ytr, gtr, METHOD)

        acc["uncalibrated"].append(_metrics(y_ev, raw_ev))
        acc["global"].append(_metrics(y_ev, glob.transform(raw_ev) if glob else raw_ev))

        for k in k_values:
            pick = _sample_k(y_s[pool], k, rng)
            if len(np.unique(y_s[pool][pick])) < 2:
                acc[k].append(_metrics(y_ev, raw_ev))
                continue
            calib = _fit_calibrator(raw[pool][pick], y_s[pool][pick], METHOD)
            acc[k].append(_metrics(y_ev, _apply_calibrator(calib, raw_ev, METHOD)))

    def mean(rows):
        arr = np.array(rows)
        return {"ece": float(arr[:, 0].mean()), "brier": float(arr[:, 1].mean())}

    return {
        "model": model,
        "eval_frac": 0.5,
        "k_values": k_values,
        "n_subjects": len(acc["uncalibrated"]),
        "uncalibrated": mean(acc["uncalibrated"]),
        "global": mean(acc["global"]),
        "fewshot": {str(k): mean(acc[k]) for k in k_values},
    }


def _plot(out, path):
    ks = out["k_values"]
    plt.figure(figsize=(6, 4))
    plt.axhline(out["uncalibrated"]["ece"], color="#95a5a6", ls=":", label="uncalibrated")
    plt.axhline(out["global"]["ece"], color="#3498db", ls="--", label="global recalibration")
    plt.plot(
        ks, [out["fewshot"][str(k)]["ece"] for k in ks], "o-", color="#2ecc71", label="few-shot"
    )
    plt.xlabel("Enrollment windows per subject")
    plt.ylabel("ECE (mean over subjects)")
    plt.title("Few-shot personalization closes the calibration gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(synthetic=False, model="rf"):
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if synthetic:
        from src.synthetic import features

        print("Using synthetic data (demo only).")
        features_df, x_raw, _ = features(n_subjects=8, block_sec=150, seed=SEED)
    else:
        cached = load_cached()
        if cached is None:
            raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
        features_df, x_raw = cached

    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])
    out = compute(X, y, groups, model=model)

    with open(RESULTS_DIR / "personalization.json", "w") as f:
        json.dump(out, f, indent=2)
    _plot(out, FIGURES_DIR / "personalization.png")

    print(f"\n{'condition':18s} {'ECE':>7s} {'Brier':>7s}")
    print(
        f"{'uncalibrated':18s} {out['uncalibrated']['ece']:>7.3f} {out['uncalibrated']['brier']:>7.3f}"
    )
    print(f"{'global':18s} {out['global']['ece']:>7.3f} {out['global']['brier']:>7.3f}")
    for k in out["k_values"]:
        f = out["fewshot"][str(k)]
        print(f"{'few-shot k=' + str(k):18s} {f['ece']:>7.3f} {f['brier']:>7.3f}")
    print(f"\nWrote {RESULTS_DIR / 'personalization.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model", default="rf")
    args = parser.parse_args()
    run(synthetic=args.synthetic, model=args.model)
