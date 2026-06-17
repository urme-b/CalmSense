"""Calibration as a fourth layer of optimism, plus leak-free recalibration and a
decision-curve safety analysis. Binary baseline-vs-stress, Random Forest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

from scripts.run_experiment import (
    RESULTS_DIR,
    _fit_params,
    build_pipeline,
    load_cached,
    nonoverlap_mask,
    prepare_task,
)
from src import calibration as cal
from src.config import FIGURES_DIR

SEED = 42
POSITIVE = "stress"
N_BINS = 15


def loso_proba(factory, X, y, groups):
    logo = LeaveOneGroupOut()
    yt, pp, gg = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe = factory()
        pipe.fit(X[train_idx], y[train_idx], **_fit_params(pipe, y[train_idx]))
        pp.append(pipe.predict_proba(X[test_idx]))
        yt.append(y[test_idx])
        gg.append(groups[test_idx])
    return np.concatenate(yt), np.concatenate(pp), np.concatenate(gg)


def within_subject_proba(factory, X, y, groups):
    """Subject-mixed 5-fold on non-overlapping windows (the optimistic baseline)."""
    keep = nonoverlap_mask(groups)
    Xk, yk, gk = X[keep], y[keep], groups[keep]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    yt, pp, gg = [], [], []
    for train_idx, test_idx in skf.split(Xk, yk):
        pipe = factory()
        pipe.fit(Xk[train_idx], yk[train_idx], **_fit_params(pipe, yk[train_idx]))
        pp.append(pipe.predict_proba(Xk[test_idx]))
        yt.append(yk[test_idx])
        gg.append(gk[test_idx])
    return np.concatenate(yt), np.concatenate(pp), np.concatenate(gg)


def _subject_brier(y, proba, g):
    return {s: cal.brier_score(y[g == s], proba[g == s]) for s in np.unique(g)}


def gap_significance(loso, within):
    """Paired test of per-subject Brier: is LOSO worse-calibrated than within-subject?"""
    subjects = sorted(set(loso) & set(within))
    a = np.array([loso[s] for s in subjects])
    b = np.array([within[s] for s in subjects])
    gap = a - b
    pval = 1.0 if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)
    rng = np.random.RandomState(SEED)
    means = [rng.choice(gap, len(gap), replace=True).mean() for _ in range(10000)]
    return {
        "n_subjects": len(subjects),
        "mean_brier_gap": float(gap.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "wilcoxon_p": pval,
        "per_subject": {s: {"loso": loso[s], "within": within[s]} for s in subjects},
    }


def _fit_calibrator(raw, y, method):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    return LogisticRegression().fit(raw.reshape(-1, 1), y)


def _apply_calibrator(model, raw, method):
    if method == "isotonic":
        return model.transform(raw)
    return model.predict_proba(raw.reshape(-1, 1))[:, 1]


def _pos_proba(estimator, X):
    """P(class==1), robust to single-class folds where proba has one column."""
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X))


def loso_recalibrated_proba(factory, X, y, groups, method="isotonic"):
    """LOSO with a calibrator fit on out-of-fold training probabilities only."""
    logo = LeaveOneGroupOut()
    yt, pp = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
        base = factory()
        base.fit(Xtr, ytr, **_fit_params(base, ytr))
        raw_te = _pos_proba(base, X[test_idx])

        n_groups = len(np.unique(gtr))
        if n_groups < 2:
            cal_pos = raw_te  # too few subjects to fit a calibrator
        else:
            oof = np.zeros(len(ytr))
            inner = GroupKFold(n_splits=min(5, n_groups))
            for itr, ical in inner.split(Xtr, ytr, gtr):
                p = factory()
                p.fit(Xtr[itr], ytr[itr], **_fit_params(p, ytr[itr]))
                oof[ical] = _pos_proba(p, Xtr[ical])
            calibrator = _fit_calibrator(oof, ytr, method)
            cal_pos = _apply_calibrator(calibrator, raw_te, method)

        pp.append(np.column_stack([1.0 - cal_pos, cal_pos]))
        yt.append(y[test_idx])
    return np.concatenate(yt), np.concatenate(pp)


def compute(X, y, groups, model="rf", n_bins=N_BINS):
    factory = lambda: build_pipeline(model)  # noqa: E731

    # Headline: all windows (the deployment-relevant calibration).
    y_loso, p_loso, g_loso = loso_proba(factory, X, y, groups)
    y_iso, p_iso = loso_recalibrated_proba(factory, X, y, groups, "isotonic")
    y_sig, p_sig = loso_recalibrated_proba(factory, X, y, groups, "sigmoid")

    # Gap: LOSO vs within-subject on the SAME non-overlapping windows, so only the
    # CV scheme differs (not the sample size).
    m = nonoverlap_mask(groups)
    y_within, p_within, g_within = within_subject_proba(factory, X, y, groups)
    y_lm, p_lm, g_lm = loso_proba(factory, X[m], y[m], groups[m])

    loso = cal.summary(y_loso, p_loso, n_bins)
    loso_matched = cal.summary(y_lm, p_lm, n_bins)
    within = cal.summary(y_within, p_within, n_bins)
    iso = cal.summary(y_iso, p_iso, n_bins)
    sig = cal.summary(y_sig, p_sig, n_bins)
    significance = gap_significance(
        _subject_brier(y_lm, p_lm, g_lm),
        _subject_brier(y_within, p_within, g_within),
    )

    # All all-window LOSO passes iterate identical splits, so labels line up; enforce it.
    assert np.array_equal(y_loso, y_iso), "LOSO label order diverged across passes"
    thresholds = np.round(np.arange(0.05, 0.61, 0.05), 2)
    prevalence = float(np.mean(y_loso == 1))
    decision = {
        "thresholds": thresholds.tolist(),
        "net_benefit_uncalibrated": cal.net_benefit(y_loso, p_loso[:, 1], thresholds).tolist(),
        "net_benefit_recalibrated": cal.net_benefit(y_loso, p_iso[:, 1], thresholds).tolist(),
        "treat_all": [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds],
    }

    return {
        "model": model,
        "positive_class": POSITIVE,
        "n_windows": int(len(y_loso)),
        "n_bins": n_bins,
        "loso": loso,
        "loso_matched": loso_matched,
        "within_subject": within,
        "recalibrated_isotonic": iso,
        "recalibrated_sigmoid": sig,
        "calibration_optimism_gap_ece": round(loso_matched["ece"] - within["ece"], 4),
        "recalibration_reduction_ece": round(loso["ece"] - iso["ece"], 4),
        "gap_significance": significance,
        "decision_curve": decision,
    }


def _plot_reliability(out, path):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k:", label="perfect")
    for key, color, label in [
        ("within_subject", "#e67e22", "within-subject"),
        ("loso", "#3498db", "LOSO"),
        ("recalibrated_isotonic", "#2ecc71", "LOSO recalibrated"),
    ]:
        rows = out[key]["reliability"]
        plt.plot(
            [r["confidence"] for r in rows],
            [r["accuracy"] for r in rows],
            "o-",
            color=color,
            label=f"{label} (ECE {out[key]['ece']:.3f})",
        )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_gap(out, path):
    keys = ["within_subject", "loso_matched", "recalibrated_isotonic"]
    labels = ["Within-subject", "LOSO", "LOSO recalibrated"]
    eces = [out[k]["ece"] for k in keys]
    plt.figure(figsize=(4.5, 4))
    bars = plt.bar(labels, eces, color=["#e67e22", "#3498db", "#2ecc71"])
    for b, v in zip(bars, eces):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
    plt.ylabel("Expected calibration error")
    plt.title("Calibration optimism and its correction")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_decision(out, path):
    d = out["decision_curve"]
    t = d["thresholds"]
    plt.figure(figsize=(5.5, 4))
    plt.plot(t, d["net_benefit_uncalibrated"], "o-", color="#3498db", label="uncalibrated")
    plt.plot(t, d["net_benefit_recalibrated"], "o-", color="#2ecc71", label="recalibrated")
    plt.plot(t, d["treat_all"], "--", color="gray", label="alert everyone")
    plt.axhline(0, color="black", lw=0.8, label="alert no one")
    plt.xlabel("Alert threshold")
    plt.ylabel("Net benefit")
    plt.title("Decision-curve analysis")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(synthetic=False, model="rf", n_bins=N_BINS):
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if synthetic:
        from src.synthetic import features

        print("Using synthetic data (demo only).")
        features_df, x_raw, _ = features(n_subjects=6, block_sec=150, seed=SEED)
    else:
        cached = load_cached()
        if cached is None:
            raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
        features_df, x_raw = cached

    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])
    out = compute(X, y, groups, model=model, n_bins=n_bins)

    _plot_reliability(out, FIGURES_DIR / "calibration_reliability.png")
    _plot_gap(out, FIGURES_DIR / "calibration_gap.png")
    _plot_decision(out, FIGURES_DIR / "calibration_decision_curve.png")

    with open(RESULTS_DIR / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"LOSO ECE {out['loso']['ece']:.3f} | within-subject ECE "
        f"{out['within_subject']['ece']:.3f} | recalibrated ECE "
        f"{out['recalibrated_isotonic']['ece']:.3f}"
    )
    print(
        f"Calibration optimism gap {out['calibration_optimism_gap_ece']:+.3f} ECE | "
        f"recalibration cuts ECE by {out['recalibration_reduction_ece']:+.3f}"
    )
    sig = out["gap_significance"]
    print(
        f"Per-subject Brier gap {sig['mean_brier_gap']:+.4f} "
        f"95% CI [{sig['ci95'][0]:.4f}, {sig['ci95'][1]:.4f}] Wilcoxon p={sig['wilcoxon_p']:.3f}"
    )
    if synthetic:
        print(
            "Note: synthetic stress is near-separable, so ECE is ~0 and the optimism gap is not "
            "meaningful. Run `make reproduce` on real WESAD for the paper's numbers."
        )
    print(f"Wrote {RESULTS_DIR / 'calibration.json'} and 3 figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="run on generated demo data")
    parser.add_argument("--model", default="rf")
    parser.add_argument("--bins", type=int, default=N_BINS)
    args = parser.parse_args()
    run(synthetic=args.synthetic, model=args.model, n_bins=args.bins)
