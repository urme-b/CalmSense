"""Cross-dataset generalization: WESAD <-> PhysioNet Non-EEG on a shared feature space."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts.run_experiment import FIGURES_DIR, RESULTS_DIR, build_pipeline, loso_evaluate
from src.config import PROCESSED_DATA_DIR
from src.datasets import non_eeg
from src.portable import wesad_portable

META = ["subject", "label"]


def _xy(df, feature_cols):
    X = df[feature_cols].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X, df["label"].to_numpy(), df["subject"].to_numpy()


def transfer(train_df, test_df, feature_cols):
    Xtr, ytr, _ = _xy(train_df, feature_cols)
    Xte, yte, _ = _xy(test_df, feature_cols)
    pipe = build_pipeline("rf")
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
    }


def within(df, feature_cols):
    X, y, groups = _xy(df, feature_cols)
    res = loso_evaluate(lambda: build_pipeline("rf"), X, y, groups)
    return {
        "accuracy": res["accuracy_mean"],
        "f1_macro": res["f1_macro_mean"],
        "balanced_accuracy": res["balanced_accuracy"],
    }


def run():
    cache_w = PROCESSED_DATA_DIR / "portable_wesad.parquet"
    cache_n = PROCESSED_DATA_DIR / "portable_noneeg.parquet"
    import pandas as pd

    wesad = pd.read_parquet(cache_w) if cache_w.exists() else wesad_portable()
    noneeg = pd.read_parquet(cache_n) if cache_n.exists() else non_eeg.build()
    if not cache_w.exists():
        wesad.to_parquet(cache_w, index=False)
    if not cache_n.exists():
        noneeg.to_parquet(cache_n, index=False)

    feature_cols = sorted(set(wesad.columns) & set(noneeg.columns) - set(META))
    print(
        f"WESAD: {len(wesad)} windows | Non-EEG: {len(noneeg)} windows | shared features: {len(feature_cols)}"
    )
    print(
        f"WESAD balance: {np.bincount(wesad['label'])} | Non-EEG balance: {np.bincount(noneeg['label'])}"
    )

    out = {
        "n_shared_features": len(feature_cols),
        "within_wesad": within(wesad, feature_cols),
        "within_noneeg": within(noneeg, feature_cols),
        "wesad_to_noneeg": transfer(wesad, noneeg, feature_cols),
        "noneeg_to_wesad": transfer(noneeg, wesad, feature_cols),
    }
    json.dump(out, open(RESULTS_DIR / "cross_dataset.json", "w"), indent=2)

    print("\n              within-LOSO   cross-dataset")
    print(
        f"WESAD          acc={out['within_wesad']['accuracy']:.3f}     -> Non-EEG f1={out['wesad_to_noneeg']['f1_macro']:.3f} (bal-acc {out['wesad_to_noneeg']['balanced_accuracy']:.3f})"
    )
    print(
        f"Non-EEG        acc={out['within_noneeg']['accuracy']:.3f}     -> WESAD   f1={out['noneeg_to_wesad']['f1_macro']:.3f} (bal-acc {out['noneeg_to_wesad']['balanced_accuracy']:.3f})"
    )

    labels = ["WESAD\n(within)", "WESAD→\nNon-EEG", "Non-EEG\n(within)", "Non-EEG→\nWESAD"]
    vals = [
        out["within_wesad"]["balanced_accuracy"],
        out["wesad_to_noneeg"]["balanced_accuracy"],
        out["within_noneeg"]["balanced_accuracy"],
        out["noneeg_to_wesad"]["balanced_accuracy"],
    ]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#e74c3c"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
    plt.axhline(0.5, color="gray", ls=":", label="chance")
    plt.ylabel("Balanced accuracy")
    plt.ylim(0, 1)
    plt.title("Within- vs cross-dataset generalization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_dataset.png", dpi=150)
    plt.close()
    print("\nWrote results/cross_dataset.json and cross_dataset.png")


if __name__ == "__main__":
    run()
