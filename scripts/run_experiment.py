"""Reproduce the full CalmSense LOSO benchmark from raw WESAD data."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.config import FIGURES_DIR, MODELS_DIR, PROJECT_ROOT
from src.dataset import WindowedDataset, load_cached
from src.models.dl.cnn_1d import CNN1DClassifier
from src.models.ml.classifiers import get_classifier

RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 42

TASKS = {
    "binary": {"keep": [1, 2], "names": ["baseline", "stress"]},
    "multiclass": {"keep": [1, 2, 3], "names": ["baseline", "stress", "amusement"]},
}

CLASSIFIERS = ["lr", "rf", "xgb", "lgbm"]
CLF_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}


def build_pipeline(clf_key: str) -> Pipeline:
    estimator = get_classifier(clf_key)._create_model()
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", estimator),
        ]
    )


def _fit_params(pipe, y_train):
    """Balance XGBoost (others balance via class_weight)."""
    if pipe.named_steps["clf"].__class__.__name__ == "XGBClassifier":
        return {"clf__sample_weight": compute_sample_weight("balanced", y_train)}
    return {}


def loso_evaluate(pipeline_factory, X, y, groups):
    """Leak-free LOSO: impute + scale fit per fold."""
    logo = LeaveOneGroupOut()
    classes = np.unique(y)
    pooled_true, pooled_pred = [], []
    per_subject = []

    for train_idx, test_idx in logo.split(X, y, groups):
        pipe = pipeline_factory()
        pipe.fit(X[train_idx], y[train_idx], **_fit_params(pipe, y[train_idx]))
        pred = pipe.predict(X[test_idx])

        pooled_true.extend(y[test_idx])
        pooled_pred.extend(pred)
        per_subject.append(
            {
                "subject": groups[test_idx][0],
                "n": len(test_idx),
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(y[test_idx], pred, average="macro"),
            }
        )

    pooled_true = np.array(pooled_true)
    pooled_pred = np.array(pooled_pred)
    subj_df = pd.DataFrame(per_subject)
    return {
        "accuracy_mean": float(subj_df["accuracy"].mean()),
        "accuracy_std": float(subj_df["accuracy"].std()),
        "f1_macro_mean": float(subj_df["f1_macro"].mean()),
        "f1_macro_std": float(subj_df["f1_macro"].std()),
        "balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_pred)),
        "per_subject": subj_df,
        "y_true": pooled_true,
        "y_pred": pooled_pred,
        "classes": classes,
    }


def cnn_loso(x_raw, y, groups):
    logo = LeaveOneGroupOut()
    pooled_true, pooled_pred, per_subject = [], [], []
    n_folds = len(np.unique(groups))

    for fold, (train_idx, test_idx) in enumerate(logo.split(x_raw, y, groups), 1):
        print(f"    1D-CNN fold {fold}/{n_folds}", flush=True)
        model = CNN1DClassifier(in_channels=x_raw.shape[1], random_state=SEED)
        model.fit(x_raw[train_idx], y[train_idx])
        pred = model.predict(x_raw[test_idx])
        pooled_true.extend(y[test_idx])
        pooled_pred.extend(pred)
        per_subject.append(
            {
                "subject": groups[test_idx][0],
                "n": len(test_idx),
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(y[test_idx], pred, average="macro"),
            }
        )

    subj_df = pd.DataFrame(per_subject)
    return {
        "accuracy_mean": float(subj_df["accuracy"].mean()),
        "accuracy_std": float(subj_df["accuracy"].std()),
        "f1_macro_mean": float(subj_df["f1_macro"].mean()),
        "f1_macro_std": float(subj_df["f1_macro"].std()),
        "balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_pred)),
        "per_subject": subj_df,
        "y_true": np.array(pooled_true),
        "y_pred": np.array(pooled_pred),
        "classes": np.unique(y),
    }


def kfold_accuracy(pipeline_factory, X, y) -> float:
    """Within-subject 5-fold accuracy for the overfitting gap."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        pipe = pipeline_factory()
        pipe.fit(X[train_idx], y[train_idx], **_fit_params(pipe, y[train_idx]))
        scores.append(accuracy_score(y[test_idx], pipe.predict(X[test_idx])))
    return float(np.mean(scores))


def plot_confusion(result, names, title, path):
    cm = confusion_matrix(result["y_true"], result["y_pred"], normalize="true")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_comparison(rows, path):
    df = pd.DataFrame(rows).sort_values("accuracy_mean")
    plt.figure(figsize=(7, 4))
    plt.barh(df["model"], df["accuracy_mean"], xerr=df["accuracy_std"], color="#3498db")
    plt.xlabel("LOSO accuracy")
    plt.title("Subject-independent model comparison")
    plt.xlim(0, 1)
    for i, (acc, _) in enumerate(zip(df["accuracy_mean"], df["accuracy_std"])):
        plt.text(acc + 0.01, i, f"{acc:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_per_subject(subj_df, path):
    df = subj_df.sort_values("accuracy")
    plt.figure(figsize=(8, 4))
    plt.bar(df["subject"], df["accuracy"], color="#2ecc71")
    plt.axhline(df["accuracy"].mean(), color="#e74c3c", ls="--", label="mean")
    plt.ylabel("LOSO accuracy")
    plt.xlabel("Held-out subject")
    plt.title("Per-subject performance (best model)")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_gap(loso_acc, kfold_acc, path):
    plt.figure(figsize=(4.5, 4))
    bars = plt.bar(
        ["LOSO\n(subject-independent)", "5-fold\n(within-subject)"],
        [loso_acc, kfold_acc],
        color=["#3498db", "#e67e22"],
    )
    for b, v in zip(bars, [loso_acc, kfold_acc]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"Optimism gap: {(kfold_acc - loso_acc) * 100:.1f} pts")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_embedding(X, y, names, path):
    from sklearn.decomposition import PCA

    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    coords = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    plt.figure(figsize=(6, 5))
    for cls, name in enumerate(names):
        m = y == np.unique(y)[cls]
        plt.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.5, label=name)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Feature space (PCA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def shap_analysis(X, y, feature_names, names, fig_dir):
    import shap

    pipe = build_pipeline("xgb")
    pipe.fit(X, y, **_fit_params(pipe, y))
    Xt = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X))
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    shap_vals = np.asarray(values)
    if shap_vals.ndim == 3:  # (samples, features, classes)
        shap_vals = shap_vals[:, :, -1]

    shap.summary_plot(shap_vals, Xt, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    arr = np.abs(shap_vals).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": arr})
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )
    return importance


def prepare_task(features_df, x_raw, keep):
    mask = features_df["label"].isin(keep).to_numpy()
    sub = features_df[mask].reset_index(drop=True)
    meta = ["subject_id", "window_id", "label", "label_name"]
    feature_cols = [c for c in sub.columns if c not in meta]
    X = sub[feature_cols].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    # Drop all-NaN features
    keep_cols = ~np.isnan(X).all(axis=0)
    X = X[:, keep_cols]
    feature_cols = [c for c, k in zip(feature_cols, keep_cols) if k]
    # 0-indexed labels in `keep` order
    remap = {label: i for i, label in enumerate(keep)}
    y = sub["label"].map(remap).to_numpy()
    groups = sub["subject_id"].to_numpy()
    return X, y, groups, feature_cols, x_raw[mask]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-cnn", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cached = None if (args.rebuild or args.subjects) else load_cached()
    if cached is None:
        print("Building dataset from raw WESAD...")
        features_df, x_raw, _ = WindowedDataset().build(subjects=args.subjects)
    else:
        print("Loaded cached dataset.")
        features_df, x_raw = cached

    print(f"Windows: {len(features_df)} | subjects: {features_df['subject_id'].nunique()}")
    summary = {}

    for task, cfg in TASKS.items():
        print(f"\n=== Task: {task} ===")
        X, y, groups, feature_cols, x_raw_task = prepare_task(features_df, x_raw, cfg["keep"])
        rows, results = [], {}

        for key in CLASSIFIERS:
            res = loso_evaluate(lambda k=key: build_pipeline(k), X, y, groups)
            results[key] = res
            rows.append(
                {
                    "model": CLF_NAMES[key],
                    "accuracy_mean": res["accuracy_mean"],
                    "accuracy_std": res["accuracy_std"],
                    "f1_macro_mean": res["f1_macro_mean"],
                    "balanced_accuracy": res["balanced_accuracy"],
                }
            )
            print(
                f"  {CLF_NAMES[key]:20s} acc={res['accuracy_mean']:.3f} f1={res['f1_macro_mean']:.3f}"
            )

        if not args.no_cnn:
            cnn_res = cnn_loso(x_raw_task, y, groups)
            results["cnn"] = cnn_res
            rows.append(
                {
                    "model": "1D-CNN",
                    "accuracy_mean": cnn_res["accuracy_mean"],
                    "accuracy_std": cnn_res["accuracy_std"],
                    "f1_macro_mean": cnn_res["f1_macro_mean"],
                    "balanced_accuracy": cnn_res["balanced_accuracy"],
                }
            )
            print(
                f"  {'1D-CNN':20s} acc={cnn_res['accuracy_mean']:.3f} f1={cnn_res['f1_macro_mean']:.3f}"
            )

        best_key = max(rows, key=lambda r: r["accuracy_mean"])["model"]
        best = max(results.items(), key=lambda kv: kv[1]["accuracy_mean"])

        # Figures
        plot_model_comparison(rows, FIGURES_DIR / f"{task}_model_comparison.png")
        plot_confusion(
            best[1], cfg["names"], f"{task} ({best_key})", FIGURES_DIR / f"{task}_confusion.png"
        )
        plot_per_subject(best[1]["per_subject"], FIGURES_DIR / f"{task}_per_subject.png")
        plot_embedding(X, y, cfg["names"], FIGURES_DIR / f"{task}_pca.png")

        if best[0] in CLASSIFIERS:
            loso_acc = best[1]["accuracy_mean"]
            kf_acc = kfold_accuracy(lambda k=best[0]: build_pipeline(k), X, y)
            plot_gap(loso_acc, kf_acc, FIGURES_DIR / f"{task}_optimism_gap.png")
        else:
            kf_acc = None

        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{task}_model_comparison.csv", index=False)
        best[1]["per_subject"].to_csv(RESULTS_DIR / f"{task}_per_subject.csv", index=False)

        summary[task] = {
            "n_windows": int(len(y)),
            "classes": cfg["names"],
            "models": rows,
            "best_model": best_key,
            "loso_accuracy": best[1]["accuracy_mean"],
            "within_subject_accuracy": kf_acc,
            "per_subject": best[1]["per_subject"].to_dict("records"),
        }

        # Serialize best classical model for the API
        if task == "binary":
            top_clf = max(
                [(k, results[k]) for k in CLASSIFIERS],
                key=lambda kv: kv[1]["accuracy_mean"],
            )[0]
            importance = shap_analysis(X, y, feature_cols, cfg["names"], FIGURES_DIR)
            importance.to_csv(RESULTS_DIR / "shap_top_features.csv", index=False)
            final = build_pipeline(top_clf)
            final.fit(X, y, **_fit_params(final, y))
            joblib.dump(
                {"pipeline": final, "features": feature_cols, "classes": cfg["names"]},
                MODELS_DIR / "stress_classifier.joblib",
            )
            print(f"  Saved API model ({CLF_NAMES[top_clf]}) + SHAP.")

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Mirror results into the dashboard
    shap_csv = RESULTS_DIR / "shap_top_features.csv"
    summary["shap"] = pd.read_csv(shap_csv).head(12).to_dict("records") if shap_csv.exists() else []
    frontend_results = PROJECT_ROOT / "frontend" / "src" / "results.json"
    if frontend_results.parent.exists():
        with open(frontend_results, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
