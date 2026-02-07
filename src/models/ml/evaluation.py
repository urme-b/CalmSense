from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ...logging_config import LoggerMixin


class ModelEvaluator(LoggerMixin):
    # Class labels for stress
    CLASS_NAMES = {
        0: "Baseline",
        1: "Stress",
        2: "Amusement",
    }

    def __init__(
        self, class_names: Optional[Dict[int, str]] = None, threshold: float = 0.5
    ):

        self.class_names = class_names or self.CLASS_NAMES
        self.threshold = threshold
        self.logger.debug("ModelEvaluator initialized")

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            matthews_corrcoef,
            roc_auc_score,
            average_precision_score,
            cohen_kappa_score,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "kappa": cohen_kappa_score(y_true, y_pred),
        }

        # Compute AUC metrics if
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            n_classes = len(np.unique(y_true))

            try:
                if n_classes == 2:
                    # Binary classification
                    metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics["auc_pr"] = average_precision_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics["auc_roc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro"
                    )
                    metrics["auc_roc_weighted"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
            except Exception as e:
                self.logger.warning(f"Could not compute AUC: {e}")
                metrics["auc_roc"] = np.nan

        self.logger.debug(
            f"Computed metrics: acc={metrics['accuracy']:.3f}, "
            f"f1={metrics['f1_macro']:.3f}"
        )

        return metrics

    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:

        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        classes = np.unique(y_true)
        results = []

        for cls in classes:
            # Binary mask for this
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            class_metrics = {
                "class": cls,
                "class_name": self.class_names.get(cls, f"Class {cls}"),
                "support": np.sum(y_true_binary),
                "precision": precision_score(
                    y_true_binary, y_pred_binary, zero_division=0
                ),
                "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
                "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            }

            # Per-class AUC if probabilities
            if y_proba is not None:
                try:
                    from sklearn.metrics import roc_auc_score

                    class_metrics["auc_roc"] = roc_auc_score(
                        y_true_binary, y_proba[:, cls]
                    )
                except Exception:
                    class_metrics["auc_roc"] = np.nan

            results.append(class_metrics)

        return pd.DataFrame(results)

    def get_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, normalize: Optional[str] = None
    ) -> np.ndarray:

        from sklearn.metrics import confusion_matrix

        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def get_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, output_dict: bool = True
    ) -> Union[str, Dict]:

        from sklearn.metrics import classification_report

        target_names = [
            self.class_names.get(i, f"Class {i}") for i in sorted(np.unique(y_true))
        ]

        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0,
        )

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        figsize: Tuple[int, int] = (8, 6),
        cmap: str = "Blues",
    ) -> Any:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = self.get_confusion_matrix(
            y_true, y_pred, normalize="true" if normalize else None
        )

        classes = sorted(np.unique(y_true))
        class_names = [self.class_names.get(c, f"Class {c}") for c in classes]

        fig, ax = plt.subplots(figsize=figsize)

        fmt = ".2%" if normalize else "d"
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

        plt.tight_layout()
        return fig

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Any:

        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        classes = sorted(np.unique(y_true))
        n_classes = len(classes)

        # Binarize labels for multi-class
        y_true_bin = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, (cls, color) in enumerate(zip(classes, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = self.class_names.get(cls, f"Class {cls}")
            ax.plot(
                fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})"
            )

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")

        plt.tight_layout()
        return fig

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Any:

        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        classes = sorted(np.unique(y_true))
        n_classes = len(classes)

        y_true_bin = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, (cls, color) in enumerate(zip(classes, colors)):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_proba[:, i]
            )
            ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])

            class_name = self.class_names.get(cls, f"Class {cls}")
            ax.plot(
                recall,
                precision,
                color=color,
                lw=2,
                label=f"{class_name} (AP = {ap:.3f})",
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="lower left")

        plt.tight_layout()
        return fig

    def compare_models(
        self, results: Dict[str, Dict[str, float]], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:

        if metrics is None:
            metrics = ["accuracy", "f1_macro", "f1_weighted", "auc_roc", "mcc"]

        comparison = []
        for model_name, model_metrics in results.items():
            row = {"model": model_name}
            for metric in metrics:
                row[metric] = model_metrics.get(metric, np.nan)
            comparison.append(row)

        df = pd.DataFrame(comparison)

        # Sort by F1 macro
        if "f1_macro" in metrics:
            df = df.sort_values("f1_macro", ascending=False)

        return df

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Any:

        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ["accuracy", "f1_macro", "auc_roc"]

        comparison_df = self.compare_models(results, metrics)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(comparison_df))
        width = 0.8 / len(metrics)

        for i, metric in enumerate(metrics):
            values = comparison_df[metric].values
            offset = (i - len(metrics) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=metric)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["model"], rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def get_summary_statistics(self, cv_results: Dict[str, Any]) -> Dict[str, str]:

        summary = {}

        for metric in ["accuracy", "f1_macro", "f1_weighted", "auc_roc"]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            ci_key = f"{metric}_ci95"

            if mean_key in cv_results:
                mean_val = cv_results[mean_key]
                std_val = cv_results.get(std_key, 0)
                ci_val = cv_results.get(ci_key, std_val * 1.96)

                summary[metric] = (
                    f"{mean_val:.3f} ± {std_val:.3f} (95% CI: ±{ci_val:.3f})"
                )

        return summary

    def evaluate_subject_variability(
        self, subject_results: pd.DataFrame
    ) -> Dict[str, Any]:

        variability = {}

        for metric in ["accuracy", "f1_macro"]:
            if metric in subject_results.columns:
                values = subject_results[metric].values

                variability[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "range": np.max(values) - np.min(values),
                    "cv": np.std(values) / np.mean(values)
                    if abs(np.mean(values)) > 1e-10
                    else np.nan,
                }

        return variability

    def statistical_comparison(
        self, model1_scores: np.ndarray, model2_scores: np.ndarray
    ) -> Dict[str, Any]:

        from scipy import stats

        model1_scores = np.asarray(model1_scores)
        model2_scores = np.asarray(model2_scores)

        if len(model1_scores) != len(model2_scores):
            raise ValueError("Score arrays must have same length")

        results = {}

        # Mean difference
        diff = model1_scores - model2_scores
        results["mean_difference"] = np.mean(diff)
        results["std_difference"] = np.std(diff)

        # Paired t-test
        try:
            t_stat, t_pvalue = stats.ttest_rel(model1_scores, model2_scores)
            results["paired_ttest"] = {
                "statistic": t_stat,
                "pvalue": t_pvalue,
                "significant": t_pvalue < 0.05,
            }
        except Exception as e:
            self.logger.warning(f"Paired t-test failed: {e}")
            results["paired_ttest"] = {"error": str(e)}

        # Wilcoxon signed-rank test (non-parametric)
        try:
            # Need at least some
            if np.any(diff != 0):
                w_stat, w_pvalue = stats.wilcoxon(model1_scores, model2_scores)
                results["wilcoxon"] = {
                    "statistic": w_stat,
                    "pvalue": w_pvalue,
                    "significant": w_pvalue < 0.05,
                }
            else:
                results["wilcoxon"] = {
                    "statistic": np.nan,
                    "pvalue": 1.0,
                    "significant": False,
                    "note": "No differences between models",
                }
        except Exception as e:
            self.logger.warning(f"Wilcoxon test failed: {e}")
            results["wilcoxon"] = {"error": str(e)}

        # Determine overall significance
        ttest_sig = results.get("paired_ttest", {}).get("significant", False)
        wilcoxon_sig = results.get("wilcoxon", {}).get("significant", False)

        results["is_significant"] = ttest_sig or wilcoxon_sig
        results["significance_level"] = 0.05

        self.logger.debug(
            f"Statistical comparison: diff={results['mean_difference']:.4f}, "
            f"significant={results['is_significant']}"
        )

        return results

    def mcnemar_test(
        self, y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray
    ) -> Dict[str, Any]:

        from scipy import stats

        y_true = np.asarray(y_true)
        y_pred1 = np.asarray(y_pred1)
        y_pred2 = np.asarray(y_pred2)

        # Build contingency table
        # b: model1 correct, model2
        # c: model1 wrong, model2
        correct1 = y_pred1 == y_true
        correct2 = y_pred2 == y_true

        b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model
        c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model

        # McNemar statistic (with continuity
        if b + c == 0:
            return {
                "statistic": np.nan,
                "pvalue": 1.0,
                "significant": False,
                "note": "No discordant pairs",
            }

        # Chi-squared approximation with continuity
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        pvalue = 1 - stats.chi2.cdf(statistic, df=1)

        return {
            "statistic": statistic,
            "pvalue": pvalue,
            "significant": pvalue < 0.05,
            "b_count": int(b),  # Model 1 better
            "c_count": int(c),  # Model 2 better
        }

    def compute_confidence_intervals(
        self, scores: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float, float]:

        from scipy import stats

        scores = np.asarray(scores)
        n = len(scores)

        if n < 2:
            return np.mean(scores), np.nan, np.nan

        mean = np.mean(scores)
        std_err = stats.sem(scores)

        # t-distribution for small samples
        t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_value * std_err

        lower = mean - margin
        upper = mean + margin

        return mean, lower, upper

    def bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: callable,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: int = 42,
    ) -> Tuple[float, float, float]:

        np.random.seed(random_state)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n_samples = len(y_true)

        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.randint(0, n_samples, size=n_samples)
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)

        bootstrap_scores = np.array(bootstrap_scores)

        # Percentile method
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_scores, alpha * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
        mean = np.mean(bootstrap_scores)

        return mean, lower, upper
