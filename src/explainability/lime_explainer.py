import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not installed. Install with: pip install lime")

from ..logging_config import LoggerMixin


class LIMEExplainer(LoggerMixin):
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_names: List[str],
        training_data: Optional[np.ndarray] = None,
        mode: str = "classification",
        kernel_width: Optional[float] = None,
        discretize_continuous: bool = True,
        discretizer: str = "quartile",
        random_state: int = 42,
    ):

        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME is required but not installed. Install with: pip install lime"
            )

        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.random_state = random_state

        # Store training data statistics
        self.training_data = training_data

        # Create LIME explainer
        if training_data is not None:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                class_names=class_names,
                mode=mode,
                kernel_width=kernel_width,
                discretize_continuous=discretize_continuous,
                discretizer=discretizer,
                random_state=random_state,
                verbose=False,
            )
        else:
            self.explainer = None
            self.logger.warning(
                "No training data provided. Call set_training_data() before explaining."
            )

    def set_training_data(
        self,
        training_data: np.ndarray,
        kernel_width: Optional[float] = None,
        discretize_continuous: bool = True,
        discretizer: str = "quartile",
    ) -> None:

        self.training_data = training_data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            kernel_width=kernel_width,
            discretize_continuous=discretize_continuous,
            discretizer=discretizer,
            random_state=self.random_state,
            verbose=False,
        )
        self.logger.info(f"Training data set with shape {training_data.shape}")

    def _get_predict_fn(self) -> Callable:

        # Check for predict_proba method
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba

        # PyTorch model
        if hasattr(self.model, "forward"):
            import torch

            def predict_proba(X):
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    if hasattr(self.model, "device"):
                        X_tensor = X_tensor.to(self.model.device)
                    outputs = self.model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    return probs.cpu().numpy()

            return predict_proba

        # Fallback: wrap predict method
        if hasattr(self.model, "predict"):

            def predict_proba(X):
                preds = self.model.predict(X)
                # Convert to one-hot
                n_classes = len(self.class_names)
                probs = np.zeros((len(X), n_classes))
                for i, p in enumerate(preds):
                    probs[i, int(p)] = 1.0
                return probs

            return predict_proba

        raise ValueError("Model must have predict_proba, forward, or predict method")

    def explain_instance(
        self,
        x: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        labels: Optional[Tuple[int, ...]] = None,
        top_labels: Optional[int] = None,
        distance_metric: str = "euclidean",
    ) -> Dict:

        if self.explainer is None:
            raise ValueError(
                "Explainer not initialized. Call set_training_data() first."
            )

        # Ensure x is 1D
        x = np.asarray(x).flatten()

        # Get prediction function
        predict_fn = self._get_predict_fn()

        # Generate explanation
        explanation = self.explainer.explain_instance(
            x,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=labels,
            top_labels=top_labels,
            distance_metric=distance_metric,
        )

        # Get prediction probabilities
        pred_proba = predict_fn(x.reshape(1, -1))[0]
        predicted_class = np.argmax(pred_proba)

        # Extract feature importance for
        feature_importance = {}
        for feat, weight in explanation.as_list(label=predicted_class):
            feature_importance[feat] = weight

        # Get local model statistics
        local_exp = explanation.local_exp.get(predicted_class, [])

        result = {
            "explanation": explanation,
            "feature_importance": feature_importance,
            "feature_weights": dict(local_exp),
            "predicted_class": predicted_class,
            "predicted_class_name": self.class_names[predicted_class],
            "prediction_proba": pred_proba,
            "score": explanation.score,
            "intercept": explanation.intercept.get(predicted_class, 0),
            "num_features": num_features,
            "instance_values": {
                self.feature_names[i]: x[i] for i in range(len(self.feature_names))
            },
        }

        return result

    def explain_batch(
        self,
        X: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:

        explanations = []
        n_samples = len(X)

        for i, x in enumerate(X):
            exp = self.explain_instance(
                x, num_features=num_features, num_samples=num_samples
            )
            explanations.append(exp)

            if progress_callback:
                progress_callback(i + 1, n_samples)

            if (i + 1) % 10 == 0:
                self.logger.info(f"Explained {i + 1}/{n_samples} instances")

        return explanations

    def plot_explanation(
        self,
        explanation: Dict,
        idx: int = 0,
        figsize: Tuple[int, int] = (10, 6),
        show_predicted_proba: bool = True,
    ) -> Figure:
        exp_obj = explanation["explanation"]
        predicted_class = explanation["predicted_class"]

        # Get feature contributions
        exp_list = exp_obj.as_list(label=predicted_class)

        if not exp_list:
            self.logger.warning("No features in explanation")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No features to display",
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

        # Prepare data
        features = [item[0] for item in exp_list]
        weights = [item[1] for item in exp_list]

        # Sort by absolute weight
        sorted_indices = np.argsort(np.abs(weights))[::-1]
        features = [features[i] for i in sorted_indices]
        weights = [weights[i] for i in sorted_indices]

        # Create figure
        if show_predicted_proba:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
            )
        else:
            fig, ax1 = plt.subplots(figsize=figsize)

        # Plot horizontal bars
        colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]
        y_pos = np.arange(len(features))

        ax1.barh(y_pos, weights, color=colors, edgecolor="black", linewidth=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize=9)
        ax1.axvline(x=0, color="black", linewidth=0.8)
        ax1.set_xlabel("Feature Contribution", fontsize=11)
        ax1.set_title(
            f"LIME Explanation for Instance {idx}\n"
            f"Predicted: {explanation['predicted_class_name']} "
            f"(Score: {explanation['score']:.3f})",
            fontsize=12,
        )
        ax1.invert_yaxis()

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#2ecc71", label="Supports prediction"),
            Patch(facecolor="#e74c3c", label="Contradicts prediction"),
        ]
        ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

        # Plot prediction probabilities
        if show_predicted_proba:
            proba = explanation["prediction_proba"]
            colors_proba = [
                "#3498db" if i != predicted_class else "#e74c3c"
                for i in range(len(proba))
            ]

            ax2.barh(
                range(len(proba)),
                proba,
                color=colors_proba,
                edgecolor="black",
                linewidth=0.5,
            )
            ax2.set_yticks(range(len(proba)))
            ax2.set_yticklabels(self.class_names, fontsize=10)
            ax2.set_xlabel("Probability", fontsize=11)
            ax2.set_title("Prediction\nProbabilities", fontsize=11)
            ax2.set_xlim(0, 1)
            ax2.invert_yaxis()

        plt.tight_layout()
        return fig

    def plot_multiple_explanations(
        self,
        explanations: List[Dict],
        max_features: int = 5,
        figsize: Tuple[int, int] = (14, 10),
    ) -> Figure:

        n_explanations = len(explanations)
        n_cols = min(3, n_explanations)
        n_rows = (n_explanations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_explanations == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, exp in enumerate(explanations):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            exp_obj = exp["explanation"]
            predicted_class = exp["predicted_class"]
            exp_list = exp_obj.as_list(label=predicted_class)[:max_features]

            if exp_list:
                features = [
                    item[0][:20] + "..." if len(item[0]) > 20 else item[0]
                    for item in exp_list
                ]
                weights = [item[1] for item in exp_list]

                colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]
                ax.barh(range(len(features)), weights, color=colors)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features, fontsize=8)
                ax.axvline(x=0, color="black", linewidth=0.8)
                ax.invert_yaxis()

            ax.set_title(f"Instance {idx}: {exp['predicted_class_name']}", fontsize=10)

        # Hide empty subplots
        for idx in range(n_explanations, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle("LIME Explanations Overview", fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    def compare_with_shap(
        self,
        lime_importances: Dict[str, float],
        shap_importances: Dict[str, float],
        top_n: int = 10,
    ) -> pd.DataFrame:

        # Get all features
        all_features = set(lime_importances.keys()) | set(shap_importances.keys())

        comparison_data = []
        for feature in all_features:
            lime_val = lime_importances.get(feature, 0)
            shap_val = shap_importances.get(feature, 0)

            comparison_data.append(
                {
                    "feature": feature,
                    "lime_importance": lime_val,
                    "shap_importance": shap_val,
                    "lime_abs": abs(lime_val),
                    "shap_abs": abs(shap_val),
                    "direction_agree": np.sign(lime_val) == np.sign(shap_val)
                    if (lime_val != 0 and shap_val != 0)
                    else None,
                }
            )

        df = pd.DataFrame(comparison_data)

        # Rank features by both
        df["lime_rank"] = df["lime_abs"].rank(ascending=False)
        df["shap_rank"] = df["shap_abs"].rank(ascending=False)
        df["rank_diff"] = abs(df["lime_rank"] - df["shap_rank"])

        # Sort by combined importance
        df["combined_importance"] = df["lime_abs"] + df["shap_abs"]
        df = df.sort_values("combined_importance", ascending=False)

        # Calculate correlation
        valid_mask = (df["lime_abs"] > 0) & (df["shap_abs"] > 0)
        if valid_mask.sum() > 2:
            from scipy import stats

            correlation, p_value = stats.spearmanr(
                df.loc[valid_mask, "lime_rank"], df.loc[valid_mask, "shap_rank"]
            )
            self.logger.info(
                f"LIME-SHAP rank correlation: {correlation:.3f} (p={p_value:.4f})"
            )

        return df.head(top_n)[
            [
                "feature",
                "lime_importance",
                "shap_importance",
                "lime_rank",
                "shap_rank",
                "direction_agree",
            ]
        ]

    def plot_comparison(
        self, comparison_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        features = comparison_df["feature"].tolist()
        y_pos = np.arange(len(features))

        # LIME importances
        ax1 = axes[0]
        lime_vals = comparison_df["lime_importance"].values
        colors1 = ["#2ecc71" if v > 0 else "#e74c3c" for v in lime_vals]
        ax1.barh(y_pos, lime_vals, color=colors1, edgecolor="black", linewidth=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize=9)
        ax1.axvline(x=0, color="black", linewidth=0.8)
        ax1.set_xlabel("LIME Weight")
        ax1.set_title("LIME Feature Importance")
        ax1.invert_yaxis()

        # SHAP importances
        ax2 = axes[1]
        shap_vals = comparison_df["shap_importance"].values
        colors2 = ["#2ecc71" if v > 0 else "#e74c3c" for v in shap_vals]
        ax2.barh(y_pos, shap_vals, color=colors2, edgecolor="black", linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features, fontsize=9)
        ax2.axvline(x=0, color="black", linewidth=0.8)
        ax2.set_xlabel("SHAP Value")
        ax2.set_title("SHAP Feature Importance")
        ax2.invert_yaxis()

        plt.suptitle("LIME vs SHAP Feature Importance Comparison", fontsize=12)
        plt.tight_layout()
        return fig

    def get_feature_statistics(
        self, explanations: List[Dict], feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:

        if feature_names is None:
            feature_names = self.feature_names

        # Collect all weights
        feature_weights = {f: [] for f in feature_names}

        for exp in explanations:
            exp_obj = exp["explanation"]
            predicted_class = exp["predicted_class"]

            # Get weights for this
            local_exp = dict(exp_obj.local_exp.get(predicted_class, []))

            for idx, name in enumerate(feature_names):
                if idx in local_exp:
                    feature_weights[name].append(local_exp[idx])

        # Compute statistics
        stats_data = []
        for feature in feature_names:
            weights = feature_weights[feature]
            if weights:
                stats_data.append(
                    {
                        "feature": feature,
                        "mean_weight": np.mean(weights),
                        "std_weight": np.std(weights),
                        "median_weight": np.median(weights),
                        "min_weight": np.min(weights),
                        "max_weight": np.max(weights),
                        "abs_mean": np.mean(np.abs(weights)),
                        "n_appearances": len(weights),
                        "pct_positive": np.mean([w > 0 for w in weights]) * 100,
                    }
                )

        df = pd.DataFrame(stats_data)
        df = df.sort_values("abs_mean", ascending=False)

        return df

    def generate_report(
        self, explanation: Dict, include_instance_values: bool = True
    ) -> str:

        lines = [
            "=" * 60,
            "LIME EXPLANATION REPORT",
            "=" * 60,
            "",
            f"Predicted Class: {explanation['predicted_class_name']}",
            f"Local Model R² Score: {explanation['score']:.4f}",
            f"Intercept: {explanation['intercept']:.4f}",
            "",
            "Prediction Probabilities:",
        ]

        for i, (name, prob) in enumerate(
            zip(self.class_names, explanation["prediction_proba"])
        ):
            indicator = " <--" if i == explanation["predicted_class"] else ""
            lines.append(f"  {name}: {prob:.4f}{indicator}")

        lines.extend(
            [
                "",
                f"Top {explanation['num_features']} Contributing Features:",
                "-" * 40,
            ]
        )

        for rank, (feature, weight) in enumerate(
            explanation["feature_importance"].items(), 1
        ):
            direction = "+" if weight > 0 else "-"
            effect = "supports" if weight > 0 else "contradicts"
            lines.append(f"  {rank}. {feature}")
            lines.append(
                f"     Weight: {direction}{abs(weight):.4f} ({effect} prediction)"
            )

        if include_instance_values:
            lines.extend(
                [
                    "",
                    "Instance Feature Values:",
                    "-" * 40,
                ]
            )
            for feature, value in explanation["instance_values"].items():
                lines.append(f"  {feature}: {value:.4f}")

        lines.append("=" * 60)

        return "\n".join(lines)
