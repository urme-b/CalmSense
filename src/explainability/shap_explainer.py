import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..logging_config import LoggerMixin


class SHAPExplainer(LoggerMixin):
    def __init__(
        self,
        model: Any,
        model_type: str = "tree",
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100,
    ):

        if not SHAP_AVAILABLE:
            raise ImportError("shap is required. Install with: pip install shap")

        self.model = model
        self.model_type = model_type
        self.background_data = background_data
        self.n_background_samples = n_background_samples
        self.explainer = None
        self._expected_value = None

        self.logger.info(f"Initialized SHAPExplainer with model_type='{model_type}'")

    def _create_explainer(self, X_background: Optional[np.ndarray] = None) -> None:

        if self.model_type == "tree":
            # TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)

        elif self.model_type == "kernel":
            # KernelExplainer for black-box models
            if X_background is None and self.background_data is None:
                raise ValueError("KernelSHAP requires background data")

            bg_data = X_background if X_background is not None else self.background_data

            # Use k-means to summarize
            if len(bg_data) > self.n_background_samples:
                bg_summary = shap.kmeans(bg_data, self.n_background_samples)
            else:
                bg_summary = bg_data

            # Get prediction function
            if hasattr(self.model, "predict_proba"):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict

            self.explainer = shap.KernelExplainer(predict_fn, bg_summary)

        elif self.model_type == "deep":
            # DeepExplainer for neural networks
            if X_background is None and self.background_data is None:
                raise ValueError("DeepSHAP requires background data")

            bg_data = X_background if X_background is not None else self.background_data
            bg_sample = bg_data[: min(100, len(bg_data))]

            self.explainer = shap.DeepExplainer(self.model, bg_sample)

        elif self.model_type == "linear":
            # LinearExplainer for linear models
            if X_background is None and self.background_data is None:
                raise ValueError("LinearSHAP requires background data")

            bg_data = X_background if X_background is not None else self.background_data
            self.explainer = shap.LinearExplainer(self.model, bg_data)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def compute_shap_values(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        check_additivity: bool = True,
    ) -> Dict[str, Any]:

        X = np.asarray(X)

        if self.explainer is None:
            self._create_explainer(X)

        self.logger.info(f"Computing SHAP values for {len(X)} samples")

        # Compute SHAP values
        if self.model_type == "tree":
            shap_values = self.explainer.shap_values(
                X, check_additivity=check_additivity
            )
        else:
            shap_values = self.explainer.shap_values(X)

        # Get expected value
        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
        else:
            base_value = None

        # Store for later use
        self._expected_value = base_value

        # Generate feature names if
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        return {
            "shap_values": shap_values,
            "base_value": base_value,
            "feature_names": feature_names,
            "data": X,
        }

    def plot_summary(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        feature_names: List[str],
        X: Optional[np.ndarray] = None,
        max_display: int = 20,
        plot_type: str = "dot",
        class_idx: Optional[int] = None,
        show: bool = False,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:

        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        # Handle multiclass
        if isinstance(shap_values, list):
            if class_idx is not None:
                sv = shap_values[class_idx]
            else:
                # Default to positive class
                sv = shap_values[1] if len(shap_values) == 2 else shap_values[-1]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == "bar":
            # Bar plot of mean
            mean_abs_shap = np.abs(sv).mean(axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[-max_display:]

            ax.barh(
                range(len(sorted_idx)),
                mean_abs_shap[sorted_idx],
                color="steelblue",
                alpha=0.8,
            )
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx])
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title("Feature Importance (SHAP)")

        else:
            # Use shap's built-in summary
            plt.close(fig)
            shap.summary_plot(
                sv,
                X,
                feature_names=feature_names,
                max_display=max_display,
                plot_type=plot_type,
                show=False,
            )
            fig = plt.gcf()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            self.logger.info(f"Summary plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_waterfall(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        idx: int,
        feature_names: List[str],
        feature_values: Optional[np.ndarray] = None,
        max_display: int = 10,
        class_idx: Optional[int] = None,
        show: bool = False,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:

        # Handle multiclass
        if isinstance(shap_values, list):
            if class_idx is not None:
                sv = shap_values[class_idx][idx]
                base = (
                    self._expected_value[class_idx]
                    if isinstance(self._expected_value, (list, np.ndarray))
                    else self._expected_value
                )
            else:
                sv = (
                    shap_values[1][idx]
                    if len(shap_values) == 2
                    else shap_values[-1][idx]
                )
                base = (
                    self._expected_value[1]
                    if isinstance(self._expected_value, (list, np.ndarray))
                    and len(self._expected_value) > 1
                    else self._expected_value
                )
        else:
            sv = shap_values[idx]
            base = self._expected_value

        # Create Explanation object for
        explanation = shap.Explanation(
            values=sv,
            base_values=base if base is not None else 0,
            data=feature_values[idx] if feature_values is not None else None,
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        fig = plt.gcf()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_force(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        idx: int,
        feature_names: List[str],
        feature_values: Optional[np.ndarray] = None,
        class_idx: Optional[int] = None,
        matplotlib: bool = True,
        show: bool = False,
        save_path: Optional[Path] = None,
    ) -> Union[plt.Figure, str]:

        # Handle multiclass
        if isinstance(shap_values, list):
            if class_idx is not None:
                sv = shap_values[class_idx][idx]
                base = (
                    self._expected_value[class_idx]
                    if isinstance(self._expected_value, (list, np.ndarray))
                    else self._expected_value
                )
            else:
                sv = (
                    shap_values[1][idx]
                    if len(shap_values) == 2
                    else shap_values[-1][idx]
                )
                base = (
                    self._expected_value[1]
                    if isinstance(self._expected_value, (list, np.ndarray))
                    and len(self._expected_value) > 1
                    else self._expected_value
                )
        else:
            sv = shap_values[idx]
            base = self._expected_value

        fv = feature_values[idx] if feature_values is not None else None

        if matplotlib:
            shap.force_plot(
                base if base is not None else 0,
                sv,
                fv,
                feature_names=feature_names,
                matplotlib=True,
                show=False,
            )
            fig = plt.gcf()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

            if show:
                plt.show()

            return fig
        else:
            # Return HTML for web
            force_plot = shap.force_plot(
                base if base is not None else 0, sv, fv, feature_names=feature_names
            )
            return shap.getjs() + force_plot.html()

    def plot_dependence(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        feature: str,
        feature_names: List[str],
        X: np.ndarray,
        interaction_feature: Optional[str] = "auto",
        class_idx: Optional[int] = None,
        show: bool = False,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:

        # Handle multiclass
        if isinstance(shap_values, list):
            sv = (
                shap_values[class_idx]
                if class_idx is not None
                else shap_values[1]
                if len(shap_values) == 2
                else shap_values[-1]
            )
        else:
            sv = shap_values

        feature_idx = (
            feature_names.index(feature) if isinstance(feature, str) else feature
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        shap.dependence_plot(
            feature_idx,
            sv,
            X,
            feature_names=feature_names,
            interaction_index=interaction_feature,
            ax=ax,
            show=False,
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_interaction(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        feature1: str,
        feature2: str,
        feature_names: List[str],
        X: np.ndarray,
        class_idx: Optional[int] = None,
        show: bool = False,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:

        # Handle multiclass
        if isinstance(shap_values, list):
            sv = (
                shap_values[class_idx]
                if class_idx is not None
                else shap_values[1]
                if len(shap_values) == 2
                else shap_values[-1]
            )
        else:
            sv = shap_values

        idx1 = feature_names.index(feature1)
        idx2 = feature_names.index(feature2)

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            X[:, idx1], sv[:, idx1], c=X[:, idx2], cmap="viridis", alpha=0.6
        )

        ax.set_xlabel(f"{feature1} Value")
        ax.set_ylabel(f"SHAP Value for {feature1}")
        ax.set_title(f"{feature1} vs SHAP (colored by {feature2})")

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(feature2)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def get_top_features(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        feature_names: List[str],
        n: int = 10,
        class_idx: Optional[int] = None,
    ) -> pd.DataFrame:

        # Handle multiclass
        if isinstance(shap_values, list):
            sv = (
                shap_values[class_idx]
                if class_idx is not None
                else shap_values[1]
                if len(shap_values) == 2
                else shap_values[-1]
            )
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": mean_abs_shap,
                "mean_shap": sv.mean(axis=0),
                "std_shap": sv.std(axis=0),
            }
        )

        df = df.sort_values("importance", ascending=False).head(n)
        df["rank"] = range(1, len(df) + 1)
        df["mean_abs_shap"] = df["importance"]

        return df[["rank", "feature", "mean_abs_shap", "mean_shap", "std_shap"]]

    def generate_clinical_report(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        idx: int,
        feature_names: List[str],
        feature_values: np.ndarray,
        normal_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        class_idx: Optional[int] = None,
    ) -> Dict[str, Any]:

        # Handle multiclass
        if isinstance(shap_values, list):
            if class_idx is not None:
                sv = shap_values[class_idx][idx]
            else:
                sv = (
                    shap_values[1][idx]
                    if len(shap_values) == 2
                    else shap_values[-1][idx]
                )
        else:
            sv = shap_values[idx]

        fv = feature_values[idx]

        # Default normal ranges based
        if normal_ranges is None:
            normal_ranges = {
                # HRV time domain
                "RMSSD": (25, 45),
                "SDNN": (50, 100),
                "pNN50": (5, 25),
                "mean_RR": (700, 1000),
                # HRV frequency domain
                "LF_power": (0.04, 0.15),
                "HF_power": (0.15, 0.4),
                "LF_HF_ratio": (1.0, 2.0),
                # EDA
                "SCL": (2, 16),
                "SCR_count": (0, 20),
                "mean_SCR_amplitude": (0.1, 1.0),
                # Temperature
                "mean_temp": (32, 36),
                "temp_slope": (-0.1, 0.1),
            }

        report = {
            "features": {},
            "summary": {
                "total_features": len(feature_names),
                "positive_contributors": 0,
                "negative_contributors": 0,
            },
        }

        for i, (name, value, shap_val) in enumerate(zip(feature_names, fv, sv)):
            # Determine status based on
            status = "normal"
            normal_range = normal_ranges.get(name)

            if normal_range:
                if value < normal_range[0]:
                    status = "below_normal"
                elif value > normal_range[1]:
                    status = "above_normal"

            # Determine impact direction
            if shap_val > 0:
                impact = "increases_stress_risk"
                report["summary"]["positive_contributors"] += 1
            elif shap_val < 0:
                impact = "decreases_stress_risk"
                report["summary"]["negative_contributors"] += 1
            else:
                impact = "neutral"

            # Generate interpretation
            interpretation = self._generate_feature_interpretation(
                name, value, shap_val, status, normal_range
            )

            report["features"][name] = {
                "value": float(value),
                "shap_impact": float(shap_val),
                "normal_range": normal_range,
                "status": status,
                "impact": impact,
                "interpretation": interpretation,
            }

        return report

    def _generate_feature_interpretation(
        self,
        feature_name: str,
        value: float,
        shap_value: float,
        status: str,
        normal_range: Optional[Tuple[float, float]],
    ) -> str:

        interpretations = {
            "RMSSD": {
                "description": "Root Mean Square of Successive Differences (parasympathetic activity)",
                "high": "Higher vagal tone, associated with relaxation",
                "low": "Reduced parasympathetic activity, may indicate stress",
            },
            "SDNN": {
                "description": "Standard Deviation of NN intervals (overall HRV)",
                "high": "Good overall heart rate variability",
                "low": "Reduced heart rate variability, potential stress marker",
            },
            "LF_HF_ratio": {
                "description": "Low Frequency to High Frequency ratio (sympathovagal balance)",
                "high": "Sympathetic dominance, often elevated during stress",
                "low": "Parasympathetic dominance, typically relaxed state",
            },
            "SCL": {
                "description": "Skin Conductance Level (tonic EDA)",
                "high": "Elevated arousal or stress",
                "low": "Lower arousal state",
            },
            "SCR_count": {
                "description": "Skin Conductance Response count (phasic EDA)",
                "high": "Frequent arousal responses, may indicate stress",
                "low": "Fewer arousal responses",
            },
        }

        base_info = interpretations.get(
            feature_name,
            {
                "description": f"Feature: {feature_name}",
                "high": "Value above typical range",
                "low": "Value below typical range",
            },
        )

        # Build interpretation string
        parts = [base_info["description"]]

        if normal_range:
            parts.append(f"Normal range: {normal_range[0]:.2f} - {normal_range[1]:.2f}")
            parts.append(f"Current value: {value:.4f}")

            if status == "above_normal":
                parts.append(base_info["high"])
            elif status == "below_normal":
                parts.append(base_info["low"])

        impact_str = "increases" if shap_value > 0 else "decreases"
        parts.append(
            f"This feature {impact_str} stress prediction by {abs(shap_value):.4f}"
        )

        return " | ".join(parts)

    def fig_to_base64(self, fig: plt.Figure) -> str:

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
