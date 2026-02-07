from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SimpleSHAPWrapper:
    def __init__(self, model: Any, model_type: str = "tree"):

        self.model = model
        self.model_type = model_type
        self.explainer = None

    def fit(self, X_background: np.ndarray) -> "SimpleSHAPWrapper":

        import shap

        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, shap.sample(X_background, 100)
            )
        elif self.model_type == "deep":
            self.explainer = shap.DeepExplainer(self.model, X_background[:100])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return self

    def explain(self, X: np.ndarray) -> np.ndarray:

        if self.explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        return self.explainer.shap_values(X)

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_display: int = 20,
    ) -> None:

        import shap

        shap.summary_plot(
            shap_values, X, feature_names=feature_names, max_display=max_display
        )

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        sample_idx: int = 0,
        feature_names: Optional[List[str]] = None,
    ) -> None:

        import shap

        shap.plots.waterfall(shap_values[sample_idx])


class SimpleLIMEWrapper:
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ):

        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ["baseline", "stress"]
        self.explainer = None

    def fit(self, X_train: np.ndarray) -> "SimpleLIMEWrapper":

        from lime.lime_tabular import LimeTabularExplainer  # noqa: F811

        self.explainer = LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
        )

        return self

    def explain_instance(self, instance: np.ndarray, num_features: int = 10) -> Any:

        if self.explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        return self.explainer.explain_instance(
            instance, self.model.predict_proba, num_features=num_features
        )

    def get_feature_weights(self, explanation: Any, label: int = 1) -> Dict[str, float]:

        return dict(explanation.as_list(label=label))


class FeatureImportance:
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):

        self.model = model
        self.feature_names = feature_names

    def get_mdi_importance(self) -> Dict[str, float]:

        importances = self.model.feature_importances_

        if self.feature_names is None:
            names = [f"feature_{i}" for i in range(len(importances))]
        else:
            names = self.feature_names

        return dict(zip(names, importances))

    def get_permutation_importance(
        self, X: np.ndarray, y: np.ndarray, n_repeats: int = 10, random_state: int = 42
    ) -> Dict[str, Tuple[float, float]]:

        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            self.model, X, y, n_repeats=n_repeats, random_state=random_state
        )

        if self.feature_names is None:
            names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            names = self.feature_names

        return {
            name: (result.importances_mean[i], result.importances_std[i])
            for i, name in enumerate(names)
        }

    def plot_importance(
        self,
        importance_dict: Dict[str, float],
        top_k: int = 20,
        title: str = "Feature Importance",
    ) -> None:

        import matplotlib.pyplot as plt

        # Sort by importance
        sorted_items = sorted(
            importance_dict.items(),
            key=lambda x: x[1] if isinstance(x[1], float) else x[1][0],
            reverse=True,
        )[:top_k]

        names = [item[0] for item in sorted_items]
        values = [
            item[1] if isinstance(item[1], float) else item[1][0]
            for item in sorted_items
        ]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(names)), values)
        plt.yticks(range(len(names)), names)
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.show()
