"""Load the trained stress classifier and serve predictions."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "stress_classifier.joblib"
_TREE_MODELS = {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier"}


def _class_shap(values, idx: int) -> np.ndarray:
    """Per-feature SHAP values for one class, across SHAP's output shapes."""
    if isinstance(values, list):  # one array per class
        return np.asarray(values[idx])[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (samples, features, classes)
        return values[0, :, idx]
    return values[0]  # (samples, features)


class StressModel:
    """Serve the trained stress classifier from its joblib bundle.

    Wraps the fitted ``impute -> scale -> clf`` pipeline plus its feature order and
    class names. Callers pass a feature-name to value mapping; unknown names are
    ignored and missing ones imputed, matching training and the in-browser ONNX path.
    """

    def __init__(self, path: Path = MODEL_PATH):
        import joblib

        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.features: List[str] = bundle["features"]
        self.classes: List[str] = bundle["classes"]
        self._feature_set = set(self.features)
        self._explainer = None

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        if not any(name in self._feature_set for name in features):
            raise ValueError(
                "No recognised feature names. Expected e.g. "
                + ", ".join(self.features[:5])
                + ", ..."
            )
        row = np.asarray([features.get(name, np.nan) for name in self.features], dtype=float)
        row[~np.isfinite(row)] = np.nan  # non-finite -> imputed, matching training and the browser
        return row.reshape(1, -1)

    def _proba(self, x: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(x)[0]

    def predict(self, features: Dict[str, float]) -> Dict:
        proba = self._proba(self._vectorize(features))
        idx = int(proba.argmax())
        return {
            "prediction": self.classes[idx],
            "confidence": float(proba[idx]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
        }

    def predict_and_explain(
        self, features: Dict[str, float], top_k: int = 8
    ) -> Tuple[Dict, List[Dict]]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        top_k = min(top_k, len(self.features))

        import shap

        x = self._vectorize(features)
        imputed = self.pipeline.named_steps["impute"].transform(x)
        scaled = self.pipeline.named_steps["scale"].transform(imputed)
        clf = self.pipeline.named_steps["clf"]

        proba = clf.predict_proba(scaled)[0]
        idx = int(proba.argmax())
        result = {
            "prediction": self.classes[idx],
            "confidence": float(proba[idx]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
        }

        # SHAP only for tree models; report the value the model actually saw (imputed)
        if clf.__class__.__name__ not in _TREE_MODELS:
            return result, []
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(clf)
        contrib = _class_shap(self._explainer.shap_values(scaled), idx)
        order = np.argsort(np.abs(contrib))[::-1][:top_k]
        contributions = [
            {"feature": self.features[i], "value": float(imputed[0, i]), "shap": float(contrib[i])}
            for i in order
        ]
        return result, contributions


_model: Optional[StressModel] = None


def get_model() -> Optional[StressModel]:
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = StressModel()
    return _model
