"""Load the trained stress classifier and serve predictions."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "stress_classifier.joblib"


def _positive_class_shap(values) -> np.ndarray:
    """Reduce any SHAP output shape to per-feature values for the positive class."""
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (samples, features, classes)
        return values[0, :, -1]
    return values[0]


class StressModel:
    def __init__(self, path: Path = MODEL_PATH):
        import joblib

        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.features: List[str] = bundle["features"]
        self.classes: List[str] = bundle["classes"]
        self._explainer = None

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        row = [features.get(name, np.nan) for name in self.features]
        return np.asarray(row, dtype=float).reshape(1, -1)

    def predict(self, features: Dict[str, float]) -> Dict:
        x = self._vectorize(features)
        proba = self.pipeline.predict_proba(x)[0]
        idx = int(proba.argmax())
        return {
            "prediction": self.classes[idx],
            "confidence": float(proba[idx]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
        }

    def explain(self, features: Dict[str, float], top_k: int = 8) -> List[Dict]:
        import shap

        x = self._vectorize(features)
        transformed = self.pipeline.named_steps["scale"].transform(
            self.pipeline.named_steps["impute"].transform(x)
        )
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.pipeline.named_steps["clf"])
        contrib = _positive_class_shap(self._explainer.shap_values(transformed))
        order = np.argsort(np.abs(contrib))[::-1][:top_k]
        return [
            {"feature": self.features[i], "value": float(x[0, i]), "shap": float(contrib[i])}
            for i in order
        ]


_model: Optional[StressModel] = None


def get_model() -> Optional[StressModel]:
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = StressModel()
    return _model
