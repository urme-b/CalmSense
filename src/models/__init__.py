from .ml import (
    BaseMLModel,
    LightGBMClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    XGBoostClassifier,
    get_classifier,
)

try:
    from .dl import CNN1DClassifier

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = [
    "BaseMLModel",
    "LogisticRegressionClassifier",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "LightGBMClassifier",
    "get_classifier",
    "CNN1DClassifier",
]
