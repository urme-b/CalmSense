from .ml import get_classifier

try:
    from .dl import CNN1DClassifier

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = ["get_classifier", "CNN1DClassifier"]
