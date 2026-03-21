import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ...logging_config import LoggerMixin

_SAFE_MODULES = frozenset(
    {
        "sklearn",
        "numpy",
        "scipy",
        "xgboost",
        "lightgbm",
        "catboost",
        "collections",
        "builtins",
        "copyreg",
        "_pickle",
        "operator",
        "types",
        "functools",
        "copy",
        "joblib",
        "numbers",
        "io",
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if module.split(".")[0] not in _SAFE_MODULES:
            raise pickle.UnpicklingError(f"Forbidden: {module}.{name}")
        return super().find_class(module, name)


class BaseMLModel(ABC, LoggerMixin):
    def __init__(self, model_name: str, random_state: int = 42, **kwargs):

        self.model_name = model_name
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self._classes = None
        self._n_features = None
        self.logger.debug(f"Initialized {model_name}")

    @abstractmethod
    def _create_model(self) -> Any:

        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "BaseMLModel":

        X = np.asarray(X)
        y = np.asarray(y)

        self._n_features = X.shape[1]
        self._classes = np.unique(y)

        if self.model is None:
            self.model = self._create_model()

        self.logger.info(f"Fitting {self.model_name} on {len(y)} samples")

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()
        X = np.asarray(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()
        X = np.asarray(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without
            predictions = self.predict(X)
            n_classes = len(self._classes)
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(predictions):
                matches = np.where(self._classes == pred)[0]
                class_idx = matches[0] if len(matches) > 0 else 0
                proba[i, class_idx] = 1.0
            return proba

    def get_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:

        self._check_is_fitted()

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # For linear models, use
            coef = self.model.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        else:
            self.logger.warning(
                f"{self.model_name} does not support feature importances"
            )
            return pd.DataFrame()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )

        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df

    def save_model(self, path: Union[str, Path]) -> Path:

        self._check_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_state = {
            "model_name": self.model_name,
            "model": self.model,
            "random_state": self.random_state,
            "kwargs": self.kwargs,
            "classes": self._classes,
            "n_features": self._n_features,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(model_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> "BaseMLModel":

        path = Path(path)

        with open(path, "rb") as f:
            model_state = _RestrictedUnpickler(f).load()

        # Create instance without calling
        instance = cls.__new__(cls)
        instance.model_name = model_state["model_name"]
        instance.model = model_state["model"]
        instance.random_state = model_state["random_state"]
        instance.kwargs = model_state["kwargs"]
        instance._classes = model_state["classes"]
        instance._n_features = model_state["n_features"]
        instance.is_fitted = model_state["is_fitted"]

        # Init logger
        import logging

        instance._logger = logging.getLogger(f"calmsense.{cls.__name__}")

        return instance

    def _check_is_fitted(self):

        if not self.is_fitted:
            raise ValueError(f"{self.model_name} is not fitted. Call fit() first.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:

        import inspect

        params = {}
        # Introspect the subclass __init__
        sig = inspect.signature(self.__class__.__init__)
        for name, param in sig.parameters.items():
            if name in ("self", "kwargs"):
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
        params.update(self.kwargs)
        return params

    def set_params(self, **params) -> "BaseMLModel":

        for key, value in params.items():
            if key == "model_name":
                self.model_name = value
            elif key == "random_state":
                self.random_state = value
            else:
                self.kwargs[key] = value

        # Recreate model with new
        self.model = None
        self.is_fitted = False

        return self

    def __repr__(self) -> str:

        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name={self.model_name}, {fitted_str})"
