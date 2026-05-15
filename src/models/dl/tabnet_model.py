from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.pretraining import TabNetPretrainer

    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

from ...logging_config import LoggerMixin


class TabNetWrapper(LoggerMixin):
    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-4,
        n_independent: int = 2,
        n_shared: int = 2,
        momentum: float = 0.02,
        clip_value: Optional[float] = 2.0,
        optimizer_fn: Any = None,
        optimizer_params: Optional[Dict] = None,
        scheduler_fn: Any = None,
        scheduler_params: Optional[Dict] = None,
        seed: int = 42,
        verbose: int = 1,
        device_name: str = "auto",
    ):

        if not TABNET_AVAILABLE:
            raise ImportError(
                "pytorch-tabnet is required for TabNet. "
                "Install with: pip install pytorch-tabnet"
            )

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.momentum = momentum
        self.clip_value = clip_value
        self.seed = seed
        self.verbose = verbose
        self.device_name = device_name

        # Default optimizer params
        if optimizer_params is None:
            optimizer_params = dict(lr=2e-2, weight_decay=1e-5)

        # Default scheduler params
        if scheduler_params is None:
            scheduler_params = dict(
                mode="min",
                patience=5,
                min_lr=1e-5,
                factor=0.5,
            )

        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        # Initialize model
        self.model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            n_independent=n_independent,
            n_shared=n_shared,
            momentum=momentum,
            clip_value=clip_value,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            seed=seed,
            verbose=verbose,
            device_name=device_name,
        )

        self._is_fitted = False
        self._feature_names: Optional[List[str]] = None
        self._classes: Optional[np.ndarray] = None

        self.logger.debug(
            f"Initialized TabNetWrapper (n_d={n_d}, n_a={n_a}, n_steps={n_steps})"
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 256,
        virtual_batch_size: int = 128,
        num_workers: int = 0,
        drop_last: bool = False,
        feature_names: Optional[List[str]] = None,
        weights: Optional[Union[int, Dict[int, float]]] = None,
        pretraining_ratio: float = 0.0,
    ) -> Dict[str, Any]:

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)

        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)
            eval_set = [(X_val, y_val)]
            eval_name = ["val"]
        else:
            eval_set = None
            eval_name = None

        self._feature_names = feature_names
        self._classes = np.unique(y_train)

        self.logger.info(
            f"Training TabNet on {len(y_train)} samples "
            f"with {X_train.shape[1]} features"
        )

        # Optional self-supervised pretraining
        if pretraining_ratio > 0:
            pretrain_epochs = int(max_epochs * pretraining_ratio)
            self._pretrain(X_train, pretrain_epochs, batch_size)

        # Train classifier
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            weights=weights,
        )

        self._is_fitted = True

        # Collect training history
        history = {
            "train_loss": self.model.history["loss"],
            "train_accuracy": self.model.history.get("train_accuracy", []),
        }
        if eval_set is not None:
            history["val_loss"] = self.model.history["val_0_loss"]
            history["val_accuracy"] = self.model.history.get("val_0_accuracy", [])

        self.logger.info(
            f"Training completed. "
            f"Best val loss: {min(history.get('val_loss', [float('inf')])):.4f}"
        )

        return history

    def _pretrain(self, X: np.ndarray, epochs: int, batch_size: int) -> None:

        self.logger.info(f"Self-supervised pretraining for {epochs} epochs")

        pretrainer = TabNetPretrainer(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            momentum=self.momentum,
            seed=self.seed,
            verbose=self.verbose,
            device_name=self.device_name,
        )

        pretrainer.fit(
            X_train=X,
            max_epochs=epochs,
            batch_size=batch_size,
            pretraining_ratio=0.8,  # Mask 80% of features
        )

        # Transfer pretrained weights
        self.model.load_weights_from_unsupervised(pretrainer)

    def predict(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X)

    def get_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> np.ndarray:

        self._check_is_fitted()

        # TabNet stores feature importance
        importance = self.model.feature_importances_

        if feature_names is not None or self._feature_names is not None:
            names = feature_names or self._feature_names
            if len(names) == len(importance):
                return dict(zip(names, importance))

        return importance

    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)

        # Get masks for each
        masks = self.model.explain(X)

        # Aggregate across steps
        aggregated = np.sum(masks, axis=0)

        # Normalize to [0, 1]
        aggregated = aggregated / (aggregated.sum(axis=1, keepdims=True) + 1e-10)

        return masks, aggregated

    def get_local_importance(
        self, X: np.ndarray, sample_idx: Optional[int] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:

        _, aggregated = self.explain(X)

        if sample_idx is not None:
            importance = aggregated[sample_idx]
            if self._feature_names is not None:
                return dict(zip(self._feature_names, importance))
            return importance

        return aggregated

    def save_model(self, path: Union[str, Path]) -> Path:

        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))
        self.logger.info(f"Model saved to {path}")

        return path

    def load_model(self, path: Union[str, Path]) -> "TabNetWrapper":

        path = Path(path)
        self.model.load_model(str(path))
        self._is_fitted = True
        self.logger.info(f"Model loaded from {path}")

        return self

    def _check_is_fitted(self) -> None:

        if not self._is_fitted:
            raise ValueError("TabNet is not fitted. Call fit() first.")

    def count_parameters(self) -> int:

        if hasattr(self.model, "network"):
            return sum(
                p.numel() for p in self.model.network.parameters() if p.requires_grad
            )
        return 0

    def summary(self) -> str:

        lines = [
            "=" * 50,
            "TabNet Summary",
            "=" * 50,
            f"n_d (decision): {self.n_d}",
            f"n_a (attention): {self.n_a}",
            f"n_steps: {self.n_steps}",
            f"gamma (sparsity coeff): {self.gamma}",
            f"lambda_sparse: {self.lambda_sparse}",
            f"n_independent: {self.n_independent}",
            f"n_shared: {self.n_shared}",
            f"Total parameters: {self.count_parameters():,}",
            f"Fitted: {self._is_fitted}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:

        return (
            f"TabNetWrapper(n_d={self.n_d}, n_a={self.n_a}, "
            f"n_steps={self.n_steps}, fitted={self._is_fitted})"
        )


class TabNetRegressor:
    def __init__(self, **kwargs):

        if not TABNET_AVAILABLE:
            raise ImportError(
                "pytorch-tabnet is required for TabNet. "
                "Install with: pip install pytorch-tabnet"
            )

        from pytorch_tabnet.tab_model import TabNetRegressor as _TabNetRegressor

        # Remove classification-specific params
        kwargs.pop("weights", None)

        self.model = _TabNetRegressor(**kwargs)
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

        eval_set = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            eval_set = [(X_val, y_val)]

        self.model.fit(X_train=X_train, y_train=y_train, eval_set=eval_set, **kwargs)

        self._is_fitted = True
        return {"history": self.model.history}

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self._is_fitted:
            raise ValueError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X).flatten()

    def get_feature_importance(self) -> np.ndarray:

        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self.model.feature_importances_

    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if not self._is_fitted:
            raise ValueError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        masks = self.model.explain(X)
        aggregated = np.sum(masks, axis=0)
        aggregated = aggregated / (aggregated.sum(axis=1, keepdims=True) + 1e-10)
        return masks, aggregated
