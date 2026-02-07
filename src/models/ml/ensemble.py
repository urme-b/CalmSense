from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...logging_config import LoggerMixin
from .base_model import BaseMLModel
from .classifiers import get_classifier


class StackingEnsemble(LoggerMixin):
    def __init__(
        self,
        base_models: List[Tuple[str, BaseMLModel]],
        meta_learner: Union[str, BaseMLModel] = "lr",
        use_probas: bool = True,
        passthrough: bool = False,
        cv: int = 5,
        random_state: int = 42,
    ):

        self.base_models = base_models
        self.use_probas = use_probas
        self.passthrough = passthrough
        self.cv = cv
        self.random_state = random_state

        # Create meta-learner
        if isinstance(meta_learner, str):
            self.meta_learner = get_classifier(meta_learner, random_state=random_state)
        else:
            self.meta_learner = meta_learner

        self._fitted_base_models = []
        self._classes = None
        self._n_features = None
        self.is_fitted = False

        self.logger.debug(
            f"StackingEnsemble initialized with {len(base_models)} base models"
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> "StackingEnsemble":

        from sklearn.model_selection import StratifiedKFold

        X = np.asarray(X)
        y = np.asarray(y)

        self._classes = np.unique(y)
        self._n_features = X.shape[1]
        n_samples = len(y)
        n_classes = len(self._classes)

        self.logger.info(f"Fitting stacking ensemble on {n_samples} samples")

        # Generate out-of-fold predictions for
        if self.use_probas:
            meta_features = np.zeros((n_samples, len(self.base_models) * n_classes))
        else:
            meta_features = np.zeros((n_samples, len(self.base_models)))

        # Use stratified k-fold
        cv = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for model_idx, (name, model) in enumerate(self.base_models):
                # Clone model for this
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train, y_train)

                if self.use_probas:
                    preds = model_clone.predict_proba(X_val)
                    start_col = model_idx * n_classes
                    end_col = start_col + n_classes
                    meta_features[val_idx, start_col:end_col] = preds
                else:
                    preds = model_clone.predict(X_val)
                    meta_features[val_idx, model_idx] = preds

        # Fit base models on
        self._fitted_base_models = []
        for name, model in self.base_models:
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X, y)
            self._fitted_base_models.append((name, model_clone))

        # Optionally add original features
        if self.passthrough:
            meta_features = np.hstack([meta_features, X])

        # Fit meta-learner
        self.meta_learner.fit(meta_features, y)
        self.is_fitted = True

        self.logger.info("Stacking ensemble fitted successfully")
        return self

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:

        n_samples = len(X)
        n_classes = len(self._classes)

        if self.use_probas:
            meta_features = np.zeros(
                (n_samples, len(self._fitted_base_models) * n_classes)
            )
        else:
            meta_features = np.zeros((n_samples, len(self._fitted_base_models)))

        for model_idx, (name, model) in enumerate(self._fitted_base_models):
            if self.use_probas:
                preds = model.predict_proba(X)
                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                meta_features[:, start_col:end_col] = preds
            else:
                preds = model.predict(X)
                meta_features[:, model_idx] = preds

        if self.passthrough:
            meta_features = np.hstack([meta_features, X])

        return meta_features

    def predict(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()
        X = np.asarray(X)
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()
        X = np.asarray(X)
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)

    def _check_is_fitted(self):

        if not self.is_fitted:
            raise ValueError("StackingEnsemble is not fitted. Call fit() first.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:

        return {
            "base_models": self.base_models,
            "use_probas": self.use_probas,
            "passthrough": self.passthrough,
            "cv": self.cv,
            "random_state": self.random_state,
        }


class VotingEnsemble(LoggerMixin):
    def __init__(
        self,
        models: List[Tuple[str, BaseMLModel]],
        voting: str = "soft",
        weights: Optional[List[float]] = None,
        random_state: int = 42,
    ):

        self.models = models
        self.voting = voting
        self.weights = weights
        self.random_state = random_state

        self._fitted_models = []
        self._classes = None
        self.is_fitted = False

        # Validate weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                self.logger.warning("Weights do not sum to 1, normalizing...")
                self.weights = [w / sum(weights) for w in weights]

        self.logger.debug(
            f"VotingEnsemble initialized with {len(models)} models, voting={voting}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":

        X = np.asarray(X)
        y = np.asarray(y)

        self._classes = np.unique(y)

        self.logger.info(f"Fitting voting ensemble on {len(y)} samples")

        self._fitted_models = []
        for name, model in self.models:
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X, y)
            self._fitted_models.append((name, model_clone))
            self.logger.debug(f"Fitted base model: {name}")

        self.is_fitted = True
        self.logger.info("Voting ensemble fitted successfully")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()
        X = np.asarray(X)

        if self.voting == "soft":
            probas = self.predict_proba(X)
            return self._classes[np.argmax(probas, axis=1)]
        else:
            # Hard voting
            predictions = np.array(
                [model.predict(X) for name, model in self._fitted_models]
            )

            # Weighted majority vote
            n_samples = len(X)
            n_classes = len(self._classes)
            votes = np.zeros((n_samples, n_classes))

            weights = self.weights or [1.0 / len(self._fitted_models)] * len(
                self._fitted_models
            )

            for model_idx, preds in enumerate(predictions):
                for sample_idx, pred in enumerate(preds):
                    class_idx = np.where(self._classes == pred)[0][0]
                    votes[sample_idx, class_idx] += weights[model_idx]

            return self._classes[np.argmax(votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_is_fitted()
        X = np.asarray(X)

        n_samples = len(X)
        n_classes = len(self._classes)

        weights = self.weights or [1.0 / len(self._fitted_models)] * len(
            self._fitted_models
        )

        # Aggregate weighted probabilities
        probas = np.zeros((n_samples, n_classes))

        for model_idx, (name, model) in enumerate(self._fitted_models):
            model_probas = model.predict_proba(X)
            probas += weights[model_idx] * model_probas

        return probas

    def _check_is_fitted(self):

        if not self.is_fitted:
            raise ValueError("VotingEnsemble is not fitted. Call fit() first.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:

        return {
            "models": self.models,
            "voting": self.voting,
            "weights": self.weights,
            "random_state": self.random_state,
        }

    def get_model_contributions(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:

        self._check_is_fitted()

        from sklearn.metrics import accuracy_score

        contributions = {}
        for name, model in self._fitted_models:
            preds = model.predict(X)
            contributions[name] = accuracy_score(y, preds)

        return contributions


def create_default_stacking_ensemble(
    random_state: int = 42, use_probas: bool = True
) -> StackingEnsemble:

    from .classifiers import (
        RandomForestClassifier,
        XGBoostClassifier,
        LightGBMClassifier,
        SVMClassifier,
    )

    base_models = [
        ("rf", RandomForestClassifier(random_state=random_state)),
        ("xgb", XGBoostClassifier(random_state=random_state)),
        ("lgbm", LightGBMClassifier(random_state=random_state)),
        ("svm", SVMClassifier(random_state=random_state)),
    ]

    return StackingEnsemble(
        base_models=base_models,
        meta_learner="lr",
        use_probas=use_probas,
        random_state=random_state,
    )


def create_default_voting_ensemble(
    random_state: int = 42, voting: str = "soft"
) -> VotingEnsemble:

    from .classifiers import (
        RandomForestClassifier,
        XGBoostClassifier,
        LightGBMClassifier,
    )

    models = [
        ("rf", RandomForestClassifier(random_state=random_state)),
        ("xgb", XGBoostClassifier(random_state=random_state)),
        ("lgbm", LightGBMClassifier(random_state=random_state)),
    ]

    return VotingEnsemble(models=models, voting=voting, random_state=random_state)
