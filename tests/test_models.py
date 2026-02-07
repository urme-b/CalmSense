import numpy as np
import pytest

from src.models.ml import (
    XGBoostClassifier,
    LightGBMClassifier,
    CatBoostClassifier,
    VotingEnsemble as EnsembleClassifier,
)


class TestMLClassifiers:
    @pytest.fixture
    def sample_data(self):

        np.random.seed(42)
        n_samples = 200
        n_features = 50

        # Create separable classes
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return X, y

    @pytest.fixture
    def train_test_split(self, sample_data):

        X, y = sample_data
        split_idx = 150

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    def test_xgboost_classifier(self, train_test_split):

        X_train, X_test, y_train, y_test = train_test_split

        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)

        # Test predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})

        # Test probabilities
        probas = model.predict_proba(X_test)
        assert probas.shape == (len(y_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

        # Test feature importance
        importance_df = model.get_feature_importance()
        assert len(importance_df) == X_train.shape[1]

    def test_lightgbm_classifier(self, train_test_split):

        X_train, X_test, y_train, y_test = train_test_split

        model = LightGBMClassifier(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

        probas = model.predict_proba(X_test)
        assert probas.shape == (len(y_test), 2)

    def test_catboost_classifier(self, train_test_split):

        X_train, X_test, y_train, y_test = train_test_split

        model = CatBoostClassifier(iterations=10, depth=3)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

        probas = model.predict_proba(X_test)
        assert probas.shape == (len(y_test), 2)

    def test_ensemble_classifier(self, train_test_split):

        X_train, X_test, y_train, y_test = train_test_split

        base_classifiers = [
            ("xgb", XGBoostClassifier(n_estimators=5)),
            ("lgbm", LightGBMClassifier(n_estimators=5)),
        ]

        model = EnsembleClassifier(models=base_classifiers)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

        probas = model.predict_proba(X_test)
        assert probas.shape == (len(y_test), 2)

    def test_model_accuracy(self, train_test_split):

        X_train, X_test, y_train, y_test = train_test_split

        model = XGBoostClassifier(n_estimators=50, max_depth=5)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        # Should achieve decent accuracy
        assert accuracy > 0.6
