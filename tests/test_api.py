"""The API serves a real trained model end-to-end."""

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import api.model as model_module
from api.main import explain, health, predict
from api.model import StressModel
from api.schemas import PredictionRequest


@pytest.fixture
def trained_model(tmp_path):
    rng = np.random.RandomState(0)
    features = ["HRV_RMSSD", "EDA_SCL_mean", "TEMP_mean"]
    X = rng.randn(120, 3)
    y = (X[:, 0] > 0).astype(int)
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=20, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    path = tmp_path / "stress_classifier.joblib"
    joblib.dump({"pipeline": pipe, "features": features, "classes": ["baseline", "stress"]}, path)

    model_module._model = StressModel(path)
    yield
    model_module._model = None


def test_health_always_responds():
    assert health().status == "ok"


def test_predict_returns_probabilities(trained_model):
    req = PredictionRequest(features={"HRV_RMSSD": 2.0, "EDA_SCL_mean": 0.1, "TEMP_mean": 33.0})
    result = predict(req)
    assert result.prediction in ("baseline", "stress")
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6


def test_explain_returns_contributions(trained_model):
    req = PredictionRequest(features={"HRV_RMSSD": 2.0, "EDA_SCL_mean": 0.1, "TEMP_mean": 33.0})
    result = explain(req)
    assert len(result.contributions) >= 1
    assert result.contributions[0].feature in ("HRV_RMSSD", "EDA_SCL_mean", "TEMP_mean")
