"""API contract tests on a representative pipeline, plus a real-model smoke test."""

import joblib
import numpy as np
import pytest
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import api.model as model_module
from api.main import explain, health, predict
from api.model import MODEL_PATH, StressModel
from api.schemas import PredictionRequest


@pytest.fixture
def trained_model(tmp_path):
    """A small but real RF pipeline with the same bundle shape as production."""
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


def test_health_reflects_model_presence():
    h = health()
    assert h.status == "ok"
    assert h.model_loaded == MODEL_PATH.exists()  # not a hardcoded constant


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


def test_unknown_features_rejected(trained_model):
    req = PredictionRequest(features={"not_a_feature": 1.0})
    with pytest.raises(HTTPException) as exc:
        predict(req)
    assert exc.value.status_code == 422


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not built")
def test_real_model_smoke():
    model = StressModel(MODEL_PATH)
    # feed real feature names with placeholder values; missing ones are imputed
    result = model.predict({name: 0.0 for name in model.features[:5]})
    assert result["prediction"] in model.classes
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
