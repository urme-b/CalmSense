"""The exported ONNX model reproduces the sklearn pipeline the browser relies on."""

import json

import numpy as np
import pytest

from src.config import MODELS_DIR, PROJECT_ROOT

MODEL = MODELS_DIR / "stress_classifier.joblib"
ONNX = PROJECT_ROOT / "frontend" / "public" / "model.onnx"
META = PROJECT_ROOT / "frontend" / "src" / "model_meta.json"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and ONNX.exists() and META.exists()),
    reason="run scripts/export_onnx.py first",
)


def test_meta_feature_order_matches_model():
    import joblib

    bundle = joblib.load(MODEL)
    meta = json.load(open(META))
    # the browser builds its input vector in this exact order
    assert meta["features"] == bundle["features"]
    assert (
        len(meta["medians"]) == len(meta["mean"]) == len(meta["scale"]) == len(bundle["features"])
    )


def test_onnx_matches_sklearn_including_nonfinite():
    import joblib
    import onnxruntime as ort

    bundle = joblib.load(MODEL)
    pipe, n = bundle["pipeline"], len(bundle["features"])
    meta = json.load(open(META))

    rng = np.random.RandomState(0)
    X = rng.randn(40, n)
    X[rng.rand(*X.shape) < 0.3] = np.nan  # missing values
    X[rng.rand(*X.shape) < 0.05] = np.inf  # the ±Inf path the JS guards against

    # sklearn imputer handles NaN; non-finite is mapped to NaN upstream (as in _vectorize)
    ref = pipe.predict_proba(np.where(np.isfinite(X), X, np.nan))

    # reproduce services/onnx.ts vectorize(): non-finite -> median, then standardize
    medians, mean, scale = (np.array(meta[k]) for k in ("medians", "mean", "scale"))
    filled = np.where(np.isfinite(X), X, medians)
    standardized = ((filled - mean) / scale).astype(np.float32)

    sess = ort.InferenceSession(ONNX.read_bytes())
    outputs = sess.run(None, {"input": standardized})
    proba = next(np.asarray(o) for o in outputs if np.asarray(o).ndim == 2)
    assert np.abs(proba - ref).max() < 1e-4
