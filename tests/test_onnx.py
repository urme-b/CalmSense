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


def test_onnx_and_joblib_agree_on_fixed_vectors():
    """Browser ONNX and the Python joblib pipeline agree on BOTH probabilities and the
    predicted label for fixed, human-meaningful inputs (the label is what a visitor sees).
    """
    import joblib
    import onnxruntime as ort

    bundle = joblib.load(MODEL)
    pipe, features = bundle["pipeline"], bundle["features"]
    meta = json.load(open(META))
    medians, mean, scale = (np.array(meta[k]) for k in ("medians", "mean", "scale"))

    baseline = dict(zip(features, medians.tolist()))  # median profile
    stress = {
        **baseline,
        "HRV_MeanNN": 680.0,
        "HRV_MedianNN": 675.0,
        "HRV_RMSSD": 22.0,
        "EDA_SCR_count": 9.0,
        "EDA_SCR_rate": 0.12,
        "EDA_SCL_mean": 6.5,
        "RESP_rate": 20.0,
        "ACC_std": 0.09,
    }
    sparse = {"HRV_MeanNN": 690.0, "EDA_SCR_count": 8.0, "RESP_rate": float("inf")}

    sess = ort.InferenceSession(ONNX.read_bytes())
    for name, feats in [("baseline", baseline), ("stress", stress), ("sparse", sparse)]:
        row = np.array([feats.get(f, np.nan) for f in features], dtype=float)

        # browser services/onnx.ts vectorize(): non-finite -> median, then standardize
        filled = np.where(np.isfinite(row), row, medians)
        standardized = ((filled - mean) / scale).astype(np.float32).reshape(1, -1)
        onnx_proba = next(
            np.asarray(o)
            for o in sess.run(None, {"input": standardized})
            if np.asarray(o).ndim == 2
        )[0]

        # joblib pipeline: non-finite -> NaN -> median imputer (equivalent by construction)
        jl_proba = pipe.predict_proba(np.where(np.isfinite(row), row, np.nan).reshape(1, -1))[0]

        assert np.abs(onnx_proba - jl_proba).max() < 1e-4, f"{name}: probabilities diverge"
        assert int(np.argmax(onnx_proba)) == int(np.argmax(jl_proba)), f"{name}: label diverges"
