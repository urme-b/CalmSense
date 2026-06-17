"""Export the trained Random Forest to ONNX so the dashboard runs it in-browser."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.config import MODELS_DIR, PROJECT_ROOT

FRONTEND = PROJECT_ROOT / "frontend"


def run():
    bundle = joblib.load(MODELS_DIR / "stress_classifier.joblib")
    pipe, features, classes = bundle["pipeline"], bundle["features"], bundle["classes"]
    imputer, scaler, rf = (
        pipe.named_steps["impute"],
        pipe.named_steps["scale"],
        pipe.named_steps["clf"],
    )
    n = len(features)

    onx = convert_sklearn(
        rf,
        initial_types=[("input", FloatTensorType([None, n]))],
        options={id(rf): {"zipmap": False}},
        target_opset=18,
    )
    (FRONTEND / "public").mkdir(parents=True, exist_ok=True)
    (FRONTEND / "public" / "model.onnx").write_bytes(onx.SerializeToString())

    meta = {
        "features": features,
        "medians": imputer.statistics_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "classes": classes,
    }
    with open(FRONTEND / "src" / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Verify ONNX matches the sklearn pipeline
    rng = np.random.RandomState(0)
    X = rng.randn(50, n)
    X[rng.rand(*X.shape) < 0.3] = np.nan  # exercise the imputation path
    ref = pipe.predict_proba(X)

    medians = np.array(meta["medians"])
    mean = np.array(meta["mean"])
    scale = np.array(meta["scale"])
    Xf = np.where(np.isnan(X), medians, X)
    Xs = ((Xf - mean) / scale).astype(np.float32)
    sess = ort.InferenceSession((FRONTEND / "public" / "model.onnx").read_bytes())
    out = sess.run(None, {"input": Xs})
    proba = out[1]

    max_err = float(np.abs(np.asarray(proba) - ref).max())
    print(
        f"Exported model.onnx ({(FRONTEND / 'public' / 'model.onnx').stat().st_size // 1024} KB), "
        f"{n} features, classes {classes}"
    )
    print(f"ONNX vs sklearn max probability error: {max_err:.6f}")
    assert max_err < 1e-4, "ONNX output diverges from the sklearn pipeline"
    print("Match OK — the browser will reproduce the trained model exactly.")


if __name__ == "__main__":
    run()
