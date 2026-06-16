"""Assemble every result file into the single JSON the dashboard reads."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
FRONTEND = PROJECT_ROOT / "frontend" / "src" / "results.json"

# Keys the dashboard consumes per task (per-subject lists stay out of the bundle)
TASK_KEYS = [
    "n_windows",
    "n_features",
    "classes",
    "models",
    "best_model",
    "loso_accuracy",
    "loso_pooled_accuracy",
    "within_subject_accuracy",
    "optimism_gap_pts",
]


def _load_json(name):
    path = RESULTS_DIR / name
    return json.load(open(path)) if path.exists() else None


def _load_csv(name):
    path = RESULTS_DIR / name
    return pd.read_csv(path).to_dict("records") if path.exists() else None


def run():
    metrics = _load_json("metrics.json")
    if metrics is None:
        raise SystemExit("results/metrics.json missing. Run scripts/run_experiment.py first.")

    out = {}
    for task in ("binary", "multiclass"):
        if task in metrics:
            out[task] = {k: metrics[task].get(k) for k in TASK_KEYS}

    shap = _load_csv("shap_top_features.csv")
    if shap:
        out["shap"] = shap[:12]
    for key, fname in [
        ("stats", "stats.json"),
        ("wrist", "wrist.json"),
        ("cross_dataset", "cross_dataset.json"),
        ("calibration", "calibration.json"),
        ("personalization", "personalization.json"),
        ("tuning", "tuning.json"),
    ]:
        data = _load_json(fname)
        if data is not None:
            out[key] = data
    ablation = _load_csv("ablation.csv")
    if ablation:
        out["ablation"] = ablation

    if not FRONTEND.parent.exists():
        raise SystemExit(f"{FRONTEND.parent} missing")
    json.dump(out, open(FRONTEND, "w"), indent=2)
    print(f"Wrote {FRONTEND} with sections: {sorted(out)}")


if __name__ == "__main__":
    run()
