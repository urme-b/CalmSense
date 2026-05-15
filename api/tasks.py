from typing import Any, Dict, List

from celery import Celery

from .settings import settings

celery_app = Celery(
    "calmsense",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
)


@celery_app.task(name="calmsense.shap_explanation")
def compute_shap_explanation(
    features: List[float],
    model_name: str,
    feature_names: List[str],
    num_features: int,
) -> Dict[str, Any]:
    import numpy as np
    from api.dependencies import app_state
    from src.explainability import SHAPExplainer

    mm = app_state.model_manager
    model = mm.models.get(model_name)
    if model is None:
        return {"error": f"Model {model_name} not loaded"}

    arr = np.array(features)
    model_type = "tree" if hasattr(model, "feature_importances_") else "kernel"
    background = np.random.randn(100, len(arr))

    explainer = SHAPExplainer(
        model=model, model_type=model_type, background_data=background
    )
    shap_values = explainer.compute_shap_values(
        arr.reshape(1, -1), feature_names=feature_names
    )
    top = explainer.get_top_features(
        shap_values["shap_values"], feature_names, n=num_features
    )

    return {
        "importances": [
            {
                "feature_name": row["feature"],
                "importance": float(row["mean_abs_shap"]),
                "direction": "positive" if row["mean_shap"] > 0 else "negative",
            }
            for _, row in top.iterrows()
        ]
    }


@celery_app.task(name="calmsense.batch_predict")
def batch_predict_task(
    samples: List[List[float]], model_name: str
) -> List[Dict[str, Any]]:
    import numpy as np
    from api.dependencies import app_state

    mm = app_state.model_manager
    features = np.array(samples)
    return mm.predict_batch(model_name=model_name, features=features)
