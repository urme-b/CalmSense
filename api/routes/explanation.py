import asyncio
import time
import base64
import io

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..schemas import (
    ExplanationRequest,
    ExplanationResponse,
    ExplanationType,
    FeatureImportance,
    ClinicalInterpretationRequest,
    ClinicalInterpretationResponse,
    ErrorResponse,
)
from ..dependencies import get_model_manager
from ..model_manager import ModelManager


router = APIRouter()


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Get model explanation",
    description="Get feature importance explanation for a prediction.",
)
async def explain_prediction(
    request: ExplanationRequest,
    model_manager: ModelManager = Depends(get_model_manager),
):
    try:
        start_time = time.time()

        features = np.array(request.features.values)
        feature_names = request.features.feature_names or [
            f"feature_{i}" for i in range(len(features))
        ]

        prediction_result = model_manager.predict(
            model_name=request.model_name, features=features, return_probabilities=True
        )

        model_name = prediction_result["model_used"]
        model = model_manager.models.get(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not loaded")

        feature_importances = []
        visualization_base64 = None
        clinical_interpretation = None

        if request.explanation_type == ExplanationType.SHAP:
            feature_importances, visualization_base64 = await _get_shap_explanation(
                model,
                features,
                feature_names,
                request.num_features,
                request.include_visualization,
                model_manager,
            )

        elif request.explanation_type == ExplanationType.LIME:
            feature_importances, visualization_base64 = await _get_lime_explanation(
                model,
                features,
                feature_names,
                request.num_features,
                request.include_visualization,
                model_manager,
            )

        elif request.explanation_type == ExplanationType.GRADCAM:
            feature_importances, visualization_base64 = await _get_gradcam_explanation(
                model,
                features,
                feature_names,
                request.num_features,
                request.include_visualization,
            )

        elif request.explanation_type == ExplanationType.ATTENTION:
            (
                feature_importances,
                visualization_base64,
            ) = await _get_attention_explanation(
                model,
                features,
                feature_names,
                request.num_features,
                request.include_visualization,
            )

        elif request.explanation_type == ExplanationType.CLINICAL:
            (
                feature_importances,
                clinical_interpretation,
            ) = await _get_clinical_explanation(
                features,
                feature_names,
                prediction_result["prediction"],
                prediction_result["probabilities"],
            )

        computation_time = (time.time() - start_time) * 1000

        return ExplanationResponse(
            prediction=prediction_result["prediction"],
            class_name=prediction_result["class_name"],
            probabilities=prediction_result["probabilities"],
            explanation_type=request.explanation_type.value,
            feature_importances=feature_importances,
            clinical_interpretation=clinical_interpretation,
            visualization_base64=visualization_base64,
            model_used=model_name,
            computation_time_ms=computation_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _get_shap_explanation(
    model, features, feature_names, num_features, include_viz, model_manager
):
    try:
        from src.explainability import SHAPExplainer

        def _compute():
            model_type = "kernel"
            if hasattr(model, "get_booster"):
                model_type = "tree"
            elif hasattr(model, "feature_importances_"):
                model_type = "tree"
            elif hasattr(model, "forward"):
                model_type = "deep"

            background_data = getattr(model_manager, "training_data", None)
            if background_data is None:
                background_data = np.random.randn(100, len(features))

            explainer = SHAPExplainer(
                model=model, model_type=model_type, background_data=background_data
            )

            shap_values = explainer.compute_shap_values(
                features.reshape(1, -1), feature_names=feature_names
            )

            top_features = explainer.get_top_features(
                shap_values["shap_values"], feature_names, n=num_features
            )

            importances = [
                FeatureImportance(
                    feature_name=row["feature"],
                    importance=row["mean_abs_shap"],
                    direction="positive" if row["mean_shap"] > 0 else "negative",
                )
                for _, row in top_features.iterrows()
            ]

            viz = None
            if include_viz:
                fig = explainer.plot_summary(
                    shap_values["shap_values"], feature_names, max_display=num_features
                )
                viz = fig_to_base64(fig)
                import matplotlib.pyplot as plt

                plt.close(fig)

            return importances, viz

        return await asyncio.to_thread(_compute)

    except ImportError:
        return _get_basic_feature_importance(model, feature_names, num_features), None


async def _get_lime_explanation(
    model, features, feature_names, num_features, include_viz, model_manager
):
    try:
        from src.explainability import LIMEExplainer

        def _compute():
            training_data = getattr(model_manager, "training_data", None)
            if training_data is None:
                training_data = np.random.randn(100, len(features))

            explainer = LIMEExplainer(
                model=model,
                feature_names=feature_names,
                class_names=["Baseline", "Stress", "Amusement"],
                training_data=training_data,
            )

            explanation = explainer.explain_instance(
                features, num_features=num_features
            )

            importances = [
                FeatureImportance(
                    feature_name=feat,
                    importance=abs(weight),
                    direction="positive" if weight > 0 else "negative",
                )
                for feat, weight in explanation["feature_importance"].items()
            ]

            viz = None
            if include_viz:
                fig = explainer.plot_explanation(explanation)
                viz = fig_to_base64(fig)
                import matplotlib.pyplot as plt

                plt.close(fig)

            return importances, viz

        return await asyncio.to_thread(_compute)

    except ImportError:
        return _get_basic_feature_importance(model, feature_names, num_features), None


async def _get_gradcam_explanation(
    model, features, feature_names, num_features, include_viz
):
    try:
        import torch  # noqa: F401
        from src.explainability import GradCAMExplainer

        if not hasattr(model, "forward"):
            raise ValueError("Grad-CAM requires a PyTorch model")

        explainer = GradCAMExplainer(model)
        heatmap = explainer.generate_heatmap(features)

        if len(heatmap) == len(feature_names):
            feature_importances = [
                FeatureImportance(
                    feature_name=name, importance=float(val), direction="positive"
                )
                for name, val in sorted(
                    zip(feature_names, heatmap), key=lambda x: x[1], reverse=True
                )[:num_features]
            ]
        else:
            feature_importances = []

        visualization_base64 = None
        if include_viz:
            fig = explainer.plot_heatmap(features, heatmap)
            visualization_base64 = fig_to_base64(fig)
            import matplotlib.pyplot as plt

            plt.close(fig)

        return feature_importances, visualization_base64

    except ImportError:
        return [], None


async def _get_attention_explanation(
    model, features, feature_names, num_features, include_viz
):
    try:
        import torch  # noqa: F401
        from src.explainability import AttentionVisualizer

        if not hasattr(model, "forward"):
            raise ValueError("Attention visualization requires a PyTorch model")

        visualizer = AttentionVisualizer(model)
        attention_weights = visualizer.get_attention_weights(features)

        feature_importances = []
        for layer_name, weights in attention_weights.items():
            if weights.ndim >= 2:
                avg_attention = weights.mean(axis=tuple(range(weights.ndim - 1)))
                if len(avg_attention) == len(feature_names):
                    for name, val in sorted(
                        zip(feature_names, avg_attention),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:num_features]:
                        feature_importances.append(
                            FeatureImportance(
                                feature_name=name,
                                importance=float(val),
                                direction="positive",
                            )
                        )
                    break

        visualization_base64 = None
        if include_viz and attention_weights:
            first_layer = list(attention_weights.values())[0]
            fig = visualizer.plot_attention_heatmap(first_layer)
            visualization_base64 = fig_to_base64(fig)
            import matplotlib.pyplot as plt

            plt.close(fig)

        return feature_importances, visualization_base64

    except ImportError:
        return [], None


async def _get_clinical_explanation(features, feature_names, prediction, probabilities):
    try:
        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter(
            class_names=["Baseline", "Stress", "Amusement"]
        )

        feature_values = dict(zip(feature_names, features))
        proba_list = [
            probabilities.get(c, 0) for c in ["Baseline", "Stress", "Amusement"]
        ]

        report = interpreter.interpret_prediction(
            prediction=prediction,
            probabilities=proba_list,
            feature_values=feature_values,
        )

        feature_importances = [
            FeatureImportance(
                feature_name=f["feature"],
                importance=f["confidence"],
                direction="positive"
                if "stress" in f["stress_implication"].lower()
                else "negative",
            )
            for f in report["findings"]
            if f["deviation"] != "normal"
        ]

        return feature_importances, report

    except ImportError:
        return [], None


def _get_basic_feature_importance(model, feature_names, num_features):
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = (
            np.abs(model.coef_).mean(axis=0)
            if model.coef_.ndim > 1
            else np.abs(model.coef_)
        )

    if importances is None:
        return []

    sorted_idx = np.argsort(importances)[::-1][:num_features]

    return [
        FeatureImportance(
            feature_name=feature_names[i] if i < len(feature_names) else f"feature_{i}",
            importance=float(importances[i]),
            direction="positive",
        )
        for i in sorted_idx
    ]


@router.post(
    "/explain/clinical",
    response_model=ClinicalInterpretationResponse,
    summary="Get clinical interpretation",
    description="Get physiologically-grounded interpretation of stress prediction.",
)
async def get_clinical_interpretation(request: ClinicalInterpretationRequest):
    try:
        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter(
            class_names=["Baseline", "Stress", "Amusement"]
        )

        feature_values = dict(
            zip(
                request.features.feature_names
                or [f"feature_{i}" for i in range(len(request.features.values))],
                request.features.values,
            )
        )

        report = interpreter.interpret_prediction(
            prediction=request.prediction,
            probabilities=request.probabilities,
            feature_values=feature_values,
        )

        return ClinicalInterpretationResponse(
            stress_level=report["stress_assessment"]["stress_level"],
            stress_score=report["stress_assessment"]["stress_score"],
            findings=report["findings"],
            summary=report["summary"],
            recommendations=report["recommendations"]
            if request.include_recommendations
            else [],
            disclaimer=report["disclaimer"],
        )

    except ImportError:
        raise HTTPException(
            status_code=501, detail="Clinical interpreter not available"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/explain/types",
    summary="List explanation types",
    description="Get list of supported explanation types.",
)
async def list_explanation_types():
    return {
        "types": [
            {
                "type": "shap",
                "name": "SHAP",
                "description": "SHapley Additive exPlanations - Game theory-based feature importance",
                "supports_ml": True,
                "supports_dl": True,
            },
            {
                "type": "lime",
                "name": "LIME",
                "description": "Local Interpretable Model-agnostic Explanations",
                "supports_ml": True,
                "supports_dl": True,
            },
            {
                "type": "gradcam",
                "name": "Grad-CAM",
                "description": "Gradient-weighted Class Activation Mapping for CNNs",
                "supports_ml": False,
                "supports_dl": True,
            },
            {
                "type": "attention",
                "name": "Attention",
                "description": "Attention weight visualization for transformers",
                "supports_ml": False,
                "supports_dl": True,
            },
            {
                "type": "clinical",
                "name": "Clinical",
                "description": "Physiologically-grounded clinical interpretation",
                "supports_ml": True,
                "supports_dl": True,
            },
        ]
    }
