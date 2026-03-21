import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from ..schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureVector,
    TimeSeriesData,
    UncertaintyInfo,
    ErrorResponse,
)
from ..dependencies import get_model_manager, get_feature_store
from ..model_manager import ModelManager
from ..feature_schema import FeatureSchemaStore
from ..auth import require_auth, TokenPayload


router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Make stress prediction",
    description="Predict stress level from physiological features or time series data.",
)
async def predict(
    request: PredictionRequest,
    user: TokenPayload = Depends(require_auth),
    model_manager: ModelManager = Depends(get_model_manager),
):
    try:
        if isinstance(request.features, FeatureVector):
            features = np.array(request.features.values)
        elif isinstance(request.features, TimeSeriesData):
            features = np.array(request.features.signal)
        else:
            raise ValueError("Invalid feature format")

        result = model_manager.predict(
            model_name=request.model_name,
            features=features,
            return_probabilities=request.return_probabilities,
        )

        explanation = None
        if request.include_explanation:
            feature_names = None
            if isinstance(request.features, FeatureVector):
                feature_names = request.features.feature_names

            if feature_names:
                model = model_manager.models.get(result["model_used"])
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    explanation = {
                        name: float(imp)
                        for name, imp in zip(feature_names, importances)
                    }

        return PredictionResponse(
            prediction=result["prediction"],
            class_name=result["class_name"],
            probabilities=result.get("probabilities"),
            confidence=result["confidence"],
            model_used=result["model_used"],
            inference_time_ms=result["inference_time_ms"],
            explanation=explanation,
            timestamp=datetime.now(timezone.utc),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Make batch predictions",
    description="Predict stress levels for multiple samples.",
)
async def predict_batch(
    request: BatchPredictionRequest,
    user: TokenPayload = Depends(require_auth),
    model_manager: ModelManager = Depends(get_model_manager),
):
    try:
        start_time = time.time()

        feature_matrix = np.array([s.values for s in request.samples])
        results = model_manager.predict_batch(
            model_name=request.model_name,
            features=feature_matrix,
            return_probabilities=request.return_probabilities,
        )

        predictions = [
            PredictionResponse(
                prediction=r["prediction"],
                class_name=r["class_name"],
                probabilities=r.get("probabilities"),
                confidence=r["confidence"],
                model_used=r["model_used"],
                inference_time_ms=r["inference_time_ms"],
                timestamp=datetime.now(timezone.utc),
            )
            for r in results
        ]

        total_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions),
            total_inference_time_ms=total_time,
            model_used=predictions[0].model_used if predictions else "",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from ..settings import settings as api_settings

MAX_CSV_SIZE = api_settings.csv_max_size
MAX_CSV_ROWS = api_settings.csv_max_rows


@router.post(
    "/predict/csv",
    response_model=BatchPredictionResponse,
    summary="Predict from CSV file",
    description="Upload a CSV file with features for batch prediction.",
)
async def predict_from_csv(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    user: TokenPayload = Depends(require_auth),
    model_manager: ModelManager = Depends(get_model_manager),
    feature_store: FeatureSchemaStore = Depends(get_feature_store),
):
    try:
        import pandas as pd
        import io

        if file.content_type and file.content_type not in [
            "text/csv",
            "application/csv",
            "text/plain",
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected CSV.",
            )

        # Chunked read
        chunks = []
        total_size = 0
        while chunk := await file.read(64 * 1024):
            total_size += len(chunk)
            if total_size > MAX_CSV_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size is {MAX_CSV_SIZE // (1024 * 1024)}MB",
                )
            chunks.append(chunk)
        content = b"".join(chunks)

        try:
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File encoding error. Please use UTF-8 encoded CSV.",
            )

        if len(df) > MAX_CSV_ROWS:
            raise HTTPException(
                status_code=400, detail=f"Too many rows. Maximum is {MAX_CSV_ROWS}"
            )

        features = df.values.astype(np.float64)

        # Validate feature dimensions
        expected = model_manager.get_feature_names(model_name)
        if expected and features.shape[1] != len(expected):
            raise HTTPException(
                status_code=400,
                detail=f"Feature mismatch: CSV has {features.shape[1]} columns, model expects {len(expected)}",
            )

        # Validate feature ranges
        target = model_name or model_manager.default_model
        if target:
            errors = feature_store.validate(target, features)
            if errors:
                raise HTTPException(status_code=400, detail=f"Feature validation: {'; '.join(errors)}")

        start_time = time.time()
        results = model_manager.predict_batch(
            model_name=model_name,
            features=features,
            return_probabilities=True,
        )

        predictions = [
            PredictionResponse(
                prediction=r["prediction"],
                class_name=r["class_name"],
                probabilities=r.get("probabilities"),
                confidence=r["confidence"],
                model_used=r["model_used"],
                inference_time_ms=r["inference_time_ms"],
                timestamp=datetime.now(timezone.utc),
            )
            for r in results
        ]

        total_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions),
            total_inference_time_ms=total_time,
            model_used=predictions[0].model_used if predictions else "",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict/ensemble",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Ensemble prediction with uncertainty",
)
async def predict_ensemble(
    request: PredictionRequest,
    user: TokenPayload = Depends(require_auth),
    model_manager: ModelManager = Depends(get_model_manager),
):
    try:
        if isinstance(request.features, FeatureVector):
            features = np.array(request.features.values)
        elif isinstance(request.features, TimeSeriesData):
            features = np.array(request.features.signal)
        else:
            raise ValueError("Invalid feature format")

        result = model_manager.predict_with_uncertainty(
            features=features, return_probabilities=request.return_probabilities
        )

        return PredictionResponse(
            prediction=result["prediction"],
            class_name=result["class_name"],
            probabilities=result.get("probabilities"),
            confidence=result["confidence"],
            model_used=", ".join(result["models_used"]),
            inference_time_ms=result["inference_time_ms"],
            uncertainty=UncertaintyInfo(**result["uncertainty"]),
            timestamp=datetime.now(timezone.utc),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/predict/classes",
    summary="Get prediction classes",
    description="Get the available prediction class names.",
)
async def get_classes(model_manager: ModelManager = Depends(get_model_manager)):
    return {
        "classes": model_manager.CLASS_NAMES,
        "num_classes": len(model_manager.CLASS_NAMES),
    }
