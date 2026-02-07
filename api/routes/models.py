from fastapi import APIRouter, Depends, HTTPException

from ..schemas import (
    ModelInfo,
    ModelListResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    ErrorResponse,
)
from ..dependencies import get_model_manager
from ..model_manager import ModelManager


router = APIRouter()


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="Get list of all available models with their status.",
)
async def list_models(model_manager: ModelManager = Depends(get_model_manager)):
    models = model_manager.list_models()

    model_infos = [
        ModelInfo(
            name=m["name"],
            model_type=m["model_type"],
            version=m["version"],
            num_classes=m["num_classes"],
            input_dim=m.get("input_dim"),
            num_parameters=m.get("num_parameters"),
            is_loaded=m["is_loaded"],
            load_time_ms=m.get("load_time_ms"),
            last_used=m.get("last_used"),
            metadata=None,
        )
        for m in models
    ]

    return ModelListResponse(
        models=model_infos,
        default_model=model_manager.default_model or "",
        total_models=len(model_infos),
    )


@router.get(
    "/models/statistics",
    summary="Get model statistics",
    description="Get inference statistics for loaded models.",
)
async def get_statistics(model_manager: ModelManager = Depends(get_model_manager)):
    return model_manager.get_statistics()


@router.get(
    "/models/default",
    summary="Get default model",
    description="Get the current default model.",
)
async def get_default_model(model_manager: ModelManager = Depends(get_model_manager)):
    return {
        "default_model": model_manager.default_model,
        "is_loaded": model_manager.is_model_loaded(model_manager.default_model)
        if model_manager.default_model
        else False,
    }


@router.get(
    "/models/{model_name}",
    response_model=ModelInfo,
    responses={404: {"model": ErrorResponse}},
    summary="Get model info",
    description="Get detailed information about a specific model.",
)
async def get_model_info(
    model_name: str, model_manager: ModelManager = Depends(get_model_manager)
):
    info = model_manager.get_model_info(model_name)

    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return ModelInfo(
        name=model_name,
        model_type=info.get("type", "unknown"),
        version=info.get("version", "1.0.0"),
        num_classes=info.get("num_classes", 3),
        input_dim=info.get("input_dim"),
        num_parameters=info.get("num_parameters"),
        is_loaded=info.get("is_loaded", False),
        load_time_ms=info.get("load_time_ms"),
        last_used=info.get("last_used"),
        metadata=info,
    )


@router.post(
    "/models/load",
    response_model=ModelLoadResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Load a model",
    description="Load a model into memory for inference.",
)
async def load_model(
    request: ModelLoadRequest, model_manager: ModelManager = Depends(get_model_manager)
):
    if request.model_path:
        if ".." in request.model_path or request.model_path.startswith("/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid model path: absolute paths and '..' are not allowed",
            )

        try:
            models_dir = model_manager.models_dir.resolve()
            requested_path = (models_dir / request.model_path).resolve()
            if not str(requested_path).startswith(str(models_dir)):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid model path: must be within models directory",
                )
        except (OSError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid model path")

    success, load_time = model_manager.load_model(
        model_name=request.model_name,
        model_path=request.model_path,
        force_reload=request.force_reload,
    )

    if not success:
        raise HTTPException(
            status_code=500, detail=f"Failed to load model '{request.model_name}'"
        )

    return ModelLoadResponse(
        success=True,
        model_name=request.model_name,
        message=f"Model loaded successfully in {load_time:.2f}ms",
        load_time_ms=load_time,
    )


@router.post(
    "/models/{model_name}/unload",
    response_model=ModelLoadResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Unload a model",
    description="Unload a model from memory to free resources.",
)
async def unload_model(
    model_name: str, model_manager: ModelManager = Depends(get_model_manager)
):
    if not model_manager.is_model_loaded(model_name):
        raise HTTPException(
            status_code=404, detail=f"Model '{model_name}' is not loaded"
        )

    success = model_manager.unload_model(model_name)

    return ModelLoadResponse(
        success=success,
        model_name=model_name,
        message="Model unloaded successfully" if success else "Failed to unload model",
        load_time_ms=0.0,
    )


@router.post(
    "/models/default/{model_name}",
    summary="Set default model",
    description="Set the default model for predictions.",
)
async def set_default_model(
    model_name: str, model_manager: ModelManager = Depends(get_model_manager)
):
    success = model_manager.set_default_model(model_name)

    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return {
        "success": True,
        "default_model": model_name,
        "message": f"Default model set to '{model_name}'",
    }


@router.get(
    "/models/{model_name}/features",
    summary="Get model feature names",
    description="Get the expected feature names for a model.",
)
async def get_feature_names(
    model_name: str, model_manager: ModelManager = Depends(get_model_manager)
):
    feature_names = model_manager.get_feature_names(model_name)

    if feature_names is None:
        return {
            "model_name": model_name,
            "feature_names": None,
            "message": "Feature names not available for this model",
        }

    return {
        "model_name": model_name,
        "feature_names": feature_names,
        "num_features": len(feature_names),
    }
