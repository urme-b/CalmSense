"""CalmSense prediction API."""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import MODEL_PATH, get_model
from .schemas import (
    ExplanationResponse,
    Health,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="CalmSense API",
    description="Subject-independent stress detection from wearable physiology.",
    version="0.1.0",
)

# Restrict CORS to the dashboard origins; override with CALMSENSE_CORS_ORIGINS (comma-separated)
_default_origins = "http://localhost:3000,https://urme-b.github.io"
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("CALMSENSE_CORS_ORIGINS", _default_origins).split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_model():
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}. Run scripts/run_experiment.py first.",
        )
    return model


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", model_loaded=MODEL_PATH.exists())


@app.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    model = _require_model()
    return ModelInfo(classes=model.classes, n_features=len(model.features), features=model.features)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = _require_model()
    try:
        return PredictionResponse(**model.predict(request.features))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: PredictionRequest) -> ExplanationResponse:
    model = _require_model()
    try:
        result, contributions = model.predict_and_explain(request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ExplanationResponse(prediction=result["prediction"], contributions=contributions)


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
