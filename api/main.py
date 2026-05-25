"""CalmSense prediction API."""

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
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
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
    return PredictionResponse(**model.predict(request.features))


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: PredictionRequest) -> ExplanationResponse:
    model = _require_model()
    result = model.predict(request.features)
    return ExplanationResponse(
        prediction=result["prediction"],
        contributions=model.explain(request.features),
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
