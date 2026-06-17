"""Request and response models for the API."""

from typing import Dict, List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    # Bounded to reject malformed payloads before inference. Unknown names are
    # ignored and missing ones imputed by the pipeline, so the cap only guards
    # against oversized requests, not feature identity (there are 58 real ones).
    features: Dict[str, float] = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Feature name -> value (e.g. HRV_RMSSD, EDA_SCR_count)",
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]


class Contribution(BaseModel):
    feature: str
    value: float
    shap: float


class ExplanationResponse(BaseModel):
    prediction: str
    contributions: List[Contribution]


class ModelInfo(BaseModel):
    classes: List[str]
    n_features: int
    features: List[str]


class Health(BaseModel):
    status: str
    model_loaded: bool
