"""Request and response models for the API."""

from typing import Dict, List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Feature name -> value (e.g. HRV_RMSSD, EDA_SCR_count)"
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
