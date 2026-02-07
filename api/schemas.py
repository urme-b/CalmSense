import math
import re
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    CNN1D = "cnn1d"
    BILSTM = "bilstm"
    CNN_LSTM = "cnn_lstm"
    TRANSFORMER = "transformer"
    CROSS_MODAL = "cross_modal"


class ExplanationType(str, Enum):
    SHAP = "shap"
    LIME = "lime"
    GRADCAM = "gradcam"
    ATTENTION = "attention"
    CLINICAL = "clinical"


class StressClass(str, Enum):
    BASELINE = "baseline"
    STRESS = "stress"
    AMUSEMENT = "amusement"


MAX_FEATURE_VALUES = 10000
MAX_FEATURE_NAME_LENGTH = 256
MAX_SUBJECT_ID_LENGTH = 128


class FeatureVector(BaseModel):
    values: List[float] = Field(
        ..., description="Feature values", max_length=MAX_FEATURE_VALUES
    )
    feature_names: Optional[List[str]] = Field(
        None, description="Feature names", max_length=MAX_FEATURE_VALUES
    )
    timestamp: Optional[datetime] = Field(None, description="Timestamp of measurement")
    subject_id: Optional[str] = Field(
        None, description="Subject identifier", max_length=MAX_SUBJECT_ID_LENGTH
    )

    @field_validator("values")
    @classmethod
    def validate_values(cls, v):
        if len(v) == 0:
            raise ValueError("Feature values cannot be empty")
        if len(v) > MAX_FEATURE_VALUES:
            raise ValueError(
                f"Feature values cannot exceed {MAX_FEATURE_VALUES} elements"
            )
        for i, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Feature value at index {i} is NaN or Inf")
            if abs(val) > 1e15:
                raise ValueError(
                    f"Feature value at index {i} exceeds safe range (|val| > 1e15)"
                )
        return v

    @field_validator("feature_names")
    @classmethod
    def validate_feature_names(cls, v):
        if v is not None:
            for name in v:
                if len(name) > MAX_FEATURE_NAME_LENGTH:
                    raise ValueError(
                        f"Feature name exceeds max length ({MAX_FEATURE_NAME_LENGTH})"
                    )
        return v


class TimeSeriesData(BaseModel):
    signal: List[List[float]] = Field(
        ..., description="Signal data [channels, timesteps]"
    )
    sampling_rate: Optional[float] = Field(None, description="Sampling rate in Hz")
    channel_names: Optional[List[str]] = Field(None, description="Channel names")


class PredictionRequest(BaseModel):
    features: Union[FeatureVector, TimeSeriesData] = Field(
        ..., description="Input features or time series"
    )
    model_name: Optional[str] = Field(None, description="Specific model to use")
    return_probabilities: bool = Field(True, description="Return class probabilities")
    include_explanation: bool = Field(False, description="Include basic explanation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "values": [65.0, 0.05, 45.0, 30.0, 0.8, 3.5, 15.0],
                    "feature_names": [
                        "hr_mean",
                        "hrv_rmssd",
                        "hrv_sdnn",
                        "hrv_pnn50",
                        "lf_hf_ratio",
                        "eda_mean",
                        "resp_rate",
                    ],
                },
                "model_name": "random_forest",
                "return_probabilities": True,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class index")
    class_name: str = Field(..., description="Predicted class name")
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Class probabilities"
    )
    confidence: float = Field(..., description="Prediction confidence")
    model_used: str = Field(..., description="Model that made prediction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    explanation: Optional[Dict[str, Any]] = Field(
        None, description="Basic feature importance"
    )
    timestamp: datetime = Field(default_factory=utc_now)


class BatchPredictionRequest(BaseModel):
    samples: List[FeatureVector] = Field(..., description="List of feature vectors")
    model_name: Optional[str] = Field(None, description="Model to use")
    return_probabilities: bool = Field(True)

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v):
        if len(v) == 0:
            raise ValueError("Samples list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 samples per batch")
        return v


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(
        ..., description="Predictions for each sample"
    )
    total_samples: int = Field(..., description="Number of samples processed")
    total_inference_time_ms: float = Field(..., description="Total inference time")
    model_used: str = Field(..., description="Model used for predictions")


class ExplanationRequest(BaseModel):
    features: FeatureVector = Field(..., description="Input features")
    explanation_type: ExplanationType = Field(
        ExplanationType.SHAP, description="Type of explanation"
    )
    model_name: Optional[str] = Field(None, description="Model to explain")
    num_features: int = Field(10, description="Number of top features")
    include_visualization: bool = Field(
        False, description="Include base64 encoded plots"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "values": [65.0, 0.05, 45.0, 30.0, 0.8, 3.5, 15.0],
                    "feature_names": [
                        "hr_mean",
                        "hrv_rmssd",
                        "hrv_sdnn",
                        "hrv_pnn50",
                        "lf_hf_ratio",
                        "eda_mean",
                        "resp_rate",
                    ],
                },
                "explanation_type": "shap",
                "num_features": 10,
            }
        }
    }


class FeatureImportance(BaseModel):
    feature_name: str
    importance: float
    direction: str = Field(..., description="positive or negative")


class ExplanationResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: Dict[str, float]
    explanation_type: str
    feature_importances: List[FeatureImportance]
    clinical_interpretation: Optional[Dict[str, Any]] = None
    visualization_base64: Optional[str] = None
    model_used: str
    computation_time_ms: float


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Model architecture type")
    version: str = Field(..., description="Model version")
    num_classes: int = Field(..., description="Number of output classes")
    input_dim: Optional[int] = Field(None, description="Input dimension")
    num_parameters: Optional[int] = Field(None, description="Parameter count")
    is_loaded: bool = Field(..., description="Whether model is in memory")
    load_time_ms: Optional[float] = Field(None, description="Time to load model")
    last_used: Optional[datetime] = Field(None, description="Last inference time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str
    total_models: int


class ModelLoadRequest(BaseModel):
    model_name: str = Field(
        ..., description="Model identifier", min_length=1, max_length=256
    )
    model_path: Optional[str] = Field(
        None, description="Path to model file", max_length=1024
    )
    force_reload: bool = Field(False, description="Force reload if already loaded")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if not re.match(r"^[\w\-\.]+$", v):
            raise ValueError(
                "Model name can only contain alphanumeric characters, underscores, hyphens, and dots"
            )
        return v

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v):
        if v is not None:
            if ".." in v:
                raise ValueError("Path traversal not allowed")
            if v.startswith("/") or v.startswith("\\"):
                raise ValueError("Absolute paths not allowed")
        return v


class ModelLoadResponse(BaseModel):
    success: bool
    model_name: str
    message: str
    load_time_ms: float


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of loaded models")
    gpu_available: bool = Field(..., description="GPU availability")
    uptime_seconds: float = Field(..., description="Service uptime")
    timestamp: datetime = Field(default_factory=utc_now)


class DetailedHealthResponse(HealthResponse):
    memory_usage_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    active_connections: int
    requests_per_minute: float


class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=utc_now)


class StreamingPredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Feature values")
    sequence_id: Optional[str] = Field(None, description="Sequence identifier")


class StreamingPredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    sequence_id: Optional[str]
    timestamp: datetime = Field(default_factory=utc_now)


class ClinicalInterpretationRequest(BaseModel):
    features: FeatureVector
    prediction: int
    probabilities: List[float]
    include_recommendations: bool = Field(True)


class ClinicalInterpretationResponse(BaseModel):
    stress_level: str
    stress_score: float
    findings: List[Dict[str, Any]]
    summary: str
    recommendations: List[str]
    disclaimer: str


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error info")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=utc_now)


class ValidationErrorResponse(BaseModel):
    error: str = "Validation Error"
    details: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=utc_now)
