"""API serving package for customer churn prediction."""

from churn_prediction.api.app import create_app
from churn_prediction.api.config import load_serving_config
from churn_prediction.api.schemas import (
    CustomerRecord,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    ReadinessResponse,
)
from churn_prediction.api.service import InferenceService, ModelNotLoadedError

__all__ = [
    "CustomerRecord",
    "ErrorResponse",
    "HealthResponse",
    "InferenceService",
    "ModelNotLoadedError",
    "PredictionResponse",
    "ReadinessResponse",
    "create_app",
    "load_serving_config",
]
