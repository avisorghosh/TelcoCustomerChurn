"""Inference service logic for single-customer predictions."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from sklearn.pipeline import Pipeline

from churn_prediction.api.schemas import CustomerRecord, PredictionResponse
from churn_prediction.models.serialization import load_artifacts

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Exception raised when inference is attempted on an unloaded model."""

    pass


class ModelCorruptedError(RuntimeError):
    """Exception raised when a loaded model artifact is corrupted or invalid."""

    pass


class InferenceService:
    """Service class for managing model lifecycle and performing predictions."""

    def __init__(
        self,
        model_dir: str | Path = "models",
        pipeline_filename: str = "baseline_pipeline.joblib",
        metadata_filename: str = "baseline_metadata.json",
        decision_threshold: float = 0.50,
    ) -> None:
        """Initialize InferenceService instance.

        Args:
            model_dir: Path to directory containing serialized model artifacts.
            pipeline_filename: Filename for the pipeline joblib artifact.
            metadata_filename: Filename for the metadata JSON artifact.
            decision_threshold: Classification probability threshold.
        """
        self.model_dir = Path(model_dir)
        self.pipeline_filename = pipeline_filename
        self.metadata_filename = metadata_filename
        self.decision_threshold = float(decision_threshold)

        self.pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] = {}
        self.is_loaded: bool = False
        self.load_error: str | None = None

    def load_model(self) -> bool:
        """Load trained pipeline and metadata from configured artifact directory.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            pipeline, metadata = load_artifacts(
                output_dir=self.model_dir,
                pipeline_filename=self.pipeline_filename,
                metadata_filename=self.metadata_filename,
            )

            # Validate pipeline structure
            if not hasattr(pipeline, "predict_proba"):
                raise ModelCorruptedError(
                    "Loaded pipeline object is missing 'predict_proba' method."
                )

            self.pipeline = pipeline
            self.metadata = metadata
            self.is_loaded = True
            self.load_error = None
            logger.info("Successfully loaded model artifact from '%s'", self.model_dir)
            return True
        except FileNotFoundError as e:
            self.is_loaded = False
            self.load_error = f"Model artifact missing: {e}"
            logger.warning(self.load_error)
            return False
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load model artifact: {e}"
            logger.error("Error loading model artifact: %s", e)
            return False

    @property
    def model_version(self) -> str:
        """Return model version string from loaded metadata or environment override."""
        import os

        env_version = os.getenv("CHURN_MODEL_VERSION")
        if env_version:
            return env_version
        if self.metadata and "schema_version" in self.metadata:
            return str(self.metadata["schema_version"])
        if self.metadata and "model_name" in self.metadata:
            return str(self.metadata["model_name"])
        return "1.0.0"

    def predict(
        self,
        record: CustomerRecord,
        correlation_id: str | None = None,
    ) -> PredictionResponse:
        """Execute single-customer inference.

        Args:
            record: Validated CustomerRecord request payload.
            correlation_id: Optional correlation ID for request tracing.

        Returns:
            Structured PredictionResponse object.

        Raises:
            ModelNotLoadedError: If model is not loaded.
            RuntimeError: If unexpected error occurs during prediction.
        """
        if not self.is_loaded or self.pipeline is None:
            raise ModelNotLoadedError("Model artifact is not loaded or unavailable.")

        cid = correlation_id or str(uuid4())
        record_dict = record.model_dump()

        try:
            df = pd.DataFrame([record_dict])
            probas = self.pipeline.predict_proba(df)[:, 1]
            churn_probability = float(probas[0])
            predicted_class = int(churn_probability >= self.decision_threshold)
            timestamp = datetime.now(timezone.utc).isoformat()

            return PredictionResponse(
                churn_probability=round(churn_probability, 4),
                predicted_class=predicted_class,
                model_version=self.model_version,
                correlation_id=cid,
                prediction_timestamp=timestamp,
            )
        except Exception as e:
            logger.error("Unexpected error during single-customer inference: %s", e)
            raise RuntimeError("Inference execution failed safely.") from e
