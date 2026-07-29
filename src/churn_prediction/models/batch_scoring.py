"""Batch scoring pipeline module.

Provides data validation, model prediction, row integrity, determinism,
quarantine error handling, and output generation for batch scoring datasets.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from churn_prediction.data.contract import ValidationReport
from churn_prediction.data.validator import validate_data
from churn_prediction.models.serialization import load_artifacts
from churn_prediction.monitoring import (
    generate_quality_report,
    log_event,
    metrics_manager,
)

logger = logging.getLogger(__name__)


class BatchScoringError(Exception):
    """Base exception for batch scoring pipeline errors."""


class BatchValidationError(BatchScoringError):
    """Exception raised when input dataset fails validation checks."""

    def __init__(self, message: str, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


class ModelLoadError(BatchScoringError):
    """Exception raised when model artifacts cannot be loaded or are invalid."""


def get_default_serving_config_path() -> Path:
    """Return the absolute path to the default serving.yaml config."""
    root_dir = Path(__file__).resolve().parents[3]
    return root_dir / "configs" / "serving.yaml"


def load_serving_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the serving/batch-scoring YAML configuration.

    Args:
        config_path: Optional path to YAML config. Uses default if None.

    Returns:
        Dictionary containing serving configuration settings.
    """
    path = Path(config_path) if config_path else get_default_serving_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Serving config not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f)
        return config or {}
    except Exception as exc:
        raise BatchScoringError(
            f"Failed to load serving configuration at '{path}': {exc}"
        ) from exc


def generate_batch_id(prefix: str = "batch") -> str:
    """Generate a default unique timestamped batch identifier.

    Args:
        prefix: Optional prefix for the batch identifier.

    Returns:
        String batch ID (e.g., batch_20260729_194106).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{short_uuid}"


def validate_scoring_batch(
    df: pd.DataFrame,
    contract_config_path: str | Path | None = None,
    raise_on_error: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate incoming batch scoring dataset against data contract.

    Args:
        df: Input pandas DataFrame to score.
        contract_config_path: Optional path to data contract configuration.
        raise_on_error: If True, raises BatchValidationError on failure.

    Returns:
        Tuple of (parsed DataFrame, ValidationReport).

    Raises:
        BatchValidationError: If raise_on_error is True and validation fails.
    """
    try:
        parsed_df, report = validate_data(
            df,
            config_path=contract_config_path,
            is_training=False,
            raise_on_error=False,
        )
    except Exception as exc:
        msg = f"Unexpected error during batch data validation: {exc}"
        raise BatchScoringError(msg) from exc

    if not report.is_valid and raise_on_error:
        raise BatchValidationError(
            f"Batch validation failed with {len(report.errors)} error(s).",
            report=report,
        )

    return parsed_df, report


def quarantine_batch(
    report: ValidationReport,
    quarantine_dir: str | Path,
    batch_id: str,
    quarantine_filename: str | None = None,
) -> Path:
    """Quarantine an invalid batch by persisting its validation report.

    Guarantees no partial scoring outputs are produced.

    Args:
        report: ValidationReport detailing the failure diagnostics.
        quarantine_dir: Directory where quarantine reports are stored.
        batch_id: Identifier of the quarantined batch.
        quarantine_filename: Optional filename override.

    Returns:
        Path to the saved quarantine report JSON file.
    """
    q_dir = Path(quarantine_dir)
    q_dir.mkdir(parents=True, exist_ok=True)

    if not quarantine_filename:
        filename = f"{batch_id}_quarantine.json"
    else:
        filename = quarantine_filename

    quarantine_path = q_dir / filename

    report_data = report.model_dump()
    report_data["batch_id"] = batch_id

    try:
        with open(quarantine_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise BatchScoringError(
            f"Failed to write quarantine report to '{quarantine_path}': {exc}"
        ) from exc

    return quarantine_path


def score_batch(
    df: pd.DataFrame,
    pipeline: Pipeline,
    model_version: str = "1.0.0",
    decision_threshold: float = 0.50,
    batch_id: str | None = None,
    id_column: str = "customerID",
) -> pd.DataFrame:
    """Score a validated customer batch deterministically using a trained pipeline.

    Preserves row ordering and 1-to-1 input-to-output mapping.

    Args:
        df: Validated customer DataFrame.
        pipeline: Trained scikit-learn Pipeline artifact.
        model_version: Model version identifier string.
        decision_threshold: Decision boundary for binary classification.
        batch_id: Batch identifier string.
        id_column: Name of customer primary key column.

    Returns:
        Scored pandas DataFrame with output schema.
    """
    if id_column not in df.columns:
        raise BatchScoringError(
            f"Primary key column '{id_column}' missing from input DataFrame."
        )

    if not (0.0 <= decision_threshold <= 1.0):
        raise ValueError(
            f"Decision threshold must be in range [0, 1], got {decision_threshold}"
        )

    b_id = batch_id or generate_batch_id()
    now_utc = datetime.now(timezone.utc).isoformat()

    try:
        probas = pipeline.predict_proba(df)[:, 1]
    except Exception as exc:
        raise BatchScoringError(f"Error during model prediction: {exc}") from exc

    predictions = (probas >= decision_threshold).astype(int)

    results_df = pd.DataFrame(
        {
            id_column: df[id_column].values,
            "churn_probability": pd.Series(probas, index=df.index).round(4).values,
            "predicted_class": predictions,
            "decision_threshold": decision_threshold,
            "model_version": model_version,
            "scoring_timestamp": now_utc,
            "batch_id": b_id,
        },
        index=df.index,
    )

    return results_df


def run_batch_scoring(
    input_path: str | Path,
    config_path: str | Path | None = None,
    output_path_override: str | Path | None = None,
    batch_id_override: str | None = None,
    decision_threshold_override: float | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Execute complete end-to-end batch scoring pipeline with observability.

    Steps:
      1. Load serving configuration.
      2. Load model pipeline and metadata (wrap errors in ModelLoadError).
      3. Read input dataset CSV (fail fast if unreadable).
      4. Validate dataset against data contract & generate operational quality report.
      5. If validation fails: quarantine batch and raise BatchValidationError.
      6. If validation passes: score all rows and write predictions output CSV.

    Args:
        input_path: Path to input customer CSV file.
        config_path: Optional path to serving configuration YAML.
        output_path_override: Optional path to output predictions CSV override.
        batch_id_override: Optional batch ID string override.
        decision_threshold_override: Optional decision threshold override.

    Returns:
        Tuple of (scored DataFrame, Path to saved predictions output CSV).
    """
    config = load_serving_config(config_path)

    batch_prefix = config.get("scoring", {}).get("batch_id_prefix", "batch")
    batch_id = (
        batch_id_override
        or config.get("scoring", {}).get("batch_id")
        or generate_batch_id(prefix=batch_prefix)
    )

    log_event(
        logger,
        logging.INFO,
        event="batch_scoring_started",
        message="Starting batch scoring pipeline execution",
        batch_id=batch_id,
        input_path=str(input_path),
    )

    # Resolve paths from config
    model_config = config.get("model", {})
    model_dir = model_config.get("model_dir", "models")
    pipeline_fn = model_config.get("pipeline_filename", "baseline_pipeline.joblib")
    metadata_fn = model_config.get("metadata_filename", "baseline_metadata.json")

    data_config = config.get("data", {})
    contract_config_path = data_config.get(
        "contract_config_path", "configs/data_contract.yaml"
    )

    scoring_config = config.get("scoring", {})
    if decision_threshold_override is not None:
        decision_threshold = float(decision_threshold_override)
    else:
        decision_threshold = float(scoring_config.get("decision_threshold", 0.50))

    id_column = str(scoring_config.get("id_column", "customerID"))

    output_config = config.get("output", {})
    output_dir = Path(output_config.get("output_dir", "reports/scoring"))
    output_filename = output_config.get("output_filename", "batch_predictions.csv")
    quarantine_dir = Path(output_config.get("quarantine_dir", "reports/quarantine"))

    # Load artifacts
    model_dir_path = Path(model_dir)
    try:
        pipeline, metadata = load_artifacts(
            output_dir=model_dir_path,
            pipeline_filename=pipeline_fn,
            metadata_filename=metadata_fn,
        )
    except FileNotFoundError as exc:
        metrics_manager.record_batch_failure()
        log_event(
            logger,
            logging.ERROR,
            event="batch_scoring_failed",
            message="Model artifact missing for batch scoring",
            batch_id=batch_id,
            error=str(exc),
        )
        raise ModelLoadError(
            f"Model artifact file not found in '{model_dir_path}': {exc}"
        ) from exc
    except Exception as exc:
        metrics_manager.record_batch_failure()
        log_event(
            logger,
            logging.ERROR,
            event="batch_scoring_failed",
            message="Failed to load model artifacts for batch scoring",
            batch_id=batch_id,
            error=str(exc),
        )
        raise ModelLoadError(
            f"Failed to load model artifacts from '{model_dir_path}': {exc}"
        ) from exc

    model_version = str(
        metadata.get("schema_version", metadata.get("version", "1.0.0"))
    )

    # Read input data
    inp_p = Path(input_path)
    if not inp_p.is_file():
        metrics_manager.record_batch_failure()
        raise FileNotFoundError(f"Input batch file not found at: {inp_p}")

    try:
        raw_df = pd.read_csv(inp_p)
    except Exception as exc:
        metrics_manager.record_batch_failure()
        log_event(
            logger,
            logging.ERROR,
            event="batch_scoring_failed",
            message="Failed to read input batch CSV file",
            batch_id=batch_id,
            error=str(exc),
        )
        raise BatchScoringError(
            f"Failed to read input CSV file '{inp_p}': {exc}"
        ) from exc

    # Generate operational quality report and validate data
    quality_report = generate_quality_report(
        raw_df,
        contract_config_path=contract_config_path,
        is_training=False,
    )

    parsed_df, report = validate_scoring_batch(
        raw_df,
        contract_config_path=contract_config_path,
        raise_on_error=False,
    )

    if not report.is_valid:
        quarantine_path = quarantine_batch(
            report=report,
            quarantine_dir=quarantine_dir,
            batch_id=batch_id,
        )
        log_event(
            logger,
            logging.WARNING,
            event="batch_validation_failed",
            message="Batch scoring dataset failed contract validation",
            batch_id=batch_id,
            total_records=quality_report.total_records,
            rejected_records=quality_report.rejected_records,
            quarantine_path=str(quarantine_path),
        )
        raise BatchValidationError(
            f"Batch validation failed for batch '{batch_id}'. "
            f"Quarantined report saved to: {quarantine_path}",
            report=report,
        )

    # Score valid batch
    scored_df = score_batch(
        df=parsed_df,
        pipeline=pipeline,
        model_version=model_version,
        decision_threshold=decision_threshold,
        batch_id=batch_id,
        id_column=id_column,
    )

    # Write output
    if output_path_override:
        final_output_path = Path(output_path_override)
    else:
        final_output_path = output_dir / output_filename

    try:
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(final_output_path, index=False)
    except Exception as exc:
        metrics_manager.record_batch_failure()
        log_event(
            logger,
            logging.ERROR,
            event="batch_scoring_failed",
            message="Failed to write output predictions CSV",
            batch_id=batch_id,
            error=str(exc),
        )
        raise BatchScoringError(
            f"Failed to write predictions to output file '{final_output_path}': {exc}"
        ) from exc

    log_event(
        logger,
        logging.INFO,
        event="batch_scoring_completed",
        message="Batch scoring pipeline completed successfully",
        batch_id=batch_id,
        model_version=model_version,
        record_count=len(scored_df),
        output_path=str(final_output_path),
    )

    return scored_df, final_output_path
