"""Unit tests for batch scoring pipeline module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from churn_prediction.data.contract import ValidationReport
from churn_prediction.models.batch_scoring import (
    BatchValidationError,
    ModelLoadError,
    generate_batch_id,
    load_serving_config,
    quarantine_batch,
    run_batch_scoring,
    score_batch,
    validate_scoring_batch,
)
from churn_prediction.models.serialization import load_artifacts


def test_load_serving_config(tmp_path: Path) -> None:
    """Test loading valid and non-existent serving config."""
    config = load_serving_config()
    assert "schema_version" in config
    assert "model" in config
    assert "scoring" in config
    assert config["model"]["pipeline_filename"] == "serving_pipeline.joblib"

    with pytest.raises(FileNotFoundError):
        load_serving_config(tmp_path / "non_existent.yaml")


def test_generate_batch_id() -> None:
    """Test batch ID generation formatting and uniqueness."""
    id1 = generate_batch_id("test_batch")
    id2 = generate_batch_id("test_batch")

    assert id1.startswith("test_batch_")
    assert id1 != id2


def test_validate_scoring_batch_valid(sample_valid_df: pd.DataFrame) -> None:
    """Test validation of valid customer scoring batch."""
    scoring_df = sample_valid_df.drop(columns=["Churn"])
    parsed_df, report = validate_scoring_batch(scoring_df, raise_on_error=False)

    assert report.is_valid is True
    assert report.total_rows == len(scoring_df)
    assert len(report.errors) == 0


def test_validate_scoring_batch_duplicate_id(sample_valid_df: pd.DataFrame) -> None:
    """Test rejection of duplicate customer IDs in scoring batch."""
    scoring_df = sample_valid_df.drop(columns=["Churn"]).copy()
    scoring_df = pd.concat([scoring_df, scoring_df.iloc[[0]]], ignore_index=True)

    parsed_df, report = validate_scoring_batch(scoring_df, raise_on_error=False)
    assert report.is_valid is False
    assert any(err.check_type == "duplicate_primary_key" for err in report.errors)

    with pytest.raises(BatchValidationError):
        validate_scoring_batch(scoring_df, raise_on_error=True)


def test_validate_scoring_batch_invalid_schema(
    sample_valid_df: pd.DataFrame,
) -> None:
    """Test rejection of batch with missing required column."""
    invalid_df = sample_valid_df.drop(columns=["Churn", "Contract"])
    parsed_df, report = validate_scoring_batch(invalid_df, raise_on_error=False)

    assert report.is_valid is False
    assert any("Contract" in err.message for err in report.errors)


def test_quarantine_batch(tmp_path: Path) -> None:
    """Test quarantine report generation and file persistence."""
    dummy_report = ValidationReport(
        is_valid=False,
        schema_version="1.0.0",
        total_rows=10,
        valid_rows=8,
        errors=[],
        summary="Test quarantine summary",
    )

    batch_id = "batch_test_123"
    q_path = quarantine_batch(dummy_report, tmp_path, batch_id)

    assert q_path.is_file()
    with open(q_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["batch_id"] == batch_id
    assert data["is_valid"] is False
    assert data["summary"] == "Test quarantine summary"


def test_score_batch_determinism_and_row_integrity(
    sample_valid_df: pd.DataFrame,
    trained_model_dir: Path,
) -> None:
    """Test score_batch row ordering, 1-to-1 mapping, and determinism."""
    pipeline, _ = load_artifacts(
        trained_model_dir,
        pipeline_filename="serving_pipeline.joblib",
        metadata_filename="serving_metadata.json",
    )

    input_df = sample_valid_df.drop(columns=["Churn"]).copy()
    scored_1 = score_batch(input_df, pipeline, batch_id="b1", decision_threshold=0.5)
    scored_2 = score_batch(input_df, pipeline, batch_id="b1", decision_threshold=0.5)

    assert len(scored_1) == len(input_df)
    assert list(scored_1["customerID"]) == list(input_df["customerID"])
    assert list(scored_1["churn_probability"]) == list(scored_2["churn_probability"])
    assert list(scored_1["predicted_class"]) == list(scored_2["predicted_class"])


def test_score_batch_threshold_override(
    sample_valid_df: pd.DataFrame,
    trained_model_dir: Path,
) -> None:
    """Test that decision threshold correctly alters predicted_class outputs."""
    pipeline, _ = load_artifacts(
        trained_model_dir,
        pipeline_filename="serving_pipeline.joblib",
        metadata_filename="serving_metadata.json",
    )
    input_df = sample_valid_df.drop(columns=["Churn"]).copy()

    scored_low = score_batch(input_df, pipeline, decision_threshold=0.01)
    scored_high = score_batch(input_df, pipeline, decision_threshold=0.99)

    assert scored_low["predicted_class"].sum() >= scored_high["predicted_class"].sum()


def test_run_batch_scoring_missing_model(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    """Test error handling when model artifact directory is missing."""
    input_csv = tmp_path / "input.csv"
    sample_valid_df.drop(columns=["Churn"]).to_csv(input_csv, index=False)

    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        f"""
schema_version: "1.0.0"
model:
  model_dir: "{tmp_path / "non_existent_models"}"
  pipeline_filename: "serving_pipeline.joblib"
  metadata_filename: "serving_metadata.json"
data:
  contract_config_path: "configs/data_contract.yaml"
scoring:
  decision_threshold: 0.50
output:
  output_dir: "{tmp_path}"
  quarantine_dir: "{tmp_path / "quarantine"}"
"""
    )

    with pytest.raises(ModelLoadError):
        run_batch_scoring(input_path=input_csv, config_path=config_path)


def test_run_batch_scoring_corrupt_model(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    """Test error handling when model artifacts are corrupt."""
    models_dir = tmp_path / "corrupt_models"
    models_dir.mkdir()
    (models_dir / "serving_pipeline.joblib").write_text("corrupted content")
    (models_dir / "serving_metadata.json").write_text("{}")

    input_csv = tmp_path / "input.csv"
    sample_valid_df.drop(columns=["Churn"]).to_csv(input_csv, index=False)

    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        f"""
schema_version: "1.0.0"
model:
  model_dir: "{models_dir}"
  pipeline_filename: "serving_pipeline.joblib"
  metadata_filename: "serving_metadata.json"
data:
  contract_config_path: "configs/data_contract.yaml"
scoring:
  decision_threshold: 0.50
output:
  output_dir: "{tmp_path}"
  quarantine_dir: "{tmp_path / "quarantine"}"
"""
    )

    with pytest.raises(ModelLoadError):
        run_batch_scoring(input_path=input_csv, config_path=config_path)
