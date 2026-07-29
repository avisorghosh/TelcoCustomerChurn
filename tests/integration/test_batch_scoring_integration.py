"""Integration tests for batch scoring workflow and acceptance criteria."""

import json
from pathlib import Path

import pandas as pd
import pytest

from churn_prediction.models.batch_scoring import (
    BatchValidationError,
    run_batch_scoring,
)
from churn_prediction.models.serialization import save_artifacts
from churn_prediction.models.trainer import train_baseline


def test_integration_successful_batch_scoring(tmp_path: Path) -> None:
    """Test full batch scoring pipeline with real dataset artifact."""
    model_dir = tmp_path / "models"
    pipeline, metadata, _ = train_baseline(
        data_path_override="Telco-Customer-Churn.csv",
        log_to_mlflow=False,
        output_dir_override=model_dir,
    )
    save_artifacts(
        pipeline=pipeline,
        metadata=metadata,
        output_dir=model_dir,
        pipeline_filename="serving_pipeline.joblib",
        metadata_filename="serving_metadata.json",
    )

    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        f"""
schema_version: "1.0.0"
model:
  model_dir: "{model_dir}"
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

    output_csv = tmp_path / "predictions.csv"
    scored_df, output_path = run_batch_scoring(
        input_path="Telco-Customer-Churn.csv",
        config_path=config_path,
        output_path_override=output_csv,
        batch_id_override="integration_test_batch_001",
    )

    assert output_path.is_file()
    saved_df = pd.read_csv(output_path)
    assert len(saved_df) == len(scored_df) == 7043
    assert "churn_probability" in saved_df.columns
    assert "predicted_class" in saved_df.columns
    assert (saved_df["batch_id"] == "integration_test_batch_001").all()


def test_integration_invalid_batch_quarantine_no_partial_output(
    tmp_path: Path,
    sample_valid_df: pd.DataFrame,
    trained_model_dir: Path,
) -> None:
    """Test acceptance criterion: invalid batch produces NO partial predictions."""
    invalid_df = sample_valid_df.drop(columns=["Churn"]).copy()
    invalid_df = pd.concat([invalid_df, invalid_df.iloc[[0]]], ignore_index=True)

    input_csv = tmp_path / "invalid_batch.csv"
    invalid_df.to_csv(input_csv, index=False)

    output_csv = tmp_path / "should_not_exist.csv"
    quarantine_dir = tmp_path / "quarantine_isolated"

    config_path = tmp_path / "serving_quarantine.yaml"
    config_path.write_text(
        f"""
schema_version: "1.0.0"
model:
  model_dir: "{trained_model_dir}"
  pipeline_filename: "serving_pipeline.joblib"
  metadata_filename: "serving_metadata.json"
data:
  contract_config_path: "configs/data_contract.yaml"
scoring:
  decision_threshold: 0.50
output:
  output_dir: "{tmp_path}"
  quarantine_dir: "{quarantine_dir}"
""",
        encoding="utf-8",
    )

    with pytest.raises(BatchValidationError):
        run_batch_scoring(
            input_path=input_csv,
            config_path=config_path,
            output_path_override=output_csv,
            batch_id_override="quarantine_batch_999",
        )

    assert not output_csv.exists()

    q_file = quarantine_dir / "quarantine_batch_999_quarantine.json"
    assert q_file.is_file()

    with open(q_file, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    assert q_data["is_valid"] is False
    assert q_data["batch_id"] == "quarantine_batch_999"
    assert len(q_data["errors"]) > 0


def test_integration_idempotent_execution(
    tmp_path: Path,
    sample_valid_df: pd.DataFrame,
    serving_config_for_tmp_model: Path,
) -> None:
    """Test that scoring the same dataset twice produces identical outputs."""
    input_csv = tmp_path / "idempotent_input.csv"
    sample_valid_df.drop(columns=["Churn"]).to_csv(input_csv, index=False)

    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"

    scored1, path1 = run_batch_scoring(
        input_path=input_csv,
        config_path=serving_config_for_tmp_model,
        output_path_override=out1,
        batch_id_override="fixed_batch_id",
    )
    scored2, path2 = run_batch_scoring(
        input_path=input_csv,
        config_path=serving_config_for_tmp_model,
        output_path_override=out2,
        batch_id_override="fixed_batch_id",
    )

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    pd.testing.assert_series_equal(df1["churn_probability"], df2["churn_probability"])
    pd.testing.assert_series_equal(df1["predicted_class"], df2["predicted_class"])
    pd.testing.assert_series_equal(df1["customerID"], df2["customerID"])
