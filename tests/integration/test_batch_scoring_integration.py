"""Integration tests for batch scoring workflow and acceptance criteria."""

import json
from pathlib import Path

import pandas as pd
import pytest

from churn_prediction.models.batch_scoring import (
    BatchValidationError,
    run_batch_scoring,
)


def test_integration_successful_batch_scoring(tmp_path: Path) -> None:
    """Test full batch scoring pipeline with real dataset artifact."""
    input_csv = "Telco-Customer-Churn.csv"
    output_csv = tmp_path / "predictions.csv"

    scored_df, output_path = run_batch_scoring(
        input_path=input_csv,
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
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    """Test acceptance criterion: invalid batch produces NO partial predictions."""
    invalid_df = sample_valid_df.drop(columns=["Churn"]).copy()
    invalid_df = pd.concat([invalid_df, invalid_df.iloc[[0]]], ignore_index=True)

    input_csv = tmp_path / "invalid_batch.csv"
    invalid_df.to_csv(input_csv, index=False)

    output_csv = tmp_path / "should_not_exist.csv"
    quarantine_dir = tmp_path / "quarantine"

    config_path = tmp_path / "test_serving_config.yaml"
    config_path.write_text(
        f"""
schema_version: "1.0.0"
model:
  model_dir: "models"
  pipeline_filename: "baseline_pipeline.joblib"
  metadata_filename: "baseline_metadata.json"
data:
  contract_config_path: "configs/data_contract.yaml"
scoring:
  decision_threshold: 0.50
output:
  output_dir: "{tmp_path}"
  output_filename: "should_not_exist.csv"
  quarantine_dir: "{quarantine_dir}"
"""
    )

    with pytest.raises(BatchValidationError):
        run_batch_scoring(
            input_path=input_csv,
            config_path=config_path,
            output_path_override=output_csv,
            batch_id_override="quarantine_batch_999",
        )

    # Acceptance criterion 1: NO partial scoring output file generated
    assert not output_csv.exists()

    # Acceptance criterion 2: Quarantine report generated
    q_file = quarantine_dir / "quarantine_batch_999_quarantine.json"
    assert q_file.is_file()

    with open(q_file, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    assert q_data["is_valid"] is False
    assert q_data["batch_id"] == "quarantine_batch_999"
    assert len(q_data["errors"]) > 0


def test_integration_idempotent_execution(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    """Test that scoring the same dataset twice produces identical outputs."""
    input_csv = tmp_path / "idempotent_input.csv"
    sample_valid_df.drop(columns=["Churn"]).to_csv(input_csv, index=False)

    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"

    scored1, path1 = run_batch_scoring(
        input_path=input_csv,
        output_path_override=out1,
        batch_id_override="fixed_batch_id",
    )

    scored2, path2 = run_batch_scoring(
        input_path=input_csv,
        output_path_override=out2,
        batch_id_override="fixed_batch_id",
    )

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    pd.testing.assert_series_equal(df1["churn_probability"], df2["churn_probability"])
    pd.testing.assert_series_equal(df1["predicted_class"], df2["predicted_class"])
    pd.testing.assert_series_equal(df1["customerID"], df2["customerID"])
