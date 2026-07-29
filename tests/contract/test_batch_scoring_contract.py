"""Contract tests for batch scoring output schema and contract guarantees."""

from pathlib import Path

import pandas as pd
import pandera.typing as pt
from pandera.pandas import Check, Column, DataFrameSchema

from churn_prediction.models.batch_scoring import run_batch_scoring, score_batch
from churn_prediction.models.serialization import load_artifacts


def get_batch_scoring_output_schema() -> DataFrameSchema:
    """Build Pandera schema for batch scoring output validation."""
    return DataFrameSchema(
        columns={
            "customerID": Column(
                pt.String,
                nullable=False,
                checks=[Check(lambda s: s.str.strip().str.len() > 0)],
                required=True,
            ),
            "churn_probability": Column(
                pt.Float,
                nullable=False,
                checks=[
                    Check.greater_than_or_equal_to(0.0),
                    Check.less_than_or_equal_to(1.0),
                ],
                required=True,
            ),
            "predicted_class": Column(
                pt.Int,
                nullable=False,
                checks=[Check.isin([0, 1])],
                required=True,
            ),
            "decision_threshold": Column(
                pt.Float,
                nullable=False,
                checks=[
                    Check.greater_than_or_equal_to(0.0),
                    Check.less_than_or_equal_to(1.0),
                ],
                required=True,
            ),
            "model_version": Column(
                pt.String,
                nullable=False,
                required=True,
            ),
            "scoring_timestamp": Column(
                pt.String,
                nullable=False,
                required=True,
            ),
            "batch_id": Column(
                pt.String,
                nullable=False,
                required=True,
            ),
        },
        strict=True,
        coerce=False,
    )


def test_batch_scoring_output_contract(
    sample_valid_df: pd.DataFrame,
    trained_model_dir: Path,
) -> None:
    """Validate that score_batch produces output adhering strictly to output schema."""
    pipeline, metadata = load_artifacts(
        trained_model_dir,
        pipeline_filename="serving_pipeline.joblib",
        metadata_filename="serving_metadata.json",
    )
    input_df = sample_valid_df.drop(columns=["Churn"]).copy()

    scored_df = score_batch(
        df=input_df,
        pipeline=pipeline,
        model_version=metadata.get("schema_version", "1.0.0"),
        decision_threshold=0.50,
        batch_id="contract_test_batch",
    )

    schema = get_batch_scoring_output_schema()
    validated_output = schema.validate(scored_df)

    assert len(validated_output) == len(input_df)
    assert not validated_output.isna().any().any()


def test_end_to_end_scoring_output_contract(
    tmp_path: Path,
    sample_valid_df: pd.DataFrame,
    serving_config_for_tmp_model: Path,
) -> None:
    """Validate end-to-end run_batch_scoring output file contract."""
    input_csv = tmp_path / "valid_input.csv"
    sample_valid_df.drop(columns=["Churn"]).to_csv(input_csv, index=False)

    scored_df, output_path = run_batch_scoring(
        input_path=input_csv,
        config_path=serving_config_for_tmp_model,
        output_path_override=tmp_path / "output_predictions.csv",
    )

    saved_df = pd.read_csv(output_path)
    schema = get_batch_scoring_output_schema()
    validated_output = schema.validate(saved_df)

    assert len(validated_output) == len(sample_valid_df)
