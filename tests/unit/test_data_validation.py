"""Unit tests for data contract validation rules and functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_prediction.data import (
    DataValidationError,
    create_source_manifest,
    parse_total_charges,
    validate_data,
)


def get_valid_sample_df() -> pd.DataFrame:
    """Return a single-row valid pandas DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
                "Churn": "No",
            }
        ]
    )


def test_create_source_manifest_computes_sha256_and_row_count(
    tmp_path: Path,
) -> None:
    """Manifest generator calculates file sha256 and row count correctly."""
    sample_file = tmp_path / "sample.csv"
    sample_file.write_text("header1,header2\nval1,val2\nval3,val4\n")

    manifest = create_source_manifest(sample_file, schema_version="1.0.0")

    assert manifest.row_count == 2
    assert len(manifest.file_sha256) == 64
    assert manifest.schema_version == "1.0.0"


def test_parse_total_charges_converts_whitespace_to_nan() -> None:
    """Whitespace-only strings in TotalCharges are explicitly converted to NaN."""
    df = pd.DataFrame({"TotalCharges": ["29.85", " ", "  ", "100.5"]})

    parsed_df, errors = parse_total_charges(df)

    assert len(errors) == 0
    assert parsed_df["TotalCharges"].iloc[0] == 29.85
    assert np.isnan(parsed_df["TotalCharges"].iloc[1])
    assert np.isnan(parsed_df["TotalCharges"].iloc[2])
    assert parsed_df["TotalCharges"].iloc[3] == 100.5


def test_parse_total_charges_rejects_non_numeric_strings() -> None:
    """Non-numeric string values in TotalCharges emit error messages."""
    df = pd.DataFrame({"TotalCharges": ["29.85", "INVALID_TEXT"]})

    _, errors = parse_total_charges(df)

    assert len(errors) == 1
    assert "Non-numeric string 'INVALID_TEXT'" in errors[0]


def test_accepts_valid_dataframe() -> None:
    """Valid dataframe passes validation without errors."""
    df = get_valid_sample_df()

    validated_df, report = validate_data(df, raise_on_error=True)

    assert report.is_valid is True
    assert report.total_rows == 1
    assert len(report.errors) == 0
    assert validated_df["TotalCharges"].iloc[0] == 29.85


def test_rejects_blank_customer_id() -> None:
    """DataFrame with blank customerID fails validation."""
    df = get_valid_sample_df()
    df.loc[0, "customerID"] = "  "

    with pytest.raises(DataValidationError) as exc_info:
        validate_data(df, raise_on_error=True)

    assert exc_info.value.report is not None
    assert exc_info.value.report.is_valid is False
    assert any(
        err.check_type == "null_primary_key" for err in exc_info.value.report.errors
    )


def test_rejects_duplicate_customer_ids() -> None:
    """DataFrame with duplicate customerIDs fails validation."""
    row1 = get_valid_sample_df()
    row2 = get_valid_sample_df()
    row2.loc[0, "MonthlyCharges"] = 50.0  # Different charges, same customerID
    df = pd.concat([row1, row2], ignore_index=True)

    with pytest.raises(DataValidationError) as exc_info:
        validate_data(df, raise_on_error=True)

    assert exc_info.value.report is not None
    assert exc_info.value.report.is_valid is False
    assert any(
        err.check_type == "duplicate_primary_key"
        for err in exc_info.value.report.errors
    )


def test_rejects_invalid_category_domain() -> None:
    """DataFrame with invalid category value fails validation."""
    df = get_valid_sample_df()
    df.loc[0, "Contract"] = "InvalidContract"

    with pytest.raises(DataValidationError) as exc_info:
        validate_data(df, raise_on_error=True)

    assert exc_info.value.report is not None
    assert exc_info.value.report.is_valid is False


def test_rejects_negative_numeric_values() -> None:
    """DataFrame with negative tenure fails validation."""
    df = get_valid_sample_df()
    df.loc[0, "tenure"] = -5

    with pytest.raises(DataValidationError) as exc_info:
        validate_data(df, raise_on_error=True)

    assert exc_info.value.report is not None
    assert exc_info.value.report.is_valid is False
