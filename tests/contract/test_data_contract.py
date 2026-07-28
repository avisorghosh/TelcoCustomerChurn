"""Contract tests for source dataset and malformed fixtures."""

from pathlib import Path

import pandas as pd

from churn_prediction.data import (
    create_source_manifest,
    validate_data,
)


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    return Path(__file__).resolve().parents[2]


def test_telco_customer_churn_csv_satisfies_contract() -> None:
    """The root dataset Telco-Customer-Churn.csv satisfies the data contract."""
    data_path = get_project_root() / "Telco-Customer-Churn.csv"
    assert data_path.is_file(), f"Dataset file not found at: {data_path}"

    df = pd.read_csv(data_path)
    validated_df, report = validate_data(df, is_training=True, raise_on_error=False)

    assert report.is_valid is True, f"Validation failed: {report.summary}"
    assert report.total_rows == 7043

    num_null_total_charges = validated_df["TotalCharges"].isna().sum()
    assert num_null_total_charges == 11

    manifest = create_source_manifest(data_path, schema_version=report.schema_version)
    assert manifest.row_count == 7043
    assert len(manifest.file_sha256) == 64


def test_malformed_fixture_invalid_total_charges_fails() -> None:
    """Fixture with non-numeric TotalCharges fails contract validation cleanly."""
    fixture_path = (
        get_project_root() / "tests" / "fixtures" / "invalid_total_charges.csv"
    )
    df = pd.read_csv(fixture_path)

    _, report = validate_data(df, raise_on_error=False)

    assert report.is_valid is False
    assert any(
        err.check_type == "invalid_type_conversion"
        and "TotalCharges" in (err.column or "")
        for err in report.errors
    )


def test_malformed_fixture_duplicate_customer_id_fails() -> None:
    """Fixture with duplicate customerID fails contract validation cleanly."""
    fixture_path = (
        get_project_root() / "tests" / "fixtures" / "duplicate_customer_id.csv"
    )
    df = pd.read_csv(fixture_path)

    _, report = validate_data(df, raise_on_error=False)

    assert report.is_valid is False
    assert any(
        err.check_type in ("duplicate_primary_key", "duplicate_rows")
        for err in report.errors
    )


def test_malformed_fixture_invalid_category_fails() -> None:
    """Fixture with invalid categorical value fails contract validation cleanly."""
    fixture_path = get_project_root() / "tests" / "fixtures" / "invalid_category.csv"
    df = pd.read_csv(fixture_path)

    _, report = validate_data(df, raise_on_error=False)

    assert report.is_valid is False
    assert len(report.errors) > 0


def test_malformed_fixture_blank_customer_id_fails() -> None:
    """Fixture with blank customerID fails contract validation cleanly."""
    fixture_path = get_project_root() / "tests" / "fixtures" / "blank_customer_id.csv"
    df = pd.read_csv(fixture_path)

    _, report = validate_data(df, raise_on_error=False)

    assert report.is_valid is False
    assert any(err.check_type == "null_primary_key" for err in report.errors)
