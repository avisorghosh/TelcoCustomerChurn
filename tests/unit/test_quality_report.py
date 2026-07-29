"""Unit tests for operational data quality reporting."""

from pathlib import Path

import pandas as pd

from churn_prediction.monitoring.quality_report import generate_quality_report


def test_generate_quality_report_valid_data(tmp_path: Path) -> None:
    """Test generating operational quality report on valid dataset."""
    df = pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 24],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["Yes", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [358.20, 1366.80],
        }
    )

    out_file = tmp_path / "quality_report.json"
    report = generate_quality_report(df, is_training=False, output_path=out_file)

    assert report.is_valid is True
    assert report.total_records == 2
    assert report.accepted_records == 2
    assert report.rejected_records == 0
    assert report.rejection_rate == 0.0
    assert report.duplicate_records["duplicate_rows"] == 0
    assert report.duplicate_records["duplicate_primary_keys"] == 0
    assert out_file.is_file()


def test_generate_quality_report_synthetic_bad_input(tmp_path: Path) -> None:
    """Test generating quality report on synthetic bad input with errors."""
    df = pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0001-AAA"],  # Duplicate ID
            "gender": ["Female", "InvalidGender"],  # Invalid domain
            "SeniorCitizen": [0, 999],  # Invalid domain
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [-5, 24],  # Invalid negative tenure
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["Yes", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "InvalidContract"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [358.20, "INVALID_TEXT"],  # Invalid non-numeric string
        }
    )

    out_file = tmp_path / "bad_quality_report.json"
    report = generate_quality_report(df, is_training=False, output_path=out_file)

    assert report.is_valid is False
    assert report.total_records == 2
    assert report.rejected_records > 0
    assert report.duplicate_records["duplicate_primary_keys"] == 2
    assert len(report.schema_violations) > 0
    assert out_file.is_file()
