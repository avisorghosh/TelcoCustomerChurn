"""Pytest shared fixtures configuration."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from churn_prediction.models.serialization import save_artifacts
from churn_prediction.models.trainer import train_baseline


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture providing a valid sample customer DataFrame with target Churn column."""
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
                "TotalCharges": 29.85,
                "Churn": "No",
            },
            {
                "customerID": "5575-GNVDE",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 34,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 56.95,
                "TotalCharges": 1889.50,
                "Churn": "No",
            },
            {
                "customerID": "3668-QPYBK",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 53.85,
                "TotalCharges": 108.15,
                "Churn": "Yes",
            },
        ]
    )


@pytest.fixture
def sample_customer_dict(sample_valid_df: pd.DataFrame) -> dict[str, Any]:
    """Fixture providing a dictionary of a single valid customer record."""
    row: dict[str, Any] = sample_valid_df.iloc[0].to_dict()
    row.pop("Churn", None)
    return row


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    """Fixture providing synthetic customer DataFrame with target for splitting."""
    rows = []
    for i in range(100):
        churn_val = "Yes" if i % 4 == 0 else "No"
        rows.append(
            {
                "customerID": f"ID-{i:04d}",
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": 1 if i % 5 == 0 else 0,
                "Partner": "Yes" if i % 3 == 0 else "No",
                "Dependents": "No",
                "tenure": (i % 60) + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL" if i % 2 == 0 else "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes" if i % 2 == 0 else "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if i % 2 == 0 else "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 30.0 + (i % 70),
                "TotalCharges": 100.0 + (i * 50),
                "Churn": churn_val,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_model_dir(tmp_path: Path, synthetic_dataset: pd.DataFrame) -> Path:
    """Train a baseline model into a temporary directory (never mutates models/)."""
    csv_path = tmp_path / "train_data.csv"
    synthetic_dataset.to_csv(csv_path, index=False)
    model_dir = tmp_path / "models"
    pipeline, metadata, _ = train_baseline(
        data_path_override=csv_path,
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
    return model_dir


@pytest.fixture
def serving_config_for_tmp_model(tmp_path: Path, trained_model_dir: Path) -> Path:
    """Write a serving YAML that points at the temporary trained model directory."""
    config = {
        "schema_version": "1.0.0",
        "model_name": "baseline_logistic_regression",
        "model": {
            "model_dir": str(trained_model_dir),
            "pipeline_filename": "serving_pipeline.joblib",
            "metadata_filename": "serving_metadata.json",
        },
        "data": {"contract_config_path": "configs/data_contract.yaml"},
        "scoring": {
            "decision_threshold": 0.50,
            "batch_id_prefix": "batch",
            "id_column": "customerID",
        },
        "output": {
            "output_dir": str(tmp_path / "scoring"),
            "output_filename": "batch_predictions.csv",
            "quarantine_dir": str(tmp_path / "quarantine"),
        },
        "api": {"host": "127.0.0.1", "port": 8000},
    }
    config_path = tmp_path / "serving.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path
