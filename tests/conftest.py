"""Pytest shared fixtures configuration."""

from typing import Any

import pandas as pd
import pytest


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
