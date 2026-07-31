"""Unit tests for baseline preprocessing transformer and pipeline construction."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_prediction.features.pipeline import (
    build_model_pipeline,
    build_preprocessing_transformer,
    extract_transformed_feature_names,
    load_training_config,
)


@pytest.fixture
def sample_config() -> dict:
    """Fixture providing training configuration dictionary."""
    return load_training_config()


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """Fixture providing a sample customer DataFrame."""
    return pd.DataFrame(
        [
            {
                "customerID": "1234-TESTA",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
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
                "MonthlyCharges": 55.85,
                "TotalCharges": 670.20,
            },
            {
                "customerID": "5678-TESTB",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.00,
                "TotalCharges": 85.00,
            },
        ]
    )


def test_build_preprocessing_transformer_excludes_sensitive_and_id(sample_config):
    """Test that ColumnTransformer excludes customerID, gender, and SeniorCitizen."""
    preprocessor = build_preprocessing_transformer(sample_config)

    for _name, _transformer, cols in preprocessor.transformers:
        assert "customerID" not in cols, "customerID must be excluded from features."
        assert "gender" not in cols, "gender must be excluded from features."
        assert "SeniorCitizen" not in cols, (
            "SeniorCitizen must be excluded from features."
        )


def test_build_model_pipeline_structure(sample_config):
    """Test that build_model_pipeline constructs a valid scikit-learn Pipeline."""
    pipeline = build_model_pipeline(sample_config)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_pipeline_fit_and_feature_names_out(sample_config, sample_customer_df):
    """Test fitting pipeline on sample data and checking extracted feature names out."""
    pipeline = build_model_pipeline(sample_config)

    X = sample_customer_df.copy()
    y = [0, 1]

    pipeline.fit(X, y)

    feature_names = extract_transformed_feature_names(pipeline)

    assert len(feature_names) > 0
    for name in feature_names:
        assert "customerID" not in name
        assert "gender" not in name
        assert "SeniorCitizen" not in name
