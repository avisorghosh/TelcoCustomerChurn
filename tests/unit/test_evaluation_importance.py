"""Unit tests for feature importance extraction."""

import pandas as pd
import pytest

from churn_prediction.evaluation.importance import extract_feature_importance
from churn_prediction.features.pipeline import (
    build_model_pipeline,
    load_training_config,
)


def test_extract_feature_importance_fitted_pipeline() -> None:
    """Verify feature importance extraction on a fitted baseline pipeline."""
    config = load_training_config()
    pipeline = build_model_pipeline(config)

    # Create dummy training data
    df = pd.DataFrame(
        {
            "customerID": ["1", "2", "3", "4"],
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "tenure": [1, 12, 24, 36],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["Yes", "No", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "No", "Yes", "No internet service"],
            "TechSupport": ["No", "Yes", "No", "No internet service"],
            "StreamingTV": ["No", "No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 42.30, 70.35],
            "TotalCharges": [29.85, 683.40, 1015.20, 2532.60],
        }
    )
    y = [1, 0, 1, 0]

    # Fit pipeline
    pipeline.fit(df, y)

    importance = extract_feature_importance(pipeline)

    assert isinstance(importance, list)
    assert len(importance) > 0

    first_item = importance[0]
    assert "feature" in first_item
    assert "coefficient" in first_item
    assert "abs_coefficient" in first_item
    assert "odds_ratio" in first_item

    # Verify descending sort order by absolute coefficient
    for i in range(len(importance) - 1):
        assert importance[i]["abs_coefficient"] >= importance[i + 1]["abs_coefficient"]


def test_extract_feature_importance_no_classifier_raises() -> None:
    """Verify ValueError is raised if pipeline lacks classifier step."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scaler", StandardScaler())])
    with pytest.raises(ValueError, match="does not contain a 'classifier' step"):
        extract_feature_importance(pipe)
