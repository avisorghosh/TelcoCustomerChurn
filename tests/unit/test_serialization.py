"""Unit tests for model serialization and deserialization."""

import pandas as pd
import pytest

from churn_prediction.features.pipeline import (
    build_baseline_pipeline,
    load_training_config,
)
from churn_prediction.models.serialization import (
    load_artifacts,
    load_metadata,
    load_pipeline,
    save_artifacts,
)


@pytest.fixture
def fitted_pipeline_and_metadata():
    """Fixture returning a fitted dummy pipeline and sample metadata."""
    config = load_training_config()
    pipeline = build_baseline_pipeline(config)

    df = pd.DataFrame(
        [
            {
                "customerID": "9999-TESTC",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 20.0,
                "TotalCharges": 100.0,
            },
            {
                "customerID": "8888-TESTD",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 50,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 110.0,
                "TotalCharges": 5500.0,
            },
        ]
    )
    pipeline.fit(df, [0, 1])

    metadata = {
        "model_name": "test_model",
        "random_seed": 42,
        "training_metrics": {"train_accuracy": 1.0},
    }

    return pipeline, metadata, df


def test_save_and_load_artifacts(tmp_path, fitted_pipeline_and_metadata):
    """Test saving and reloading model artifacts."""
    pipeline, metadata, df = fitted_pipeline_and_metadata

    pipe_path, meta_path = save_artifacts(
        pipeline=pipeline,
        metadata=metadata,
        output_dir=tmp_path,
        pipeline_filename="test_pipe.joblib",
        metadata_filename="test_meta.json",
    )

    assert pipe_path.is_file()
    assert meta_path.is_file()

    reloaded_pipe, reloaded_meta = load_artifacts(
        output_dir=tmp_path,
        pipeline_filename="test_pipe.joblib",
        metadata_filename="test_meta.json",
    )

    assert reloaded_meta["model_name"] == "test_model"
    assert reloaded_meta["random_seed"] == 42

    orig_preds = pipeline.predict_proba(df)
    reloaded_preds = reloaded_pipe.predict_proba(df)

    import numpy as np

    np.testing.assert_allclose(orig_preds, reloaded_preds)


def test_load_pipeline_file_not_found(tmp_path):
    """Test raising FileNotFoundError for missing pipeline file."""
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "non_existent.joblib")


def test_load_metadata_file_not_found(tmp_path):
    """Test raising FileNotFoundError for missing metadata file."""
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "non_existent.json")
