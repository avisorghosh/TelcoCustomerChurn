"""Unit tests for API inference service and schema validation."""

import json
from pathlib import Path

import pydantic
import pytest

from churn_prediction.api.config import load_serving_config
from churn_prediction.api.schemas import CustomerRecord, PredictionResponse
from churn_prediction.api.service import (
    InferenceService,
    ModelNotLoadedError,
)
from churn_prediction.features.pipeline import load_training_config
from churn_prediction.models.serialization import save_artifacts
from churn_prediction.models.trainer import (
    build_baseline_pipeline,
    prepare_features_and_target,
)


@pytest.fixture
def trained_artifacts_dir(tmp_path: Path, sample_valid_df):
    """Fixture to train baseline pipeline on sample_valid_df and save artifacts."""
    config = load_training_config()
    X, y = prepare_features_and_target(sample_valid_df, config)
    pipeline = build_baseline_pipeline(config)
    pipeline.fit(X, y)

    metadata = {
        "model_name": "baseline_logistic_regression",
        "schema_version": "1.0.0",
        "timestamp": "2026-07-29T12:00:00Z",
    }

    save_artifacts(
        pipeline=pipeline,
        metadata=metadata,
        output_dir=tmp_path,
        pipeline_filename="baseline_pipeline.joblib",
        metadata_filename="baseline_metadata.json",
    )
    return tmp_path


def test_customer_record_valid(sample_customer_dict):
    """Test that a valid customer dictionary parses successfully."""
    record = CustomerRecord(**sample_customer_dict)
    assert record.customerID == "7590-VHVEG"
    assert record.gender == "Female"
    assert record.tenure == 1
    assert record.MonthlyCharges == 29.85


def test_customer_record_missing_required_field(sample_customer_dict):
    """Test that missing required fields raise Pydantic ValidationError."""
    incomplete_dict = sample_customer_dict.copy()
    incomplete_dict.pop("MonthlyCharges")
    with pytest.raises(pydantic.ValidationError):
        CustomerRecord(**incomplete_dict)


def test_customer_record_invalid_categorical(sample_customer_dict):
    """Test that invalid categorical values raise ValidationError."""
    bad_dict = sample_customer_dict.copy()
    bad_dict["gender"] = "Other"
    with pytest.raises(pydantic.ValidationError):
        CustomerRecord(**bad_dict)


def test_customer_record_negative_tenure(sample_customer_dict):
    """Test that negative tenure raises ValidationError."""
    bad_dict = sample_customer_dict.copy()
    bad_dict["tenure"] = -5
    with pytest.raises(pydantic.ValidationError):
        CustomerRecord(**bad_dict)


def test_customer_record_forbidden_extra_field(sample_customer_dict):
    """Test that extra unrecognised fields raise ValidationError."""
    bad_dict = sample_customer_dict.copy()
    bad_dict["unsupported_extra_field"] = "value"
    with pytest.raises(pydantic.ValidationError):
        CustomerRecord(**bad_dict)


def test_customer_record_null_total_charges(sample_customer_dict):
    """Test that TotalCharges can be None for new customers."""
    null_charges_dict = sample_customer_dict.copy()
    null_charges_dict["TotalCharges"] = None
    record = CustomerRecord(**null_charges_dict)
    assert record.TotalCharges is None


def test_inference_service_unloaded_state():
    """Test InferenceService initial state when model is not loaded."""
    service = InferenceService(model_dir="non_existent_dir_12345")
    assert not service.is_loaded
    assert service.pipeline is None

    dummy_record = CustomerRecord(
        customerID="1234-TEST",
        gender="Male",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=10,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="No",
        PaymentMethod="Mailed check",
        MonthlyCharges=50.0,
        TotalCharges=500.0,
    )
    with pytest.raises(ModelNotLoadedError):
        service.predict(dummy_record)


def test_inference_service_load_model_failure(tmp_path):
    """Test load_model failure handling when files are missing."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    service = InferenceService(model_dir=empty_dir)

    success = service.load_model()
    assert not success
    assert not service.is_loaded
    assert service.load_error is not None
    assert "missing" in service.load_error.lower()


def test_inference_service_corrupted_model_failure(tmp_path):
    """Test load_model failure when joblib artifact is corrupted."""
    corrupt_dir = tmp_path / "corrupt_dir"
    corrupt_dir.mkdir()

    pipeline_file = corrupt_dir / "baseline_pipeline.joblib"
    pipeline_file.write_text("corrupted content not joblib")

    metadata_file = corrupt_dir / "baseline_metadata.json"
    metadata_file.write_text(json.dumps({"model_name": "baseline"}))

    service = InferenceService(model_dir=corrupt_dir)
    success = service.load_model()

    assert not success
    assert not service.is_loaded
    assert service.load_error is not None


def test_inference_service_successful_prediction(
    trained_artifacts_dir, sample_customer_dict
):
    """Test successful model loading and prediction execution."""
    service = InferenceService(model_dir=trained_artifacts_dir)
    loaded = service.load_model()

    assert loaded
    assert service.is_loaded
    assert service.model_version == "1.0.0"

    record = CustomerRecord(**sample_customer_dict)
    response = service.predict(record, correlation_id="test-corr-id-123")

    assert isinstance(response, PredictionResponse)
    assert 0.0 <= response.churn_probability <= 1.0
    assert response.predicted_class in (0, 1)
    assert response.model_version == "1.0.0"
    assert response.correlation_id == "test-corr-id-123"
    assert response.prediction_timestamp is not None


def test_load_serving_config_fallback(tmp_path):
    """Test load_serving_config fallback when config path is non-existent."""
    config = load_serving_config(tmp_path / "non_existent_config.yaml")
    assert "schema_version" in config
    assert "api" in config
    assert config["api"]["port"] == 8000
