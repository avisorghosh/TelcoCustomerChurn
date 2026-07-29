"""Contract tests for API service endpoints."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.app import create_app
from churn_prediction.api.service import InferenceService
from churn_prediction.features.pipeline import (
    build_baseline_pipeline,
    load_training_config,
)
from churn_prediction.models.serialization import save_artifacts
from churn_prediction.models.trainer import prepare_features_and_target


@pytest.fixture
def trained_artifacts(tmp_path: Path, sample_valid_df):
    """Fixture to train baseline pipeline and output artifacts to tmp_path."""
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
        pipeline_filename="serving_pipeline.joblib",
        metadata_filename="serving_metadata.json",
    )
    return tmp_path


def test_health_endpoint_healthy(trained_artifacts):
    """Test GET /health returns status ok when model is loaded."""
    service = InferenceService(model_dir=trained_artifacts)
    app = create_app(service=service)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


def test_health_endpoint_degraded(tmp_path):
    """Test GET /health returns status degraded when model is missing."""
    service = InferenceService(model_dir=tmp_path / "missing_dir")
    app = create_app(service=service)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


def test_ready_endpoint_ready(trained_artifacts):
    """Test GET /ready returns status 200 ready when model is loaded."""
    service = InferenceService(model_dir=trained_artifacts)
    app = create_app(service=service)

    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True


def test_ready_endpoint_not_ready(tmp_path):
    """Test GET /ready returns status 503 not ready when model is missing."""
    service = InferenceService(model_dir=tmp_path / "missing_dir")
    app = create_app(service=service)

    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["model_loaded"] is False
        assert "detail" in data


def test_predict_endpoint_valid_request(trained_artifacts, sample_customer_dict):
    """Test POST /predict returns successful prediction with model version."""
    service = InferenceService(model_dir=trained_artifacts)
    app = create_app(service=service)

    with TestClient(app) as client:
        headers = {"X-Correlation-ID": "test-req-999"}
        response = client.post("/predict", json=sample_customer_dict, headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert "churn_probability" in data
        assert "predicted_class" in data
        assert "model_version" in data
        assert "correlation_id" in data
        assert "prediction_timestamp" in data

        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["predicted_class"] in (0, 1)
        assert data["model_version"] == "1.0.0"
        assert data["correlation_id"] == "test-req-999"


def test_predict_endpoint_auto_generated_correlation_id(
    trained_artifacts, sample_customer_dict
):
    """Test POST /predict generates a correlation ID when header is omitted."""
    service = InferenceService(model_dir=trained_artifacts)
    app = create_app(service=service)

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_customer_dict)
        assert response.status_code == 200
        data = response.json()
        assert data["correlation_id"] is not None
        assert len(data["correlation_id"]) > 0


def test_predict_endpoint_invalid_schema_4xx(trained_artifacts, sample_customer_dict):
    """Test POST /predict with invalid request schema returns HTTP 422."""
    service = InferenceService(model_dir=trained_artifacts)
    app = create_app(service=service)

    invalid_payload = sample_customer_dict.copy()
    invalid_payload["Contract"] = "InvalidContractTerm"

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
        assert "message" in data
        assert "details" in data


def test_predict_endpoint_missing_required_field_4xx(
    trained_artifacts, sample_customer_dict
):
    """Test POST /predict with missing required field returns HTTP 422."""
    service = InferenceService(model_dir=trained_artifacts)
    app = create_app(service=service)

    incomplete_payload = sample_customer_dict.copy()
    incomplete_payload.pop("tenure")

    with TestClient(app) as client:
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 422


def test_predict_endpoint_forbidden_extra_field_4xx(
    trained_artifacts, sample_customer_dict
):
    """Test POST /predict with extra unallowed fields returns HTTP 422."""
    service = InferenceService(model_dir=trained_artifacts)
    app = create_app(service=service)

    extra_payload = sample_customer_dict.copy()
    extra_payload["unauthorized_column"] = 123

    with TestClient(app) as client:
        response = client.post("/predict", json=extra_payload)
        assert response.status_code == 422


def test_predict_endpoint_missing_model_5xx(tmp_path, sample_customer_dict):
    """Test POST /predict when model artifact is missing returns safe HTTP 503."""
    service = InferenceService(model_dir=tmp_path / "missing_dir")
    app = create_app(service=service)

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_customer_dict)
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert "message" in data
        assert "traceback" not in data
        assert "Exception" not in data.get("message", "")


def test_predict_endpoint_corrupted_model_handling(tmp_path, sample_customer_dict):
    """Test POST /predict handling when model artifact file is corrupted."""
    corrupt_dir = tmp_path / "corrupt_dir"
    corrupt_dir.mkdir()
    (corrupt_dir / "serving_pipeline.joblib").write_text("invalid content")
    meta_path = corrupt_dir / "serving_metadata.json"
    meta_path.write_text(json.dumps({"model_name": "test"}))

    service = InferenceService(model_dir=corrupt_dir)
    app = create_app(service=service)

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_customer_dict)
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert "message" in data
