"""Integration tests for observability, metrics, quality, and drift reporting."""

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from churn_prediction.api.app import create_app
from churn_prediction.api.service import InferenceService
from churn_prediction.features.pipeline import (
    build_baseline_pipeline,
    load_training_config,
)
from churn_prediction.models.batch_scoring import run_batch_scoring


def test_api_metrics_endpoint_and_privacy_logging(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    """Test API /metrics endpoint, structured logging, and prediction serving."""
    training_config = load_training_config()
    X = sample_valid_df.drop(columns=["customerID", "Churn"])
    y = (sample_valid_df["Churn"] == "Yes").astype(int)
    pipeline = build_baseline_pipeline(training_config)
    pipeline.fit(X, y)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "baseline_pipeline.joblib")

    metadata = {"schema_version": "1.0.0", "model_name": "baseline_logistic_regression"}
    with open(model_dir / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    service = InferenceService(
        model_dir=model_dir,
        pipeline_filename="baseline_pipeline.joblib",
        metadata_filename="baseline_metadata.json",
    )
    app = create_app(service=service)

    with TestClient(app) as client:
        # Check health
        resp = client.get("/health")
        assert resp.status_code == 200

        # Check /metrics endpoint
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "text/plain" in metrics_resp.headers["content-type"]
        metrics_content = metrics_resp.text
        assert "telco_churn_api_requests_total" in metrics_content

        # Predict single customer
        record_payload = {
            "customerID": "7590-VHVEG",
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
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 55.85,
            "TotalCharges": 670.20,
        }

        pred_resp = client.post(
            "/predict",
            json=record_payload,
            headers={"X-Correlation-ID": "test-corr-999"},
        )
        assert pred_resp.status_code == 200
        data = pred_resp.json()
        assert "churn_probability" in data
        assert data["correlation_id"] == "test-corr-999"

        # Check updated metrics
        updated_metrics = client.get("/metrics").text
        assert "telco_churn_predictions_total" in updated_metrics


def test_batch_scoring_observability_integration(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    """Test batch scoring execution produces quality report and records metrics."""
    training_config = load_training_config()
    X = sample_valid_df.drop(columns=["customerID", "Churn"])
    y = (sample_valid_df["Churn"] == "Yes").astype(int)
    pipeline = build_baseline_pipeline(training_config)
    pipeline.fit(X, y)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "baseline_pipeline.joblib")

    metadata = {"schema_version": "1.0.0", "model_name": "baseline_logistic_regression"}
    with open(model_dir / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    inp_csv = tmp_path / "batch_input.csv"
    sample_valid_df.to_csv(inp_csv, index=False)

    config_dict = {
        "model": {
            "model_dir": str(model_dir),
            "pipeline_filename": "baseline_pipeline.joblib",
            "metadata_filename": "baseline_metadata.json",
        },
        "scoring": {"decision_threshold": 0.50, "batch_id": "test_batch_obs"},
        "output": {
            "output_dir": str(tmp_path),
            "output_filename": "output_predictions.csv",
        },
    }
    cfg_file = tmp_path / "serving.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    scored_df, saved_path = run_batch_scoring(input_path=inp_csv, config_path=cfg_file)

    assert len(scored_df) == len(sample_valid_df)
    assert saved_path.is_file()
