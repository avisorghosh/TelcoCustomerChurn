"""Integration test for MLflow experiment tracking and local model registry workflow."""

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from churn_prediction.features.pipeline import load_training_config
from churn_prediction.models.trainer import predict_churn, train_baseline
from churn_prediction.tracking.tracker import (
    load_registered_model,
    score_with_registered_model,
    setup_mlflow,
)


def test_full_mlflow_training_and_registry_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify full workflow: train model, log to MLflow, register, reload, and score."""
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    exp_name = "integration_test_churn_exp"
    model_name = "integration_test_churn_model"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", exp_name)
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", model_name)

    config = load_training_config()
    config["mlflow"] = {
        "enabled": True,
        "tracking_uri": tracking_uri,
        "experiment_name": exp_name,
        "registered_model_name": model_name,
    }

    # 1. Train baseline pipeline with MLflow tracking enabled
    pipeline, _metadata, _artifact_paths = train_baseline(
        config_path=None,
        data_path_override="Telco-Customer-Churn.csv",
        log_to_mlflow=True,
        output_dir_override=tmp_path / "models",
    )

    # 2. Verify MLflow run creation and lineage
    setup_mlflow(config)
    exp = mlflow.get_experiment_by_name(exp_name)
    assert exp is not None

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 1
    run = runs[0]

    # Verify recorded parameters, tags, and metrics
    assert "data.sha256" in run.data.params
    assert run.data.params.get("random_seed") == "42"
    assert "git_commit" in run.data.tags
    assert "train_accuracy" in run.data.metrics

    # 3. Verify Model Registry
    versions = client.search_model_versions(f"name='{model_name}'")
    assert len(versions) >= 1
    latest_version = max(int(v.version) for v in versions)

    # 4. Load registered model from MLflow registry
    loaded_pipeline = load_registered_model(
        model_name=model_name,
        version=latest_version,
        config=config,
    )
    assert loaded_pipeline is not None

    # 5. Score sample record using loaded registered model
    sample_df = pd.read_csv("Telco-Customer-Churn.csv").head(5)
    direct_preds = predict_churn(pipeline, sample_df)
    registry_preds = score_with_registered_model(
        df=sample_df,
        model_name=model_name,
        version=latest_version,
        config=config,
    )

    pd.testing.assert_series_equal(
        direct_preds["churn_prediction"], registry_preds["churn_prediction"]
    )
    pd.testing.assert_series_equal(
        direct_preds["churn_probability"], registry_preds["churn_probability"]
    )
