"""Unit tests for MLflow experiment tracking and local model registry utilities."""

from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_prediction.tracking.tracker import (
    create_and_log_data_manifest,
    create_model_signature,
    get_git_revision,
    load_registered_model,
    log_experiment_run,
    setup_mlflow,
)


@pytest.fixture
def dummy_pipeline() -> Pipeline:
    """Create a fitted simple dummy pipeline for testing."""
    X = pd.DataFrame(
        {"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [4.0, 3.0, 2.0, 1.0]}
    )
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def tmp_mlflow_dir(tmp_path: Path) -> str:
    """Create a temporary tracking URI for testing."""
    mlruns_path = tmp_path / "mlruns"
    return f"file:{mlruns_path}"


def test_get_git_revision_success():
    """Test get_git_revision returns a valid commit hash or string."""
    rev = get_git_revision()
    assert isinstance(rev, str)
    assert len(rev) > 0


def test_get_git_revision_failure():
    """Test get_git_revision returns 'unknown' when git command fails."""
    with patch("subprocess.run", side_effect=Exception("Git not found")):
        rev = get_git_revision()
        assert rev == "unknown"


def test_setup_mlflow_defaults(tmp_mlflow_dir: str, monkeypatch: pytest.MonkeyPatch):
    """Test setup_mlflow configures default tracking URI and experiment."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_mlflow_dir)
    config = {
        "mlflow": {
            "tracking_uri": tmp_mlflow_dir,
            "experiment_name": "test_exp",
            "registered_model_name": "test_model",
        }
    }
    settings = setup_mlflow(config)
    assert settings["tracking_uri"] == tmp_mlflow_dir
    assert settings["experiment_name"] == "test_exp"
    assert settings["registered_model_name"] == "test_model"

    active_exp = mlflow.get_experiment_by_name("test_exp")
    assert active_exp is not None
    assert active_exp.name == "test_exp"


def test_create_model_signature(dummy_pipeline: Pipeline):
    """Test create_model_signature extracts valid input/output schemas."""
    sample_df = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]})
    sig = create_model_signature(dummy_pipeline, sample_df)
    assert sig is not None
    assert sig.inputs is not None
    assert sig.outputs is not None


def test_create_and_log_data_manifest(tmp_path: Path, tmp_mlflow_dir: str):
    """Test data manifest creation and MLflow parameter logging."""
    mlflow.set_tracking_uri(tmp_mlflow_dir)
    mlflow.set_experiment("test_manifest_exp")

    data_file = tmp_path / "sample.csv"
    data_file.write_text("col1,col2\n1,2\n3,4\n", encoding="utf-8")

    with mlflow.start_run() as run:
        manifest = create_and_log_data_manifest(data_file, schema_version="1.0.0")
        assert manifest.row_count == 2
        assert len(manifest.file_sha256) == 64

        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run.info.run_id).data
        assert run_data.params.get("data.schema_version") == "1.0.0"
        assert run_data.metrics.get("data.row_count") == 2.0
        assert run_data.tags.get("data_checksum") == manifest.file_sha256


def test_log_experiment_run(
    dummy_pipeline: Pipeline, tmp_path: Path, tmp_mlflow_dir: str
):
    """Test log_experiment_run records tags, parameters, metrics, model,
    and artifacts.
    """
    config = {
        "mlflow": {
            "tracking_uri": tmp_mlflow_dir,
            "experiment_name": "test_run_exp",
            "registered_model_name": "test_run_model",
        },
        "model": {"type": "LogisticRegression", "hyperparameters": {"C": 1.0}},
        "split": {
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "random_seed": 42,
            "stratify": True,
        },
        "features": {"target_column": "Churn"},
    }

    metadata = {
        "model_name": "baseline_test",
        "schema_version": "1.0.0",
        "random_seed": 42,
        "split_counts": {"train": 10, "val": 2, "test": 2, "total": 14},
        "training_metrics": {"train_accuracy": 0.9, "val_accuracy": 0.85},
        "timestamp": "2026-07-29T20:00:00Z",
    }

    dummy_artifact = tmp_path / "metadata.json"
    dummy_artifact.write_text("{}", encoding="utf-8")
    artifact_paths = {"metadata_path": str(dummy_artifact)}

    sample_df = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]})

    run = log_experiment_run(
        pipeline=dummy_pipeline,
        metadata=metadata,
        config=config,
        sample_df=sample_df,
        artifact_paths=artifact_paths,
        register_model=True,
    )

    assert run is not None
    client = mlflow.tracking.MlflowClient()
    fetched_run = client.get_run(run.info.run_id)

    assert fetched_run.data.params.get("random_seed") == "42"
    assert fetched_run.data.params.get("hyperparameters.C") == "1.0"
    assert fetched_run.data.metrics.get("train_accuracy") == 0.9
    assert fetched_run.data.tags.get("model_name") == "baseline_test"

    # Verify model registration
    registered_models = client.search_model_versions("name='test_run_model'")
    assert len(registered_models) > 0


def test_load_registered_model_missing_raises(tmp_mlflow_dir: str):
    """Test load_registered_model raises ValueError when model is not found."""
    config = {
        "mlflow": {
            "tracking_uri": tmp_mlflow_dir,
            "experiment_name": "non_existent_exp",
            "registered_model_name": "non_existent_model",
        }
    }
    with pytest.raises(ValueError, match="No registered versions found"):
        load_registered_model(config=config)
