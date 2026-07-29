"""Unit tests for candidate model training workflow and pipeline builder."""

from pathlib import Path

import numpy as np
import pytest

from churn_prediction.features.pipeline import (
    build_model_pipeline,
    load_training_config,
)
from churn_prediction.models.trainer import train_candidate


def test_build_model_pipeline_supported_types():
    """Verify build_model_pipeline instantiates appropriate scikit-learn classifiers."""
    base_config = load_training_config()

    lr_pipe = build_model_pipeline(base_config)
    assert lr_pipe.named_steps["classifier"].__class__.__name__ == "LogisticRegression"

    gb_config = base_config.copy()
    gb_config["model"] = {
        "type": "GradientBoosting",
        "hyperparameters": {"n_estimators": 10, "random_state": 42},
    }
    gb_pipe = build_model_pipeline(gb_config)
    assert (
        gb_pipe.named_steps["classifier"].__class__.__name__
        == "GradientBoostingClassifier"
    )


def test_build_model_pipeline_unsupported_type_raises():
    """Verify ValueError raised on unknown model type."""
    base_config = load_training_config()
    bad_config = base_config.copy()
    bad_config["model"] = {"type": "UnknownModelType"}
    with pytest.raises(ValueError, match="Unsupported model type"):
        build_model_pipeline(bad_config)


def test_train_candidate_creates_artifacts_and_metadata(synthetic_dataset, tmp_path):
    """Verify train_candidate trains candidate model and persists expected artifacts."""
    csv_path = tmp_path / "sample_data.csv"
    synthetic_dataset.to_csv(csv_path, index=False)
    model_dir = tmp_path / "models"

    pipeline, metadata, artifact_paths = train_candidate(
        config_path=Path("configs/candidate_training.yaml"),
        data_path_override=csv_path,
        log_to_mlflow=False,
        output_dir_override=model_dir,
    )

    assert pipeline is not None
    assert metadata["model_name"] == "candidate_gradient_boosting"
    assert metadata["random_seed"] == 42
    assert "training_metrics" in metadata
    assert Path(artifact_paths["pipeline_path"]).exists()
    assert Path(artifact_paths["metadata_path"]).exists()
    assert model_dir in Path(artifact_paths["pipeline_path"]).parents


def test_candidate_training_deterministic_execution(synthetic_dataset, tmp_path):
    """Verify candidate model execution is fully deterministic."""
    csv_path = tmp_path / "sample_data.csv"
    synthetic_dataset.to_csv(csv_path, index=False)

    pipe1, _, _ = train_candidate(
        config_path=Path("configs/candidate_training.yaml"),
        data_path_override=csv_path,
        log_to_mlflow=False,
        output_dir_override=tmp_path / "run1",
    )
    pipe2, _, _ = train_candidate(
        config_path=Path("configs/candidate_training.yaml"),
        data_path_override=csv_path,
        log_to_mlflow=False,
        output_dir_override=tmp_path / "run2",
    )

    prob1 = pipe1.predict_proba(synthetic_dataset)[:, 1]
    prob2 = pipe2.predict_proba(synthetic_dataset)[:, 1]
    np.testing.assert_allclose(prob1, prob2, rtol=1e-5, atol=1e-5)
