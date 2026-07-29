"""Integration tests for model registration, rollback, and restoration workflows."""

from churn_prediction.api.service import InferenceService
from churn_prediction.models.restore import restore_model_from_registry
from churn_prediction.models.trainer import train_baseline, train_candidate


def test_end_to_end_model_rollback_integration(
    tmp_path, synthetic_dataset, monkeypatch
):
    """Verify training version 1, candidate version 2, and rolling back to version 1."""
    raw_csv = tmp_path / "Telco-Customer-Churn.csv"
    synthetic_dataset.to_csv(raw_csv, index=False)

    mlruns_dir = tmp_path / "mlruns"
    active_models_dir = tmp_path / "active_models"
    registered_model_name = "test_rollback_model"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{mlruns_dir}")
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", registered_model_name)

    # 1. Train version 1 (Baseline)
    _, meta_v1, _ = train_baseline(
        data_path_override=raw_csv,
        log_to_mlflow=True,
        output_dir_override=tmp_path / "baseline_artifacts",
    )
    assert meta_v1["model_name"] == "baseline_logistic_regression"

    # 2. Train version 2 (Candidate Gradient Boosting)
    _, meta_v2, _ = train_candidate(
        data_path_override=raw_csv,
        log_to_mlflow=True,
        output_dir_override=tmp_path / "candidate_artifacts",
    )
    assert meta_v2["model_name"] == "candidate_gradient_boosting"

    # 3. Roll back to MLflow version 1
    pipe_p, meta_p, restored_meta = restore_model_from_registry(
        version=1,
        registered_model_name=registered_model_name,
        target_dir=active_models_dir,
    )

    assert pipe_p.exists()
    assert meta_p.exists()
    assert restored_meta["restored_from_version"] == "1"

    # 4. Verify InferenceService loads restored version 1 model
    service = InferenceService(
        model_dir=active_models_dir,
        pipeline_filename="baseline_pipeline.joblib",
        metadata_filename="baseline_metadata.json",
    )
    assert service.load_model() is True
    assert service.metadata.get("restored_from_version") == "1"
