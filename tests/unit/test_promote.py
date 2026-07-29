"""Unit tests for model promotion helpers."""

import json

import pytest
import yaml

from churn_prediction.models.promote import PromotionError, promote_selected_model
from churn_prediction.models.trainer import train_baseline, train_candidate


def test_promote_selected_candidate_copies_serving_artifacts(
    tmp_path, synthetic_dataset
):
    """Promote selected candidate into serving artifact filenames."""
    csv_path = tmp_path / "data.csv"
    synthetic_dataset.to_csv(csv_path, index=False)
    model_dir = tmp_path / "models"

    train_baseline(
        data_path_override=csv_path,
        log_to_mlflow=False,
        output_dir_override=model_dir,
    )
    train_candidate(
        data_path_override=csv_path,
        log_to_mlflow=False,
        output_dir_override=model_dir,
    )

    decision = {
        "final_decision": {
            "selected_model_name": "candidate_gradient_boosting",
            "selected_model_type": "GradientBoosting",
            "all_gates_passed": True,
        }
    }
    decision_path = tmp_path / "decision_record.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    serving_config = {
        "schema_version": "1.0.0",
        "model_name": "baseline_logistic_regression",
        "model": {
            "model_dir": str(model_dir),
            "pipeline_filename": "serving_pipeline.joblib",
            "metadata_filename": "serving_metadata.json",
        },
    }
    serving_path = tmp_path / "serving.yaml"
    serving_path.write_text(yaml.safe_dump(serving_config), encoding="utf-8")

    paths = promote_selected_model(
        decision_record_path=decision_path,
        model_dir=model_dir,
        serving_config_path=serving_path,
    )

    assert paths["pipeline_path"].is_file()
    assert paths["metadata_path"].is_file()

    updated = yaml.safe_load(serving_path.read_text(encoding="utf-8"))
    assert updated["model_name"] == "candidate_gradient_boosting"
    assert updated["model"]["pipeline_filename"] == "serving_pipeline.joblib"


def test_promote_missing_decision_record_raises(tmp_path):
    """Promotion fails clearly when decision record is missing."""
    with pytest.raises(PromotionError, match="Decision record not found"):
        promote_selected_model(
            decision_record_path=tmp_path / "missing.json",
            model_dir=tmp_path / "models",
            serving_config_path=None,
            update_serving_config=False,
        )
