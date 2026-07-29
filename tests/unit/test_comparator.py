"""Unit tests for candidate comparison and Decision Record generation."""

import json
from pathlib import Path

import pytest

from churn_prediction.evaluation.comparator import compare_baseline_and_candidate
from churn_prediction.models.trainer import train_baseline, train_candidate


def test_compare_baseline_and_candidate_workflow(synthetic_dataset, tmp_path):
    """Verify compare_baseline_and_candidate executes comparison."""
    data_path = tmp_path / "test_churn_data.csv"
    synthetic_dataset.to_csv(data_path, index=False)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train baseline and save into tmp_path/models
    base_config_path = Path("configs/training.yaml")
    train_baseline(
        config_path=base_config_path,
        data_path_override=data_path,
        log_to_mlflow=False,
    )

    # 2. Train candidate and save into tmp_path/models
    cand_config_path = Path("configs/candidate_training.yaml")
    train_candidate(
        config_path=cand_config_path,
        data_path_override=data_path,
        log_to_mlflow=False,
    )

    # 3. Run comparison
    reports_dir = tmp_path / "reports"
    record, output_paths = compare_baseline_and_candidate(
        config_path="configs/evaluation.yaml",
        baseline_model_dir="models",
        data_path_override=data_path,
        output_dir=reports_dir,
    )

    assert "title" in record
    assert "models_compared" in record
    assert len(record["models_compared"]) == 2
    assert "metrics_summary" in record
    assert "acceptance_gates" in record
    assert "final_decision" in record

    assert Path(output_paths["decision_record_json"]).exists()
    assert Path(output_paths["decision_record_md"]).exists()

    with open(output_paths["decision_record_json"], "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["title"] == record["title"]


def test_compare_baseline_and_candidate_missing_model_raises(tmp_path):
    """Verify FileNotFoundError raised if model file is missing."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        compare_baseline_and_candidate(
            baseline_model_dir=empty_dir,
            baseline_pipeline_filename="non_existent_baseline.joblib",
        )
