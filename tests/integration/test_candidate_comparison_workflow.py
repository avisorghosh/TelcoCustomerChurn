"""Integration tests for end-to-end candidate comparison workflow."""

from pathlib import Path

from churn_prediction.evaluation.comparator import compare_baseline_and_candidate
from churn_prediction.models.trainer import train_baseline, train_candidate


def test_full_candidate_comparison_integration_workflow(tmp_path):
    """Verify complete end-to-end milestone 10 pipeline execution."""
    data_path = Path("Telco-Customer-Churn.csv")
    if not data_path.is_file():
        return

    model_dir = tmp_path / "models"

    train_baseline(
        config_path="configs/training.yaml",
        data_path_override=data_path,
        log_to_mlflow=False,
        output_dir_override=model_dir,
    )
    train_candidate(
        config_path="configs/candidate_training.yaml",
        data_path_override=data_path,
        log_to_mlflow=False,
        output_dir_override=model_dir,
    )

    output_dir = tmp_path / "evaluation_reports"
    decision_record, output_paths = compare_baseline_and_candidate(
        config_path="configs/evaluation.yaml",
        baseline_model_dir=model_dir,
        data_path_override=data_path,
        output_dir=output_dir,
    )

    assert decision_record["selection_split"] == "val"
    assert decision_record["evaluation_split"] == "test"
    assert decision_record["sample_size"] == 1057
    assert decision_record["selection_sample_size"] == 1057

    metrics = decision_record["metrics_summary"]
    assert "baseline" in metrics
    assert "candidate" in metrics
    assert "difference_candidate_minus_baseline" in metrics

    gates = decision_record["acceptance_gates"]
    assert gates["selection_split"] == "val"
    assert "pr_auc_improvement_gate" in gates
    assert "calibration_integrity_gate" in gates
    assert "fairness_parity_gate" in gates
    assert "operational_reproducibility_gate" in gates

    final_decision = decision_record["final_decision"]
    assert final_decision["selected_model_type"] in {
        "GradientBoosting",
        "LogisticRegression",
    }
    assert "LightGBM" not in final_decision["selected_model_type"]

    json_file = Path(output_paths["decision_record_json"])
    md_file = Path(output_paths["decision_record_md"])
    assert json_file.is_file()
    assert md_file.is_file()
    assert md_file.stat().st_size > 500
    assert "LightGBM" not in md_file.read_text(encoding="utf-8")
