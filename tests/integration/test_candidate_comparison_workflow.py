"""Integration tests for end-to-end candidate comparison workflow."""

from pathlib import Path

from churn_prediction.evaluation.comparator import compare_baseline_and_candidate
from churn_prediction.models.trainer import train_baseline, train_candidate


def test_full_candidate_comparison_integration_workflow(tmp_path):
    """Verify complete end-to-end milestone 10 pipeline execution."""
    data_path = Path("Telco-Customer-Churn.csv")
    if not data_path.is_file():
        return

    # Train Baseline Model
    train_baseline(
        config_path="configs/training.yaml",
        data_path_override=data_path,
        log_to_mlflow=False,
    )

    # Train Candidate Boosted Tree Model
    train_candidate(
        config_path="configs/candidate_training.yaml",
        data_path_override=data_path,
        log_to_mlflow=False,
    )

    # Run Model Comparison & Decision Record Generation
    output_dir = tmp_path / "evaluation_reports"
    decision_record, output_paths = compare_baseline_and_candidate(
        config_path="configs/evaluation.yaml",
        baseline_model_dir="models",
        data_path_override=data_path,
        output_dir=output_dir,
    )

    # Verify decision record schema and contents
    assert decision_record["evaluation_split"] == "test"
    assert decision_record["sample_size"] == 1057

    metrics = decision_record["metrics_summary"]
    assert "baseline" in metrics
    assert "candidate" in metrics
    assert "difference_candidate_minus_baseline" in metrics

    diffs = metrics["difference_candidate_minus_baseline"]
    assert "pr_auc" in diffs
    assert "brier_score" in diffs
    assert "precision_at_capacity" in diffs
    assert "recall_at_capacity" in diffs

    gates = decision_record["acceptance_gates"]
    assert gates["pr_auc_improvement_gate"]["passed"] is True
    assert gates["calibration_integrity_gate"]["passed"] is True
    assert gates["fairness_parity_gate"]["passed"] is True
    assert gates["operational_reproducibility_gate"]["passed"] is True

    final_decision = decision_record["final_decision"]
    assert final_decision["all_gates_passed"] is True
    assert final_decision["selected_model_type"] == "GradientBoosting"

    json_file = Path(output_paths["decision_record_json"])
    md_file = Path(output_paths["decision_record_md"])
    assert json_file.is_file()
    assert md_file.is_file()
    assert md_file.stat().st_size > 500
