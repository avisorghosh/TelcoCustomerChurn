"""Integration tests for post-deployment evaluation workflow."""

import json
from pathlib import Path

import pandas as pd

from churn_prediction.post_deployment import run_post_deployment_evaluation


def test_end_to_end_post_deployment_evaluation(tmp_path: Path) -> None:
    """Test full end-to-end post-deployment evaluation workflow execution."""
    preds_file = tmp_path / "batch_predictions.csv"
    labels_file = tmp_path / "delayed_labels.csv"
    output_dir = tmp_path / "post_deployment_reports"

    # 1. Create mock predictions CSV
    probs = [0.10 + (i * 0.015) for i in range(50)]
    classes = [0 if p < 0.50 else 1 for p in probs]
    preds_df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(50)],
            "churn_probability": probs,
            "predicted_class": classes,
            "decision_threshold": [0.50] * 50,
            "model_version": ["1.0.0"] * 50,
            "scoring_timestamp": ["2026-07-29T10:00:00+00:00"] * 50,
            "batch_id": ["batch-test"] * 50,
        }
    )
    preds_df.to_csv(preds_file, index=False)

    # 2. Create mock delayed labels CSV
    treatments = ["treatment" if i % 2 == 0 else "control" for i in range(50)]
    labels_df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(50)],
            "observed_churn": [1 if i % 3 == 0 else 0 for i in range(50)],
            "treatment_group": treatments,
            "observation_date": ["2026-07-29"] * 50,
        }
    )
    labels_df.to_csv(labels_file, index=False)

    # 3. Execute post-deployment evaluation orchestrator
    summary, artifacts = run_post_deployment_evaluation(
        predictions_path_override=preds_file,
        delayed_labels_path_override=labels_file,
        output_dir_override=output_dir,
    )

    # 4. Verify summary output structure
    assert summary["matching_statistics"]["matched_records"] == 50
    assert summary["matching_statistics"]["match_rate"] == 1.0
    assert "pr_auc" in summary["model_performance"]["metrics"]
    assert summary["campaign_effectiveness"]["campaign_data_available"] is True
    assert "retrain_recommended" in summary["retraining_decision"]
    assert "policy_note" in summary["retraining_decision"]

    # 5. Verify generated report artifacts exist on disk
    json_path = artifacts["report_summary_json"]
    md_path = artifacts["report_markdown"]

    assert Path(json_path).is_file()
    assert Path(md_path).is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    assert json_data["matching_statistics"]["matched_records"] == 50
    assert json_data["retraining_decision"]["decision_code"] in [
        "MAINTAIN_INCUMBENT",
        "RETRAIN_RECOMMENDED",
    ]

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "# Post-Deployment Learning & Model Performance Review" in md_text
    assert "Retraining Decision:" in md_text
    assert "Governance Policy Note:" in md_text
