"""Unit tests for post-deployment evaluation modules."""

import json

import pandas as pd
import pytest

from churn_prediction.post_deployment import (
    evaluate_campaign_effectiveness,
    evaluate_delayed_predictions,
    generate_synthetic_delayed_labels,
    load_and_match_delayed_labels,
    make_retraining_decision,
)


@pytest.fixture
def sample_predictions_df() -> pd.DataFrame:
    """Fixture providing sample historical batch predictions."""
    return pd.DataFrame(
        {
            "customerID": ["CUST-001", "CUST-002", "CUST-003", "CUST-004", "CUST-005"],
            "churn_probability": [0.85, 0.15, 0.70, 0.30, 0.90],
            "predicted_class": [1, 0, 1, 0, 1],
            "decision_threshold": [0.50] * 5,
            "model_version": ["1.0.0"] * 5,
            "scoring_timestamp": ["2026-07-29T10:00:00+00:00"] * 5,
            "batch_id": ["batch-001"] * 5,
        }
    )


@pytest.fixture
def sample_delayed_labels_df() -> pd.DataFrame:
    """Fixture providing sample delayed ground-truth labels."""
    return pd.DataFrame(
        {
            "customerID": ["CUST-001", "CUST-002", "CUST-003", "CUST-004", "CUST-005"],
            "observed_churn": ["Yes", "No", "Yes", "No", "No"],
            "treatment_group": [
                "treatment",
                "control",
                "treatment",
                "control",
                "treatment",
            ],
            "observation_date": ["2026-07-29"] * 5,
        }
    )


def test_load_and_match_delayed_labels_success(
    sample_predictions_df: pd.DataFrame, sample_delayed_labels_df: pd.DataFrame
) -> None:
    """Test matching historical predictions with delayed labels on customer key."""
    matched_df, stats = load_and_match_delayed_labels(
        sample_predictions_df, sample_delayed_labels_df
    )

    assert len(matched_df) == 5
    assert stats["matched_records"] == 5
    assert stats["match_rate"] == 1.0
    assert "observed_churn" in matched_df.columns
    assert matched_df["observed_churn"].tolist() == [1, 0, 1, 0, 0]


def test_load_and_match_handles_unmatched_records(
    sample_predictions_df: pd.DataFrame,
) -> None:
    """Test matching with partial customer ID overlap."""
    partial_labels = pd.DataFrame(
        {
            "customerID": ["CUST-001", "CUST-002", "CUST-999"],
            "observed_churn": [1, 0, 1],
        }
    )

    matched_df, stats = load_and_match_delayed_labels(
        sample_predictions_df, partial_labels
    )

    assert len(matched_df) == 2
    assert stats["matched_records"] == 2
    assert stats["total_historical_predictions"] == 5
    assert stats["unmatched_predictions"] == 3
    assert stats["match_rate"] == 0.40


def test_evaluate_delayed_predictions_metrics(
    sample_predictions_df: pd.DataFrame, sample_delayed_labels_df: pd.DataFrame
) -> None:
    """Test computing evaluation metrics on matched predictions and delayed labels."""
    matched_df, _ = load_and_match_delayed_labels(
        sample_predictions_df, sample_delayed_labels_df
    )

    result = evaluate_delayed_predictions(matched_df, threshold=0.50)

    assert result["sample_size"] == 5
    assert result["observed_churn_prevalence"] == 0.40
    assert "pr_auc" in result["metrics"]
    assert "roc_auc" in result["metrics"]
    assert "brier_score" in result["metrics"]
    assert "capacity_metrics" in result
    assert "calibration_metrics" in result


def test_evaluate_campaign_effectiveness_treatment_vs_control() -> None:
    """Test campaign outcome evaluation between Treatment and Control groups."""
    # 20% treatment churn rate, 50% control churn rate
    churn_outcomes = [0] * 40 + [1] * 10 + [0] * 25 + [1] * 25
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(100)],
            "observed_churn": churn_outcomes,
            "treatment_group": ["treatment"] * 50 + ["control"] * 50,
        }
    )

    result = evaluate_campaign_effectiveness(
        df, unit_contact_cost=10.0, unit_customer_value=500.0
    )

    assert result["campaign_data_available"] is True
    assert result["treatment_group"]["churn_rate"] == 0.20
    assert result["control_group"]["churn_rate"] == 0.50

    eff = result["effectiveness"]
    assert eff["absolute_churn_reduction"] == 0.30
    assert eff["relative_churn_reduction_pct"] == 60.0
    assert eff["estimated_incremental_retained_customers"] == 15.0

    fin = result["financial_summary"]
    assert fin["total_campaign_cost_usd"] == 500.0
    assert fin["gross_retained_margin_usd"] == 7500.0
    assert fin["net_campaign_value_usd"] == 7000.0
    assert fin["campaign_roi_percent"] == 1400.0


def test_evaluate_campaign_effectiveness_missing_column() -> None:
    """Test campaign evaluation when treatment column is missing."""
    df = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "observed_churn": [1, 0],
        }
    )

    result = evaluate_campaign_effectiveness(df)
    assert result["campaign_data_available"] is False
    assert "assumptions_documented" in result


def test_make_retraining_decision_maintain_incumbent() -> None:
    """Test retraining decision engine when performance remains strong."""
    summary = {
        "metrics": {"pr_auc": 0.65, "brier_score": 0.12},
    }
    decision = make_retraining_decision(
        delayed_eval_summary=summary,
        matching_stats={"match_rate": 1.0},
    )

    assert decision["retrain_recommended"] is False
    assert decision["decision_code"] == "MAINTAIN_INCUMBENT"
    assert "policy_note" in decision


def test_make_retraining_decision_pr_auc_degradation(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Test retraining decision trigger on PR-AUC drop."""
    base_metrics_file = tmp_path / "baseline_metrics.json"  # type: ignore[operator]
    with open(base_metrics_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": {"pr_auc": 0.75}}, f)

    summary = {
        "metrics": {"pr_auc": 0.60, "brier_score": 0.12},
    }

    decision = make_retraining_decision(
        delayed_eval_summary=summary,
        baseline_metrics_path=base_metrics_file,
    )

    assert decision["retrain_recommended"] is True
    assert decision["decision_code"] == "RETRAIN_RECOMMENDED"
    assert "PR_AUC_DEGRADATION" in decision["evidence"]["triggers_fired"]


def test_make_retraining_decision_feature_drift(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Test retraining decision trigger on feature drift alert."""
    drift_file = tmp_path / "drift_report.json"  # type: ignore[operator]
    with open(drift_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {"max_psi": 0.35, "significant_drift_count": 2},
                "feature_drift": {"Contract": {"psi_alert": True}},
            },
            f,
        )

    summary = {
        "metrics": {"pr_auc": 0.64, "brier_score": 0.22},
    }

    decision = make_retraining_decision(
        delayed_eval_summary=summary,
        drift_report_path=drift_file,
    )

    assert decision["retrain_recommended"] is True
    assert "FEATURE_DRIFT_ALERT" in decision["evidence"]["triggers_fired"]


def test_generate_synthetic_delayed_labels(
    sample_predictions_df: pd.DataFrame,
) -> None:
    """Test generation of synthetic delayed labels for local testing."""
    synth_df = generate_synthetic_delayed_labels(sample_predictions_df, seed=123)

    assert len(synth_df) == len(sample_predictions_df)
    assert set(synth_df.columns) == {
        "customerID",
        "observed_churn",
        "treatment_group",
        "observation_date",
    }
    assert synth_df["observed_churn"].isin([0, 1]).all()
