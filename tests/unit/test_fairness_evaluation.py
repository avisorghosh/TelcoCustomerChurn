"""Unit tests for lightweight fairness review module."""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.evaluation.fairness import evaluate_fairness_review


def test_evaluate_fairness_review_basic():
    """Verify fairness review calculates subgroup and parity metrics."""
    df = pd.DataFrame(
        {
            "gender": ["Female", "Female", "Male", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1, 0, 0],
        }
    )

    y_true = np.array([1, 0, 1, 0, 0, 0])
    y_prob = np.array([0.7, 0.2, 0.8, 0.3, 0.1, 0.4])

    res = evaluate_fairness_review(df, y_true, y_prob, threshold=0.5)

    assert "gender" in res
    assert "SeniorCitizen" in res

    gender_res = res["gender"]
    assert "subgroups" in gender_res
    assert "Female" in gender_res["subgroups"]
    assert "Male" in gender_res["subgroups"]

    female_stats = gender_res["subgroups"]["Female"]
    assert female_stats["count"] == 3
    assert female_stats["churn_count"] == 1
    assert "selection_rate" in female_stats
    assert "tpr" in female_stats
    assert "fpr" in female_stats

    # Check parity metrics
    disp_metrics = gender_res["disparity_metrics"]
    assert "demographic_parity_difference" in disp_metrics
    assert "demographic_parity_ratio" in disp_metrics
    assert "equalized_odds_tpr_difference" in disp_metrics
    assert "equalized_odds_fpr_difference" in disp_metrics


def test_evaluate_fairness_review_shape_mismatch_raises():
    """Verify ValueError raised on mismatched DataFrame length."""
    df = pd.DataFrame({"gender": ["Female", "Male"]})
    y_true = np.array([1])
    y_prob = np.array([0.9])

    with pytest.raises(ValueError, match="identical lengths"):
        evaluate_fairness_review(df, y_true, y_prob)
