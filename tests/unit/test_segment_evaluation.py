"""Unit tests for customer segment evaluation module."""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.evaluation.segment import evaluate_segment_performance


def test_evaluate_segment_performance_basic():
    """Verify segment evaluation returns expected dimensions and metrics."""
    df = pd.DataFrame(
        {
            "tenure": [5, 15, 30, 60, 10, 50],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "No"],
            "MonthlyCharges": [25.0, 55.0, 85.0, 40.0, 95.0, 20.0],
        }
    )

    y_true = np.array([1, 0, 0, 1, 0, 0])
    y_prob = np.array([0.8, 0.2, 0.1, 0.7, 0.3, 0.1])

    res = evaluate_segment_performance(df, y_true, y_prob, capacity_threshold=0.5)

    assert "tenure_band" in res
    assert "Contract" in res
    assert "InternetService" in res
    assert "monthly_charges_band" in res

    # Verify subgroup contents for Contract
    m2m_stats = res["Contract"]["Month-to-month"]
    assert m2m_stats["count"] == 2
    assert m2m_stats["churn_count"] == 2
    assert m2m_stats["churn_rate"] == 1.0
    assert "mean_predicted_probability" in m2m_stats


def test_evaluate_segment_performance_shape_mismatch_raises():
    """Verify ValueError raised when array lengths mismatch."""
    df = pd.DataFrame({"tenure": [10, 20]})
    y_true = np.array([1, 0, 0])
    y_prob = np.array([0.8, 0.2])

    with pytest.raises(ValueError, match="identical lengths"):
        evaluate_segment_performance(df, y_true, y_prob)
