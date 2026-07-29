"""Unit tests for campaign capacity evaluation module."""

import pytest

from churn_prediction.evaluation.capacity import compute_capacity_metrics


def test_compute_capacity_metrics_top_10_percent() -> None:
    """Verify campaign capacity evaluation for top 10% population."""
    # 10 samples, top 10% = 1 sample
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.95]

    res = compute_capacity_metrics(y_true, y_prob, capacity_fraction=0.10)

    assert res["campaign_capacity_fraction"] == 0.10
    assert res["num_targeted_customers"] == 1
    assert res["total_customers"] == 10
    assert res["retained_churners_captured"] == 1
    assert res["total_actual_churners"] == 2
    assert res["precision_at_capacity"] == 1.0
    assert res["recall_at_capacity"] == 0.5
    assert res["capacity_threshold"] == 0.95


def test_compute_capacity_metrics_top_50_percent() -> None:
    """Verify campaign capacity evaluation for top 50% population."""
    # 10 samples, top 50% = 5 samples
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.95]

    res = compute_capacity_metrics(y_true, y_prob, capacity_fraction=0.50)

    assert res["num_targeted_customers"] == 5
    assert res["retained_churners_captured"] == 2
    assert res["precision_at_capacity"] == 0.4
    assert res["recall_at_capacity"] == 1.0
    assert res["capacity_threshold"] == 0.6


def test_compute_capacity_metrics_invalid_fraction() -> None:
    """Verify ValueError is raised for invalid capacity_fraction values."""
    with pytest.raises(ValueError, match="capacity_fraction must be between"):
        compute_capacity_metrics([0, 1], [0.2, 0.8], capacity_fraction=0.0)

    with pytest.raises(ValueError, match="capacity_fraction must be between"):
        compute_capacity_metrics([0, 1], [0.2, 0.8], capacity_fraction=1.5)
