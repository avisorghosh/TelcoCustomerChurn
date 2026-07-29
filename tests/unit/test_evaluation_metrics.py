"""Unit tests for binary classification metrics calculation."""

import pytest

from churn_prediction.evaluation.metrics import (
    compute_binary_classification_metrics,
)


def test_compute_binary_classification_metrics_correctness() -> None:
    """Verify calculated metrics for known binary classification outputs."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.8]

    res = compute_binary_classification_metrics(y_true, y_prob, threshold=0.5)

    assert "pr_auc" in res
    assert "roc_auc" in res
    assert "precision" in res
    assert "recall" in res
    assert "f1_score" in res
    assert "brier_score" in res
    assert "confusion_matrix" in res

    assert res["roc_auc"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_custom_decision_threshold() -> None:
    """Verify decision threshold affects precision, recall, and predictions."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.8]

    # At threshold 0.35, prediction for index 1 becomes 1 (FP)
    res_low = compute_binary_classification_metrics(y_true, y_prob, threshold=0.35)
    assert res_low["threshold"] == 0.35
    assert res_low["confusion_matrix"]["fp"] == 1
    assert res_low["precision"] == round(2 / 3, 4)

    # At threshold 0.70, prediction for index 2 becomes 0 (FN)
    res_high = compute_binary_classification_metrics(y_true, y_prob, threshold=0.70)
    assert res_high["threshold"] == 0.70
    assert res_high["confusion_matrix"]["fn"] == 1
    assert res_high["recall"] == 0.5


def test_compute_metrics_shape_mismatch() -> None:
    """Verify ValueError is raised when length of y_true and y_prob differ."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_binary_classification_metrics([0, 1], [0.5])


def test_compute_metrics_single_class_raises() -> None:
    """Verify ValueError is raised when y_true contains only a single class."""
    with pytest.raises(ValueError, match="must contain both positive"):
        compute_binary_classification_metrics([1, 1, 1], [0.8, 0.9, 0.7])
