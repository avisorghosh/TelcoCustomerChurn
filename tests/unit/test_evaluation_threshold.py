"""Unit tests for decision threshold analysis module."""

import pytest

from churn_prediction.evaluation.threshold import (
    DEFAULT_THRESHOLDS,
    compute_threshold_analysis,
)


def test_compute_threshold_analysis_defaults() -> None:
    """Verify threshold analysis with default threshold range."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]

    res = compute_threshold_analysis(y_true, y_prob)

    assert len(res) == len(DEFAULT_THRESHOLDS)
    for entry in res:
        assert "threshold" in entry
        assert "precision" in entry
        assert "recall" in entry
        assert "f1_score" in entry
        assert "accuracy" in entry
        assert "tp" in entry
        assert "fp" in entry
        assert "tn" in entry
        assert "fn" in entry


def test_compute_threshold_analysis_custom_list() -> None:
    """Verify threshold analysis with custom threshold list."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.8]

    custom_th = [0.2, 0.5, 0.7]
    res = compute_threshold_analysis(y_true, y_prob, thresholds=custom_th)

    assert len(res) == 3
    assert res[0]["threshold"] == 0.2
    assert res[1]["threshold"] == 0.5
    assert res[2]["threshold"] == 0.7


def test_compute_threshold_analysis_mismatch_raises() -> None:
    """Verify ValueError is raised on mismatched input lengths."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_threshold_analysis([0, 1], [0.5])
