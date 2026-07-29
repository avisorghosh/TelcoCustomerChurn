"""Unit tests for calibration and Brier score evaluation module."""

import pytest

from churn_prediction.evaluation.calibration import compute_calibration_curve


def test_compute_calibration_curve_structure() -> None:
    """Verify calibration curve output structure and keys."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    res = compute_calibration_curve(y_true, y_prob, n_bins=5)

    assert "prob_true" in res
    assert "prob_pred" in res
    assert "brier_score" in res
    assert "n_bins" in res
    assert res["n_bins"] == 5
    assert len(res["prob_true"]) == len(res["prob_pred"])
    assert 0.0 <= res["brier_score"] <= 1.0


def test_compute_calibration_curve_mismatch_raises() -> None:
    """Verify ValueError is raised on mismatched input lengths."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_calibration_curve([0, 1], [0.5])
