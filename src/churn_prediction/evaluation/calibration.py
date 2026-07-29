"""Probability calibration and Brier score evaluation module."""

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_calibration_curve(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Evaluate model probability calibration and Brier score.

    Computes empirical bin proportions (prob_true) vs average predicted probability
    (prob_pred) across probability bins.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in range [0, 1].
        n_bins: Number of probability bins for calibration curve.
        strategy: Binning strategy ('uniform' or 'quantile').

    Returns:
        Dictionary containing calibration points, Brier score, and bin details.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Shape mismatch: y_true has length {len(y_true_arr)} "
            f"but y_prob has length {len(y_prob_arr)}."
        )

    prob_true, prob_pred = calibration_curve(
        y_true_arr,
        y_prob_arr,
        n_bins=n_bins,
        strategy=strategy,
    )
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "prob_true": [round(float(p), 4) for p in prob_true],
        "prob_pred": [round(float(p), 4) for p in prob_pred],
        "brier_score": round(brier, 4),
        "n_bins": int(n_bins),
        "strategy": str(strategy),
    }
