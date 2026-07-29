"""Campaign capacity metrics and risk-ranking evaluation module."""

from typing import Any

import numpy as np


def compute_capacity_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    capacity_fraction: float = 0.10,
) -> dict[str, Any]:
    """Compute precision, recall, and risk threshold at campaign capacity.

    Ranks customers by predicted probability descending and evaluates retention yield
    for the top `capacity_fraction` highest-risk population (e.g. top 10%).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in range [0, 1].
        capacity_fraction: Fraction of population to intervene on (0.0 < f <= 1.0).

    Returns:
        Dictionary containing capacity evaluation metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Shape mismatch: y_true has length {len(y_true_arr)} "
            f"but y_prob has length {len(y_prob_arr)}."
        )

    if not (0.0 < capacity_fraction <= 1.0):
        raise ValueError(
            f"capacity_fraction must be between 0.0 (exclusive) and 1.0 (inclusive), "
            f"got {capacity_fraction}."
        )

    total_customers = len(y_true_arr)
    num_targeted = max(1, int(np.round(total_customers * capacity_fraction)))

    # Sort descending by predicted churn probability
    sort_indices = np.argsort(-y_prob_arr)
    y_true_sorted = y_true_arr[sort_indices]
    y_prob_sorted = y_prob_arr[sort_indices]

    targeted_true = y_true_sorted[:num_targeted]
    retained_churners = int(np.sum(targeted_true))
    total_actual_churners = int(np.sum(y_true_arr))

    precision_at_capacity = float(retained_churners / num_targeted)
    recall_at_capacity = (
        float(retained_churners / total_actual_churners)
        if total_actual_churners > 0
        else 0.0
    )
    capacity_threshold = float(y_prob_sorted[num_targeted - 1])

    return {
        "campaign_capacity_fraction": float(capacity_fraction),
        "num_targeted_customers": num_targeted,
        "total_customers": total_customers,
        "precision_at_capacity": round(precision_at_capacity, 4),
        "recall_at_capacity": round(recall_at_capacity, 4),
        "capacity_threshold": round(capacity_threshold, 4),
        "retained_churners_captured": retained_churners,
        "total_actual_churners": total_actual_churners,
    }
