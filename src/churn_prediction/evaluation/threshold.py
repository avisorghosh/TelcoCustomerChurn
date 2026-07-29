"""Decision threshold sweep and metric sensitivity analysis module."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DEFAULT_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


def compute_threshold_analysis(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate classification metrics across a sweep of decision thresholds.

    Allows stakeholders to evaluate metric trade-offs (precision vs. recall,
    false positives vs. false negatives) across candidate operating points.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in range [0, 1].
        thresholds: Optional list of thresholds to evaluate. Uses defaults if None.

    Returns:
        List of dictionaries, each containing metrics at a specific threshold.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Shape mismatch: y_true has length {len(y_true_arr)} "
            f"but y_prob has length {len(y_prob_arr)}."
        )

    eval_thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS

    results: list[dict[str, Any]] = []
    for th in eval_thresholds:
        y_pred = (y_prob_arr >= th).astype(int)

        prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true_arr, y_pred))

        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (
            int(cm[0, 0]),
            int(cm[0, 1]),
            int(cm[1, 0]),
            int(cm[1, 1]),
        )

        results.append(
            {
                "threshold": round(float(th), 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "accuracy": round(acc, 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    return results
