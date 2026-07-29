"""Binary classification metrics computation module."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute offline binary classification metrics at a decision threshold.

    Includes PR-AUC (primary metric), ROC-AUC, Precision, Recall, F1 Score,
    Accuracy, Brier Score, Confusion Matrix, and Classification Report.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in range [0, 1].
        threshold: Decision threshold for converting probabilities to predictions.

    Returns:
        Dictionary containing calculated metrics and reports.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Shape mismatch: y_true has length {len(y_true_arr)} "
            f"but y_prob has length {len(y_prob_arr)}."
        )

    if len(np.unique(y_true_arr)) < 2:
        raise ValueError(
            "y_true must contain both positive (1) and negative (0) classes."
        )

    y_pred = (y_prob_arr >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true_arr, y_pred))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (
        int(cm[0, 0]),
        int(cm[0, 1]),
        int(cm[1, 0]),
        int(cm[1, 1]),
    )
    cm_dict = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    report_dict = classification_report(
        y_true_arr,
        y_pred,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": cm_dict,
        "classification_report": report_dict,
    }
