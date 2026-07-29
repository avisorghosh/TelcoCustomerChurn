"""Lightweight fairness review module for sensitive attributes."""

from typing import Any

import numpy as np
import pandas as pd

from churn_prediction.evaluation.calibration import compute_calibration_curve
from churn_prediction.evaluation.metrics import compute_binary_classification_metrics


def evaluate_fairness_review(
    df: pd.DataFrame,
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.50,
    capacity_threshold: float | None = None,
    sensitive_attributes: list[str] | None = None,
) -> dict[str, Any]:
    """Perform lightweight fairness review across sensitive or protected attributes.

    IMPORTANT: Sensitive attributes MUST NOT be included as model features.
    This review is conducted post-hoc for audit and equity monitoring only.

    Default sensitive attributes evaluated:
    - gender ('Female', 'Male')
    - SeniorCitizen (0, 1)

    Args:
        df: Input evaluation DataFrame containing sensitive attribute columns.
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in range [0, 1].
        threshold: Decision threshold for binary prediction assignment (default 0.50).
        capacity_threshold: Optional threshold for capacity metric.
        sensitive_attributes: List of attribute names to review
            (defaults to ['gender', 'SeniorCitizen']).

    Returns:
        Dictionary containing subgroup metrics and disparity metrics for attributes.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(df) != len(y_true_arr) or len(y_true_arr) != len(y_prob_arr):
        raise ValueError("DataFrame, y_true, and y_prob must have identical lengths.")

    if sensitive_attributes is None:
        sensitive_attributes = ["gender", "SeniorCitizen"]

    eval_df = df.copy()
    eval_df["_y_true"] = y_true_arr
    eval_df["_y_prob"] = y_prob_arr
    eval_df["_y_pred"] = (y_prob_arr >= threshold).astype(int)

    results: dict[str, Any] = {}

    for attr in sensitive_attributes:
        if attr not in eval_df.columns:
            continue

        attr_summary: dict[str, Any] = {"subgroups": {}, "disparity_metrics": {}}

        subgroup_stats: dict[str, dict[str, float]] = {}

        grouped = eval_df.groupby(attr, observed=True)

        for name, group in grouped:
            str_name = str(name)
            sub_y_true = group["_y_true"].to_numpy()
            sub_y_prob = group["_y_prob"].to_numpy()
            sub_y_pred = group["_y_pred"].to_numpy()

            count = len(group)
            churn_count = int(sub_y_true.sum())
            churn_rate = float(round(churn_count / count, 4)) if count > 0 else 0.0
            mean_prob = float(round(sub_y_prob.mean(), 4)) if count > 0 else 0.0

            selection_count = int(sub_y_pred.sum())
            selection_rate = (
                float(round(selection_count / count, 4)) if count > 0 else 0.0
            )

            # Compute confusion matrix terms for equalized odds
            tp = int(((sub_y_pred == 1) & (sub_y_true == 1)).sum())
            fp = int(((sub_y_pred == 1) & (sub_y_true == 0)).sum())
            tn = int(((sub_y_pred == 0) & (sub_y_true == 0)).sum())
            fn = int(((sub_y_pred == 0) & (sub_y_true == 1)).sum())

            tpr = float(round(tp / (tp + fn), 4)) if (tp + fn) > 0 else 0.0
            fpr = float(round(fp / (fp + tn), 4)) if (fp + tn) > 0 else 0.0

            b_metrics = (
                compute_binary_classification_metrics(
                    sub_y_true, sub_y_prob, threshold=threshold
                )
                if len(np.unique(sub_y_true)) > 1
                else {}
            )
            cal_metrics = compute_calibration_curve(sub_y_true, sub_y_prob)

            sub_info = {
                "count": count,
                "churn_count": churn_count,
                "churn_rate": churn_rate,
                "mean_predicted_probability": mean_prob,
                "selection_rate": selection_rate,
                "tpr": tpr,
                "fpr": fpr,
                "pr_auc": b_metrics.get("pr_auc"),
                "roc_auc": b_metrics.get("roc_auc"),
                "brier_score": cal_metrics["brier_score"],
            }

            if capacity_threshold is not None:
                cap_flag = sub_y_prob >= capacity_threshold
                cap_count = int(cap_flag.sum())
                cap_rate = float(round(cap_count / count, 4)) if count > 0 else 0.0
                sub_info["capacity_selection_rate"] = cap_rate

            attr_summary["subgroups"][str_name] = sub_info
            subgroup_stats[str_name] = {
                "selection_rate": selection_rate,
                "tpr": tpr,
                "fpr": fpr,
                "brier_score": cal_metrics["brier_score"],
            }

        # Calculate disparity metrics if multiple subgroups exist
        if len(subgroup_stats) >= 2:
            sel_rates = [v["selection_rate"] for v in subgroup_stats.values()]
            tprs = [v["tpr"] for v in subgroup_stats.values()]
            fprs = [v["fpr"] for v in subgroup_stats.values()]
            briers = [v["brier_score"] for v in subgroup_stats.values()]

            dp_diff = float(round(max(sel_rates) - min(sel_rates), 4))
            dp_ratio = (
                float(round(min(sel_rates) / max(sel_rates), 4))
                if max(sel_rates) > 0
                else 1.0
            )
            tpr_diff = float(round(max(tprs) - min(tprs), 4))
            fpr_diff = float(round(max(fprs) - min(fprs), 4))
            brier_diff = float(round(max(briers) - min(briers), 4))

            attr_summary["disparity_metrics"] = {
                "demographic_parity_difference": dp_diff,
                "demographic_parity_ratio": dp_ratio,
                "equalized_odds_tpr_difference": tpr_diff,
                "equalized_odds_fpr_difference": fpr_diff,
                "brier_score_disparity": brier_diff,
            }

        results[attr] = attr_summary

    return results
