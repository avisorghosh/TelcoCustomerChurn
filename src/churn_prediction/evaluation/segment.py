"""Customer segment evaluation module."""

from typing import Any

import numpy as np
import pandas as pd

from churn_prediction.evaluation.metrics import compute_binary_classification_metrics


def _assign_tenure_band(tenure: int | float) -> str:
    """Group continuous tenure into discrete analytical bands."""
    if tenure <= 12:
        return "0-12 months"
    elif tenure <= 24:
        return "13-24 months"
    elif tenure <= 48:
        return "25-48 months"
    else:
        return "49+ months"


def _assign_monthly_charges_band(charges: float) -> str:
    """Group continuous MonthlyCharges into discrete analytical bands."""
    if charges < 35.0:
        return "< $35"
    elif charges <= 70.0:
        return "$35 - $70"
    else:
        return "> $70"


def evaluate_segment_performance(
    df: pd.DataFrame,
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.50,
    capacity_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate classification performance across key customer business segments.

    Segments analyzed:
    - tenure_band (0-12, 13-24, 25-48, 49+ months)
    - Contract (Month-to-month, One year, Two year)
    - InternetService (DSL, Fiber optic, No)
    - monthly_charges_band (< $35, $35 - $70, > $70)

    Args:
        df: Input DataFrame containing feature columns.
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in range [0, 1].
        threshold: Decision threshold for classification metrics (default 0.50).
        capacity_threshold: Optional threshold for capacity metric.

    Returns:
        Dictionary mapping segment dimension names to subgroup metric summaries.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(df) != len(y_true_arr) or len(y_true_arr) != len(y_prob_arr):
        raise ValueError("DataFrame, y_true, and y_prob must have identical lengths.")

    eval_df = df.copy()
    eval_df["_y_true"] = y_true_arr
    eval_df["_y_prob"] = y_prob_arr

    # Create derived segment columns if raw features exist
    if "tenure" in eval_df.columns:
        eval_df["tenure_band"] = eval_df["tenure"].apply(_assign_tenure_band)

    if "MonthlyCharges" in eval_df.columns:
        eval_df["monthly_charges_band"] = eval_df["MonthlyCharges"].apply(
            _assign_monthly_charges_band
        )

    segment_cols = [
        "tenure_band",
        "Contract",
        "InternetService",
        "monthly_charges_band",
    ]
    segment_columns = [col for col in segment_cols if col in eval_df.columns]

    results: dict[str, Any] = {}

    for col in segment_columns:
        subgroups: dict[str, Any] = {}
        grouped = eval_df.groupby(col, observed=True)

        for name, group in grouped:
            str_name = str(name)
            sub_y_true = group["_y_true"].to_numpy()
            sub_y_prob = group["_y_prob"].to_numpy()

            count = len(group)
            churn_count = int(sub_y_true.sum())
            churn_rate = float(round(churn_count / count, 4)) if count > 0 else 0.0
            mean_prob = float(round(sub_y_prob.mean(), 4)) if count > 0 else 0.0

            sub_metrics: dict[str, Any] = {
                "count": count,
                "churn_count": churn_count,
                "churn_rate": churn_rate,
                "mean_predicted_probability": mean_prob,
            }

            # If subgroup has both positive and negative samples, compute full metrics
            if len(np.unique(sub_y_true)) > 1:
                b_metrics = compute_binary_classification_metrics(
                    sub_y_true, sub_y_prob, threshold=threshold
                )

                sub_metrics.update(
                    {
                        "pr_auc": b_metrics["pr_auc"],
                        "roc_auc": b_metrics["roc_auc"],
                        "brier_score": b_metrics["brier_score"],
                        "accuracy": b_metrics["accuracy"],
                        "f1_score": b_metrics["f1_score"],
                    }
                )
            else:
                sub_metrics.update(
                    {
                        "pr_auc": None,
                        "roc_auc": None,
                        "brier_score": float(
                            round(((sub_y_prob - sub_y_true) ** 2).mean(), 4)
                        ),
                        "accuracy": float(
                            round(
                                (
                                    (sub_y_prob >= threshold).astype(int) == sub_y_true
                                ).mean(),
                                4,
                            )
                        ),
                        "f1_score": 0.0,
                    }
                )

            # Capacity metrics under global capacity threshold
            if capacity_threshold is not None:
                selected_flag = sub_y_prob >= capacity_threshold
                selected_count = int(selected_flag.sum())
                selected_tp = int((selected_flag & (sub_y_true == 1)).sum())

                sub_prec_cap = (
                    round(selected_tp / selected_count, 4)
                    if selected_count > 0
                    else 0.0
                )
                sub_rec_cap = (
                    round(selected_tp / churn_count, 4) if churn_count > 0 else 0.0
                )

                sub_metrics["selected_count_at_capacity"] = selected_count
                sub_metrics["precision_at_capacity"] = sub_prec_cap
                sub_metrics["recall_at_capacity"] = sub_rec_cap

            subgroups[str_name] = sub_metrics

        results[col] = subgroups

    return results
