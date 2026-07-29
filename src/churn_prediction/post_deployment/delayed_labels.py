"""Delayed ground-truth label integration and evaluation module."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn_prediction.evaluation.calibration import compute_calibration_curve
from churn_prediction.evaluation.capacity import compute_capacity_metrics
from churn_prediction.evaluation.metrics import compute_binary_classification_metrics

logger = logging.getLogger(__name__)


def load_and_match_delayed_labels(
    predictions_input: str | Path | pd.DataFrame,
    delayed_labels_input: str | Path | pd.DataFrame,
    id_column: str = "customerID",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest historical predictions and delayed ground-truth labels and merge on customer key.

    Args:
        predictions_input: File path or DataFrame containing historical predictions.
        delayed_labels_input: File path or DataFrame containing delayed ground-truth labels.
        id_column: Customer identifier column name.

    Returns:
        Tuple of (matched_dataframe, operational_matching_stats_dict).

    Raises:
        ValueError: If required columns are missing or input datasets are invalid.
    """
    # Load predictions DataFrame
    if isinstance(predictions_input, (str, Path)):
        pred_path = Path(predictions_input)
        if not pred_path.is_file():
            raise FileNotFoundError(f"Predictions file not found at: {pred_path}")
        preds_df = pd.read_csv(pred_path)
    elif isinstance(predictions_input, pd.DataFrame):
        preds_df = predictions_input.copy()
    else:
        raise ValueError("predictions_input must be a file path or pandas DataFrame")

    # Load delayed labels DataFrame
    if isinstance(delayed_labels_input, (str, Path)):
        labels_path = Path(delayed_labels_input)
        if not labels_path.is_file():
            raise FileNotFoundError(f"Delayed labels file not found at: {labels_path}")
        labels_df = pd.read_csv(labels_path)
    elif isinstance(delayed_labels_input, pd.DataFrame):
        labels_df = delayed_labels_input.copy()
    else:
        raise ValueError("delayed_labels_input must be a file path or pandas DataFrame")

    # Validate required ID column presence
    if id_column not in preds_df.columns:
        raise ValueError(
            f"Predictions dataset missing required ID column: '{id_column}'"
        )
    if id_column not in labels_df.columns:
        raise ValueError(
            f"Delayed labels dataset missing required ID column: '{id_column}'"
        )

    # Check prediction probability column
    prob_col = "churn_probability"
    if prob_col not in preds_df.columns:
        raise ValueError(
            f"Predictions dataset missing probability column: '{prob_col}'"
        )

    # Normalize delayed label churn target column
    target_col = None
    for candidate in ["observed_churn", "Churn", "churn", "target"]:
        if candidate in labels_df.columns:
            target_col = candidate
            break

    if target_col is None:
        raise ValueError(
            "Delayed labels dataset missing target outcome column. "
            "Expected one of ['observed_churn', 'Churn', 'churn', 'target']"
        )

    # Standardize ground truth column to binary integer (0 or 1)
    target_series = labels_df[target_col]
    if target_series.dtype == object or isinstance(target_series.iloc[0], str):
        binary_target = (
            target_series.astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "1", "true", "churn"])
        ).astype(int)
    else:
        binary_target = target_series.astype(int)

    labels_df["observed_churn"] = binary_target

    # Count records before matching
    total_preds = len(preds_df)
    total_labels = len(labels_df)

    # Deduplicate predictions & labels on customerID if necessary
    preds_df = preds_df.drop_duplicates(subset=[id_column], keep="last")
    labels_df = labels_df.drop_duplicates(subset=[id_column], keep="last")

    # Perform inner merge on customer key
    merged_df = pd.merge(
        preds_df,
        labels_df[
            [col for col in labels_df.columns if col != id_column or col == id_column]
        ],
        on=id_column,
        how="inner",
        suffixes=("_pred", "_observed"),
    )

    matched_records = len(merged_df)
    unmatched_preds = total_preds - matched_records
    match_rate = matched_records / total_preds if total_preds > 0 else 0.0

    stats = {
        "total_historical_predictions": total_preds,
        "total_delayed_labels_received": total_labels,
        "matched_records": matched_records,
        "unmatched_predictions": unmatched_preds,
        "match_rate": round(match_rate, 4),
    }

    logger.info(
        f"Matched {matched_records}/{total_preds} predictions with delayed labels "
        f"(match rate: {match_rate:.2%})"
    )

    return merged_df, stats


def evaluate_delayed_predictions(
    matched_df: pd.DataFrame,
    threshold: float = 0.50,
    capacity_fraction: float = 0.10,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute prediction evaluation metrics on matched historical predictions and delayed labels.

    Reuses existing evaluation contract utilities for binary metrics, capacity, and calibration.

    Args:
        matched_df: DataFrame output from load_and_match_delayed_labels.
        threshold: Classification decision threshold.
        capacity_fraction: Target campaign capacity fraction.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dictionary containing binary classification metrics, campaign capacity metrics,
        and probability calibration diagnostics.
    """
    if "observed_churn" not in matched_df.columns:
        raise ValueError("matched_df must contain 'observed_churn' column")
    if "churn_probability" not in matched_df.columns:
        raise ValueError("matched_df must contain 'churn_probability' column")

    y_true = matched_df["observed_churn"].to_numpy(dtype=int)
    y_prob = matched_df["churn_probability"].to_numpy(dtype=float)

    # Calculate standard evaluation metrics
    classification_metrics = compute_binary_classification_metrics(
        y_true, y_prob, threshold=threshold
    )
    capacity_metrics = compute_capacity_metrics(
        y_true, y_prob, capacity_fraction=capacity_fraction
    )
    calibration_metrics = compute_calibration_curve(y_true, y_prob, n_bins=n_bins)

    return {
        "sample_size": len(y_true),
        "observed_churn_prevalence": round(float(np.mean(y_true)), 4),
        "primary_metric": {
            "name": "PR-AUC",
            "value": classification_metrics["pr_auc"],
        },
        "metrics": classification_metrics,
        "capacity_metrics": capacity_metrics,
        "calibration_metrics": calibration_metrics,
    }


def generate_synthetic_delayed_labels(
    predictions_df: pd.DataFrame,
    seed: int = 42,
    treatment_ratio: float = 0.80,
    treatment_effect: float = 0.15,
    id_column: str = "customerID",
) -> pd.DataFrame:
    """Generate reproducible synthetic delayed ground-truth labels for local testing & rehearsal.

    Simulates eventual observed outcomes based on model predicted probabilities and optional
    retention campaign intervention effects.

    Assumptions (Documented):
    - Ground truth outcomes follow prediction probabilities + random noise.
    - Targeted treatment group receives a retention intervention with documented effectiveness.

    Args:
        predictions_df: DataFrame of historical batch predictions.
        seed: Random seed for deterministic reproducibility.
        treatment_ratio: Fraction of high-risk customers assigned to treatment vs control.
        treatment_effect: Relative reduction in churn probability for treatment group.
        id_column: Customer key column name.

    Returns:
        DataFrame containing customerID, observed_churn, treatment_group, and observation_date.
    """
    rng = np.random.default_rng(seed)
    df = predictions_df.copy()

    if id_column not in df.columns:
        raise ValueError(f"predictions_df missing required column: '{id_column}'")

    probs = (
        df["churn_probability"].to_numpy()
        if "churn_probability" in df.columns
        else rng.uniform(0.05, 0.85, size=len(df))
    )

    # Determine campaign assignment: high risk gets targeted, split into treatment and control
    high_risk_mask = probs >= np.quantile(probs, 1.0 - treatment_ratio)
    treatment_assignment = []
    for is_high in high_risk_mask:
        if is_high:
            # 80% treatment, 20% control holdout
            assignment = "treatment" if rng.random() < 0.80 else "control"
        else:
            assignment = "control"
        treatment_assignment.append(assignment)

    treatment_arr = np.array(treatment_assignment)

    # Adjust observed churn probability for treatment group (simulated campaign effect)
    observed_probs = probs.copy()
    treatment_mask = treatment_arr == "treatment"
    observed_probs[treatment_mask] = np.clip(
        observed_probs[treatment_mask] * (1.0 - treatment_effect), 0.0, 1.0
    )

    # Draw binary outcome from Bernoulli trial
    observed_outcomes = (rng.random(size=len(df)) < observed_probs).astype(int)

    delayed_df = pd.DataFrame(
        {
            id_column: df[id_column].values,
            "observed_churn": observed_outcomes,
            "treatment_group": treatment_arr,
            "observation_date": "2026-07-29",
        }
    )

    return delayed_df
