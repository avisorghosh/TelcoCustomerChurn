"""Lightweight data drift reporting module."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from churn_prediction.monitoring.config import load_observability_config


class FeatureDriftDetail(BaseModel):
    """Detailed drift statistics for a single feature."""

    feature_name: str = Field(description="Name of feature evaluated.")
    feature_type: str = Field(
        description="Feature type ('numerical' or 'categorical')."
    )
    psi: float = Field(ge=0.0, description="Population Stability Index (PSI).")
    ks_statistic: float | None = Field(
        default=None, description="Kolmogorov-Smirnov statistic for numerical feature."
    )
    ks_p_value: float | None = Field(
        default=None, description="KS test p-value for numerical feature."
    )
    missing_rate_reference: float = Field(
        ge=0.0, le=1.0, description="Missing value rate in reference dataset."
    )
    missing_rate_scoring: float = Field(
        ge=0.0, le=1.0, description="Missing value rate in scoring dataset."
    )
    missing_rate_delta: float = Field(
        ge=0.0, description="Absolute difference in missing value rates."
    )
    drift_status: str = Field(description="Feature drift status.")


class DriftReport(BaseModel):
    """Data drift report comparing current scoring batch against training reference."""

    schema_version: str = Field(
        default="1.0.0", description="Observability schema version."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp of drift report execution.",
    )
    reference_record_count: int = Field(
        ge=0, description="Row count of baseline training reference data."
    )
    scoring_record_count: int = Field(
        ge=0, description="Row count of current scoring batch data."
    )
    overall_drift_status: str = Field(description="Overall dataset drift status.")
    max_psi: float = Field(
        ge=0.0, description="Maximum PSI observed across all features."
    )
    drifted_features_count: int = Field(
        ge=0, description="Number of features exhibiting significant drift."
    )
    feature_reports: list[FeatureDriftDetail] = Field(
        default_factory=list, description="Per-feature drift details."
    )
    summary: str = Field(description="Human-readable summary of drift analysis.")


def calculate_numerical_psi(
    ref_s: pd.Series, target_s: pd.Series, num_bins: int = 10, eps: float = 1e-4
) -> float:
    """Calculate Population Stability Index (PSI) for numerical series.

    Args:
        ref_s: Baseline reference series.
        target_s: Current scoring target series.
        num_bins: Number of bins for discretization.
        eps: Epsilon value to prevent division/log by zero.

    Returns:
        PSI float value.
    """
    ref_clean = ref_s.dropna()
    target_clean = target_s.dropna()

    if len(ref_clean) == 0 or len(target_clean) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(ref_clean, percentiles)
    bins = np.unique(bins)  # Deduplicate bin edges if values are sparse

    if len(bins) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    target_counts, _ = np.histogram(target_clean, bins=bins)

    ref_pct = ref_counts / len(ref_clean)
    target_pct = target_counts / len(target_clean)

    # Apply epsilon smoothing
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    target_pct = np.where(target_pct == 0, eps, target_pct)

    psi_val = np.sum((target_pct - ref_pct) * np.log(target_pct / ref_pct))
    return float(np.round(max(0.0, psi_val), 4))


def calculate_categorical_psi(
    ref_s: pd.Series, target_s: pd.Series, eps: float = 1e-4
) -> float:
    """Calculate Population Stability Index (PSI) for categorical series.

    Args:
        ref_s: Baseline reference categorical series.
        target_s: Current scoring target categorical series.
        eps: Epsilon value to prevent division/log by zero.

    Returns:
        PSI float value.
    """
    ref_clean = ref_s.fillna("__MISSING__").astype(str)
    target_clean = target_s.fillna("__MISSING__").astype(str)

    all_categories = sorted(
        list(set(ref_clean.unique()).union(set(target_clean.unique())))
    )
    if not all_categories:
        return 0.0

    ref_counts = ref_clean.value_counts(normalize=True).to_dict()
    target_counts = target_clean.value_counts(normalize=True).to_dict()

    psi_sum = 0.0
    for cat in all_categories:
        p_pct = ref_counts.get(cat, eps)
        q_pct = target_counts.get(cat, eps)
        if p_pct == 0:
            p_pct = eps
        if q_pct == 0:
            q_pct = eps
        psi_sum += (q_pct - p_pct) * np.log(q_pct / p_pct)

    return float(np.round(max(0.0, psi_sum), 4))


def calculate_ks_stat(ref_s: pd.Series, target_s: pd.Series) -> tuple[float, float]:
    """Calculate Kolmogorov-Smirnov statistic and p-value for numerical features."""
    ref_clean = ref_s.dropna().values
    target_clean = target_s.dropna().values

    if len(ref_clean) == 0 or len(target_clean) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(ref_clean, target_clean)
    return float(np.round(res.statistic, 4)), float(np.round(res.pvalue, 4))


def detect_data_drift(
    scoring_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> DriftReport:
    """Detect data drift between current scoring batch and baseline training reference.

    Args:
        scoring_df: Current dataset to evaluate for drift.
        reference_df: Baseline reference training dataset.
        config_path: Optional path to observability config YAML.
        output_path: Optional path to write drift report JSON.

    Returns:
        DriftReport containing detailed feature-level drift statistics.
    """
    obs_config = load_observability_config(config_path)
    drift_config = obs_config.get("drift", {})
    thresholds = drift_config.get(
        "thresholds", {"psi_warning": 0.10, "psi_critical": 0.25}
    )
    psi_warning = float(thresholds.get("psi_warning", 0.10))
    psi_critical = float(thresholds.get("psi_critical", 0.25))

    num_cols = drift_config.get(
        "numerical_features", ["tenure", "MonthlyCharges", "TotalCharges"]
    )
    cat_cols = drift_config.get(
        "categorical_features",
        [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "OnlineSecurity",
            "TechSupport",
            "PaperlessBilling",
        ],
    )

    feature_reports: list[FeatureDriftDetail] = []
    max_psi = 0.0
    drifted_count = 0

    # Evaluate Numerical Features
    for col in num_cols:
        if col in scoring_df.columns and col in reference_df.columns:
            ref_s = pd.to_numeric(reference_df[col], errors="coerce")
            tar_s = pd.to_numeric(scoring_df[col], errors="coerce")

            psi_val = calculate_numerical_psi(ref_s, tar_s)
            ks_stat, ks_p = calculate_ks_stat(ref_s, tar_s)

            ref_missing = float(ref_s.isna().mean())
            tar_missing = float(tar_s.isna().mean())
            missing_delta = float(abs(tar_missing - ref_missing))

            if psi_val >= psi_critical:
                status_str = "significant_drift"
                drifted_count += 1
            elif psi_val >= psi_warning:
                status_str = "moderate_drift"
            else:
                status_str = "no_drift"

            max_psi = max(max_psi, psi_val)

            feature_reports.append(
                FeatureDriftDetail(
                    feature_name=col,
                    feature_type="numerical",
                    psi=psi_val,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p,
                    missing_rate_reference=round(ref_missing, 4),
                    missing_rate_scoring=round(tar_missing, 4),
                    missing_rate_delta=round(missing_delta, 4),
                    drift_status=status_str,
                )
            )

    # Evaluate Categorical Features
    for col in cat_cols:
        if col in scoring_df.columns and col in reference_df.columns:
            ref_s = reference_df[col]
            tar_s = scoring_df[col]

            psi_val = calculate_categorical_psi(ref_s, tar_s)

            ref_missing = float(ref_s.isna().mean())
            tar_missing = float(tar_s.isna().mean())
            missing_delta = float(abs(tar_missing - ref_missing))

            if psi_val >= psi_critical:
                status_str = "significant_drift"
                drifted_count += 1
            elif psi_val >= psi_warning:
                status_str = "moderate_drift"
            else:
                status_str = "no_drift"

            max_psi = max(max_psi, psi_val)

            feature_reports.append(
                FeatureDriftDetail(
                    feature_name=col,
                    feature_type="categorical",
                    psi=psi_val,
                    ks_statistic=None,
                    ks_p_value=None,
                    missing_rate_reference=round(ref_missing, 4),
                    missing_rate_scoring=round(tar_missing, 4),
                    missing_rate_delta=round(missing_delta, 4),
                    drift_status=status_str,
                )
            )

    max_psi = round(max_psi, 4)
    if max_psi >= psi_critical:
        overall_status = "significant_drift"
    elif max_psi >= psi_warning:
        overall_status = "moderate_drift"
    else:
        overall_status = "no_drift"

    summary = (
        f"Drift Analysis: overall status={overall_status}, max PSI={max_psi}, "
        f"{drifted_count}/{len(feature_reports)} features exhibited significant drift."
    )

    report = DriftReport(
        schema_version="1.0.0",
        reference_record_count=len(reference_df),
        scoring_record_count=len(scoring_df),
        overall_drift_status=overall_status,
        max_psi=max_psi,
        drifted_features_count=drifted_count,
        feature_reports=feature_reports,
        summary=summary,
    )

    if output_path is None:
        out_dir = Path(drift_config.get("output_dir", "reports/drift"))
        filename = drift_config.get("report_filename", "drift_report.json")
        output_path = out_dir / filename

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    return report
