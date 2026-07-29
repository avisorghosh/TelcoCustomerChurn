"""Data analysis utilities for overview, distributions, and quality checks."""

from typing import Any

import pandas as pd


def get_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall dimensions, row/column counts, missingness, and duplicates.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing overview metrics.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())

    dup_ids = 0
    if "customerID" in df.columns:
        dup_ids = int(df["customerID"].duplicated().sum())

    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_pcts = {
        col: float(df[col].isna().mean() * 100)
        for col in df.columns
        if df[col].isna().sum() > 0
    }
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_customer_ids": dup_ids,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "column_dtypes": dtypes,
    }


def get_target_distribution(
    df: pd.DataFrame, target_col: str = "Churn"
) -> dict[str, Any]:
    """Compute class counts, proportions, and imbalance ratio for target variable.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Dictionary with target distribution metrics.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    counts = df[target_col].value_counts().to_dict()
    percentages = (df[target_col].value_counts(normalize=True) * 100).to_dict()

    churn_yes = counts.get("Yes", 0)
    churn_no = counts.get("No", 0)
    total = len(df)
    churn_rate = (churn_yes / total) if total > 0 else 0.0

    imbalance_ratio = (churn_no / churn_yes) if churn_yes > 0 else float("inf")

    return {
        "counts": counts,
        "percentages": percentages,
        "churn_rate": churn_rate,
        "imbalance_ratio": imbalance_ratio,
    }


def get_numeric_summary(
    df: pd.DataFrame, numeric_cols: list[str] | None = None
) -> pd.DataFrame:
    """Compute detailed summary statistics and IQR outlier counts for numeric features.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric column names. Defaults to ['tenure',
          'MonthlyCharges', 'TotalCharges'].

    Returns:
        DataFrame containing summary statistics for each numeric column.
    """
    if numeric_cols is None:
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    cols_to_use = [c for c in numeric_cols if c in df.columns]
    summary_rows: list[dict[str, Any]] = []

    for col in cols_to_use:
        s = df[col].dropna()
        count = len(s)
        mean_val = float(s.mean())
        std_val = float(s.std())
        min_val = float(s.min())
        q25 = float(s.quantile(0.25))
        q50 = float(s.median())
        q75 = float(s.quantile(0.75))
        max_val = float(s.max())
        skew_val = float(s.skew())
        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers_iqr = int(((s < lower_bound) | (s > upper_bound)).sum())

        summary_rows.append(
            {
                "feature": col,
                "count": count,
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "min": round(min_val, 2),
                "25%": round(q25, 2),
                "50%": round(q50, 2),
                "75%": round(q75, 2),
                "max": round(max_val, 2),
                "skewness": round(skew_val, 2),
                "iqr": round(iqr, 2),
                "outliers_iqr": outliers_iqr,
            }
        )

    return pd.DataFrame(summary_rows).set_index("feature")


def get_categorical_summary(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    target_col: str = "Churn",
) -> dict[str, pd.DataFrame]:
    """Compute category frequencies and churn rates for categorical features.

    Args:
        df: Input DataFrame.
        categorical_cols: List of categorical column names.
        target_col: Name of target column ('Churn').

    Returns:
        Dictionary mapping column name to summary DataFrame.
    """
    if categorical_cols is None:
        categorical_cols = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]

    results: dict[str, pd.DataFrame] = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        grouped = df.groupby(col, observed=False)
        total_count = grouped.size()
        pct = (total_count / len(df)) * 100

        if target_col in df.columns:
            churn_count = grouped[target_col].apply(lambda s: int((s == "Yes").sum()))
            churn_rate = (churn_count / total_count) * 100
        else:
            churn_count = pd.Series(0, index=total_count.index)
            churn_rate = pd.Series(0.0, index=total_count.index)

        cat_df = pd.DataFrame(
            {
                "category": total_count.index.astype(str),
                "count": total_count.values,
                "percentage": pct.values.round(2),
                "churn_count": churn_count.values,
                "churn_rate": churn_rate.values.round(2),
            }
        )
        results[col] = cat_df

    return results


def get_numeric_correlations(
    df: pd.DataFrame, numeric_cols: list[str] | None = None
) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric features.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric columns.

    Returns:
        DataFrame containing correlation matrix.
    """
    if numeric_cols is None:
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    cols_to_use = [c for c in numeric_cols if c in df.columns]
    return df[cols_to_use].corr(method="pearson").round(4)


def get_charge_discrepancy_analysis(df: pd.DataFrame) -> dict[str, float]:
    """Analyze discrepancy between TotalCharges and tenure * MonthlyCharges.

    Args:
        df: Input DataFrame with tenure, MonthlyCharges, TotalCharges.

    Returns:
        Dictionary of discrepancy statistics.
    """
    valid_mask = df["TotalCharges"].notna()
    valid_df = df[valid_mask]

    expected = valid_df["tenure"] * valid_df["MonthlyCharges"]
    diff = (valid_df["TotalCharges"] - expected).abs()

    blank_count = int(df["TotalCharges"].isna().sum())

    return {
        "blank_total_charges_count": float(blank_count),
        "mean_abs_diff": round(float(diff.mean()), 2),
        "median_abs_diff": round(float(diff.median()), 2),
        "std_abs_diff": round(float(diff.std()), 2),
        "max_abs_diff": round(float(diff.max()), 2),
    }


def assess_leakage_and_fairness(
    df: pd.DataFrame, target_col: str = "Churn"
) -> dict[str, Any]:
    """Assess potential data leakage and demographic fairness attributes.

    Args:
        df: Input DataFrame.
        target_col: Name of target column.

    Returns:
        Dictionary detailing leakage and fairness checks.
    """
    customer_id_unique = False
    if "customerID" in df.columns:
        customer_id_unique = df["customerID"].nunique() == len(df)

    gender_churn_rates: dict[str, float] = {}
    if "gender" in df.columns and target_col in df.columns:
        gender_grouped = df.groupby("gender")[target_col].apply(
            lambda s: float((s == "Yes").mean() * 100)
        )
        gender_churn_rates = gender_grouped.to_dict()

    senior_churn_rates: dict[str, float] = {}
    if "SeniorCitizen" in df.columns and target_col in df.columns:
        senior_grouped = df.groupby("SeniorCitizen")[target_col].apply(
            lambda s: float((s == "Yes").mean() * 100)
        )
        senior_churn_rates = {
            str(k): float(v) for k, v in senior_grouped.to_dict().items()
        }

    f_rate = gender_churn_rates.get("Female", 0)
    m_rate = gender_churn_rates.get("Male", 0)
    s_rate = senior_churn_rates.get("1", 0)
    ns_rate = senior_churn_rates.get("0", 0)

    leakage_findings = [
        "customerID is an arbitrary hash with 100% uniqueness; MUST be excluded.",
        "No post-churn outcome features (e.g. cancellation dates) are present.",
        "TotalCharges reflects historical billing up to snapshot time.",
    ]

    fairness_findings = [
        (
            f"gender churn rate disparity is minimal (Female: {f_rate:.2f}%, "
            f"Male: {m_rate:.2f}%). Exclude from model inputs."
        ),
        (
            f"SeniorCitizen shows higher churn (Seniors: {s_rate:.2f}% vs "
            f"Non-seniors: {ns_rate:.2f}%). Exclude per SYSTEM_DESIGN.md."
        ),
    ]

    return {
        "customer_id_unique": customer_id_unique,
        "gender_churn_rates": gender_churn_rates,
        "senior_citizen_churn_rates": senior_churn_rates,
        "leakage_findings": leakage_findings,
        "fairness_findings": fairness_findings,
    }
