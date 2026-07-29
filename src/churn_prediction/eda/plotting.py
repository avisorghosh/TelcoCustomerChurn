"""Visualization utilities for EDA, generating publication-grade plots saved to disk."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ensure_output_dir(output_dir: Path | str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_target_distribution(
    df: pd.DataFrame,
    output_dir: Path | str = "docs/images",
    target_col: str = "Churn",
) -> Path:
    """Plot target variable distribution and save to disk.

    Args:
        df: Input DataFrame.
        output_dir: Target output directory path.
        target_col: Target column name.

    Returns:
        Path to saved plot file.
    """
    out_dir = _ensure_output_dir(output_dir)
    file_path = out_dir / "target_distribution.png"

    plt.figure(figsize=(6, 4), dpi=300)
    palette = {"No": "#2b5c8f", "Yes": "#d9534f"}

    ax = sns.countplot(
        x=target_col, data=df, palette=palette, hue=target_col, legend=False
    )
    plt.title(
        "Target Distribution (Churn vs Retained)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Churn Status", fontsize=10, fontweight="bold")
    plt.ylabel("Customer Count", fontsize=10, fontweight="bold")

    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2.0),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    sns.despine()
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_numeric_distributions(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    output_dir: Path | str = "docs/images",
) -> Path:
    """Plot distribution histograms and boxplots for numeric features.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric columns.
        output_dir: Target output directory path.

    Returns:
        Path to saved plot file.
    """
    if numeric_cols is None:
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    out_dir = _ensure_output_dir(output_dir)
    file_path = out_dir / "numeric_distributions.png"

    cols_to_use = [c for c in numeric_cols if c in df.columns]
    n_cols = len(cols_to_use)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), dpi=300)

    palette = {"No": "#2b5c8f", "Yes": "#d9534f"}

    for idx, col in enumerate(cols_to_use):
        # Histogram / KDE
        ax_hist = axes[0, idx]
        sns.histplot(
            data=df,
            x=col,
            hue="Churn",
            kde=True,
            palette=palette,
            ax=ax_hist,
            bins=30,
            alpha=0.5,
        )
        ax_hist.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel(col, fontsize=9)
        ax_hist.set_ylabel("Count", fontsize=9)

        # Boxplot
        ax_box = axes[1, idx]
        sns.boxplot(
            data=df,
            x="Churn",
            y=col,
            palette=palette,
            ax=ax_box,
            hue="Churn",
            legend=False,
        )
        ax_box.set_title(f"{col} by Churn", fontsize=11, fontweight="bold")
        ax_box.set_xlabel("Churn", fontsize=9)
        ax_box.set_ylabel(col, fontsize=9)

    sns.despine()
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    output_dir: Path | str = "docs/images",
) -> Path:
    """Plot correlation matrix heatmap for numeric features.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric columns.
        output_dir: Target output directory path.

    Returns:
        Path to saved plot file.
    """
    if numeric_cols is None:
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    out_dir = _ensure_output_dir(output_dir)
    file_path = out_dir / "correlation_heatmap.png"

    cols_to_use = [c for c in numeric_cols if c in df.columns]
    corr = df[cols_to_use].corr()

    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        annot_kws={"size": 10, "weight": "bold"},
    )
    plt.title(
        "Numeric Feature Correlation Matrix", fontsize=12, fontweight="bold", pad=12
    )
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_categorical_churn_rates(
    df: pd.DataFrame,
    key_features: list[str] | None = None,
    output_dir: Path | str = "docs/images",
) -> Path:
    """Plot churn rates across key categorical features.

    Args:
        df: Input DataFrame.
        key_features: List of categorical columns to plot.
        output_dir: Target output directory path.

    Returns:
        Path to saved plot file.
    """
    if key_features is None:
        key_features = [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "TechSupport",
            "OnlineSecurity",
        ]

    out_dir = _ensure_output_dir(output_dir)
    file_path = out_dir / "categorical_churn_rates.png"

    cols_to_use = [c for c in key_features if c in df.columns]
    n_features = len(cols_to_use)

    fig, axes = plt.subplots(
        (n_features + 1) // 2, 2, figsize=(12, 3.5 * ((n_features + 1) // 2)), dpi=300
    )
    axes = axes.flatten()

    for idx, col in enumerate(cols_to_use):
        ax = axes[idx]
        churn_rates = (
            df.groupby(col, observed=False)["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        churn_rates.columns = [col, "ChurnRate"]

        sns.barplot(
            data=churn_rates,
            x=col,
            y="ChurnRate",
            palette="Reds_r",
            ax=ax,
            hue=col,
            legend=False,
        )
        ax.set_title(f"Churn Rate by {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylim(0, 60)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1f}%",
                    (p.get_x() + p.get_width() / 2.0, h + 1),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    # Hide unused subplot axes if n_features is odd
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    sns.despine()
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_tenure_vs_charges(
    df: pd.DataFrame,
    output_dir: Path | str = "docs/images",
) -> Path:
    """Plot scatter relationship of tenure vs TotalCharges by Churn.

    Args:
        df: Input DataFrame.
        output_dir: Target output directory path.

    Returns:
        Path to saved plot file.
    """
    out_dir = _ensure_output_dir(output_dir)
    file_path = out_dir / "tenure_vs_charges.png"

    plt.figure(figsize=(7, 5), dpi=300)
    palette = {"No": "#2b5c8f", "Yes": "#d9534f"}

    sns.scatterplot(
        data=df,
        x="tenure",
        y="TotalCharges",
        hue="Churn",
        palette=palette,
        alpha=0.5,
        s=25,
    )
    plt.title(
        "Tenure vs Total Charges by Churn Status",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Tenure (Months)", fontsize=10, fontweight="bold")
    plt.ylabel("Total Charges ($)", fontsize=10, fontweight="bold")

    sns.despine()
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_fairness_attributes(
    df: pd.DataFrame,
    output_dir: Path | str = "docs/images",
) -> Path:
    """Plot churn rates across sensitive/fairness attributes (gender, SeniorCitizen).

    Args:
        df: Input DataFrame.
        output_dir: Target output directory path.

    Returns:
        Path to saved plot file.
    """
    out_dir = _ensure_output_dir(output_dir)
    file_path = out_dir / "fairness_attributes.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=300)

    # Gender
    if "gender" in df.columns and "Churn" in df.columns:
        g_rates = (
            df.groupby("gender")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        sns.barplot(
            data=g_rates,
            x="gender",
            y="Churn",
            palette="Set2",
            ax=ax1,
            hue="gender",
            legend=False,
        )
        ax1.set_title("Churn Rate by Gender", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Churn Rate (%)", fontsize=9)
        ax1.set_ylim(0, 50)
        for p in ax1.patches:
            h = p.get_height()
            ax1.annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h + 1),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Senior Citizen
    if "SeniorCitizen" in df.columns and "Churn" in df.columns:
        df_senior = df.copy()
        df_senior["SeniorStatus"] = df_senior["SeniorCitizen"].map(
            {0: "Non-Senior", 1: "Senior"}
        )
        s_rates = (
            df_senior.groupby("SeniorStatus")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        sns.barplot(
            data=s_rates,
            x="SeniorStatus",
            y="Churn",
            palette="Set2",
            ax=ax2,
            hue="SeniorStatus",
            legend=False,
        )
        ax2.set_title(
            "Churn Rate by Senior Citizen Status", fontsize=11, fontweight="bold"
        )
        ax2.set_ylabel("Churn Rate (%)", fontsize=9)
        ax2.set_ylim(0, 50)
        for p in ax2.patches:
            h = p.get_height()
            ax2.annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h + 1),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    sns.despine()
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def generate_all_eda_plots(
    df: pd.DataFrame,
    output_dir: Path | str = "docs/images",
) -> dict[str, Path]:
    """Generate and save all EDA plots to specified directory.

    Args:
        df: Input DataFrame.
        output_dir: Destination directory.

    Returns:
        Dictionary mapping plot name to saved file Path.
    """
    out_dir = _ensure_output_dir(output_dir)

    return {
        "target_distribution": plot_target_distribution(df, out_dir),
        "numeric_distributions": plot_numeric_distributions(df, output_dir=out_dir),
        "correlation_heatmap": plot_correlation_heatmap(df, output_dir=out_dir),
        "categorical_churn_rates": plot_categorical_churn_rates(df, output_dir=out_dir),
        "tenure_vs_charges": plot_tenure_vs_charges(df, output_dir=out_dir),
        "fairness_attributes": plot_fairness_attributes(df, output_dir=out_dir),
    }
