"""Operator entry point for executing reproducible Exploratory Data Analysis (EDA)."""

import argparse
from pathlib import Path

import pandas as pd

from churn_prediction.data.validator import parse_total_charges
from churn_prediction.eda.analysis import (
    assess_leakage_and_fairness,
    get_categorical_summary,
    get_charge_discrepancy_analysis,
    get_dataset_overview,
    get_numeric_correlations,
    get_numeric_summary,
    get_target_distribution,
)
from churn_prediction.eda.plotting import generate_all_eda_plots


def run_eda(
    data_path: str | Path = "Telco-Customer-Churn.csv",
    output_dir: str | Path = "docs/images",
) -> None:
    """Run full EDA pipeline on target dataset and export summary artifacts.

    Args:
        data_path: Path to raw input dataset.
        output_dir: Path to directory where plots will be saved.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source dataset not found at: {path}")

    print(f"Loading dataset from: {path}...")
    df_raw = pd.read_csv(path)
    df, errors = parse_total_charges(df_raw)

    if errors:
        print(f"Warning encountered during TotalCharges parsing: {errors}")

    print("\n" + "=" * 60)
    print("1. DATASET OVERVIEW & QUALITY")
    print("=" * 60)
    overview = get_dataset_overview(df)
    print(f"Total Rows: {overview['total_rows']:,}")
    print(f"Total Columns: {overview['total_cols']}")
    print(f"Duplicate Rows: {overview['duplicate_rows']}")
    print(f"Duplicate Customer IDs: {overview['duplicate_customer_ids']}")
    print(f"Missing Value Counts: {overview['missing_counts']}")

    print("\n" + "=" * 60)
    print("2. TARGET DISTRIBUTION (CHURN)")
    print("=" * 60)
    target_dist = get_target_distribution(df)
    print(f"Churn Counts: {target_dist['counts']}")
    print(f"Churn Percentages: {target_dist['percentages']}")
    print(f"Overall Churn Rate: {target_dist['churn_rate'] * 100:.2f}%")
    print(f"Imbalance Ratio (No/Yes): {target_dist['imbalance_ratio']:.2f}")

    print("\n" + "=" * 60)
    print("3. NUMERIC SUMMARY STATISTICS")
    print("=" * 60)
    numeric_df = get_numeric_summary(df)
    print(numeric_df.to_string())

    print("\n" + "=" * 60)
    print("4. CHARGE DISCREPANCY ANALYSIS (|TotalCharges - tenure*MonthlyCharges|)")
    print("=" * 60)
    disc = get_charge_discrepancy_analysis(df)
    for k, v in disc.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("5. NUMERIC PEARSON CORRELATIONS")
    print("=" * 60)
    corr_df = get_numeric_correlations(df)
    print(corr_df.to_string())

    print("\n" + "=" * 60)
    print("6. KEY CATEGORICAL CHURN RATES")
    print("=" * 60)
    cat_summaries = get_categorical_summary(
        df,
        categorical_cols=[
            "Contract",
            "InternetService",
            "PaymentMethod",
            "TechSupport",
        ],
    )
    for col_name, cat_df in cat_summaries.items():
        print(f"\nFeature: {col_name}")
        print(cat_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("7. LEAKAGE & FAIRNESS ASSESSMENT")
    print("=" * 60)
    lf = assess_leakage_and_fairness(df)
    print("Leakage Findings:")
    for finding in lf["leakage_findings"]:
        print(f"  - {finding}")
    print("\nFairness Findings:")
    for finding in lf["fairness_findings"]:
        print(f"  - {finding}")

    print("\n" + "=" * 60)
    print("8. GENERATING EDA PLOTS")
    print("=" * 60)
    plot_paths = generate_all_eda_plots(df, output_dir=output_dir)
    for plot_name, file_path in plot_paths.items():
        print(f"  Saved {plot_name}: {file_path}")

    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis (EDA).")
    parser.add_argument(
        "--data-path",
        default="Telco-Customer-Churn.csv",
        help="Path to raw source dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/images",
        help="Path to directory where generated figures will be stored.",
    )
    args = parser.parse_args()
    run_eda(data_path=args.data_path, output_dir=args.output_dir)
