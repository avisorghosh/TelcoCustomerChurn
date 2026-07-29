"""Exploratory Data Analysis (EDA) module for churn prediction dataset."""

from churn_prediction.eda.analysis import (
    assess_leakage_and_fairness,
    get_categorical_summary,
    get_charge_discrepancy_analysis,
    get_dataset_overview,
    get_numeric_correlations,
    get_numeric_summary,
    get_target_distribution,
)
from churn_prediction.eda.plotting import (
    generate_all_eda_plots,
    plot_categorical_churn_rates,
    plot_correlation_heatmap,
    plot_fairness_attributes,
    plot_numeric_distributions,
    plot_target_distribution,
    plot_tenure_vs_charges,
)

__all__ = [
    "get_dataset_overview",
    "get_target_distribution",
    "get_numeric_summary",
    "get_categorical_summary",
    "get_numeric_correlations",
    "get_charge_discrepancy_analysis",
    "assess_leakage_and_fairness",
    "plot_target_distribution",
    "plot_numeric_distributions",
    "plot_correlation_heatmap",
    "plot_categorical_churn_rates",
    "plot_tenure_vs_charges",
    "plot_fairness_attributes",
    "generate_all_eda_plots",
]
