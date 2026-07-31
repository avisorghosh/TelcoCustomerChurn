"""Unit tests for EDA analysis and plotting modules."""

from pathlib import Path

import pandas as pd
import pytest

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
)


@pytest.fixture
def sample_eda_df() -> pd.DataFrame:
    """Fixture providing a sample DataFrame for EDA unit tests."""
    return pd.DataFrame(
        {
            "customerID": [f"ID-{i}" for i in range(10)],
            "gender": ["Female", "Male"] * 5,
            "SeniorCitizen": [0, 1] * 5,
            "Partner": ["Yes", "No"] * 5,
            "Dependents": ["No", "Yes"] * 5,
            "tenure": [1, 12, 24, 36, 48, 60, 72, 0, 10, 20],
            "PhoneService": ["Yes"] * 10,
            "MultipleLines": ["No"] * 10,
            "InternetService": ["Fiber optic", "DSL"] * 5,
            "OnlineSecurity": ["No", "Yes"] * 5,
            "OnlineBackup": ["Yes", "No"] * 5,
            "DeviceProtection": ["No", "Yes"] * 5,
            "TechSupport": ["No", "Yes"] * 5,
            "StreamingTV": ["Yes", "No"] * 5,
            "StreamingMovies": ["No", "Yes"] * 5,
            "Contract": ["Month-to-month", "One year"] * 5,
            "PaperlessBilling": ["Yes", "No"] * 5,
            "PaymentMethod": ["Electronic check", "Mailed check"] * 5,
            "MonthlyCharges": [
                70.0,
                50.0,
                80.0,
                90.0,
                60.0,
                100.0,
                40.0,
                20.0,
                75.0,
                85.0,
            ],
            "TotalCharges": [
                70.0,
                600.0,
                1920.0,
                3240.0,
                2880.0,
                6000.0,
                2880.0,
                None,
                750.0,
                1700.0,
            ],
            "Churn": ["Yes", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No"],
        }
    )


def test_get_dataset_overview(sample_eda_df: pd.DataFrame) -> None:
    """Test get_dataset_overview computes metrics accurately."""
    overview = get_dataset_overview(sample_eda_df)
    assert overview["total_rows"] == 10
    assert overview["total_cols"] == 21
    assert overview["duplicate_rows"] == 0
    assert overview["duplicate_customer_ids"] == 0
    assert overview["missing_counts"]["TotalCharges"] == 1
    assert overview["missing_percentages"]["TotalCharges"] == 10.0


def test_get_target_distribution(sample_eda_df: pd.DataFrame) -> None:
    """Test get_target_distribution computes target metrics accurately."""
    dist = get_target_distribution(sample_eda_df)
    assert dist["counts"]["Yes"] == 3
    assert dist["counts"]["No"] == 7
    assert dist["churn_rate"] == 0.3
    assert pytest.approx(dist["imbalance_ratio"], 0.01) == 2.33


def test_get_target_distribution_invalid_col(sample_eda_df: pd.DataFrame) -> None:
    """Test get_target_distribution raises ValueError for missing target."""
    with pytest.raises(ValueError, match="Target column 'NonExistent' not found"):
        get_target_distribution(sample_eda_df, target_col="NonExistent")


def test_get_numeric_summary(sample_eda_df: pd.DataFrame) -> None:
    """Test get_numeric_summary computes summary statistics."""
    summary = get_numeric_summary(sample_eda_df)
    assert "tenure" in summary.index
    assert "MonthlyCharges" in summary.index
    assert "TotalCharges" in summary.index
    assert summary.loc["tenure", "count"] == 10
    assert summary.loc["TotalCharges", "count"] == 9


def test_get_categorical_summary(sample_eda_df: pd.DataFrame) -> None:
    """Test get_categorical_summary returns category frequency tables."""
    cat_summary = get_categorical_summary(sample_eda_df)
    assert "Contract" in cat_summary
    contract_df = cat_summary["Contract"]
    assert "category" in contract_df.columns
    assert "churn_rate" in contract_df.columns


def test_get_numeric_correlations(sample_eda_df: pd.DataFrame) -> None:
    """Test get_numeric_correlations computes correlation matrix."""
    corr = get_numeric_correlations(sample_eda_df)
    assert corr.shape == (3, 3)
    assert corr.loc["tenure", "tenure"] == 1.0


def test_get_charge_discrepancy_analysis(sample_eda_df: pd.DataFrame) -> None:
    """Test get_charge_discrepancy_analysis calculates discrepancy stats."""
    disc = get_charge_discrepancy_analysis(sample_eda_df)
    assert disc["blank_total_charges_count"] == 1.0
    assert "mean_abs_diff" in disc
    assert "median_abs_diff" in disc


def test_assess_leakage_and_fairness(sample_eda_df: pd.DataFrame) -> None:
    """Test assess_leakage_and_fairness evaluates leakage and sensitive features."""
    lf = assess_leakage_and_fairness(sample_eda_df)
    assert lf["customer_id_unique"] is True
    assert "Female" in lf["gender_churn_rates"]
    assert len(lf["fairness_findings"]) > 0
    assert len(lf["leakage_findings"]) > 0


def test_generate_all_eda_plots(sample_eda_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test all EDA plot generation functions save files to specified directory."""
    plots = generate_all_eda_plots(sample_eda_df, output_dir=tmp_path)
    assert len(plots) == 6
    for _plot_name, file_path in plots.items():
        assert file_path.exists()
        assert file_path.stat().st_size > 0
