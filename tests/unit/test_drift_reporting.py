"""Unit tests for lightweight data drift reporting."""

from pathlib import Path

import numpy as np
import pandas as pd

from churn_prediction.monitoring.drift import (
    calculate_categorical_psi,
    calculate_ks_stat,
    calculate_numerical_psi,
    detect_data_drift,
)


def test_psi_and_ks_calculations() -> None:
    """Test PSI and KS statistic calculation functions."""
    ref_num = pd.Series(np.random.normal(loc=50, scale=10, size=1000))
    same_num = pd.Series(np.random.normal(loc=50, scale=10, size=1000))
    drifted_num = pd.Series(np.random.normal(loc=100, scale=10, size=1000))

    psi_same = calculate_numerical_psi(ref_num, same_num)
    psi_drifted = calculate_numerical_psi(ref_num, drifted_num)

    assert psi_same < 0.10
    assert psi_drifted > 0.25

    ks_stat, ks_p = calculate_ks_stat(ref_num, drifted_num)
    assert ks_stat > 0.5
    assert ks_p < 0.05

    ref_cat = pd.Series(["Month-to-month"] * 700 + ["Two year"] * 300)
    same_cat = pd.Series(["Month-to-month"] * 700 + ["Two year"] * 300)
    drifted_cat = pd.Series(["Month-to-month"] * 100 + ["Two year"] * 900)

    cat_psi_same = calculate_categorical_psi(ref_cat, same_cat)
    cat_psi_drifted = calculate_categorical_psi(ref_cat, drifted_cat)

    assert cat_psi_same < 0.10
    assert cat_psi_drifted > 0.25


def test_detect_data_drift_no_drift(tmp_path: Path) -> None:
    """Test drift detection when scoring data matches reference data."""
    ref_df = pd.DataFrame(
        {
            "tenure": np.random.randint(1, 72, size=500),
            "MonthlyCharges": np.random.uniform(20, 120, size=500),
            "TotalCharges": np.random.uniform(100, 5000, size=500),
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"], size=500
            ),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=500),
            "PaymentMethod": np.random.choice(
                ["Electronic check", "Mailed check"], size=500
            ),
        }
    )

    out_file = tmp_path / "drift_no_drift.json"
    report = detect_data_drift(
        scoring_df=ref_df, reference_df=ref_df, output_path=out_file
    )

    assert report.overall_drift_status == "no_drift"
    assert report.max_psi < 0.10
    assert report.drifted_features_count == 0
    assert out_file.is_file()


def test_detect_data_drift_synthetic_drift(tmp_path: Path) -> None:
    """Test drift detection on synthetic drifted dataset."""
    ref_df = pd.DataFrame(
        {
            "tenure": [12] * 200,
            "MonthlyCharges": [30.0] * 200,
            "TotalCharges": [360.0] * 200,
            "Contract": ["Month-to-month"] * 200,
            "InternetService": ["DSL"] * 200,
            "PaymentMethod": ["Mailed check"] * 200,
        }
    )

    drifted_df = pd.DataFrame(
        {
            "tenure": [72] * 200,  # Extreme tenure shift
            "MonthlyCharges": [150.0] * 200,  # Extreme charge shift
            "TotalCharges": [10800.0] * 200,
            "Contract": ["Two year"] * 200,  # Extreme contract shift
            "InternetService": ["Fiber optic"] * 200,
            "PaymentMethod": ["Bank transfer (automatic)"] * 200,
        }
    )

    out_file = tmp_path / "drift_synthetic.json"
    report = detect_data_drift(
        scoring_df=drifted_df, reference_df=ref_df, output_path=out_file
    )

    assert report.overall_drift_status == "significant_drift"
    assert report.max_psi >= 0.25
    assert report.drifted_features_count > 0
    assert out_file.is_file()
