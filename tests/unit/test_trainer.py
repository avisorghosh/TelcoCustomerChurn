"""Unit tests for baseline model trainer and dataset splitting."""

import pandas as pd
import pytest

from churn_prediction.features.pipeline import load_training_config
from churn_prediction.models.trainer import (
    predict_churn,
    prepare_features_and_target,
    split_dataset,
    train_baseline,
)


@pytest.fixture
def sample_config():
    """Fixture providing default training config."""
    return load_training_config()


@pytest.fixture
def synthetic_dataset():
    """Fixture providing synthetic customer DataFrame with target."""
    rows = []
    for i in range(100):
        churn_val = "Yes" if i % 4 == 0 else "No"
        rows.append(
            {
                "customerID": f"ID-{i:04d}",
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": 1 if i % 5 == 0 else 0,
                "Partner": "Yes" if i % 3 == 0 else "No",
                "Dependents": "No",
                "tenure": (i % 60) + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL" if i % 2 == 0 else "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes" if i % 2 == 0 else "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if i % 2 == 0 else "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 30.0 + (i % 70),
                "TotalCharges": 100.0 + (i * 50),
                "Churn": churn_val,
            }
        )
    return pd.DataFrame(rows)


def test_split_dataset_proportions_and_reproducibility(
    synthetic_dataset, sample_config
):
    """Test stratified splitting dataset proportions and reproducibility."""
    train_df1, val_df1, test_df1 = split_dataset(synthetic_dataset, sample_config)
    train_df2, val_df2, test_df2 = split_dataset(synthetic_dataset, sample_config)

    total_len = len(synthetic_dataset)
    assert len(train_df1) + len(val_df1) + len(test_df1) == total_len

    # Reproducibility check with same seed
    pd.testing.assert_frame_equal(train_df1, train_df2)
    pd.testing.assert_frame_equal(val_df1, val_df2)
    pd.testing.assert_frame_equal(test_df1, test_df2)


def test_prepare_features_and_target(synthetic_dataset, sample_config):
    """Test feature and target separation."""
    X, y = prepare_features_and_target(synthetic_dataset, sample_config)

    assert "Churn" not in X.columns
    assert len(y) == len(synthetic_dataset)
    assert set(y.unique()).issubset({0, 1})
    assert y.iloc[0] == 1  # ID-0000 has Churn="Yes"
    assert y.iloc[1] == 0  # ID-0001 has Churn="No"


def test_deterministic_training_with_fixed_seed(
    tmp_path, sample_config, synthetic_dataset
):
    """Test that training twice with fixed seed produces identical predictions."""
    csv_path = tmp_path / "synthetic_telco.csv"
    synthetic_dataset.to_csv(csv_path, index=False)

    config1 = sample_config.copy()
    config1["data"] = config1.get("data", {}).copy()
    config1["data"]["raw_data_path"] = str(csv_path)
    config1["artifacts"] = config1.get("artifacts", {}).copy()
    config1["artifacts"]["output_dir"] = str(tmp_path / "run1")

    config2 = sample_config.copy()
    config2["data"] = config2.get("data", {}).copy()
    config2["data"]["raw_data_path"] = str(csv_path)
    config2["artifacts"] = config2.get("artifacts", {}).copy()
    config2["artifacts"]["output_dir"] = str(tmp_path / "run2")

    pipeline1, meta1, _ = train_baseline(data_path_override=str(csv_path))
    pipeline2, meta2, _ = train_baseline(data_path_override=str(csv_path))

    sample_customer = synthetic_dataset.iloc[:5].copy()
    preds1 = predict_churn(pipeline1, sample_customer)
    preds2 = predict_churn(pipeline2, sample_customer)

    pd.testing.assert_frame_equal(preds1, preds2)
    assert meta1["random_seed"] == meta2["random_seed"]


def test_predict_churn_single_valid_customer(synthetic_dataset, sample_config):
    """Test predict_churn on a single valid customer input."""
    X, y = prepare_features_and_target(synthetic_dataset, sample_config)
    pipeline, _, _ = train_baseline(data_path_override="Telco-Customer-Churn.csv")

    single_row = synthetic_dataset.iloc[:1].copy()
    results = predict_churn(pipeline, single_row)

    assert "churn_prediction" in results.columns
    assert "churn_probability" in results.columns
    assert len(results) == 1

    prob = results["churn_probability"].iloc[0]
    pred = results["churn_prediction"].iloc[0]

    assert 0.0 <= prob <= 1.0
    assert pred in [0, 1]
