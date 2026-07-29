"""Integration test for end-to-end baseline training and inference."""

import pandas as pd

from churn_prediction.models.serialization import load_artifacts
from churn_prediction.models.trainer import predict_churn, train_baseline


def test_end_to_end_baseline_training_and_reload(tmp_path):
    """Test end-to-end training, persistence, reload, and feature exclusion."""

    # 1. Execute baseline training outputting to tmp_path
    pipeline, metadata, artifact_paths = train_baseline(
        data_path_override="Telco-Customer-Churn.csv",
    )

    assert artifact_paths["pipeline_path"].is_file()
    assert artifact_paths["metadata_path"].is_file()

    # 2. Check metadata fields
    assert metadata["model_name"] == "baseline_logistic_regression"
    assert metadata["random_seed"] == 42
    assert "excluded_features" in metadata
    assert set(metadata["excluded_features"]) == {
        "customerID",
        "gender",
        "SeniorCitizen",
    }

    # Verify feature names in & out do not contain excluded features
    for col in metadata["feature_names_in"]:
        assert col not in metadata["excluded_features"]

    for col in metadata["feature_names_out"]:
        assert "customerID" not in col
        assert "gender" not in col
        assert "SeniorCitizen" not in col

    # 3. Test deserialization and reload
    output_dir = artifact_paths["pipeline_path"].parent
    reloaded_pipeline, reloaded_metadata = load_artifacts(output_dir)

    # 4. Load a sample customer from the actual dataset
    raw_df = pd.read_csv("Telco-Customer-Churn.csv")
    sample_df = raw_df.iloc[:10].copy()

    orig_predictions = predict_churn(pipeline, sample_df)
    reloaded_predictions = predict_churn(reloaded_pipeline, sample_df)

    # 5. Pipeline reload produces identical predictions
    pd.testing.assert_frame_equal(orig_predictions, reloaded_predictions)
