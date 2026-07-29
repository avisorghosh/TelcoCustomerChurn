"""Baseline model training workflow, data splitting, and inference helpers."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_prediction.data.validator import validate_data
from churn_prediction.features.pipeline import (
    build_baseline_pipeline,
    extract_transformed_feature_names,
    load_training_config,
)
from churn_prediction.models.serialization import save_artifacts


def load_and_validate_dataset(
    data_path: str | Path,
    contract_config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw dataset CSV and validate against data contract.

    Args:
        data_path: Path to input CSV file.
        contract_config_path: Optional path to data contract YAML config.

    Returns:
        Parsed and validated pandas DataFrame.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset file not found at: {path}")

    raw_df = pd.read_csv(path)
    validated_df, report = validate_data(
        raw_df,
        config_path=contract_config_path,
        is_training=True,
        raise_on_error=True,
    )
    return validated_df


def split_dataset(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform reproducible stratified train/validation/test splitting.

    Args:
        df: Validated input DataFrame.
        config: Training configuration dictionary.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    split_config = config.get("split", {})
    features_config = config.get("features", {})

    test_size = float(split_config.get("test_size", 0.15))
    val_size = float(split_config.get("val_size", 0.15))
    random_seed = int(split_config.get("random_seed", 42))
    do_stratify = bool(split_config.get("stratify", True))
    target_col = str(features_config.get("target_column", "Churn"))

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' missing from DataFrame.")

    stratify_col = df[target_col] if do_stratify else None

    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_col,
    )

    # Second split off validation set from train_val
    relative_val_size = val_size / (1.0 - test_size)
    val_stratify_col = train_val_df[target_col] if do_stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=val_stratify_col,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_features_and_target(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features X and binary target y from DataFrame.

    Converts target string ('Yes'/'No') to 1/0 integers.

    Args:
        df: Input DataFrame.
        config: Training configuration dictionary.

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    features_config = config.get("features", {})
    target_col = str(features_config.get("target_column", "Churn"))
    pos_val = str(features_config.get("target_positive_value", "Yes"))

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' missing from DataFrame.")

    y = (df[target_col] == pos_val).astype(int)
    X = df.drop(columns=[target_col]).copy()

    return X, y


def train_baseline(
    config_path: str | Path | None = None,
    data_path_override: str | Path | None = None,
) -> tuple[Pipeline, dict[str, Any], dict[str, Path]]:
    """Execute complete baseline training workflow and serialize artifacts.

    Args:
        config_path: Optional path to training YAML config.
        data_path_override: Optional path to dataset override.

    Returns:
        Tuple of (fitted_pipeline, metadata_dict, artifact_paths_dict).
    """
    config = load_training_config(config_path)

    data_config = config.get("data", {})
    data_path = data_path_override or data_config.get(
        "raw_data_path", "Telco-Customer-Churn.csv"
    )
    contract_config_path = data_config.get("contract_config_path")

    # 1. Ingest and validate data
    validated_df = load_and_validate_dataset(
        data_path=data_path,
        contract_config_path=contract_config_path,
    )

    # 2. Stratified train / validation / test split
    train_df, val_df, test_df = split_dataset(validated_df, config)

    # 3. Separate features and target
    X_train, y_train = prepare_features_and_target(train_df, config)
    X_val, y_val = prepare_features_and_target(val_df, config)
    X_test, y_test = prepare_features_and_target(test_df, config)

    # 4. Build pipeline
    pipeline = build_baseline_pipeline(config)

    # 5. Fit preprocessing and classifier ONLY on training split (no leakage)
    pipeline.fit(X_train, y_train)

    # 6. Evaluate basic training validation scores
    train_acc = float(pipeline.score(X_train, y_train))
    val_acc = float(pipeline.score(X_val, y_val))
    test_acc = float(pipeline.score(X_test, y_test))

    features_config = config.get("features", {})
    numeric_features = list(features_config.get("numeric_features", []))
    categorical_features = list(features_config.get("categorical_features", []))
    excluded_features = list(features_config.get("excluded_features", []))

    feature_names_in = numeric_features + categorical_features
    feature_names_out = extract_transformed_feature_names(pipeline)

    # 7. Construct metadata
    metadata = {
        "model_name": config.get("model_name", "baseline_logistic_regression"),
        "schema_version": config.get("schema_version", "1.0.0"),
        "random_seed": int(config.get("split", {}).get("random_seed", 42)),
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(validated_df),
        },
        "feature_names_in": feature_names_in,
        "feature_names_out": feature_names_out,
        "excluded_features": excluded_features,
        "training_metrics": {
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
        },
        "training_config": config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 8. Save artifacts
    artifacts_config = config.get("artifacts", {})
    output_dir = artifacts_config.get("output_dir", "models")
    pipeline_filename = artifacts_config.get(
        "pipeline_filename", "baseline_pipeline.joblib"
    )
    metadata_filename = artifacts_config.get(
        "metadata_filename", "baseline_metadata.json"
    )

    pipeline_path, metadata_path = save_artifacts(
        pipeline=pipeline,
        metadata=metadata,
        output_dir=output_dir,
        pipeline_filename=pipeline_filename,
        metadata_filename=metadata_filename,
    )

    artifact_paths = {
        "pipeline_path": pipeline_path,
        "metadata_path": metadata_path,
    }

    return pipeline, metadata, artifact_paths


def predict_churn(
    pipeline: Pipeline,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run inference using trained pipeline on input customer DataFrame.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        df: DataFrame containing customer features.

    Returns:
        DataFrame with columns 'churn_prediction' (0/1) and 'churn_probability' (float).
    """
    probas = pipeline.predict_proba(df)[:, 1]
    predictions = pipeline.predict(df)

    results_df = pd.DataFrame(
        {
            "churn_prediction": predictions,
            "churn_probability": np.round(probas, 4),
        },
        index=df.index,
    )

    return results_df
