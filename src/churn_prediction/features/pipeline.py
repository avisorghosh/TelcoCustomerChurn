"""Preprocessing transformer and baseline pipeline builders."""

from pathlib import Path
from typing import Any

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_default_training_config_path() -> Path:
    """Return the absolute path to the default training.yaml config."""
    root_dir = Path(__file__).resolve().parents[3]
    return root_dir / "configs" / "training.yaml"


def load_training_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the training YAML configuration.

    Args:
        config_path: Optional path to YAML config. Uses default if None.

    Returns:
        Dictionary containing training configuration.
    """
    path = Path(config_path) if config_path else get_default_training_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found at: {path}")

    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def build_preprocessing_transformer(config: dict[str, Any]) -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for baseline preprocessing.

    Applies median imputation + StandardScaler to numeric features and
    most frequent imputation + OneHotEncoder to categorical features.
    Explicitly excludes sensitive and identifier features.

    Args:
        config: Training configuration dictionary.

    Returns:
        Configured ColumnTransformer instance.
    """
    features_config = config.get("features", {})
    numeric_features = list(features_config.get("numeric_features", []))
    categorical_features = list(features_config.get("categorical_features", []))
    excluded_features = set(features_config.get("excluded_features", []))

    # Enforce strict feature exclusion
    numeric_features = [f for f in numeric_features if f not in excluded_features]
    categorical_features = [
        f for f in categorical_features if f not in excluded_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def build_model_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build scikit-learn training pipeline based on configuration settings.

    Combines ColumnTransformer preprocessor and specified classifier
    (LogisticRegression or GradientBoosting).

    Args:
        config: Training configuration dictionary.

    Returns:
        Complete scikit-learn Pipeline instance.
    """
    preprocessor = build_preprocessing_transformer(config)

    model_config = config.get("model", {})
    model_type = str(model_config.get("type", "LogisticRegression"))
    hyperparams = {
        k: v
        for k, v in model_config.get("hyperparameters", {}).items()
        if v is not None
    }

    m_type = model_type.lower()
    if m_type in ("gradientboosting", "gradientboostingclassifier", "gb", "gbc"):
        from sklearn.ensemble import GradientBoostingClassifier

        classifier = GradientBoostingClassifier(**hyperparams)
    elif m_type in ("histgradientboosting", "histgradientboostingclassifier"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        classifier = HistGradientBoostingClassifier(**hyperparams)
    elif m_type in ("logisticregression", "logistic_regression"):
        classifier = LogisticRegression(**hyperparams)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def extract_transformed_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Extract encoded feature names out of a fitted preprocessing ColumnTransformer.

    Args:
        fitted_pipeline: Fitted scikit-learn Pipeline.

    Returns:
        List of feature names produced by the preprocessing transformer.
    """
    preprocessor = fitted_pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        return []
    try:
        names: list[str] = list(preprocessor.get_feature_names_out())
        return names
    except Exception:
        return []
