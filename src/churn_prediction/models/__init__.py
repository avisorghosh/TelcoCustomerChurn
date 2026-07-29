"""Model training, serialization, and inference module."""

from churn_prediction.models.serialization import (
    load_artifacts,
    load_metadata,
    load_pipeline,
    save_artifacts,
)
from churn_prediction.models.trainer import (
    load_and_validate_dataset,
    predict_churn,
    prepare_features_and_target,
    split_dataset,
    train_baseline,
)

__all__ = [
    "load_and_validate_dataset",
    "split_dataset",
    "prepare_features_and_target",
    "train_baseline",
    "predict_churn",
    "save_artifacts",
    "load_pipeline",
    "load_metadata",
    "load_artifacts",
]
