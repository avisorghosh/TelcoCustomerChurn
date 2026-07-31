"""Model training, serialization, and inference module."""

from churn_prediction.models.batch_scoring import (
    BatchScoringError,
    BatchValidationError,
    ModelLoadError,
    generate_batch_id,
    load_serving_config,
    quarantine_batch,
    run_batch_scoring,
    score_batch,
    validate_scoring_batch,
)
from churn_prediction.models.promote import (
    PromotionError,
    promote_selected_model,
)
from churn_prediction.models.restore import (
    restore_model_from_dir,
    restore_model_from_registry,
)
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
    train_candidate,
    train_model,
)

__all__ = [
    "BatchScoringError",
    "BatchValidationError",
    "ModelLoadError",
    "PromotionError",
    "generate_batch_id",
    "load_and_validate_dataset",
    "load_artifacts",
    "load_metadata",
    "load_pipeline",
    "load_serving_config",
    "predict_churn",
    "prepare_features_and_target",
    "promote_selected_model",
    "quarantine_batch",
    "restore_model_from_dir",
    "restore_model_from_registry",
    "run_batch_scoring",
    "save_artifacts",
    "score_batch",
    "split_dataset",
    "train_baseline",
    "train_candidate",
    "train_model",
    "validate_scoring_batch",
]
