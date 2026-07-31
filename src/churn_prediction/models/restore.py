"""Model rollback and artifact restoration utilities."""

import logging
import shutil
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from churn_prediction.models.serialization import load_artifacts, save_artifacts
from churn_prediction.tracking.tracker import load_registered_model, setup_mlflow

logger = logging.getLogger(__name__)


def restore_model_from_registry(
    version: int | str,
    registered_model_name: str | None = None,
    target_dir: str | Path = "models",
    pipeline_filename: str = "baseline_pipeline.joblib",
    metadata_filename: str = "baseline_metadata.json",
    config: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Restore a registered model version from MLflow to target directory.

    Args:
        version: Model version number in MLflow registry.
        registered_model_name: Optional MLflow model name. Defaults to config setting.
        target_dir: Destination directory for restored model artifacts.
        pipeline_filename: Target pipeline filename.
        metadata_filename: Target metadata filename.
        config: Training configuration dictionary.

    Returns:
        Tuple of (pipeline_path, metadata_path, metadata_dict).
    """
    settings = setup_mlflow(config)
    model_name = registered_model_name or settings["registered_model_name"]
    version_str = str(version)

    pipeline = load_registered_model(
        model_name=model_name,
        version=version_str,
        config=config,
    )

    client = MlflowClient()
    mv = client.get_model_version(name=model_name, version=version_str)
    run_id = mv.run_id
    run = client.get_run(run_id)

    metadata = {
        "model_name": run.data.tags.get("model_name", model_name),
        "schema_version": run.data.tags.get("schema_version", "1.0.0"),
        "model_version": version_str,
        "mlflow_run_id": run_id,
        "mlflow_model_name": model_name,
        "timestamp": run.data.tags.get(
            "training_timestamp", str(mv.creation_timestamp)
        ),
        "restored_from_version": version_str,
        "training_metrics": {k: float(v) for k, v in run.data.metrics.items()},
        "random_seed": int(run.data.params.get("random_seed", 42)),
    }

    pipe_path, meta_path = save_artifacts(
        pipeline=pipeline,
        metadata=metadata,
        output_dir=target_dir,
        pipeline_filename=pipeline_filename,
        metadata_filename=metadata_filename,
    )

    logger.info(f"Restored version {version_str} of '{model_name}' to {target_dir}")
    return pipe_path, meta_path, metadata


def restore_model_from_dir(
    source_dir: str | Path,
    target_dir: str | Path = "models",
    pipeline_filename: str = "baseline_pipeline.joblib",
    metadata_filename: str = "baseline_metadata.json",
) -> tuple[Path, Path, dict[str, Any]]:
    """Restore model artifacts from local backup directory to target active directory.

    Args:
        source_dir: Source directory containing model artifact files.
        target_dir: Target active artifact directory.
        pipeline_filename: Filename of pipeline artifact.
        metadata_filename: Filename of metadata artifact.

    Returns:
        Tuple of (pipeline_path, metadata_path, metadata_dict).
    """
    src_dir = Path(source_dir)
    tgt_dir = Path(target_dir)

    src_pipeline = src_dir / pipeline_filename
    src_metadata = src_dir / metadata_filename

    if not src_pipeline.exists() or not src_metadata.exists():
        msg = f"Source directory {src_dir} missing required artifact files."
        raise FileNotFoundError(msg)

    tgt_dir.mkdir(parents=True, exist_ok=True)
    tgt_pipeline = tgt_dir / pipeline_filename
    tgt_metadata = tgt_dir / metadata_filename

    shutil.copy2(src_pipeline, tgt_pipeline)
    shutil.copy2(src_metadata, tgt_metadata)

    _pipeline, metadata = load_artifacts(
        output_dir=tgt_dir,
        pipeline_filename=pipeline_filename,
        metadata_filename=metadata_filename,
    )

    logger.info(f"Restored model artifacts from {src_dir} to {tgt_dir}")
    return tgt_pipeline, tgt_metadata, metadata
