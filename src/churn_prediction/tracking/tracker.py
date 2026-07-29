"""MLflow experiment tracking, lineage recording, and local model registry utilities."""

import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from churn_prediction.data.manifest import SourceManifest, create_source_manifest
from churn_prediction.features.pipeline import load_training_config


def get_git_revision(cwd: Path | str | None = None) -> str:
    """Retrieve current Git commit hash cleanly.

    Returns 'unknown' if Git is unavailable, uninitialized, or fails.

    Args:
        cwd: Optional working directory for git command.

    Returns:
        Git commit hash string or 'unknown'.
    """
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def setup_mlflow(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Configure MLflow tracking URI, experiment name, and registered model name.

    Prefers environment variables over configuration values to allow
    flexible runtime overrides.

    Args:
        config: Training configuration dictionary. Uses default if None.

    Returns:
        Dictionary with active MLflow settings.
    """
    if config is None:
        config = load_training_config()

    mlflow_cfg = config.get("mlflow", {})

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        mlflow_cfg.get("tracking_uri", "file:./mlruns"),
    )
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        mlflow_cfg.get("experiment_name", "telco_customer_churn"),
    )
    registered_model_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME",
        mlflow_cfg.get("registered_model_name", "telco_churn_model"),
    )
    artifact_location = mlflow_cfg.get("artifact_location")

    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    mlflow.set_experiment(experiment_name)

    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
        "artifact_location": artifact_location,
    }


def create_and_log_data_manifest(
    data_path: str | Path,
    schema_version: str = "1.0.0",
) -> SourceManifest:
    """Generate SourceManifest and log dataset lineage parameters to MLflow.

    Args:
        data_path: Path to raw dataset CSV file.
        schema_version: Schema version string.

    Returns:
        SourceManifest object containing dataset checksum and metadata.
    """
    manifest = create_source_manifest(
        file_path=data_path, schema_version=schema_version
    )

    mlflow.log_params(
        {
            "data.source_path": manifest.source_path,
            "data.filename": Path(manifest.source_path).name,
            "data.sha256": manifest.file_sha256,
            "data.schema_version": manifest.schema_version,
            "data.received_timestamp": manifest.received_timestamp,
        }
    )
    mlflow.log_metric("data.row_count", manifest.row_count)
    mlflow.set_tag("data_checksum", manifest.file_sha256)
    mlflow.set_tag("data_filename", Path(manifest.source_path).name)

    return manifest


def create_model_signature(
    pipeline: Pipeline,
    sample_df: pd.DataFrame,
) -> Any:
    """Create MLflow ModelSignature from fitted pipeline and sample input DataFrame.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        sample_df: Sample input DataFrame containing feature columns.

    Returns:
        MLflow ModelSignature containing input and output schema specifications.
    """
    predictions = pipeline.predict(sample_df)
    signature = infer_signature(sample_df, predictions)
    return signature


def log_experiment_run(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    config: dict[str, Any],
    data_path: str | Path | None = None,
    sample_df: pd.DataFrame | None = None,
    artifact_paths: dict[str, Path | str] | None = None,
    register_model: bool = True,
    run_name: str | None = None,
) -> mlflow.ActiveRun:
    """Record complete experiment lineage and log model artifacts in MLflow.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        metadata: Dictionary containing training metadata and metrics.
        config: Training configuration dictionary.
        data_path: Path to dataset used for training.
        sample_df: Optional sample input DataFrame for inferring model signature.
        artifact_paths: Map of artifact names to file paths.
        register_model: Whether to register the model in the local MLflow registry.
        run_name: Optional custom run name.

    Returns:
        Active MLflow run context.
    """
    settings = setup_mlflow(config)
    registered_model_name = settings["registered_model_name"]

    data_cfg = config.get("data", {})
    actual_data_path = data_path or data_cfg.get(
        "raw_data_path", "Telco-Customer-Churn.csv"
    )

    git_commit = get_git_revision()

    if not run_name:
        model_n = metadata.get("model_name", "baseline")
        schema_v = metadata.get("schema_version", "1.0.0")
        run_name = f"{model_n}_{schema_v}"

    active_run = mlflow.start_run(run_name=run_name)

    try:
        # 1. Tags
        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "model_name": metadata.get(
                    "model_name", "baseline_logistic_regression"
                ),
                "model_type": config.get("model", {}).get("type", "LogisticRegression"),
                "schema_version": metadata.get("schema_version", "1.0.0"),
                "training_timestamp": str(metadata.get("timestamp", "")),
            }
        )

        # 2. Data Manifest
        if Path(actual_data_path).exists():
            create_and_log_data_manifest(
                data_path=actual_data_path,
                schema_version=metadata.get("schema_version", "1.0.0"),
            )

        # 3. Parameters
        mlflow.log_params(
            {
                "model_name": str(metadata.get("model_name")),
                "schema_version": str(metadata.get("schema_version")),
                "random_seed": int(metadata.get("random_seed", 42)),
                "model_type": str(config.get("model", {}).get("type")),
                "split.train_size": float(
                    config.get("split", {}).get("train_size", 0.70)
                ),
                "split.val_size": float(config.get("split", {}).get("val_size", 0.15)),
                "split.test_size": float(
                    config.get("split", {}).get("test_size", 0.15)
                ),
                "split.stratify": bool(config.get("split", {}).get("stratify", True)),
                "features.target_column": str(
                    config.get("features", {}).get("target_column", "Churn")
                ),
            }
        )

        # Hyperparameters
        hyperparams = config.get("model", {}).get("hyperparameters", {})
        for k, v in hyperparams.items():
            if v is not None:
                mlflow.log_param(f"hyperparameters.{k}", str(v))

        # 4. Metrics
        metrics = metadata.get("training_metrics", {})
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        split_counts = metadata.get("split_counts", {})
        for k, v in split_counts.items():
            mlflow.log_metric(f"split_counts.{k}", int(v))

        # 5. Model Signature & Model Logging
        signature = None
        if sample_df is not None:
            signature = create_model_signature(pipeline, sample_df)

        model_reg_name = registered_model_name if register_model else None
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_reg_name,
        )

        # 6. Artifact Logging
        if artifact_paths:
            for _, path in artifact_paths.items():
                p = Path(path)
                if p.exists():
                    mlflow.log_artifact(str(p))

        # Log evaluation/plots artifacts if present
        reports_dir = Path("reports")
        if reports_dir.exists():
            for file_path in reports_dir.glob("**/*"):
                if file_path.is_file():
                    rel_dir = file_path.parent.relative_to(reports_dir)
                    artifact_dir = (
                        f"reports/{rel_dir}" if str(rel_dir) != "." else "reports"
                    )
                    mlflow.log_artifact(str(file_path), artifact_path=artifact_dir)

    finally:
        mlflow.end_run()

    return active_run


def load_registered_model(
    model_name: str | None = None,
    version: str | int | None = None,
    stage: str | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Load a registered model from the local MLflow Model Registry.

    Args:
        model_name: Registered model name. Defaults to config setting.
        version: Specific model version number or 'latest'.
        stage: Optional model stage ('Staging', 'Production').
        config: Training configuration dictionary.

    Returns:
        Loaded scikit-learn Pipeline model.
    """
    settings = setup_mlflow(config)
    target_name = model_name or settings["registered_model_name"]

    client = mlflow.tracking.MlflowClient()

    if stage:
        model_uri = f"models:/{target_name}/{stage}"
    elif version and str(version).lower() != "latest":
        model_uri = f"models:/{target_name}/{version}"
    else:
        # Determine latest version
        versions = client.search_model_versions(f"name='{target_name}'")
        if not versions:
            raise ValueError(f"No registered versions found for model '{target_name}'.")
        latest_ver = max(int(v.version) for v in versions)
        model_uri = f"models:/{target_name}/{latest_ver}"

    loaded_model = mlflow.sklearn.load_model(model_uri)
    return loaded_model


def score_with_registered_model(
    df: pd.DataFrame,
    model_name: str | None = None,
    version: str | int | None = None,
    stage: str | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load registered model from MLflow registry and compute churn predictions.

    Args:
        df: Input DataFrame containing customer features.
        model_name: Registered model name.
        version: Model version number or 'latest'.
        stage: Optional model stage.
        config: Training configuration dictionary.

    Returns:
        DataFrame with prediction results (churn_prediction, churn_probability).
    """
    from churn_prediction.models.trainer import predict_churn

    model = load_registered_model(
        model_name=model_name,
        version=version,
        stage=stage,
        config=config,
    )
    return predict_churn(model, df)
