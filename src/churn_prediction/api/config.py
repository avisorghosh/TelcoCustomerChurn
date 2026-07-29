"""Configuration loader for serving and API settings."""

import os
from pathlib import Path
from typing import Any

import yaml


def get_default_serving_config_path() -> Path:
    """Return default path to serving.yaml."""
    root_dir = Path(__file__).resolve().parents[3]
    return root_dir / "configs" / "serving.yaml"


def load_serving_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load serving configuration YAML file and apply environment variable overrides.

    Args:
        config_path: Optional path to YAML config. Uses env var CHURN_CONFIG_PATH
            or default path if None.

    Returns:
        Dictionary of serving configuration with environment variable overrides.
    """
    if config_path is None and os.getenv("CHURN_CONFIG_PATH"):
        config_path = os.getenv("CHURN_CONFIG_PATH")

    path = Path(config_path) if config_path else get_default_serving_config_path()
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        config = {
            "schema_version": "1.0.0",
            "model_name": "baseline_logistic_regression",
            "model": {
                "model_dir": "models",
                "pipeline_filename": "serving_pipeline.joblib",
                "metadata_filename": "serving_metadata.json",
            },
            "scoring": {
                "decision_threshold": 0.50,
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "title": "Telco Customer Churn Prediction API",
                "version": "1.0.0",
            },
        }

    # Ensure required section dictionaries exist
    config.setdefault("api", {})
    config.setdefault("model", {})
    config.setdefault("scoring", {})

    # Apply environment variable overrides
    env_host = os.getenv("CHURN_API_HOST") or os.getenv("HOST")
    if env_host:
        config["api"]["host"] = env_host

    env_port = os.getenv("CHURN_API_PORT") or os.getenv("PORT")
    if env_port:
        try:
            config["api"]["port"] = int(env_port)
        except ValueError:
            pass

    env_model_dir = os.getenv("CHURN_MODEL_DIR")
    if env_model_dir:
        config["model"]["model_dir"] = env_model_dir

    env_pipeline_file = os.getenv("CHURN_PIPELINE_FILENAME")
    if env_pipeline_file:
        config["model"]["pipeline_filename"] = env_pipeline_file

    env_metadata_file = os.getenv("CHURN_METADATA_FILENAME")
    if env_metadata_file:
        config["model"]["metadata_filename"] = env_metadata_file

    env_threshold = os.getenv("CHURN_DECISION_THRESHOLD")
    if env_threshold:
        try:
            config["scoring"]["decision_threshold"] = float(env_threshold)
        except ValueError:
            pass

    env_version = os.getenv("CHURN_MODEL_VERSION") or os.getenv("CHURN_API_VERSION")
    if env_version:
        config["api"]["version"] = env_version

    return config
