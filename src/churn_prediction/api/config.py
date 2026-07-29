"""Configuration loader for serving and API settings."""

from pathlib import Path
from typing import Any

import yaml


def get_default_serving_config_path() -> Path:
    """Return default path to serving.yaml."""
    root_dir = Path(__file__).resolve().parents[3]
    return root_dir / "configs" / "serving.yaml"


def load_serving_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load serving configuration YAML file.

    Args:
        config_path: Optional path to YAML config. Uses default if None.

    Returns:
        Dictionary of serving configuration.
    """
    path = Path(config_path) if config_path else get_default_serving_config_path()
    if not path.is_file():
        return {
            "schema_version": "1.0.0",
            "model_name": "baseline_logistic_regression",
            "model": {
                "model_dir": "models",
                "pipeline_filename": "baseline_pipeline.joblib",
                "metadata_filename": "baseline_metadata.json",
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

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}
    return config
