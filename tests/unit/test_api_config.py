"""Unit tests for API and serving configuration loading with environment variables."""

import os
from pathlib import Path
from unittest.mock import patch

from churn_prediction.api.config import (
    get_default_serving_config_path,
    load_serving_config,
)


def test_default_serving_config_path_exists() -> None:
    """Test that default serving config path points to valid location."""
    path = get_default_serving_config_path()
    assert isinstance(path, Path)
    assert path.name == "serving.yaml"


def test_load_serving_config_defaults() -> None:
    """Test loading serving config from default YAML file."""
    config = load_serving_config()
    assert "api" in config
    assert "model" in config
    assert "scoring" in config
    assert config["api"]["port"] == 8000


def test_load_serving_config_env_overrides(tmp_path: Path) -> None:
    """Test environment variables override configuration values."""
    env_vars = {
        "CHURN_API_HOST": "0.0.0.0",
        "CHURN_API_PORT": "9090",
        "CHURN_MODEL_DIR": str(tmp_path / "custom_models"),
        "CHURN_PIPELINE_FILENAME": "custom_pipeline.joblib",
        "CHURN_METADATA_FILENAME": "custom_metadata.json",
        "CHURN_DECISION_THRESHOLD": "0.35",
        "CHURN_MODEL_VERSION": "2.0.0",
    }
    with patch.dict(os.environ, env_vars):
        config = load_serving_config()
        assert config["api"]["host"] == "0.0.0.0"
        assert config["api"]["port"] == 9090
        assert config["model"]["model_dir"] == str(tmp_path / "custom_models")
        assert config["model"]["pipeline_filename"] == "custom_pipeline.joblib"
        assert config["model"]["metadata_filename"] == "custom_metadata.json"
        assert config["scoring"]["decision_threshold"] == 0.35
        assert config["api"]["version"] == "2.0.0"


def test_load_serving_config_missing_file(tmp_path: Path) -> None:
    """Test fallback configuration dictionary when file does not exist."""
    missing_path = tmp_path / "non_existent.yaml"
    config = load_serving_config(missing_path)
    assert config["schema_version"] == "1.0.0"
    assert config["api"]["port"] == 8000
    assert config["model"]["model_dir"] == "models"
