"""Unit tests for evaluation configuration loading."""

import pytest

from churn_prediction.evaluation.config import (
    get_default_evaluation_config_path,
    load_evaluation_config,
)


def test_get_default_evaluation_config_path_exists() -> None:
    """Verify default evaluation configuration file path exists."""
    path = get_default_evaluation_config_path()
    assert path.is_file()
    assert path.name == "evaluation.yaml"


def test_load_evaluation_config_structure() -> None:
    """Verify loaded evaluation configuration structure and keys."""
    config = load_evaluation_config()
    assert "evaluation" in config
    assert "paths" in config
    assert "split" in config

    eval_cfg = config["evaluation"]
    assert "evaluation_threshold" in eval_cfg
    assert "campaign_capacity" in eval_cfg
    assert "evaluation_split" in eval_cfg
    assert eval_cfg["campaign_capacity"] == 0.10


def test_load_evaluation_config_file_not_found() -> None:
    """Verify FileNotFoundError is raised for non-existent config path."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_evaluation_config("non_existent_config.yaml")
