"""Evaluation pipeline configuration loader and validator."""

from pathlib import Path
from typing import Any

import yaml


def get_default_evaluation_config_path() -> Path:
    """Return the absolute path to the default evaluation.yaml config."""
    root_dir = Path(__file__).resolve().parents[3]
    return root_dir / "configs" / "evaluation.yaml"


def load_evaluation_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the evaluation YAML configuration file.

    Args:
        config_path: Optional path to YAML config. Uses default if None.

    Returns:
        Dictionary containing evaluation parameters, paths, and split config.
    """
    path = Path(config_path) if config_path else get_default_evaluation_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
