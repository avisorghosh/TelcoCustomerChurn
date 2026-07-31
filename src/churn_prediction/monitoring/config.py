"""Observability configuration loader."""

from pathlib import Path
from typing import Any

import yaml


def get_default_observability_config_path() -> Path:
    """Return absolute path to default observability configuration YAML."""
    root_dir = Path(__file__).resolve().parents[3]
    return root_dir / "configs" / "observability.yaml"


def load_observability_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load observability configuration dictionary.

    Args:
        config_path: Optional path to YAML config. Uses default if None.

    Returns:
        Dictionary containing observability configuration settings.
    """
    path = Path(config_path) if config_path else get_default_observability_config_path()
    if not path.is_file():
        # Fallback to sensible default configuration if config file missing
        return {
            "logging": {"level": "INFO", "format": "json", "privacy_safe": True},
            "metrics": {"enabled": True, "endpoint_path": "/metrics"},
            "data_quality": {"output_dir": "reports/quality"},
            "drift": {
                "output_dir": "reports/drift",
                "thresholds": {"psi_warning": 0.10, "psi_critical": 0.25},
            },
        }

    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
