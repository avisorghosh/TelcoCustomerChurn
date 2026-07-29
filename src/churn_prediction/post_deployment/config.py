"""Post-deployment evaluation configuration loader and validator."""

from pathlib import Path
from typing import Any

import yaml


def get_default_post_deployment_config_path() -> Path:
    """Return the absolute path to the default post_deployment.yaml config."""
    root_dir = Path(__file__).resolve().parents[3]
    return root_dir / "configs" / "post_deployment.yaml"


def load_post_deployment_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the post-deployment evaluation YAML configuration.

    Args:
        config_path: Optional path to YAML config. Uses default if None.

    Returns:
        Dictionary containing post-deployment paths, evaluation rules, campaign settings,
        and retraining decision thresholds.
    """
    path = (
        Path(config_path) if config_path else get_default_post_deployment_config_path()
    )

    if not path.is_file():
        # Sensible fallback defaults if configuration file is not found
        return {
            "paths": {
                "historical_predictions_path": "reports/scoring/batch_predictions.csv",
                "delayed_labels_path": "data/delayed_labels.csv",
                "baseline_metrics_path": "reports/evaluation/evaluation_metrics.json",
                "drift_report_path": "reports/drift/drift_report.json",
                "output_dir": "reports/post_deployment",
                "report_markdown_filename": "post_deployment_report.md",
                "report_json_filename": "post_deployment_summary.json",
            },
            "evaluation": {
                "evaluation_date": "2026-07-29",
                "decision_threshold": 0.50,
                "campaign_capacity": 0.10,
                "n_bins": 10,
            },
            "campaign": {
                "enabled": True,
                "unit_contact_cost": 10.0,
                "unit_customer_value": 500.0,
                "default_treatment_ratio": 0.80,
            },
            "retraining_decision": {
                "pr_auc_drop_threshold": 0.05,
                "brier_score_max": 0.20,
                "psi_alert_threshold": 0.25,
                "min_match_rate": 0.80,
            },
        }

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    return config
