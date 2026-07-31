"""Contract tests for post-deployment configuration schema and rules."""

from pathlib import Path

import yaml

from churn_prediction.post_deployment import (
    get_default_post_deployment_config_path,
    load_post_deployment_config,
)


def test_default_config_file_exists_and_valid_yaml() -> None:
    """Verify default post_deployment.yaml config file exists and is valid YAML."""
    config_path = get_default_post_deployment_config_path()
    assert config_path.is_file(), f"Config file missing at: {config_path}"

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict)
    assert "paths" in data
    assert "evaluation" in data
    assert "campaign" in data
    assert "retraining_decision" in data


def test_load_post_deployment_config_structure() -> None:
    """Verify load_post_deployment_config loads required keys and valid types."""
    config = load_post_deployment_config()

    paths = config.get("paths", {})
    assert "historical_predictions_path" in paths
    assert "delayed_labels_path" in paths
    assert "output_dir" in paths

    eval_cfg = config.get("evaluation", {})
    assert 0.0 <= float(eval_cfg.get("decision_threshold", 0.5)) <= 1.0
    assert 0.0 < float(eval_cfg.get("campaign_capacity", 0.1)) <= 1.0

    campaign_cfg = config.get("campaign", {})
    assert float(campaign_cfg.get("unit_contact_cost", 0)) >= 0.0
    assert float(campaign_cfg.get("unit_customer_value", 0)) >= 0.0

    retrain_cfg = config.get("retraining_decision", {})
    assert 0.0 <= float(retrain_cfg.get("pr_auc_drop_threshold", 0.05)) <= 1.0
    assert 0.0 <= float(retrain_cfg.get("brier_score_max", 0.20)) <= 1.0
    assert 0.0 <= float(retrain_cfg.get("psi_alert_threshold", 0.25)) <= 1.0
    assert 0.0 <= float(retrain_cfg.get("min_match_rate", 0.80)) <= 1.0


def test_config_fallback_if_missing_file(tmp_path: Path) -> None:
    """Verify fallback dictionary returned if config file does not exist."""
    non_existent_path = tmp_path / "non_existent_config.yaml"
    config = load_post_deployment_config(non_existent_path)

    assert isinstance(config, dict)
    assert "paths" in config
    assert "retraining_decision" in config
