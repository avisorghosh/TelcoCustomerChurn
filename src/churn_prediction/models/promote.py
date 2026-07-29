"""Minimal model promotion helpers for serving consistency."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from churn_prediction.models.batch_scoring import load_serving_config


class PromotionError(Exception):
    """Raised when selected model artifacts cannot be promoted to serving."""


def _resolve_artifact_names(
    selected_model_name: str,
    baseline_pipeline: str = "baseline_pipeline.joblib",
    baseline_metadata: str = "baseline_metadata.json",
    candidate_pipeline: str = "candidate_pipeline.joblib",
    candidate_metadata: str = "candidate_metadata.json",
) -> tuple[str, str]:
    """Map a selected model name to source artifact filenames."""
    name = selected_model_name.lower()
    if "candidate" in name or "gradient" in name or "boost" in name:
        return candidate_pipeline, candidate_metadata
    return baseline_pipeline, baseline_metadata


def load_decision_record(decision_record_path: str | Path) -> dict[str, Any]:
    """Load a decision record JSON file."""
    path = Path(decision_record_path)
    if not path.is_file():
        raise PromotionError(f"Decision record not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        record: dict[str, Any] = json.load(f)
    return record


def promote_selected_model(
    decision_record_path: str | Path = "reports/evaluation/decision_record.json",
    model_dir: str | Path = "models",
    serving_config_path: str | Path | None = "configs/serving.yaml",
    serving_pipeline_filename: str = "serving_pipeline.joblib",
    serving_metadata_filename: str = "serving_metadata.json",
    update_serving_config: bool = True,
) -> dict[str, Path]:
    """Promote the decision-record winner into fixed serving artifact names.

    Copies the selected pipeline/metadata into ``serving_pipeline.joblib`` and
    ``serving_metadata.json`` under ``model_dir``, optionally updating
    ``configs/serving.yaml`` to point at those filenames.

    Args:
        decision_record_path: Path to comparison decision record JSON.
        model_dir: Directory containing trained baseline/candidate artifacts.
        serving_config_path: Optional serving YAML to update.
        serving_pipeline_filename: Destination serving pipeline filename.
        serving_metadata_filename: Destination serving metadata filename.
        update_serving_config: When True, rewrite serving model filenames.

    Returns:
        Dictionary with paths to promoted pipeline and metadata files.
    """
    record = load_decision_record(decision_record_path)
    final_decision = record.get("final_decision", {})
    selected_name = str(
        final_decision.get("selected_model_name", "baseline_logistic_regression")
    )

    src_pipeline_name, src_metadata_name = _resolve_artifact_names(selected_name)
    model_path = Path(model_dir)
    src_pipeline = model_path / src_pipeline_name
    src_metadata = model_path / src_metadata_name

    if not src_pipeline.is_file():
        raise PromotionError(f"Selected pipeline artifact missing: {src_pipeline}")
    if not src_metadata.is_file():
        raise PromotionError(f"Selected metadata artifact missing: {src_metadata}")

    model_path.mkdir(parents=True, exist_ok=True)
    dest_pipeline = model_path / serving_pipeline_filename
    dest_metadata = model_path / serving_metadata_filename
    shutil.copy2(src_pipeline, dest_pipeline)
    shutil.copy2(src_metadata, dest_metadata)

    if update_serving_config and serving_config_path is not None:
        config_path = Path(serving_config_path)
        if not config_path.is_file():
            raise PromotionError(f"Serving config not found at: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        model_cfg = config.setdefault("model", {})
        model_cfg["model_dir"] = str(model_path)
        model_cfg["pipeline_filename"] = serving_pipeline_filename
        model_cfg["metadata_filename"] = serving_metadata_filename
        config["model_name"] = selected_name
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    return {
        "pipeline_path": dest_pipeline,
        "metadata_path": dest_metadata,
    }


def get_serving_artifact_paths(
    serving_config_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return absolute paths to the currently configured serving artifacts."""
    config = load_serving_config(serving_config_path)
    model_cfg = config.get("model", {})
    model_dir = Path(model_cfg.get("model_dir", "models"))
    pipeline = model_dir / model_cfg.get("pipeline_filename", "serving_pipeline.joblib")
    metadata = model_dir / model_cfg.get("metadata_filename", "serving_metadata.json")
    return pipeline, metadata
