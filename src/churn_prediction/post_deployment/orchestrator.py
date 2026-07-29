"""Post-deployment evaluation workflow orchestrator."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from churn_prediction.post_deployment.campaign import evaluate_campaign_effectiveness
from churn_prediction.post_deployment.config import load_post_deployment_config
from churn_prediction.post_deployment.delayed_labels import (
    evaluate_delayed_predictions,
    generate_synthetic_delayed_labels,
    load_and_match_delayed_labels,
)
from churn_prediction.post_deployment.report import generate_post_deployment_report
from churn_prediction.post_deployment.retraining_decision import (
    make_retraining_decision,
)

logger = logging.getLogger(__name__)


def run_post_deployment_evaluation(
    config_path: str | Path | None = None,
    predictions_path_override: str | Path | pd.DataFrame | None = None,
    delayed_labels_path_override: str | Path | pd.DataFrame | None = None,
    output_dir_override: str | Path | None = None,
    baseline_metrics_override: str | Path | None = None,
    drift_report_override: str | Path | None = None,
    auto_generate_synthetic_if_missing: bool = True,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Execute end-to-end post-deployment evaluation pipeline.

    Steps:
      1. Load post-deployment configuration YAML.
      2. Resolve file paths for predictions, delayed labels, baseline metrics, drift report, and output dir.
      3. Ingest and match historical predictions with delayed labels on customer key.
      4. Evaluate model prediction quality, capacity metrics, and probability calibration.
      5. Evaluate retention campaign effectiveness (Treatment vs Control outcomes & ROI).
      6. Formulate evidence-based retraining decision based on performance, calibration, drift, & quality.
      7. Generate structured JSON summary and formatted Markdown scheduled report.

    Args:
        config_path: Optional path to post_deployment.yaml configuration file.
        predictions_path_override: Optional path or DataFrame for historical predictions.
        delayed_labels_path_override: Optional path or DataFrame for delayed labels.
        output_dir_override: Optional path to report output directory.
        baseline_metrics_override: Optional path to historical baseline metrics JSON.
        drift_report_override: Optional path to drift report JSON.
        auto_generate_synthetic_if_missing: If True and delayed labels file is missing,
            generate reproducible synthetic labels for local demonstration.

    Returns:
        Tuple of (complete_summary_dict, generated_artifact_paths_dict).
    """
    config = load_post_deployment_config(config_path)

    paths_config = config.get("paths", {})
    eval_config = config.get("evaluation", {})
    campaign_config = config.get("campaign", {})

    preds_input = (
        predictions_path_override
        or paths_config.get("historical_predictions_path")
        or "reports/scoring/batch_predictions.csv"
    )
    labels_input = (
        delayed_labels_path_override
        or paths_config.get("delayed_labels_path")
        or "data/delayed_labels.csv"
    )

    baseline_path = (
        baseline_metrics_override
        or paths_config.get("baseline_metrics_path")
        or "reports/evaluation/evaluation_metrics.json"
    )
    drift_path = (
        drift_report_override
        or paths_config.get("drift_report_path")
        or "reports/drift/drift_report.json"
    )
    output_dir = Path(
        output_dir_override
        or paths_config.get("output_dir")
        or "reports/post_deployment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_filename = paths_config.get(
        "report_markdown_filename", "post_deployment_report.md"
    )
    json_filename = paths_config.get(
        "report_json_filename", "post_deployment_summary.json"
    )

    threshold = float(eval_config.get("decision_threshold", 0.50))
    capacity_fraction = float(eval_config.get("campaign_capacity", 0.10))
    n_bins = int(eval_config.get("n_bins", 10))
    eval_date = str(eval_config.get("evaluation_date", "2026-07-29"))

    unit_cost = float(campaign_config.get("unit_contact_cost", 10.0))
    unit_value = float(campaign_config.get("unit_customer_value", 500.0))

    # 1. Handle missing delayed labels file in local dev by auto-generating if requested
    if isinstance(preds_input, (str, Path)) and isinstance(labels_input, (str, Path)):
        pred_p = Path(preds_input)
        label_p = Path(labels_input)

        if not pred_p.is_file():
            # If batch predictions haven't been generated yet, raise informative error or run batch scoring
            raise FileNotFoundError(
                f"Historical predictions file not found at '{pred_p}'. "
                "Run batch scoring script first: `python scripts/run_batch_scoring.py`"
            )

        if not label_p.is_file() and auto_generate_synthetic_if_missing:
            logger.info(
                f"Delayed labels file not found at '{label_p}'. "
                "Generating reproducible synthetic delayed labels for local demonstration."
            )
            preds_df = pd.read_csv(pred_p)
            synth_labels_df = generate_synthetic_delayed_labels(preds_df)
            label_p.parent.mkdir(parents=True, exist_ok=True)
            synth_labels_df.to_csv(label_p, index=False)
            labels_input = label_p

    # 2. Load and match predictions with delayed labels
    matched_df, matching_stats = load_and_match_delayed_labels(
        predictions_input=preds_input,
        delayed_labels_input=labels_input,
    )

    # 3. Evaluate prediction quality metrics
    eval_summary = evaluate_delayed_predictions(
        matched_df=matched_df,
        threshold=threshold,
        capacity_fraction=capacity_fraction,
        n_bins=n_bins,
    )

    # 4. Evaluate retention campaign effectiveness
    campaign_summary = evaluate_campaign_effectiveness(
        matched_df=matched_df,
        unit_contact_cost=unit_cost,
        unit_customer_value=unit_value,
    )

    # 5. Make evidence-based retraining decision
    retraining_decision = make_retraining_decision(
        delayed_eval_summary=eval_summary,
        baseline_metrics_path=baseline_path,
        drift_report_path=drift_path,
        matching_stats=matching_stats,
        config=config,
    )

    # 6. Generate scheduled reports
    artifacts = generate_post_deployment_report(
        eval_summary=eval_summary,
        campaign_summary=campaign_summary,
        retraining_decision=retraining_decision,
        matching_stats=matching_stats,
        output_dir=output_dir,
        report_markdown_filename=markdown_filename,
        report_json_filename=json_filename,
        evaluation_date=eval_date,
    )

    complete_summary = {
        "evaluation_date": eval_date,
        "matching_statistics": matching_stats,
        "model_performance": eval_summary,
        "campaign_effectiveness": campaign_summary,
        "retraining_decision": retraining_decision,
        "generated_artifacts": {k: str(v) for k, v in artifacts.items()},
    }

    return complete_summary, artifacts
