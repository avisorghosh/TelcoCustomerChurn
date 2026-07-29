"""Retraining decision engine module.

Evaluates observed model performance, drift alerts, data quality, and campaign outcomes
to form a documented, evidence-based recommendation on whether model retraining is required.

CRITICAL: Does NOT trigger automatic retraining.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def make_retraining_decision(
    delayed_eval_summary: dict[str, Any],
    baseline_metrics_path: str | Path | None = None,
    drift_report_path: str | Path | None = None,
    matching_stats: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate performance, calibration, drift, and quality for retraining decision.

    Args:
        delayed_eval_summary: Output dictionary from evaluate_delayed_predictions.
        baseline_metrics_path: Path to baseline evaluation_metrics.json.
        drift_report_path: Path to drift_report.json from observability.
        matching_stats: Operational matching statistics dictionary.
        config: Post-deployment configuration dictionary.

    Returns:
        Structured dictionary containing decision boolean, code, and evidence.
    """
    retrain_config = (
        config.get("retraining_decision", {})
        if config
        else {
            "pr_auc_drop_threshold": 0.05,
            "brier_score_max": 0.20,
            "psi_alert_threshold": 0.25,
            "min_match_rate": 0.80,
        }
    )

    pr_auc_drop_thresh = float(retrain_config.get("pr_auc_drop_threshold", 0.05))
    brier_max = float(retrain_config.get("brier_score_max", 0.20))
    psi_alert_thresh = float(retrain_config.get("psi_alert_threshold", 0.25))
    min_match_rate = float(retrain_config.get("min_match_rate", 0.80))

    current_metrics = delayed_eval_summary.get("metrics", {})
    current_pr_auc = float(current_metrics.get("pr_auc", 0.0))
    current_brier = float(current_metrics.get("brier_score", 0.0))

    baseline_pr_auc = None
    pr_auc_delta = None
    pr_auc_degraded = False

    if baseline_metrics_path and Path(baseline_metrics_path).is_file():
        try:
            with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                base_data = json.load(f)

            if "metrics" in base_data and "pr_auc" in base_data["metrics"]:
                baseline_pr_auc = float(base_data["metrics"]["pr_auc"])
            elif "primary_metric" in base_data:
                baseline_pr_auc = float(base_data["primary_metric"]["value"])

            if baseline_pr_auc is not None:
                pr_auc_delta = current_pr_auc - baseline_pr_auc
                if (baseline_pr_auc - current_pr_auc) > pr_auc_drop_thresh:
                    pr_auc_degraded = True
        except Exception as err:
            logger.warning(
                f"Unable to parse baseline metrics file '{baseline_metrics_path}': {err}"
            )

    if baseline_pr_auc is None:
        baseline_pr_auc = 0.6437
        pr_auc_delta = current_pr_auc - baseline_pr_auc
        if (baseline_pr_auc - current_pr_auc) > pr_auc_drop_thresh:
            pr_auc_degraded = True

    calibration_degraded = current_brier > brier_max

    max_psi = 0.0
    significant_drift = False
    drifted_features = []

    if drift_report_path and Path(drift_report_path).is_file():
        try:
            with open(drift_report_path, "r", encoding="utf-8") as f:
                drift_data = json.load(f)

            drift_summary = drift_data.get("summary", {})
            max_psi = float(drift_summary.get("max_psi", 0.0))
            if (
                max_psi >= psi_alert_thresh
                or drift_summary.get("significant_drift_count", 0) > 0
            ):
                significant_drift = True

            feature_drifts = drift_data.get("feature_drift", {})
            for feat, metrics_dict in feature_drifts.items():
                if metrics_dict.get("psi_alert", False):
                    drifted_features.append(feat)
        except Exception as err:
            logger.warning(
                f"Unable to parse drift report file '{drift_report_path}': {err}"
            )

    match_rate = float(matching_stats.get("match_rate", 1.0)) if matching_stats else 1.0
    match_rate_low = match_rate < min_match_rate

    reasons = []
    triggers = []

    if pr_auc_degraded:
        reasons.append(
            f"PR-AUC dropped by {abs(pr_auc_delta):.4f} "
            f"(baseline: {baseline_pr_auc:.4f}, current: {current_pr_auc:.4f}), "
            f"exceeding threshold drop of {pr_auc_drop_thresh:.4f}."
        )
        triggers.append("PR_AUC_DEGRADATION")

    if calibration_degraded:
        reasons.append(
            f"Brier score of {current_brier:.4f} exceeds "
            f"maximum threshold of {brier_max:.4f}."
        )
        triggers.append("CALIBRATION_DEGRADATION")

    if significant_drift:
        feat_str = (
            f" in features ({', '.join(drifted_features)})" if drifted_features else ""
        )
        reasons.append(
            f"Significant feature drift detected with max PSI of {max_psi:.4f}{feat_str}, "
            f"exceeding threshold of {psi_alert_thresh:.4f}."
        )
        triggers.append("FEATURE_DRIFT_ALERT")

    if match_rate_low:
        reasons.append(
            f"Delayed label matching rate ({match_rate:.2%}) is below "
            f"minimum required quality threshold ({min_match_rate:.2%})."
        )
        triggers.append("DATA_QUALITY_ALERT")

    retrain_recommended = pr_auc_degraded or (
        significant_drift and calibration_degraded
    )

    if retrain_recommended:
        decision_code = "RETRAIN_RECOMMENDED"
        summary_reason = (
            "Retraining IS recommended due to observed performance degradation "
            "or feature drift. The incumbent model should be re-evaluated."
        )
    else:
        decision_code = "MAINTAIN_INCUMBENT"
        if not reasons:
            reasons.append(
                f"Incumbent model PR-AUC ({current_pr_auc:.4f}) and Brier score "
                f"({current_brier:.4f}) remain within acceptable bounds."
            )
        summary_reason = (
            "Retraining is NOT recommended at this time. The incumbent production "
            "model continues to perform within expected quality bounds."
        )

    evidence = {
        "baseline_pr_auc": round(baseline_pr_auc, 4),
        "delayed_label_pr_auc": round(current_pr_auc, 4),
        "pr_auc_delta": round(pr_auc_delta, 4) if pr_auc_delta is not None else 0.0,
        "pr_auc_drop_threshold": pr_auc_drop_thresh,
        "brier_score": round(current_brier, 4),
        "brier_score_max": brier_max,
        "max_psi": round(max_psi, 4),
        "psi_alert_threshold": psi_alert_thresh,
        "match_rate": round(match_rate, 4),
        "min_match_rate": min_match_rate,
        "triggers_fired": triggers,
    }

    return {
        "retrain_recommended": retrain_recommended,
        "decision_code": decision_code,
        "summary_reason": summary_reason,
        "decision_reasons": reasons,
        "evidence": evidence,
        "policy_note": (
            "Automated retraining is explicitly DISABLED by system governance. "
            "This decision serves as an advisory signal for ML engineering review."
        ),
    }
