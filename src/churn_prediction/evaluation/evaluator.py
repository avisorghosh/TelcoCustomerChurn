"""Baseline evaluation pipeline orchestrator."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from churn_prediction.evaluation.calibration import compute_calibration_curve
from churn_prediction.evaluation.capacity import compute_capacity_metrics
from churn_prediction.evaluation.config import load_evaluation_config
from churn_prediction.evaluation.importance import extract_feature_importance
from churn_prediction.evaluation.metrics import compute_binary_classification_metrics
from churn_prediction.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_analysis,
)
from churn_prediction.evaluation.threshold import compute_threshold_analysis
from churn_prediction.models.serialization import load_metadata, load_pipeline
from churn_prediction.models.trainer import (
    load_and_validate_dataset,
    prepare_features_and_target,
    split_dataset,
)


def evaluate_model(
    config_path: str | Path | None = None,
    model_path_override: str | Path | None = None,
    data_path_override: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Execute evaluation workflow on persisted model artifact.

    Loads trained pipeline, evaluates on validation/test split, calculates
    metrics (PR-AUC, ROC-AUC, Precision, Recall, F1, Brier, Capacity, Calibration,
    Thresholds), extracts feature importances, and emits plots & JSON reports.

    Args:
        config_path: Optional path to evaluation configuration YAML.
        model_path_override: Optional path to baseline pipeline joblib file.
        data_path_override: Optional path to raw dataset CSV file.

    Returns:
        Tuple of (evaluation_summary_dict, generated_artifact_paths_dict).
    """
    config = load_evaluation_config(config_path)

    paths_config = config.get("paths", {})
    model_dir = paths_config.get("model_dir", "models")
    pipeline_filename = paths_config.get(
        "pipeline_filename", "baseline_pipeline.joblib"
    )
    metadata_filename = paths_config.get("metadata_filename", "baseline_metadata.json")

    pipeline_path = (
        Path(model_path_override)
        if model_path_override
        else Path(model_dir) / pipeline_filename
    )
    metadata_path = Path(model_dir) / metadata_filename

    raw_data_path = data_path_override or paths_config.get(
        "raw_data_path", "Telco-Customer-Churn.csv"
    )
    contract_config_path = paths_config.get("contract_config_path")
    output_dir = Path(paths_config.get("output_dir", "reports/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_config = config.get("evaluation", {})
    threshold = float(eval_config.get("evaluation_threshold", 0.50))
    capacity_fraction = float(eval_config.get("campaign_capacity", 0.10))
    eval_split_name = str(eval_config.get("evaluation_split", "test")).lower()
    n_bins = int(eval_config.get("n_bins", 10))
    thresholds_list = eval_config.get("thresholds")

    # 1. Load persisted trained pipeline and metadata (No retraining)
    pipeline = load_pipeline(pipeline_path)
    metadata = load_metadata(metadata_path)

    # 2. Ingest and validate dataset
    validated_df = load_and_validate_dataset(
        data_path=raw_data_path,
        contract_config_path=contract_config_path,
    )

    # 3. Stratified data split using training configuration settings
    train_df, val_df, test_df = split_dataset(validated_df, config)

    if eval_split_name == "val":
        eval_df = val_df
    elif eval_split_name == "train":
        eval_df = train_df
    elif eval_split_name in ("full", "all"):
        eval_df = validated_df
    else:
        eval_df = test_df

    # 4. Prepare features and target
    X_eval, y_eval = prepare_features_and_target(eval_df, config)
    y_true = y_eval.to_numpy()

    # 5. Predict probabilities
    y_prob = pipeline.predict_proba(X_eval)[:, 1]

    # 6. Compute evaluation metrics
    metrics = compute_binary_classification_metrics(y_true, y_prob, threshold=threshold)
    capacity_metrics = compute_capacity_metrics(
        y_true, y_prob, capacity_fraction=capacity_fraction
    )
    threshold_analysis = compute_threshold_analysis(
        y_true, y_prob, thresholds=thresholds_list
    )
    calibration_metrics = compute_calibration_curve(y_true, y_prob, n_bins=n_bins)
    feature_importance = extract_feature_importance(pipeline, metadata)

    # 7. Generate plots
    roc_plot_path = plot_roc_curve(
        y_true, y_prob, metrics["roc_auc"], output_dir / "roc_curve.png"
    )
    pr_plot_path = plot_precision_recall_curve(
        y_true, y_prob, metrics["pr_auc"], output_dir / "precision_recall_curve.png"
    )
    cal_plot_path = plot_calibration_curve(
        calibration_metrics["prob_true"],
        calibration_metrics["prob_pred"],
        calibration_metrics["brier_score"],
        output_dir / "calibration_curve.png",
    )
    cm_plot_path = plot_confusion_matrix(
        metrics["confusion_matrix"], output_dir / "confusion_matrix.png"
    )
    th_plot_path = plot_threshold_analysis(
        threshold_analysis, threshold, output_dir / "threshold_analysis.png"
    )
    fi_plot_path = plot_feature_importance(
        feature_importance, top_n=15, output_path=output_dir / "feature_importance.png"
    )

    # 8. Save JSON reports
    metrics_summary_path = output_dir / "evaluation_metrics.json"
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": metadata.get("model_name"),
                "schema_version": metadata.get("schema_version"),
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluation_split": eval_split_name,
                "sample_size": len(y_true),
                "primary_metric": {
                    "name": "PR-AUC",
                    "value": metrics["pr_auc"],
                },
                "metrics": metrics,
                "campaign_capacity": capacity_metrics,
                "calibration": calibration_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    threshold_json_path = output_dir / "threshold_analysis.json"
    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_analysis, f, indent=2, ensure_ascii=False)

    report_json_path = output_dir / "classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics["classification_report"], f, indent=2, ensure_ascii=False)

    importance_json_path = output_dir / "feature_importance.json"
    with open(importance_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2, ensure_ascii=False)

    generated_artifacts = {
        "metrics_summary_json": metrics_summary_path,
        "threshold_analysis_json": threshold_json_path,
        "classification_report_json": report_json_path,
        "feature_importance_json": importance_json_path,
        "roc_curve_plot": roc_plot_path,
        "precision_recall_curve_plot": pr_plot_path,
        "calibration_curve_plot": cal_plot_path,
        "confusion_matrix_plot": cm_plot_path,
        "threshold_analysis_plot": th_plot_path,
        "feature_importance_plot": fi_plot_path,
    }

    summary = {
        "model_name": metadata.get("model_name"),
        "schema_version": metadata.get("schema_version"),
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": eval_split_name,
        "sample_size": len(y_true),
        "primary_metric": {
            "name": "PR-AUC",
            "value": metrics["pr_auc"],
        },
        "metrics": metrics,
        "campaign_capacity": capacity_metrics,
        "calibration": calibration_metrics,
        "threshold_analysis": threshold_analysis,
        "feature_importance": feature_importance,
    }

    return summary, generated_artifacts
