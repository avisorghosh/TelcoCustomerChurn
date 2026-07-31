"""Evaluation package for baseline and candidate churn prediction models."""

from churn_prediction.evaluation.calibration import compute_calibration_curve
from churn_prediction.evaluation.capacity import compute_capacity_metrics
from churn_prediction.evaluation.comparator import compare_baseline_and_candidate
from churn_prediction.evaluation.config import load_evaluation_config
from churn_prediction.evaluation.evaluator import evaluate_model
from churn_prediction.evaluation.fairness import evaluate_fairness_review
from churn_prediction.evaluation.importance import extract_feature_importance
from churn_prediction.evaluation.metrics import compute_binary_classification_metrics
from churn_prediction.evaluation.segment import evaluate_segment_performance
from churn_prediction.evaluation.threshold import compute_threshold_analysis

__all__ = [
    "compare_baseline_and_candidate",
    "compute_binary_classification_metrics",
    "compute_calibration_curve",
    "compute_capacity_metrics",
    "compute_threshold_analysis",
    "evaluate_fairness_review",
    "evaluate_model",
    "evaluate_segment_performance",
    "extract_feature_importance",
    "load_evaluation_config",
]
