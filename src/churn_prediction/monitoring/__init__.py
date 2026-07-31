"""Observability package exposing structured logging, metrics, quality, and drift."""

from churn_prediction.monitoring.config import load_observability_config
from churn_prediction.monitoring.drift import (
    DriftReport,
    FeatureDriftDetail,
    detect_data_drift,
)
from churn_prediction.monitoring.logging import (
    StructuredJsonFormatter,
    log_event,
    sanitize_log_data,
    setup_structured_logging,
)
from churn_prediction.monitoring.metrics import MetricsManager, metrics_manager
from churn_prediction.monitoring.quality_report import (
    OperationalQualityReport,
    generate_quality_report,
)

__all__ = [
    "DriftReport",
    "FeatureDriftDetail",
    "MetricsManager",
    "OperationalQualityReport",
    "StructuredJsonFormatter",
    "detect_data_drift",
    "generate_quality_report",
    "load_observability_config",
    "log_event",
    "metrics_manager",
    "sanitize_log_data",
    "setup_structured_logging",
]
