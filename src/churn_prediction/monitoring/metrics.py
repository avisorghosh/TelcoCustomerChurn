"""Prometheus metrics registry and collectors for API, batch, and model."""

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

NAMESPACE = "telco_churn"


def _get_or_create_counter(
    name: str,
    documentation: str,
    labelnames: tuple[str, ...] = (),
    registry: CollectorRegistry = REGISTRY,
) -> Counter:
    """Safely retrieve or register a Prometheus Counter metric."""
    full_name = f"{NAMESPACE}_{name}"
    try:
        return Counter(
            name=full_name,
            documentation=documentation,
            labelnames=labelnames,
            registry=registry,
        )
    except ValueError:
        # Metric already registered in this collector registry
        return registry._names_to_collectors[full_name]  # type: ignore


def _get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: tuple[str, ...] = (),
    buckets: tuple[float, ...] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
    registry: CollectorRegistry = REGISTRY,
) -> Histogram:
    """Safely retrieve or register a Prometheus Histogram metric."""
    full_name = f"{NAMESPACE}_{name}"
    try:
        return Histogram(
            name=full_name,
            documentation=documentation,
            labelnames=labelnames,
            buckets=buckets,
            registry=registry,
        )
    except ValueError:
        return registry._names_to_collectors[full_name]  # type: ignore


class MetricsManager:
    """Manager for Prometheus operational and model metrics."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.registry = registry

        # API Metrics
        self.api_requests_total = _get_or_create_counter(
            name="api_requests_total",
            documentation="Total count of API HTTP requests received.",
            labelnames=("endpoint", "method", "status_code"),
            registry=registry,
        )

        self.api_request_duration_seconds = _get_or_create_histogram(
            name="api_request_duration_seconds",
            documentation="API HTTP request duration in seconds.",
            labelnames=("endpoint",),
            registry=registry,
        )

        self.predictions_total = _get_or_create_counter(
            name="predictions_total",
            documentation="Total churn predictions served.",
            labelnames=("predicted_class", "model_version"),
            registry=registry,
        )

        self.prediction_failures_total = _get_or_create_counter(
            name="prediction_failures_total",
            documentation="Total prediction request failures.",
            labelnames=("failure_reason",),
            registry=registry,
        )

        # Batch Scoring Metrics
        self.batch_scoring_batches_total = _get_or_create_counter(
            name="batch_scoring_batches_total",
            documentation="Total batch scoring jobs executed.",
            labelnames=("status",),
            registry=registry,
        )

        self.batch_scoring_records_total = _get_or_create_counter(
            name="batch_scoring_records_total",
            documentation="Total records processed during batch scoring.",
            labelnames=("status",),
            registry=registry,
        )

        self.batch_scoring_validation_failures_total = _get_or_create_counter(
            name="batch_scoring_validation_failures_total",
            documentation="Total validation failures encountered during batch scoring.",
            registry=registry,
        )

        self.batch_scoring_failures_total = _get_or_create_counter(
            name="batch_scoring_failures_total",
            documentation="Total batch scoring execution failures.",
            registry=registry,
        )

        # Model Lifecycle Metrics
        self.model_loads_total = _get_or_create_counter(
            name="model_loads_total",
            documentation="Total model artifact load attempts.",
            labelnames=("status",),
            registry=registry,
        )

        self.model_load_failures_total = _get_or_create_counter(
            name="model_load_failures_total",
            documentation="Total model artifact load failures.",
            registry=registry,
        )

    def record_api_request(
        self, endpoint: str, method: str, status_code: int, duration_seconds: float
    ) -> None:
        """Record API request count and latency."""
        self.api_requests_total.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).inc()
        self.api_request_duration_seconds.labels(endpoint=endpoint).observe(
            duration_seconds
        )

    def record_prediction(self, predicted_class: int, model_version: str) -> None:
        """Record successful prediction count by class and version."""
        self.predictions_total.labels(
            predicted_class=str(predicted_class), model_version=str(model_version)
        ).inc()

    def record_prediction_failure(self, reason: str) -> None:
        """Record prediction failure count."""
        self.prediction_failures_total.labels(failure_reason=reason).inc()

    def record_batch_job(
        self,
        status: str,
        accepted_count: int = 0,
        rejected_count: int = 0,
        validation_failures: int = 0,
    ) -> None:
        """Record batch scoring job execution metrics."""
        self.batch_scoring_batches_total.labels(status=status).inc()
        if accepted_count > 0:
            self.batch_scoring_records_total.labels(status="accepted").inc(
                accepted_count
            )
        if rejected_count > 0:
            self.batch_scoring_records_total.labels(status="rejected").inc(
                rejected_count
            )
        if validation_failures > 0:
            self.batch_scoring_validation_failures_total.inc(validation_failures)

    def record_batch_failure(self) -> None:
        """Record overall batch scoring execution failure."""
        self.batch_scoring_batches_total.labels(status="failed").inc()
        self.batch_scoring_failures_total.inc()

    def record_model_load(self, success: bool) -> None:
        """Record model load attempt result."""
        status = "success" if success else "failed"
        self.model_loads_total.labels(status=status).inc()
        if not success:
            self.model_load_failures_total.inc()

    def generate_metrics_text(self) -> str:
        """Export current Prometheus metrics as formatted text string."""
        return generate_latest(self.registry).decode("utf-8")


# Default global metrics manager instance
metrics_manager = MetricsManager(REGISTRY)
