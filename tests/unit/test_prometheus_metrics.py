"""Unit tests for Prometheus metrics collection and formatting."""

from prometheus_client import CollectorRegistry

from churn_prediction.monitoring.metrics import MetricsManager


def test_metrics_manager_initialization_and_recording() -> None:
    """Test that MetricsManager initializes metrics and increments counters."""
    registry = CollectorRegistry()
    mgr = MetricsManager(registry=registry)

    # Record API request
    mgr.record_api_request(
        endpoint="/predict", method="POST", status_code=200, duration_seconds=0.045
    )
    mgr.record_prediction(predicted_class=1, model_version="1.0.0")
    mgr.record_prediction_failure(reason="validation_error")

    # Record batch job metrics
    mgr.record_batch_job(
        status="success", accepted_count=100, rejected_count=5, validation_failures=2
    )
    mgr.record_batch_failure()

    # Record model load metrics
    mgr.record_model_load(success=True)
    mgr.record_model_load(success=False)

    text_output = mgr.generate_metrics_text()

    assert "telco_churn_api_requests_total" in text_output
    assert 'endpoint="/predict"' in text_output
    assert "telco_churn_api_request_duration_seconds" in text_output
    assert "telco_churn_predictions_total" in text_output
    assert 'predicted_class="1"' in text_output
    assert "telco_churn_prediction_failures_total" in text_output
    assert "telco_churn_batch_scoring_records_total" in text_output
    assert "telco_churn_model_loads_total" in text_output
    assert "telco_churn_model_load_failures_total" in text_output
