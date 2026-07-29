"""Unit tests for structured, privacy-safe JSON logging."""

import io
import json
import logging

from churn_prediction.monitoring.logging import (
    StructuredJsonFormatter,
    log_event,
    sanitize_log_data,
)


def test_sanitize_log_data_redacts_sensitive_keys() -> None:
    """Test that sensitive customer keys and PII are redacted from log data."""
    raw_data = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "tenure": 12,
        "nested": {
            "customer_id": "12345",
            "MonthlyCharges": 55.85,
            "raw_payload": {"secret": "data"},
        },
    }

    sanitized = sanitize_log_data(raw_data)

    assert sanitized["customerID"] == "[REDACTED]"
    assert sanitized["gender"] == "[REDACTED]"
    assert sanitized["tenure"] == 12
    assert sanitized["nested"]["customer_id"] == "[REDACTED]"
    assert sanitized["nested"]["MonthlyCharges"] == 55.85
    assert sanitized["nested"]["raw_payload"] == "[REDACTED]"


def test_structured_json_formatter_output() -> None:
    """Test that StructuredJsonFormatter emits valid structured JSON lines."""
    formatter = StructuredJsonFormatter(privacy_safe=True)
    logger = logging.getLogger("test_formatter_logger")
    record = logger.makeRecord(
        name="test_formatter",
        level=logging.INFO,
        fn="test_fn.py",
        lno=10,
        msg="Test log message",
        args=(),
        exc_info=None,
        extra={
            "event": "test_event",
            "correlation_id": "corr-123",
            "model_version": "1.0.0",
            "customerID": "9999-SECRET",
            "safe_metric": 42,
        },
    )

    formatted_str = formatter.format(record)
    parsed = json.loads(formatted_str)

    assert parsed["level"] == "INFO"
    assert parsed["event"] == "test_event"
    assert parsed["correlation_id"] == "corr-123"
    assert parsed["model_version"] == "1.0.0"
    assert parsed["message"] == "Test log message"
    assert "timestamp" in parsed

    # Privacy Check
    assert "9999-SECRET" not in formatted_str
    assert parsed["context"]["customerID"] == "[REDACTED]"
    assert parsed["context"]["safe_metric"] == 42


def test_log_event_helper() -> None:
    """Test log_event convenience helper function."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(StructuredJsonFormatter(privacy_safe=True))

    logger = logging.getLogger("test_log_event_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    log_event(
        logger,
        logging.INFO,
        event="prediction_request",
        message="Served single prediction",
        correlation_id="req-abc-789",
        model_version="2.0.0",
        customerID="7590-VHVEG",
        predicted_class=1,
    )

    output = log_capture.getvalue()
    parsed = json.loads(output.strip())

    assert parsed["event"] == "prediction_request"
    assert parsed["correlation_id"] == "req-abc-789"
    assert parsed["model_version"] == "2.0.0"
    assert parsed["context"]["predicted_class"] == 1
    assert "7590-VHVEG" not in output
    assert parsed["context"]["customerID"] == "[REDACTED]"
