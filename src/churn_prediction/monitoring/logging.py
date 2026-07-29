"""Structured, privacy-safe JSON logging utilities."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

SENSITIVE_KEYS: set[str] = {
    "customerid",
    "customer_id",
    "customer_name",
    "gender",
    "raw_payload",
    "customer_record",
    "features",
    "payload",
    "request_body",
    "input_data",
    "name",
    "email",
    "address",
}


def sanitize_log_data(data: Any) -> Any:
    """Recursively redact sensitive customer features, IDs, and PII from log dictionary.

    Args:
        data: Arbitrary object or dictionary to sanitize.

    Returns:
        Sanitized object with sensitive keys redacted or omitted.
    """
    if isinstance(data, dict):
        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            if str(key).lower() in SENSITIVE_KEYS:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_log_data(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    return data


class StructuredJsonFormatter(logging.Formatter):
    """Custom logging Formatter that outputs structured, privacy-safe JSON lines."""

    def __init__(self, privacy_safe: bool = True) -> None:
        super().__init__()
        self.privacy_safe = privacy_safe

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as a structured JSON string."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
        }

        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id is not None:
            log_entry["correlation_id"] = str(correlation_id)

        model_version = getattr(record, "model_version", None)
        if model_version is not None:
            log_entry["model_version"] = str(model_version)

        # Standard extra attributes to exclude from custom context
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "event",
            "correlation_id",
            "model_version",
            "message",
        }

        extra_context: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                extra_context[key] = value

        if extra_context:
            if self.privacy_safe:
                extra_context = sanitize_log_data(extra_context)
            log_entry["context"] = extra_context

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        if self.privacy_safe:
            log_entry = sanitize_log_data(log_entry)

        return json.dumps(log_entry)


def setup_structured_logging(
    level: str = "INFO",
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure structured JSON logging for a logger.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        logger_name: Logger name. Configures root logger if None.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output during re-configuration
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter(privacy_safe=True))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    correlation_id: str | None = None,
    model_version: str | None = None,
    **kwargs: Any,
) -> None:
    """Emit a structured, privacy-safe log event.

    Args:
        logger: Logger instance to output log.
        level: Logging level (e.g., logging.INFO).
        event: Standardized event name (e.g., 'api_startup', 'prediction_request').
        message: Human-readable message.
        correlation_id: Optional correlation ID string.
        model_version: Optional model version string.
        **kwargs: Additional contextual metadata (auto-sanitized for PII).
    """
    extra: dict[str, Any] = {
        "event": event,
    }
    if correlation_id is not None:
        extra["correlation_id"] = str(correlation_id)
    if model_version is not None:
        extra["model_version"] = str(model_version)

    sanitized_kwargs = sanitize_log_data(kwargs) if kwargs else {}
    extra.update(sanitized_kwargs)

    logger.log(level, message, extra=extra)
