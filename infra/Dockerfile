# Stage 1: Build dependencies environment
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy uv binary for fast, deterministic locked dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

# Copy dependency specifications and project source
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/

# Build virtual environment with production dependencies only
ENV UV_PROJECT_ENVIRONMENT=/install
RUN /uv/bin/uv sync --locked --no-dev --no-editable

# Stage 2: Production runtime image
FROM python:3.11-slim AS runner

# Create non-root application user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy isolated virtual environment from builder stage
COPY --from=builder --chown=appuser:appgroup /install /app/.venv

# Copy application source, configs, scripts, and pre-trained model artifacts
COPY --chown=appuser:appgroup src /app/src
COPY --chown=appuser:appgroup configs /app/configs
COPY --chown=appuser:appgroup scripts /app/scripts
COPY --chown=appuser:appgroup models /app/models
COPY --chown=appuser:appgroup pyproject.toml /app/pyproject.toml

# Set default runtime environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHURN_API_HOST=0.0.0.0 \
    CHURN_API_PORT=8000 \
    CHURN_MODEL_DIR=/app/models \
    CHURN_DECISION_THRESHOLD=0.50

# Switch to non-root container user
USER appuser

# Expose default API port
EXPOSE 8000

# Healthcheck to verify service availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Launch FastAPI inference service
CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
