"""FastAPI application factory, middleware, and API endpoint handlers."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn_prediction.api.config import load_serving_config
from churn_prediction.api.schemas import (
    CustomerRecord,
    HealthResponse,
    PredictionResponse,
    ReadinessResponse,
)
from churn_prediction.api.service import InferenceService, ModelNotLoadedError
from churn_prediction.monitoring import (
    log_event,
    metrics_manager,
    setup_structured_logging,
)

logger = logging.getLogger(__name__)


def create_app(
    service: InferenceService | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Create and configure FastAPI application instance.

    Args:
        service: Optional pre-configured InferenceService instance.
        config_path: Optional path to serving configuration YAML.

    Returns:
        Configured FastAPI application instance.
    """
    setup_structured_logging(level="INFO")
    config = load_serving_config(config_path)
    api_config = config.get("api", {})
    model_config = config.get("model", {})
    scoring_config = config.get("scoring", {})

    title = api_config.get("title", "Telco Customer Churn Prediction API")
    version = api_config.get("version", config.get("schema_version", "1.0.0"))

    # Instantiate inference service if not supplied
    if service is None:
        service = InferenceService(
            model_dir=model_config.get("model_dir", "models"),
            pipeline_filename=model_config.get(
                "pipeline_filename", "serving_pipeline.joblib"
            ),
            metadata_filename=model_config.get(
                "metadata_filename", "serving_metadata.json"
            ),
            decision_threshold=float(scoring_config.get("decision_threshold", 0.50)),
        )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        """Lifespan context manager to load model artifact on server startup."""
        log_event(
            logger,
            logging.INFO,
            event="api_startup",
            message="Initializing API application and loading model artifact",
        )
        service.load_model()
        yield
        log_event(
            logger,
            logging.INFO,
            event="api_shutdown",
            message="Shutting down API application",
        )

    app = FastAPI(
        title=title,
        version=version,
        lifespan=lifespan,
    )

    # Attach service to app state for testing and middleware access
    app.state.service = service

    # Request metrics middleware
    @app.middleware("http")
    async def measure_request_metrics(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        metrics_manager.record_api_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_seconds=duration,
        )
        return response

    # Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request schema validation errors with clean 422 response."""
        cid = request.headers.get("X-Correlation-ID")
        metrics_manager.record_prediction_failure(reason="validation_error")
        log_event(
            logger,
            logging.WARNING,
            event="request_validation_failed",
            message=f"Request validation failed for path {request.url.path}",
            correlation_id=cid,
            error=str(exc),
        )
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": "Invalid request schema or field validation failed.",
                "details": exc.errors(),
                "correlation_id": cid,
            },
        )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(
        request: Request, exc: ModelNotLoadedError
    ) -> JSONResponse:
        """Handle prediction attempts on unavailable model with safe 503 response."""
        cid = request.headers.get("X-Correlation-ID")
        metrics_manager.record_prediction_failure(reason="model_unloaded")
        log_event(
            logger,
            logging.ERROR,
            event="model_unavailable",
            message="Model unavailable during prediction request",
            correlation_id=cid,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "Model service unavailable. Artifact is not loaded.",
                "correlation_id": cid,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unhandled server exceptions safely without leaking stack traces."""
        cid = request.headers.get("X-Correlation-ID")
        metrics_manager.record_prediction_failure(reason="unhandled_exception")
        log_event(
            logger,
            logging.ERROR,
            event="major_failure",
            message="Unhandled internal server error",
            correlation_id=cid,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "An internal server error occurred.",
                "correlation_id": cid,
            },
        )

    # Route Handlers
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def get_health() -> HealthResponse:
        """Health check endpoint returning overall service and model status."""
        is_loaded = service.is_loaded
        service_status = "ok" if is_loaded else "degraded"
        log_event(
            logger,
            logging.INFO,
            event="health_check",
            message="Health check probe served",
            status=service_status,
            model_loaded=is_loaded,
        )
        return HealthResponse(
            status=service_status,
            model_loaded=is_loaded,
        )

    @app.get("/ready", response_model=ReadinessResponse, tags=["Health"])
    async def get_ready() -> ReadinessResponse | JSONResponse:
        """Readiness check endpoint returning 200 when ready or 503 when not ready."""
        if service.is_loaded:
            return ReadinessResponse(
                status="ready",
                model_loaded=True,
            )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "model_loaded": False,
                "detail": service.load_error or "Model artifact is not loaded.",
            },
        )

    @app.get("/metrics", tags=["Observability"])
    async def get_metrics() -> Response:
        """Expose Prometheus metrics endpoint."""
        metrics_data = metrics_manager.generate_metrics_text()
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        status_code=status.HTTP_200_OK,
        tags=["Inference"],
    )
    async def predict_single_customer(
        record: CustomerRecord,
        x_correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
    ) -> PredictionResponse:
        """Predict churn probability for a single customer record."""
        cid = x_correlation_id or str(uuid4())
        try:
            response = service.predict(record=record, correlation_id=cid)
            return response
        except ModelNotLoadedError as e:
            raise e
        except Exception as e:
            metrics_manager.record_prediction_failure(reason="prediction_error")
            log_event(
                logger,
                logging.ERROR,
                event="major_failure",
                message="Prediction endpoint execution failed",
                correlation_id=cid,
                error=str(e),
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": "error",
                    "message": "Inference execution failed safely.",
                    "correlation_id": cid,
                },
            )

    return app
