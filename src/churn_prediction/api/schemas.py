"""Pydantic schemas for API request validation and response formatting."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerRecord(BaseModel):
    """Schema for a single customer input record."""

    model_config = ConfigDict(extra="forbid")

    customerID: str = Field(
        ...,
        description="Unique customer identifier.",
        min_length=1,
        examples=["7590-VHVEG"],
    )
    gender: Literal["Female", "Male"] = Field(
        ..., description="Customer gender.", examples=["Female"]
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Senior citizen indicator (0 or 1).", examples=[0]
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has a partner.", examples=["Yes"]
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has dependents.", examples=["No"]
    )
    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months customer has stayed with company.",
        examples=[1],
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has phone service.", examples=["No"]
    )
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(
        ...,
        description="Whether customer has multiple lines.",
        examples=["No phone service"],
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer internet service provider.", examples=["DSL"]
    )
    OnlineSecurity: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has online security.", examples=["No"]
    )
    OnlineBackup: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has online backup.", examples=["Yes"]
    )
    DeviceProtection: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has device protection.", examples=["No"]
    )
    TechSupport: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has tech support.", examples=["No"]
    )
    StreamingTV: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has streaming TV.", examples=["No"]
    )
    StreamingMovies: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has streaming movies.", examples=["No"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Customer contract term.", examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has paperless billing.", examples=["Yes"]
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ..., description="Customer payment method.", examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(
        ..., ge=0.0, description="Monthly charge amount.", examples=[29.85]
    )
    TotalCharges: float | None = Field(
        default=None,
        ge=0.0,
        description="Total charges amount (can be null for new customers).",
        examples=[29.85],
    )


class PredictionResponse(BaseModel):
    """Schema for single customer prediction response."""

    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of customer churning.",
    )
    predicted_class: int = Field(
        ...,
        description="Predicted churn class (0 for No Churn, 1 for Churn).",
    )
    model_version: str = Field(
        ...,
        description="Version or identifier of the model artifact used for prediction.",
    )
    correlation_id: str = Field(
        ...,
        description="Correlation ID for tracing the prediction request.",
    )
    prediction_timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of prediction.",
    )


class HealthResponse(BaseModel):
    """Schema for health endpoint response."""

    status: str = Field(..., description="Overall health status of service.")
    model_loaded: bool = Field(
        ..., description="Indicates if model artifact is successfully loaded."
    )


class ReadinessResponse(BaseModel):
    """Schema for readiness endpoint response."""

    status: str = Field(..., description="Readiness status of service.")
    model_loaded: bool = Field(
        ..., description="Indicates if model artifact is ready to serve requests."
    )


class ErrorResponse(BaseModel):
    """Schema for structured error responses."""

    status: str = Field("error", description="Error indicator.")
    message: str = Field(..., description="Safe user-facing error message.")
    correlation_id: str | None = Field(
        default=None, description="Correlation ID associated with request."
    )
