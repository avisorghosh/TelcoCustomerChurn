"""Data contract definition, schema generation, and report models."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandera.typing as pt
import yaml
from pandera.pandas import Check, Column, DataFrameSchema
from pydantic import BaseModel, Field


class DataValidationError(Exception):
    """Exception raised when data validation checks fail."""

    def __init__(self, message: str, report: "ValidationReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


class ValidationErrorDetail(BaseModel):
    """Detailed information for a single data validation failure."""

    column: str | None = Field(
        default=None, description="Column associated with validation error."
    )
    check_type: str = Field(
        description="Type of check that failed (e.g., domain, type, uniqueness)."
    )
    message: str = Field(description="Actionable description of the failure.")
    invalid_count: int = Field(
        default=1, ge=0, description="Number of rows violating the check."
    )
    sample_invalid_values: list[str] = Field(
        default_factory=list, description="Sample invalid values encountered."
    )


class ValidationReport(BaseModel):
    """Data quality and contract validation report."""

    is_valid: bool = Field(description="True if dataset satisfies all contract checks.")
    schema_version: str = Field(
        default="1.0.0", description="Contract schema version applied."
    )
    total_rows: int = Field(ge=0, description="Total rows in evaluated dataset.")
    valid_rows: int = Field(ge=0, description="Number of valid rows.")
    errors: list[ValidationErrorDetail] = Field(
        default_factory=list, description="List of validation errors."
    )
    summary: str = Field(description="Human-readable summary of validation outcome.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of report generation.",
    )


def get_default_config_path() -> Path:
    """Return the absolute path to the default data_contract.yaml config."""

    root_dir = Path(__file__).resolve().parents[3]
    return root_dir / "configs" / "data_contract.yaml"


def load_contract_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the data contract YAML configuration.

    Args:
        config_path: Optional path to YAML config. Uses default if None.

    Returns:
        Dictionary containing schema configuration.
    """
    path = Path(config_path) if config_path else get_default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Data contract config not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def get_pandera_schema(
    config: dict[str, Any], is_training: bool = True
) -> DataFrameSchema:
    """Build a Pandera DataFrameSchema from configuration.

    Args:
        config: Data contract configuration dictionary.
        is_training: Whether dataset contains target column (Churn).

    Returns:
        Pandera DataFrameSchema instance.
    """
    columns: dict[str, Column] = {}

    primary_key = config.get("primary_key", "customerID")
    pk_error_msg = "Primary key 'customerID' must not contain blank strings."
    columns[primary_key] = Column(
        pt.String,
        nullable=False,
        checks=[
            Check(
                lambda s: s.str.strip().str.len() > 0,
                name="non_blank_id",
                error=pk_error_msg,
            )
        ],
        required=True,
    )

    categorical_cols = config.get("categorical_columns", {})
    for col_name, col_meta in categorical_cols.items():
        allowed = col_meta.get("allowed_values", [])
        if col_name == "SeniorCitizen":
            columns[col_name] = Column(
                pt.Int,
                checks=[Check.isin(allowed, error=f"Invalid domain for '{col_name}'.")],
                nullable=False,
                required=True,
            )
        else:
            columns[col_name] = Column(
                pt.String,
                checks=[Check.isin(allowed, error=f"Invalid domain for '{col_name}'.")],
                nullable=False,
                required=True,
            )

    numeric_cols = config.get("numeric_columns", {})
    for col_name, col_meta in numeric_cols.items():
        min_val = col_meta.get("min_value", 0)
        allow_null = col_meta.get("allow_null", False)
        if col_name == "tenure":
            columns[col_name] = Column(
                pt.Int,
                checks=[Check.greater_than_or_equal_to(min_val)],
                nullable=allow_null,
                required=True,
            )
        else:
            columns[col_name] = Column(
                pt.Float,
                checks=[Check.greater_than_or_equal_to(min_val)],
                nullable=allow_null,
                required=True,
            )

    target_col = config.get("target_column", "Churn")
    if is_training:
        target_allowed = config.get("target_allowed_values", ["Yes", "No"])
        columns[target_col] = Column(
            pt.String,
            checks=[
                Check.isin(
                    target_allowed, error=f"Invalid target domain for '{target_col}'."
                )
            ],
            nullable=False,
            required=True,
        )

    return DataFrameSchema(
        columns=columns,
        strict=True,
        coerce=False,
    )
