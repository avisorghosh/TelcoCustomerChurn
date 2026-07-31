"""Operational data quality reporting extending the validation pipeline."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from churn_prediction.data.validator import validate_data
from churn_prediction.monitoring.config import load_observability_config
from churn_prediction.monitoring.metrics import metrics_manager


class OperationalQualityReport(BaseModel):
    """Operational quality report extending contract validation results."""

    schema_version: str = Field(
        default="1.0.0", description="Data contract schema version."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp of report execution.",
    )
    total_records: int = Field(ge=0, description="Total input records evaluated.")
    accepted_records: int = Field(
        ge=0, description="Count of accepted (valid) records."
    )
    rejected_records: int = Field(
        ge=0, description="Count of rejected (quarantined) records."
    )
    rejection_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of records rejected due to contract violations.",
    )
    duplicate_records: dict[str, int] = Field(
        default_factory=dict,
        description="Count of duplicate rows and duplicate primary keys.",
    )
    missing_values: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-column missing value count and percentage.",
    )
    schema_violations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of detailed schema check violations.",
    )
    is_valid: bool = Field(
        description="True if dataset met all data contract criteria."
    )
    summary: str = Field(description="Operational summary string.")


def generate_quality_report(
    df: pd.DataFrame,
    contract_config_path: str | Path | None = None,
    is_training: bool = False,
    output_path: str | Path | None = None,
) -> OperationalQualityReport:
    """Generate operational quality report from input DataFrame.

    Reuses existing Pandera and custom validation pipeline from churn_prediction.data.

    Args:
        df: Input DataFrame to validate and evaluate.
        contract_config_path: Optional path to YAML data contract config.
        is_training: If True, requires target column 'Churn'.
        output_path: Optional path to save output JSON report.

    Returns:
        OperationalQualityReport instance containing quality metrics.
    """
    total_records = len(df)

    # Execute validation without raising exception to capture full report
    _, val_report = validate_data(
        df,
        config_path=contract_config_path,
        is_training=is_training,
        raise_on_error=False,
    )

    # Missing value analysis across all columns
    missing_values: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            missing_values[col] = {
                "null_count": null_count,
                "null_percentage": round(null_count / total_records, 4)
                if total_records > 0
                else 0.0,
            }

    # Duplicate record analysis
    num_duplicate_rows = int(df.duplicated().sum())
    num_duplicate_ids = 0
    if "customerID" in df.columns:
        num_duplicate_ids = int(df["customerID"].duplicated(keep=False).sum())

    duplicate_summary = {
        "duplicate_rows": num_duplicate_rows,
        "duplicate_primary_keys": num_duplicate_ids,
    }

    # Schema violations breakdown from validation report
    schema_violations: list[dict[str, Any]] = []
    for err in val_report.errors:
        schema_violations.append(
            {
                "column": err.column,
                "check_type": err.check_type,
                "message": err.message,
                "invalid_count": err.invalid_count,
                "sample_invalid_values": err.sample_invalid_values,
            }
        )

    accepted_records = val_report.valid_rows
    rejected_records = total_records - accepted_records
    rejection_rate = (
        round(rejected_records / total_records, 4) if total_records > 0 else 0.0
    )

    is_valid = val_report.is_valid
    summary = (
        f"Operational Quality Report: {accepted_records}/{total_records} accepted "
        f"({rejection_rate * 100:.2f}% rejected, "
        f"{len(schema_violations)} check failure(s))."
    )

    op_report = OperationalQualityReport(
        schema_version=val_report.schema_version,
        timestamp=val_report.timestamp,
        total_records=total_records,
        accepted_records=accepted_records,
        rejected_records=rejected_records,
        rejection_rate=rejection_rate,
        duplicate_records=duplicate_summary,
        missing_values=missing_values,
        schema_violations=schema_violations,
        is_valid=is_valid,
        summary=summary,
    )

    # Record Prometheus metrics for batch quality
    metrics_manager.record_batch_job(
        status="success" if is_valid else "rejected",
        accepted_count=accepted_records,
        rejected_count=rejected_records,
        validation_failures=len(schema_violations),
    )

    # Save to disk if output path provided or configured
    if output_path is None:
        obs_config = load_observability_config()
        out_dir = Path(
            obs_config.get("data_quality", {}).get("output_dir", "reports/quality")
        )
        filename = obs_config.get("data_quality", {}).get(
            "report_filename", "operational_quality_report.json"
        )
        output_path = out_dir / filename

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(op_report.model_dump_json(indent=2))

    return op_report
