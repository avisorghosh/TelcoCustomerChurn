"""Data validation execution engine and report generator."""

from pathlib import Path
from typing import Any

import pandas as pd
import pandera.errors as pa_errors

from churn_prediction.data.contract import (
    DataValidationError,
    ValidationErrorDetail,
    ValidationReport,
    get_pandera_schema,
    load_contract_config,
)


def parse_total_charges(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parse TotalCharges explicitly, converting blank whitespace to NaN.

    Never silently coerces invalid text values to zero or NaN if non-numeric.

    Args:
        df: Input DataFrame containing 'TotalCharges'.

    Returns:
        Tuple of (transformed DataFrame, error list if non-numeric invalid text exists).
    """
    df_out = df.copy()
    errors: list[str] = []

    if "TotalCharges" not in df_out.columns:
        return df_out, errors

    s = df_out["TotalCharges"]

    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):

        def _convert_val(val: Any) -> float | None:
            if pd.isna(val):
                return None
            val_str = str(val).strip()
            if val_str == "":
                return None
            try:
                return float(val_str)
            except ValueError:
                errors.append(
                    f"Non-numeric string '{val}' found in TotalCharges column."
                )
                return None

        converted = s.map(_convert_val)
        df_out["TotalCharges"] = converted.astype("float64")
    elif not pd.api.types.is_float_dtype(s) and not pd.api.types.is_integer_dtype(s):
        errors.append(f"Unsupported data type '{s.dtype}' for TotalCharges column.")

    return df_out, errors


def validate_data(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
    is_training: bool = True,
    raise_on_error: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate input DataFrame against versioned data contract.

    Args:
        df: Raw or loaded pandas DataFrame.
        config_path: Optional path to YAML contract config.
        is_training: If True, requires target column ('Churn').
        raise_on_error: If True, raises DataValidationError when validation fails.

    Returns:
        Tuple of (parsed/validated DataFrame, ValidationReport).

    Raises:
        DataValidationError: If raise_on_error=True and validation fails.
    """
    config = load_contract_config(config_path)
    schema_version = str(config.get("schema_version", "1.0.0"))
    total_rows = len(df)
    error_details: list[ValidationErrorDetail] = []

    allow_duplicates = config.get("duplicate_policy", {}).get("allow_duplicates", False)
    if not allow_duplicates:
        num_duplicates = int(df.duplicated().sum())
        if num_duplicates > 0:
            msg = (
                f"Dataset contains {num_duplicates} duplicate row(s). "
                "Duplicate rows are prohibited by contract policy."
            )
            error_details.append(
                ValidationErrorDetail(
                    column=None,
                    check_type="duplicate_rows",
                    message=msg,
                    invalid_count=num_duplicates,
                )
            )

    primary_key = config.get("primary_key", "customerID")
    if primary_key not in df.columns:
        msg = f"Required primary key column '{primary_key}' is missing from input data."
        error_details.append(
            ValidationErrorDetail(
                column=primary_key,
                check_type="missing_column",
                message=msg,
                invalid_count=total_rows,
            )
        )
    else:
        id_series = df[primary_key]
        blank_mask = id_series.isna() | (
            id_series.astype(str).str.strip().str.len() == 0
        )
        num_blank_ids = int(blank_mask.sum())
        if num_blank_ids > 0:
            msg = (
                f"Primary key '{primary_key}' contains "
                f"{num_blank_ids} null/blank value(s)."
            )
            error_details.append(
                ValidationErrorDetail(
                    column=primary_key,
                    check_type="null_primary_key",
                    message=msg,
                    invalid_count=num_blank_ids,
                )
            )

        dup_id_mask = id_series.duplicated(keep=False)
        num_dup_ids = int(dup_id_mask.sum())
        if num_dup_ids > 0:
            sample_dups = [str(x) for x in id_series[dup_id_mask].unique()[:5]]
            msg = f"Primary key '{primary_key}' contains duplicates: {sample_dups}."
            error_details.append(
                ValidationErrorDetail(
                    column=primary_key,
                    check_type="duplicate_primary_key",
                    message=msg,
                    invalid_count=num_dup_ids,
                    sample_invalid_values=sample_dups,
                )
            )

    parsed_df, total_charges_errors = parse_total_charges(df)
    if total_charges_errors:
        for err_msg in total_charges_errors:
            error_details.append(
                ValidationErrorDetail(
                    column="TotalCharges",
                    check_type="invalid_type_conversion",
                    message=err_msg,
                    invalid_count=1,
                )
            )

    schema = get_pandera_schema(config, is_training=is_training)
    try:
        validated_df = schema.validate(parsed_df, lazy=True)
    except pa_errors.SchemaErrors as exc:
        validated_df = parsed_df
        failure_cases = exc.failure_cases
        for _, row in failure_cases.iterrows():
            col_name = str(row.get("column", "DataFrame"))
            check_name = str(row.get("check", "schema_check"))
            failure_val = str(row.get("failure_case", ""))
            msg = (
                f"Pandera check '{check_name}' failed on '{col_name}'. "
                f"Invalid value: '{failure_val}'."
            )
            error_details.append(
                ValidationErrorDetail(
                    column=col_name if col_name != "DataFrame" else None,
                    check_type=check_name,
                    message=msg,
                    invalid_count=1,
                    sample_invalid_values=[failure_val] if failure_val else [],
                )
            )

    is_valid = len(error_details) == 0
    valid_rows = total_rows if is_valid else max(0, total_rows - len(error_details))
    num_errs = len(error_details)
    if is_valid:
        summary = (
            f"Validation PASSED for {total_rows} row(s) "
            f"against schema v{schema_version}."
        )
    else:
        summary = (
            f"Validation FAILED with {num_errs} error(s) across {total_rows} row(s)."
        )

    report = ValidationReport(
        is_valid=is_valid,
        schema_version=schema_version,
        total_rows=total_rows,
        valid_rows=valid_rows,
        errors=error_details,
        summary=summary,
    )

    if not is_valid and raise_on_error:
        first_err = error_details[0].message
        error_msg = f"Data validation failed: {summary}\nFirst error: {first_err}"
        raise DataValidationError(error_msg, report=report)

    return validated_df, report
