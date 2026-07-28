"""Operator script for running data validation and producing quality reports."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from churn_prediction.data import (
    create_source_manifest,
    validate_data,
)


def main() -> None:
    """Run data contract validation on a specified dataset."""
    parser = argparse.ArgumentParser(
        description="Validate dataset against data contract."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Telco-Customer-Churn.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to the data contract YAML config.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Path to save the validation report JSON.",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default=None,
        help="Path to save the source manifest JSON.",
    )
    parser.add_argument(
        "--is-training",
        action="store_true",
        default=True,
        help="Whether dataset expects target column (Churn).",
    )

    args = parser.parse_args()
    data_path = Path(args.data_path)

    if not data_path.is_file():
        print(f"Error: Input data file not found at '{data_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Reading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    try:
        _, report = validate_data(
            df,
            config_path=args.config_path,
            is_training=args.is_training,
            raise_on_error=False,
        )
    except Exception as exc:
        print(f"Validation execution error: {exc}", file=sys.stderr)
        sys.exit(1)

    manifest = create_source_manifest(data_path, quality_report_path=args.report_out)

    print("\n--- Validation Summary ---")
    print(f"Status: {'PASSED' if report.is_valid else 'FAILED'}")
    print(f"Summary: {report.summary}")
    print(f"SHA-256: {manifest.file_sha256}")
    print(f"Row count: {manifest.row_count}")

    if not report.is_valid:
        print("\nValidation Errors:")
        for idx, err in enumerate(report.errors, 1):
            col_str = f" [Column: {err.column}]" if err.column else ""
            print(f" {idx}.{col_str} ({err.check_type}): {err.message}")

    if args.report_out:
        out_p = Path(args.report_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\nValidation report saved to: {out_p}")

    if args.manifest_out:
        man_p = Path(args.manifest_out)
        man_p.parent.mkdir(parents=True, exist_ok=True)
        with open(man_p, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        print(f"Source manifest saved to: {man_p}")

    if not report.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
