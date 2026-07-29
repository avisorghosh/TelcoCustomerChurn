"""Thin operator script to validate batch scoring input dataset.

Business logic resides strictly inside src/churn_prediction/models/batch_scoring.py.
"""

import argparse
import sys

import pandas as pd

from churn_prediction.models.batch_scoring import (
    generate_batch_id,
    load_serving_config,
    quarantine_batch,
    validate_scoring_batch,
)


def main() -> None:
    """CLI entry point for validating scoring input dataset."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate input dataset against data contract prior to batch scoring."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Telco-Customer-Churn.csv",
        help="Path to input dataset CSV to validate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to serving YAML configuration file.",
    )
    parser.add_argument(
        "--contract-config",
        type=str,
        default=None,
        help="Path to data contract YAML configuration file.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=str,
        default=None,
        help="Directory to save quarantine report if validation fails.",
    )

    args = parser.parse_args()

    print("=== Validating Scoring Input Dataset ===")
    try:
        config = load_serving_config(args.config)
        contract_path = args.contract_config or config.get("data", {}).get(
            "contract_config_path"
        )
        quarantine_dir = args.quarantine_dir or config.get("output", {}).get(
            "quarantine_dir", "reports/quarantine"
        )

        df = pd.read_csv(args.input)
        _, report = validate_scoring_batch(
            df,
            contract_config_path=contract_path,
            raise_on_error=False,
        )

        if report.is_valid:
            print("Validation PASSED successfully!")
            print(f"  - Total Rows: {report.total_rows}")
            print(f"  - Summary:    {report.summary}")
        else:
            batch_id = generate_batch_id(prefix="val_failed")
            q_path = quarantine_batch(report, quarantine_dir, batch_id)
            print(
                f"Validation FAILED with {len(report.errors)} error(s).",
                file=sys.stderr,
            )
            print(f"  - Summary:           {report.summary}", file=sys.stderr)
            print(f"  - Quarantine Report: {q_path.resolve()}", file=sys.stderr)
            for err in report.errors:
                print(
                    f"    * [{err.check_type}] Column '{err.column}': {err.message}",
                    file=sys.stderr,
                )
            sys.exit(1)
    except Exception as e:
        print(f"Validation execution FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
