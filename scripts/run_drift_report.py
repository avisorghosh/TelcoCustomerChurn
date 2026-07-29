#!/usr/bin/env python
"""Thin operator script to execute lightweight data drift analysis."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to sys.path to enable imports when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn_prediction.monitoring.drift import detect_data_drift


def main() -> None:
    """CLI entry point for running data drift report."""
    parser = argparse.ArgumentParser(
        description="Run data drift analysis comparing scoring data to reference data."
    )
    parser.add_argument(
        "--scoring-path",
        type=str,
        default="reports/scoring/batch_predictions.csv",
        help="Path to scoring CSV dataset.",
    )
    parser.add_argument(
        "--reference-path",
        type=str,
        default="Telco-Customer-Churn.csv",
        help="Path to baseline reference dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="reports/drift/drift_report.json",
        help="Path to save output JSON drift report.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/observability.yaml",
        help="Path to observability config YAML.",
    )
    args = parser.parse_args()

    scoring_p = Path(args.scoring_path)
    ref_p = Path(args.reference_path)

    if not ref_p.is_file():
        print(f"Error: Reference dataset file not found at '{ref_p}'", file=sys.stderr)
        sys.exit(1)

    if not scoring_p.is_file():
        print(
            f"Warning: Scoring file missing at '{scoring_p}'. Using fallback.",
            file=sys.stderr,
        )
        scoring_p = ref_p

    ref_df = pd.read_csv(ref_p)
    scoring_df = pd.read_csv(scoring_p)

    report = detect_data_drift(
        scoring_df=scoring_df,
        reference_df=ref_df,
        config_path=args.config_path,
        output_path=args.output_path,
    )

    print("Data Drift Report generated successfully:")
    print(f"  Summary: {report.summary}")
    print(f"  Overall Drift Status: {report.overall_drift_status}")
    print(f"  Max PSI: {report.max_psi}")
    print(f"  Drifted Features Count: {report.drifted_features_count}")
    print(f"  Saved report to: {args.output_path}")


if __name__ == "__main__":
    main()
