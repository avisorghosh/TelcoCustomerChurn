"""Thin operator script to run batch scoring pipeline.

Business logic resides strictly inside src/churn_prediction/models/batch_scoring.py.
"""

import argparse
import sys

from churn_prediction.models.batch_scoring import (
    BatchScoringError,
    BatchValidationError,
    ModelLoadError,
    run_batch_scoring,
)


def main() -> None:
    """CLI entry point for running batch scoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Run batch churn prediction scoring pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Telco-Customer-Churn.csv",
        help="Path to input dataset CSV to score.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to serving YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output predictions CSV file override.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Explicit batch identifier string.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override.",
    )

    args = parser.parse_args()

    print("=== Running Batch Scoring Pipeline ===")
    try:
        scored_df, output_path = run_batch_scoring(
            input_path=args.input,
            config_path=args.config,
            output_path_override=args.output,
            batch_id_override=args.batch_id,
            decision_threshold_override=args.threshold,
        )

        print("Scoring Complete!")
        print(f"  - Total Customers Scored: {len(scored_df)}")
        print(f"  - Batch Identifier:       {scored_df['batch_id'].iloc[0]}")
        print(f"  - Model Version:          {scored_df['model_version'].iloc[0]}")
        print(f"  - Decision Threshold:     {scored_df['decision_threshold'].iloc[0]}")
        print(f"  - Output Destination:     {output_path.resolve()}")
    except BatchValidationError as e:
        print(f"\nBATCH VALIDATION FAILED: {e}", file=sys.stderr)
        if e.report:
            print(f"Summary: {e.report.summary}", file=sys.stderr)
        sys.exit(1)
    except (ModelLoadError, BatchScoringError, FileNotFoundError) as e:
        print(f"\nBATCH SCORING FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED FAILURE: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
