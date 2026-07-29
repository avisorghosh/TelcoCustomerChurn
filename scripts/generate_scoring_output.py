"""Thin operator script to generate formatted batch scoring output file.

Business logic resides strictly inside src/churn_prediction/models/batch_scoring.py.
"""

import argparse
import sys

from churn_prediction.models.batch_scoring import run_batch_scoring


def main() -> None:
    """CLI entry point for generating versioned batch scoring output."""
    parser = argparse.ArgumentParser(
        description="Generate versioned batch scoring output."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Telco-Customer-Churn.csv",
        help="Path to input CSV dataset.",
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
        help="Target output CSV file path.",
    )

    args = parser.parse_args()

    print("=== Generating Batch Scoring Output ===")
    try:
        scored_df, output_path = run_batch_scoring(
            input_path=args.input,
            config_path=args.config,
            output_path_override=args.output,
        )
        print(f"Scoring output generated successfully at: {output_path.resolve()}")
        print(f"Total rows scored: {len(scored_df)}")
    except Exception as e:
        print(f"Failed to generate scoring output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
