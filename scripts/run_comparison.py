"""Thin operator script to execute candidate comparison and generate Decision Record.

Business logic resides strictly inside src/churn_prediction/evaluation/comparator.py.
"""

import argparse
import sys

from churn_prediction.evaluation.comparator import compare_baseline_and_candidate


def main() -> None:
    """Execute model comparison CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Baseline and Candidate models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation configuration YAML file.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw dataset CSV file (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation",
        help="Output directory for generated decision records and reports.",
    )
    args = parser.parse_args()

    print("=== Running Baseline vs Candidate Model Comparison ===")
    try:
        decision_record, output_paths = compare_baseline_and_candidate(
            config_path=args.config,
            data_path_override=args.data,
            output_dir=args.output_dir,
        )

        final_decision = decision_record.get("final_decision", {})
        metrics_diff = decision_record.get("metrics_summary", {}).get(
            "difference_candidate_minus_baseline", {}
        )

        print("\nComparison Completed Successfully!")
        print("--------------------------------------------------")
        print(f"Selected Model: {final_decision.get('selected_model_name')}")
        print(f"Model Type:     {final_decision.get('selected_model_type')}")
        print(f"PR-AUC Diff:    {metrics_diff.get('pr_auc'):+.4f}")
        print(f"Brier Diff:     {metrics_diff.get('brier_score'):+.4f}")
        print(f"JSON Record:    {output_paths['decision_record_json']}")
        print(f"Markdown Record:{output_paths['decision_record_md']}")
        print("--------------------------------------------------")
        print(f"\nRationale:\n{final_decision.get('rationale')}")

    except Exception as e:
        print(f"Model comparison FAILED with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
