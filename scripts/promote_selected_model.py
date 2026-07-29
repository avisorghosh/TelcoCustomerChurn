"""Thin operator script to promote the decision-record winner to serving artifacts.

Business logic resides in churn_prediction.models.promote.
"""

import argparse
import sys

from churn_prediction.models.promote import PromotionError, promote_selected_model


def main() -> None:
    """CLI entry point for promoting the selected model into serving filenames."""
    parser = argparse.ArgumentParser(
        description=(
            "Promote the model named in the decision record to serving_pipeline.joblib "
            "and serving_metadata.json, and update configs/serving.yaml."
        )
    )
    parser.add_argument(
        "--decision-record",
        type=str,
        default="reports/evaluation/decision_record.json",
        help="Path to decision record JSON.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained baseline/candidate artifacts.",
    )
    parser.add_argument(
        "--serving-config",
        type=str,
        default="configs/serving.yaml",
        help="Serving YAML config to update.",
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Copy artifacts only; do not rewrite serving.yaml.",
    )
    args = parser.parse_args()

    try:
        paths = promote_selected_model(
            decision_record_path=args.decision_record,
            model_dir=args.model_dir,
            serving_config_path=None if args.no_update_config else args.serving_config,
            update_serving_config=not args.no_update_config,
        )
    except (PromotionError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Promotion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Promotion complete.")
    print(f"  Pipeline:  {paths['pipeline_path']}")
    print(f"  Metadata:  {paths['metadata_path']}")


if __name__ == "__main__":
    main()
