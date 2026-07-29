"""Thin operator script to restore a model artifact.

Supports MLflow Registry or local backup restoration.
Business logic resides strictly inside src/churn_prediction/models/restore.py.
"""

import argparse
import sys

from churn_prediction.models.restore import (
    restore_model_from_dir,
    restore_model_from_registry,
)


def main() -> None:
    """Execute model restoration CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Restore model artifact from MLflow registry or local directory."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version",
        type=str,
        help="MLflow registered model version number to restore (e.g. 1).",
    )
    group.add_argument(
        "--source-dir",
        type=str,
        help="Local directory path containing backup model artifacts.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="models",
        help="Destination directory for active model artifacts (default: models).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="MLflow registered model name (default: telco_churn_model).",
    )

    args = parser.parse_args()

    print("=== Model Restoration & Rollback Utility ===")
    try:
        if args.version:
            print(f"Restoring MLflow registered model version '{args.version}'...")
            pipe_p, meta_p, meta = restore_model_from_registry(
                version=args.version,
                registered_model_name=args.model_name,
                target_dir=args.target_dir,
            )
        else:
            print(f"Restoring artifacts from source dir '{args.source_dir}'...")
            pipe_p, meta_p, meta = restore_model_from_dir(
                source_dir=args.source_dir,
                target_dir=args.target_dir,
            )

        print("Model restoration completed successfully!")
        print(f"Target Directory: {args.target_dir}")
        print(f"Pipeline Path:    {pipe_p}")
        print(f"Metadata Path:    {meta_p}")
        print(f"Model Name:       {meta.get('model_name')}")
        print(
            f"Model Version:    {meta.get('model_version', meta.get('schema_version'))}"
        )
    except Exception as e:
        print(f"Model restoration FAILED with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
