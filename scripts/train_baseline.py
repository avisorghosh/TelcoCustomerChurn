"""Thin operator script to train the baseline churn prediction model.

Business logic resides strictly inside src/churn_prediction/models/trainer.py.
"""

import argparse
import sys

from churn_prediction.models.trainer import train_baseline


def main() -> None:
    """Execute training CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train the baseline churn prediction model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw dataset CSV file (overrides config).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking and model logging.",
    )
    args = parser.parse_args()

    print("=== Training Baseline Churn Prediction Model ===")
    try:
        pipeline, metadata, artifact_paths = train_baseline(
            config_path=args.config,
            data_path_override=args.data,
            log_to_mlflow=not args.no_mlflow,
        )

        print("Training completed successfully!")
        print(f"Model Name:      {metadata.get('model_name')}")
        print(f"Random Seed:     {metadata.get('random_seed')}")
        print(f"Train Rows:      {metadata.get('split_counts', {}).get('train')}")
        print(f"Val Rows:        {metadata.get('split_counts', {}).get('val')}")
        print(f"Test Rows:       {metadata.get('split_counts', {}).get('test')}")
        print("Training Metrics:")
        for metric, val in metadata.get("training_metrics", {}).items():
            print(f"  - {metric}: {val}")
        print(f"Saved Pipeline:  {artifact_paths['pipeline_path']}")
        print(f"Saved Metadata:  {artifact_paths['metadata_path']}")
    except Exception as e:
        print(f"Training FAILED with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
