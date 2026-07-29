"""Thin operator script to evaluate the baseline churn prediction model.

Business logic resides strictly inside src/churn_prediction/evaluation/.
"""

import argparse
import sys

from churn_prediction.evaluation.evaluator import evaluate_model


def main() -> None:
    """Execute model evaluation CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate the baseline churn prediction model artifact."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation configuration YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained baseline model joblib file (overrides config).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw dataset CSV file (overrides config).",
    )
    args = parser.parse_args()

    print("=== Evaluating Baseline Churn Prediction Model ===")
    try:
        summary, generated_artifacts = evaluate_model(
            config_path=args.config,
            model_path_override=args.model,
            data_path_override=args.data,
        )

        metrics = summary.get("metrics", {})
        capacity = summary.get("campaign_capacity", {})
        calibration = summary.get("calibration", {})

        print("Evaluation completed successfully!")
        print(f"Model Name:          {summary.get('model_name')}")
        print(f"Evaluation Split:    {summary.get('evaluation_split')}")
        print(f"Sample Size:         {summary.get('sample_size')} rows")
        pr_val = summary.get("primary_metric", {}).get("value")
        print("\nPrimary Metric:")
        print(f"  - PR-AUC:            {pr_val}")
        th_val = metrics.get("threshold")
        prec_val = metrics.get("precision")
        rec_val = metrics.get("recall")
        print("\nSecondary Classification Metrics:")
        print(f"  - ROC-AUC:           {metrics.get('roc_auc')}")
        print(f"  - Precision:         {prec_val} (at threshold {th_val})")
        print(f"  - Recall:            {rec_val} (at threshold {th_val})")
        print(f"  - F1 Score:          {metrics.get('f1_score')}")
        print(f"  - Accuracy:          {metrics.get('accuracy')}")
        print(f"  - Brier Score:       {calibration.get('brier_score')}")
        frac_pct = capacity.get("campaign_capacity_fraction", 0.10) * 100
        n_tgt = capacity.get("num_targeted_customers")
        print("\nCampaign Capacity Evaluation:")
        print(f"  - Target Fraction:   {frac_pct:.1f}%")
        print(f"  - Targeted Count:    {n_tgt} customers")
        print(f"  - Precision @ Cap:   {capacity.get('precision_at_capacity')}")
        print(f"  - Recall @ Cap:      {capacity.get('recall_at_capacity')}")
        print(f"  - Risk Threshold:    {capacity.get('capacity_threshold')}")

        print("\nGenerated Evaluation Artifacts:")
        for name, path in generated_artifacts.items():
            print(f"  - {name}: {path}")

    except Exception as e:
        print(f"Evaluation FAILED with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
