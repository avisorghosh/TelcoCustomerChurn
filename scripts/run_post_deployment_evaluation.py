"""Thin operator script to execute post-deployment evaluation pipeline.

Business logic resides strictly inside src/churn_prediction/post_deployment/.
"""

import argparse
import sys
from pathlib import Path

from churn_prediction.post_deployment import run_post_deployment_evaluation


def main() -> None:
    """CLI entry point for post-deployment evaluation workflow."""
    parser = argparse.ArgumentParser(
        description="Run post-deployment evaluation pipeline on delayed labels."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to post_deployment.yaml configuration file.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to historical batch predictions CSV file override.",
    )
    parser.add_argument(
        "--delayed-labels",
        type=str,
        default=None,
        help="Path to delayed ground-truth labels CSV file override.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to report output directory override.",
    )

    args = parser.parse_args()

    print("=== Running Post-Deployment Evaluation Pipeline ===")
    try:
        summary, artifacts = run_post_deployment_evaluation(
            config_path=args.config,
            predictions_path_override=args.predictions,
            delayed_labels_path_override=args.delayed_labels,
            output_dir_override=args.output_dir,
        )

        match_stats = summary.get("matching_statistics", {})
        perf = summary.get("model_performance", {}).get("metrics", {})
        camp = summary.get("campaign_effectiveness", {})
        decision = summary.get("retraining_decision", {})

        print("\nPost-Deployment Evaluation Complete!")
        matched_str = (
            f"  - Matched Records:     "
            f"{match_stats.get('matched_records', 0):,} / "
            f"{match_stats.get('total_historical_predictions', 0):,} "
            f"(Match Rate: {match_stats.get('match_rate', 0.0):.2%})"
        )
        print(matched_str)
        print(f"  - Delayed PR-AUC:      {perf.get('pr_auc', 0.0):.4f}")
        print(f"  - Delayed ROC-AUC:     {perf.get('roc_auc', 0.0):.4f}")
        print(f"  - Brier Score:         {perf.get('brier_score', 0.0):.4f}")

        if camp.get("campaign_data_available", False):
            eff = camp.get("effectiveness", {})
            fin = camp.get("financial_summary", {})
            roi_str = (
                f"  - Campaign ROI:        "
                f"{fin.get('campaign_roi_percent', 0.0):.2f}% "
                f"(Net Value: ${fin.get('net_campaign_value_usd', 0.0):,.2f})"
            )
            red_str = (
                f"  - Churn Reduction:     "
                f"-{eff.get('absolute_churn_reduction', 0.0):.2%} absolute "
                f"({eff.get('relative_churn_reduction_pct', 0.0):.2f}% relative)"
            )
            print(roi_str)
            print(red_str)

        dec_code = decision.get("decision_code")
        retrain_flag = decision.get("retrain_recommended")
        print(
            f"\n  - Retraining Decision: {dec_code} (Retrain Recommended: {retrain_flag})"
        )
        print(f"  - Summary Reason:      {decision.get('summary_reason')}")

        print("\nGenerated Reports:")
        for name, path_obj in artifacts.items():
            print(f"  - {name}: {Path(path_obj).resolve()}")

    except FileNotFoundError as e:
        print(f"\nPOST-DEPLOYMENT EVALUATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED FAILURE: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
