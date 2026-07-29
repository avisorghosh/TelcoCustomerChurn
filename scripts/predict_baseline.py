"""Thin operator script to load trained pipeline and run sample inference.

Business logic resides strictly inside src/churn_prediction/models/.
"""

import argparse
import sys

import pandas as pd

from churn_prediction.models.serialization import load_artifacts
from churn_prediction.models.trainer import predict_churn


def main() -> None:
    """Execute inference verification CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Load trained baseline pipeline and validate inference."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Path to directory containing serialized model artifacts.",
    )
    args = parser.parse_args()

    print("=== Loading Baseline Pipeline and Validating Inference ===")
    try:
        pipeline, metadata = load_artifacts(args.output_dir)
        print(f"Loaded Model:   {metadata.get('model_name')}")
        print(f"Schema Version: {metadata.get('schema_version')}")
        print(f"Timestamp:      {metadata.get('timestamp')}")

        # Sample customer record for validation
        sample_customer = pd.DataFrame(
            [
                {
                    "customerID": "7590-VHVEG",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        )

        results = predict_churn(pipeline, sample_customer)
        churn_pred = int(results["churn_prediction"].iloc[0])
        churn_prob = float(results["churn_probability"].iloc[0])

        label = "Churn" if churn_pred == 1 else "No Churn"
        print("\nInference Results for Sample Customer '7590-VHVEG':")
        print(f"  - Predicted Churn Class:       {churn_pred} ({label})")
        print(f"  - Predicted Churn Probability: {churn_prob:.4f}")
        print("\nInference validation PASSED successfully!")
    except Exception as e:
        print(f"Inference FAILED with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
