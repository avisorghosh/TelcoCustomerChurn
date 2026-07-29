"""Model feature importance and Logistic Regression coefficient extraction module."""

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from churn_prediction.features.pipeline import extract_transformed_feature_names


def extract_feature_importance(
    pipeline: Pipeline,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract and rank feature importance / coefficients from a fitted pipeline.

    Supports Logistic Regression linear coefficients and tree-based feature importances.

    Args:
        pipeline: Fitted scikit-learn Pipeline instance.
        metadata: Optional metadata dictionary containing feature_names_out.

    Returns:
        List of dictionaries with feature name, coefficient/importance value,
        absolute magnitude, and odds ratio, sorted by magnitude.
    """
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        raise ValueError("Pipeline does not contain a 'classifier' step.")

    # Retrieve transformed feature names
    feature_names: list[str] = []
    if metadata and "feature_names_out" in metadata and metadata["feature_names_out"]:
        feature_names = list(metadata["feature_names_out"])
    else:
        feature_names = extract_transformed_feature_names(pipeline)

    importance_list: list[dict[str, Any]] = []

    if hasattr(classifier, "coef_"):
        coefs = np.asarray(classifier.coef_).ravel()
        if len(feature_names) == len(coefs):
            names = feature_names
        else:
            names = [f"feature_{i}" for i in range(len(coefs))]

        for name, coef in zip(names, coefs, strict=False):
            c_val = float(coef)
            abs_val = float(abs(coef))
            # Clip odds ratio exponent to avoid numeric overflow
            odds_ratio = float(np.exp(np.clip(coef, -20.0, 20.0)))
            importance_list.append(
                {
                    "feature": name,
                    "coefficient": round(c_val, 4),
                    "abs_coefficient": round(abs_val, 4),
                    "odds_ratio": round(odds_ratio, 4),
                }
            )
    elif hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_).ravel()
        if len(feature_names) == len(importances):
            names = feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importances))]

        for name, imp in zip(names, importances, strict=False):
            imp_val = float(imp)
            importance_list.append(
                {
                    "feature": name,
                    "coefficient": round(imp_val, 4),
                    "abs_coefficient": round(imp_val, 4),
                    "odds_ratio": 1.0,
                }
            )

    # Sort descending by absolute coefficient / importance
    importance_list.sort(key=lambda x: x["abs_coefficient"], reverse=True)
    return importance_list
