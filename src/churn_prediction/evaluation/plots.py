"""Plot generation module for evaluation reports."""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc_curve(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    roc_auc: float,
    output_path: str | Path,
) -> Path:
    """Generate and save ROC Curve plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    fpr, tpr, _ = roc_curve(y_t, y_p)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot(
        [0, 1],
        [0, 1],
        color="#7f7f7f",
        lw=1.5,
        linestyle="--",
        label="Random Chance",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(
        "Receiver Operating Characteristic (ROC) Curve",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_precision_recall_curve(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    pr_auc: float,
    output_path: str | Path,
) -> Path:
    """Generate and save Precision-Recall Curve plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    precision, recall, _ = precision_recall_curve(y_t, y_p)
    no_skill = float(np.mean(y_t))

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        color="#2ca02c",
        lw=2,
        label=f"PR Curve (PR-AUC = {pr_auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [no_skill, no_skill],
        color="#7f7f7f",
        lw=1.5,
        linestyle="--",
        label=f"No Skill ({no_skill:.4f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_calibration_curve(
    prob_true: list[float],
    prob_pred: list[float],
    brier_score: float,
    output_path: str | Path,
) -> Path:
    """Generate and save Reliability / Calibration Curve plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        color="#ff7f0e",
        lw=2,
        label=f"Baseline Model (Brier = {brier_score:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="#7f7f7f",
        lw=1.5,
        linestyle="--",
        label="Perfectly Calibrated",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives (True Churn)", fontsize=11)
    plt.title("Probability Calibration Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_confusion_matrix(
    cm_dict: dict[str, int],
    output_path: str | Path,
) -> Path:
    """Generate and save Confusion Matrix heatmap plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(
        [
            [cm_dict.get("tn", 0), cm_dict.get("fp", 0)],
            [cm_dict.get("fn", 0), cm_dict.get("tp", 0)],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
    fig.colorbar(cax)

    classes = ["No Churn", "Churn"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.xaxis.set_ticks_position("bottom")

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            ax.text(
                j,
                i,
                f"{count}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_threshold_analysis(
    threshold_results: list[dict[str, Any]],
    policy_threshold: float,
    output_path: str | Path,
) -> Path:
    """Generate and save Threshold Sensitivity plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = [r["threshold"] for r in threshold_results]
    precisions = [r["precision"] for r in threshold_results]
    recalls = [r["recall"] for r in threshold_results]
    f1s = [r["f1_score"] for r in threshold_results]

    plt.figure(figsize=(8, 6))
    plt.plot(
        thresholds,
        precisions,
        marker="s",
        color="#1f77b4",
        lw=2,
        label="Precision",
    )
    plt.plot(thresholds, recalls, marker="o", color="#d62728", lw=2, label="Recall")
    plt.plot(thresholds, f1s, marker="^", color="#9467bd", lw=2, label="F1 Score")
    plt.axvline(
        x=policy_threshold,
        color="#7f7f7f",
        linestyle="--",
        lw=1.5,
        label=f"Policy Threshold ({policy_threshold:.2f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("Metrics vs. Decision Threshold", fontsize=13, fontweight="bold")
    plt.legend(loc="center right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_feature_importance(
    importance_list: list[dict[str, Any]],
    top_n: int = 15,
    output_path: str | Path = "reports/evaluation/feature_importance.png",
) -> Path:
    """Generate and save Feature Importance / Coefficients bar chart."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    top_features = importance_list[:top_n]
    top_features = top_features[::-1]

    names = [item["feature"] for item in top_features]
    coefs = [item["coefficient"] for item in top_features]
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in coefs]

    plt.figure(figsize=(9, 6))
    plt.barh(names, coefs, color=colors, alpha=0.85)
    plt.axvline(x=0, color="#000000", lw=0.8, linestyle="-")
    plt.xlabel(
        "Logistic Regression Coefficient (Impact on Churn Log-Odds)", fontsize=11
    )
    plt.ylabel("Feature", fontsize=11)
    plt.title(
        f"Top {len(top_features)} Feature Coefficients",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path
