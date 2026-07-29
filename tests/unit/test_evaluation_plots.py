"""Unit tests for plot generation functions."""

import tempfile
from pathlib import Path

from churn_prediction.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_analysis,
)


def test_plot_generation_creates_nonempty_files() -> None:
    """Verify plot generation functions save non-empty image files."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. ROC Plot
        roc_path = plot_roc_curve(
            y_true, y_prob, roc_auc=0.95, output_path=tmp_path / "roc.png"
        )
        assert roc_path.is_file()
        assert roc_path.stat().st_size > 0

        # 2. PR Plot
        pr_path = plot_precision_recall_curve(
            y_true, y_prob, pr_auc=0.92, output_path=tmp_path / "pr.png"
        )
        assert pr_path.is_file()
        assert pr_path.stat().st_size > 0

        # 3. Calibration Plot
        cal_path = plot_calibration_curve(
            [0.1, 0.8],
            [0.15, 0.75],
            brier_score=0.08,
            output_path=tmp_path / "cal.png",
        )
        assert cal_path.is_file()
        assert cal_path.stat().st_size > 0

        # 4. Confusion Matrix Plot
        cm_path = plot_confusion_matrix(
            {"tn": 3, "fp": 0, "fn": 0, "tp": 3},
            output_path=tmp_path / "cm.png",
        )
        assert cm_path.is_file()
        assert cm_path.stat().st_size > 0

        # 5. Threshold Analysis Plot
        th_data = [
            {"threshold": 0.3, "precision": 0.8, "recall": 1.0, "f1_score": 0.88},
            {"threshold": 0.5, "precision": 1.0, "recall": 0.8, "f1_score": 0.88},
        ]
        th_path = plot_threshold_analysis(
            th_data, policy_threshold=0.5, output_path=tmp_path / "th.png"
        )
        assert th_path.is_file()
        assert th_path.stat().st_size > 0

        # 6. Feature Importance Plot
        fi_data = [
            {
                "feature": "Contract_Month-to-month",
                "coefficient": 1.25,
                "abs_coefficient": 1.25,
            },
            {"feature": "tenure", "coefficient": -0.85, "abs_coefficient": 0.85},
        ]
        fi_path = plot_feature_importance(
            fi_data, top_n=10, output_path=tmp_path / "fi.png"
        )
        assert fi_path.is_file()
        assert fi_path.stat().st_size > 0
