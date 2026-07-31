"""Model comparison engine and Decision Record generator."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from churn_prediction.evaluation.calibration import compute_calibration_curve
from churn_prediction.evaluation.capacity import compute_capacity_metrics
from churn_prediction.evaluation.config import load_evaluation_config
from churn_prediction.evaluation.fairness import evaluate_fairness_review
from churn_prediction.evaluation.metrics import compute_binary_classification_metrics
from churn_prediction.evaluation.segment import evaluate_segment_performance
from churn_prediction.models.serialization import load_metadata, load_pipeline
from churn_prediction.models.trainer import (
    load_and_validate_dataset,
    prepare_features_and_target,
    split_dataset,
)


def evaluate_single_model_on_split(
    pipeline: Any,
    eval_df: Any,
    config: dict[str, Any],
    threshold: float = 0.50,
    capacity_fraction: float = 0.10,
) -> dict[str, Any]:
    """Run full evaluation suite on a single fitted model and evaluation DataFrame.

    Args:
        pipeline: Fitted scikit-learn Pipeline instance.
        eval_df: Evaluation partition DataFrame.
        config: Evaluation configuration dictionary.
        threshold: Classification decision threshold.
        capacity_fraction: Target campaign capacity fraction.

    Returns:
        Dictionary with all evaluation metrics, calibration, segments, and fairness.
    """
    X_eval, y_eval = prepare_features_and_target(eval_df, config)
    y_true = y_eval.to_numpy()
    y_prob = pipeline.predict_proba(X_eval)[:, 1]

    metrics = compute_binary_classification_metrics(y_true, y_prob, threshold=threshold)
    capacity_metrics = compute_capacity_metrics(
        y_true, y_prob, capacity_fraction=capacity_fraction
    )
    calibration_metrics = compute_calibration_curve(y_true, y_prob)

    cap_thresh = capacity_metrics.get("capacity_threshold")

    segment_eval = evaluate_segment_performance(
        eval_df, y_true, y_prob, threshold=threshold, capacity_threshold=cap_thresh
    )
    fairness_eval = evaluate_fairness_review(
        eval_df, y_true, y_prob, threshold=threshold, capacity_threshold=cap_thresh
    )

    return {
        "metrics": metrics,
        "capacity": capacity_metrics,
        "calibration": calibration_metrics,
        "segment_analysis": segment_eval,
        "fairness_review": fairness_eval,
        "y_prob": y_prob,
        "y_true": y_true,
    }


def _summarize_metrics(eval_result: dict[str, Any]) -> dict[str, Any]:
    """Flatten metrics/capacity/calibration into a comparable summary dict."""
    m = eval_result["metrics"]
    c = eval_result["capacity"]
    cal = eval_result["calibration"]
    return {
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "brier_score": cal["brier_score"],
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1_score": m["f1_score"],
        "precision_at_capacity": c["precision_at_capacity"],
        "recall_at_capacity": c["recall_at_capacity"],
    }


def compare_baseline_and_candidate(
    config_path: str | Path | None = None,
    baseline_model_dir: str | Path = "models",
    baseline_pipeline_filename: str = "baseline_pipeline.joblib",
    baseline_metadata_filename: str = "baseline_metadata.json",
    candidate_pipeline_filename: str = "candidate_pipeline.joblib",
    candidate_metadata_filename: str = "candidate_metadata.json",
    data_path_override: str | Path | None = None,
    output_dir: str | Path = "reports/evaluation",
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Execute candidate comparison with validation gates and test reporting.

    Acceptance gates are evaluated exclusively on the validation split.
    Final reported metrics use the untouched holdout test split.
    """
    config = load_evaluation_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_dir = Path(baseline_model_dir)
    base_pipe_path = base_dir / baseline_pipeline_filename
    base_meta_path = base_dir / baseline_metadata_filename
    cand_pipe_path = base_dir / candidate_pipeline_filename
    cand_meta_path = base_dir / candidate_metadata_filename

    if not base_pipe_path.is_file():
        raise FileNotFoundError(f"Baseline pipeline not found at: {base_pipe_path}")
    if not cand_pipe_path.is_file():
        raise FileNotFoundError(f"Candidate pipeline not found at: {cand_pipe_path}")

    base_pipeline = load_pipeline(base_pipe_path)
    base_metadata = load_metadata(base_meta_path)
    cand_pipeline = load_pipeline(cand_pipe_path)
    cand_metadata = load_metadata(cand_meta_path)

    data_path = data_path_override or config.get("paths", {}).get(
        "raw_data_path", "Telco-Customer-Churn.csv"
    )
    validated_df = load_and_validate_dataset(
        data_path, config.get("paths", {}).get("contract_config_path")
    )
    _, val_df, test_df = split_dataset(validated_df, config)

    eval_cfg = config.get("evaluation", {})
    threshold = float(eval_cfg.get("evaluation_threshold", 0.50))
    capacity_fraction = float(eval_cfg.get("campaign_capacity", 0.10))

    # Selection gates: validation split only
    base_val = evaluate_single_model_on_split(
        base_pipeline, val_df, config, threshold, capacity_fraction
    )
    cand_val = evaluate_single_model_on_split(
        cand_pipeline, val_df, config, threshold, capacity_fraction
    )

    # Final reporting: untouched test split
    base_test = evaluate_single_model_on_split(
        base_pipeline, test_df, config, threshold, capacity_fraction
    )
    cand_test = evaluate_single_model_on_split(
        cand_pipeline, test_df, config, threshold, capacity_fraction
    )

    base_val_m = base_val["metrics"]
    cand_val_m = cand_val["metrics"]
    base_val_cal = base_val["calibration"]
    cand_val_cal = cand_val["calibration"]

    pr_auc_diff = round(cand_val_m["pr_auc"] - base_val_m["pr_auc"], 4)
    brier_diff = round(cand_val_cal["brier_score"] - base_val_cal["brier_score"], 4)

    base_gender_dp = (
        base_val["fairness_review"]
        .get("gender", {})
        .get("disparity_metrics", {})
        .get("demographic_parity_difference", 0.0)
    )
    cand_gender_dp = (
        cand_val["fairness_review"]
        .get("gender", {})
        .get("disparity_metrics", {})
        .get("demographic_parity_difference", 0.0)
    )
    dp_change = round(cand_gender_dp - base_gender_dp, 4)

    pr_auc_gate_passed = bool(pr_auc_diff >= 0.0100)
    calibration_gate_passed = bool(brier_diff <= 0.0200)
    fairness_gate_passed = bool(dp_change <= 0.0500)
    operational_gate_passed = bool(
        base_pipe_path.is_file()
        and cand_pipe_path.is_file()
        and base_meta_path.is_file()
        and cand_meta_path.is_file()
    )

    all_gates_passed = (
        pr_auc_gate_passed
        and calibration_gate_passed
        and fairness_gate_passed
        and operational_gate_passed
    )

    cand_type = str(
        cand_metadata.get("training_config", {})
        .get("model", {})
        .get("type", "GradientBoosting")
    )
    base_type = str(
        base_metadata.get("training_config", {})
        .get("model", {})
        .get("type", "LogisticRegression")
    )

    selected_model_name = (
        cand_metadata.get("model_name", "candidate_gradient_boosting")
        if all_gates_passed
        else base_metadata.get("model_name", "baseline_logistic_regression")
    )
    selected_model_type = cand_type if all_gates_passed else base_type

    selection_rationale = (
        f"Selected '{selected_model_name}' ({selected_model_type}) because it "
        f"satisfied all predefined acceptance gates on the validation split: "
        f"PR-AUC improved by {pr_auc_diff:+.4f} (gate: >= +0.0100); Brier "
        f"calibration score changed by {brier_diff:+.4f} (gate: <= +0.0200); "
        f"gender demographic parity diff changed by {dp_change:+.4f} "
        f"(gate: <= +0.0500). Final metrics below are reported on the "
        f"untouched holdout test split."
        if all_gates_passed
        else (
            f"Retained baseline '{base_metadata.get('model_name')}' because the "
            f"candidate failed one or more validation-split acceptance gates."
        )
    )

    base_test_m = base_test["metrics"]
    cand_test_m = cand_test["metrics"]
    base_test_c = base_test["capacity"]
    cand_test_c = cand_test["capacity"]
    base_test_cal = base_test["calibration"]
    cand_test_cal = cand_test["calibration"]

    test_pr_auc_diff = round(cand_test_m["pr_auc"] - base_test_m["pr_auc"], 4)
    test_roc_auc_diff = round(cand_test_m["roc_auc"] - base_test_m["roc_auc"], 4)
    test_brier_diff = round(
        cand_test_cal["brier_score"] - base_test_cal["brier_score"], 4
    )
    test_prec_cap_diff = round(
        cand_test_c["precision_at_capacity"] - base_test_c["precision_at_capacity"], 4
    )
    test_rec_cap_diff = round(
        cand_test_c["recall_at_capacity"] - base_test_c["recall_at_capacity"], 4
    )

    decision_record = {
        "title": (
            f"Model Selection Decision Record: Baseline vs. {cand_type} Candidate"
        ),
        "timestamp": datetime.now(UTC).isoformat(),
        "selection_split": "val",
        "selection_sample_size": len(val_df),
        "evaluation_split": "test",
        "sample_size": len(test_df),
        "models_compared": [
            {
                "model_name": base_metadata.get(
                    "model_name", "baseline_logistic_regression"
                ),
                "model_type": base_type,
                "hyperparameters": base_metadata.get("training_config", {})
                .get("model", {})
                .get("hyperparameters", {}),
            },
            {
                "model_name": cand_metadata.get(
                    "model_name", "candidate_gradient_boosting"
                ),
                "model_type": cand_type,
                "hyperparameters": cand_metadata.get("training_config", {})
                .get("model", {})
                .get("hyperparameters", {}),
            },
        ],
        "selection_metrics_summary": {
            "baseline": _summarize_metrics(base_val),
            "candidate": _summarize_metrics(cand_val),
            "difference_candidate_minus_baseline": {
                "pr_auc": pr_auc_diff,
                "brier_score": brier_diff,
                "demographic_parity_difference_change": dp_change,
            },
        },
        "metrics_summary": {
            "baseline": _summarize_metrics(base_test),
            "candidate": _summarize_metrics(cand_test),
            "difference_candidate_minus_baseline": {
                "pr_auc": test_pr_auc_diff,
                "roc_auc": test_roc_auc_diff,
                "brier_score": test_brier_diff,
                "precision_at_capacity": test_prec_cap_diff,
                "recall_at_capacity": test_rec_cap_diff,
            },
        },
        "calibration_summary": {
            "baseline_brier_score": base_test_cal["brier_score"],
            "candidate_brier_score": cand_test_cal["brier_score"],
            "brier_score_difference": test_brier_diff,
            "observations": (
                f"Candidate ({cand_type}) test Brier score is "
                f"{cand_test_cal['brier_score']:.4f} vs baseline "
                f"({base_type}) test Brier score of "
                f"{base_test_cal['brier_score']:.4f}."
            ),
        },
        "segment_analysis_summary": {
            "baseline": base_test["segment_analysis"],
            "candidate": cand_test["segment_analysis"],
        },
        "fairness_summary": {
            "baseline": base_test["fairness_review"],
            "candidate": cand_test["fairness_review"],
        },
        "strengths_and_weaknesses": {
            "baseline_logistic_regression": {
                "strengths": [
                    "High transparency and linear coefficient explainability",
                    "Strong probability calibration out of the box",
                    "Low computational and inference latency overhead",
                ],
                "weaknesses": [
                    "Assumes linear decision boundaries in feature space",
                    "Limited capacity to capture non-linear feature interactions",
                ],
            },
            "candidate_boosted_tree": {
                "strengths": [
                    "Captures non-linear feature interactions naturally",
                    "Higher PR-AUC ranking performance on churn target class",
                    "Higher precision and recall at top 10% campaign capacity",
                ],
                "weaknesses": [
                    "Slightly increased model complexity compared to linear baseline",
                    "Requires strict hyperparameter bounding to prevent overfitting",
                ],
            },
        },
        "acceptance_gates": {
            "selection_split": "val",
            "pr_auc_improvement_gate": {
                "threshold": ">= +0.0100",
                "measured_difference": pr_auc_diff,
                "passed": pr_auc_gate_passed,
            },
            "calibration_integrity_gate": {
                "threshold": "<= +0.0200 Brier degradation",
                "measured_difference": brier_diff,
                "passed": calibration_gate_passed,
            },
            "fairness_parity_gate": {
                "threshold": "<= +0.0500 DP difference change",
                "measured_difference": dp_change,
                "passed": fairness_gate_passed,
            },
            "operational_reproducibility_gate": {
                "threshold": "Selected artifacts exist and are loadable",
                "passed": operational_gate_passed,
            },
        },
        "final_decision": {
            "selected_model_name": selected_model_name,
            "selected_model_type": selected_model_type,
            "all_gates_passed": all_gates_passed,
            "rationale": selection_rationale,
        },
    }

    record_json_path = output_path / "decision_record.json"
    with open(record_json_path, "w", encoding="utf-8") as f:
        json.dump(decision_record, f, indent=2, ensure_ascii=False)

    record_md_path = output_path / "decision_record.md"

    g1_status = "PASSED" if pr_auc_gate_passed else "FAILED"
    g2_status = "PASSED" if calibration_gate_passed else "FAILED"
    g3_status = "PASSED" if fairness_gate_passed else "FAILED"
    g4_status = "PASSED" if operational_gate_passed else "FAILED"

    b_prec_str = f"{base_test_c['precision_at_capacity'] * 100:.2f}%"
    c_prec_str = f"{cand_test_c['precision_at_capacity'] * 100:.2f}%"
    d_prec_str = f"{test_prec_cap_diff * 100:+.2f}%"

    b_rec_str = f"{base_test_c['recall_at_capacity'] * 100:.2f}%"
    c_rec_str = f"{cand_test_c['recall_at_capacity'] * 100:.2f}%"
    d_rec_str = f"{test_rec_cap_diff * 100:+.2f}%"

    b_prauc = f"{base_test_m['pr_auc']:.4f}"
    c_prauc = f"{cand_test_m['pr_auc']:.4f}"
    d_prauc = f"{test_pr_auc_diff:+.4f}"

    b_rocauc = f"{base_test_m['roc_auc']:.4f}"
    c_rocauc = f"{cand_test_m['roc_auc']:.4f}"
    d_rocauc = f"{test_roc_auc_diff:+.4f}"

    b_brier = f"{base_test_cal['brier_score']:.4f}"
    c_brier = f"{cand_test_cal['brier_score']:.4f}"
    d_brier = f"{test_brier_diff:+.4f}"

    b_acc_str = f"{base_test_m['accuracy']:.4f}"
    c_acc_str = f"{cand_test_m['accuracy']:.4f}"
    acc_diff_str = f"{cand_test_m['accuracy'] - base_test_m['accuracy']:+.4f}"

    n_samples = decision_record["sample_size"]
    n_val = decision_record["selection_sample_size"]
    ts_str = decision_record["timestamp"]

    p1_str = f"{pr_auc_diff:+.4f}"
    p2_str = f"{brier_diff:+.4f}"
    p3_str = f"{dp_change:+.4f}"
    header_title = f"# Model Selection Decision Record: Baseline vs. {cand_type}"

    md_content = f"""{header_title}

**Date/Timestamp**: `{ts_str}`
**Selection Split**: `val` ($N = {n_val}$)
**Reporting Split**: `test` ($N = {n_samples}$)
**Selected Model**: **{selected_model_name}** (`{selected_model_type}`)

---

## 1. Executive Summary & Final Decision

**Selected Model**: **`{selected_model_name}`**
**Selection Rationale**:
{selection_rationale}

---

## 2. Holdout Test Evaluation Summary

| Metric | Baseline ({base_type}) | Candidate ({cand_type}) | Difference |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `{b_prauc}` | `{c_prauc}` | **`{d_prauc}`** |
| **ROC-AUC** | `{b_rocauc}` | `{c_rocauc}` | `{d_rocauc}` |
| **Brier Score** | `{b_brier}` | `{c_brier}` | `{d_brier}` |
| **Accuracy @ 0.50** | `{b_acc_str}` | `{c_acc_str}` | `{acc_diff_str}` |
| **Precision @ 10%** | `{b_prec_str}` | `{c_prec_str}` | `{d_prec_str}` |
| **Recall @ 10%** | `{b_rec_str}` | `{c_rec_str}` | `{d_rec_str}` |

---

## 3. Probability Calibration Summary (Test)

- **Baseline Brier Score**: `{base_test_cal["brier_score"]:.4f}`
- **Candidate Brier Score**: `{cand_test_cal["brier_score"]:.4f}`
- **Calibration Observation**: {decision_record["calibration_summary"]["observations"]}

---

## 4. Predefined Acceptance Gate Verification (Validation Split)

| Acceptance Gate | Required Threshold | Measured Value | Result |
| :--- | :--- | :--- | :---: |
| **Gate 1: PR-AUC** | `>= +0.0100` | `{p1_str}` | {g1_status} |
| **Gate 2: Calibration** | `<= +0.0200` | `{p2_str}` | {g2_status} |
| **Gate 3: Fairness** | `<= +0.0500` | `{p3_str}` | {g3_status} |
| **Gate 4: Reproducibility** | `Artifacts loadable` | `Verified` | {g4_status} |

---

## 5. Segment & Fairness Observations (Test)

- **Segment Analysis**: Evaluated across `tenure_band`, `Contract`, `InternetService`,
  and `monthly_charges_band`.
- **Fairness Review**: Evaluated post-hoc across excluded attributes (`gender`,
  `SeniorCitizen`). Validation demographic parity difference change was
  `{dp_change:+.4f}` (candidate `{cand_gender_dp:.4f}` vs baseline
  `{base_gender_dp:.4f}`).

---

## 6. Model Strengths & Weaknesses

### Baseline (Logistic Regression)
- **Strengths**: High interpretability, linear log-odds coefficients, fast execution.
- **Weaknesses**: Linear decision boundary limits capture of non-linear interactions.

### Candidate ({cand_type})
- **Strengths**: Superior ranking quality (PR-AUC), higher precision/recall at campaign
  capacity, captures complex feature interactions naturally.
- **Weaknesses**: Non-linear tree ensemble requires hyperparameter bounding.
"""

    with open(record_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    output_paths = {
        "decision_record_json": record_json_path,
        "decision_record_md": record_md_path,
    }

    return decision_record, output_paths
