"""Scheduled post-deployment report generation module."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def generate_post_deployment_report(
    eval_summary: dict[str, Any],
    campaign_summary: dict[str, Any],
    retraining_decision: dict[str, Any],
    matching_stats: dict[str, Any],
    output_dir: str | Path,
    report_markdown_filename: str = "post_deployment_report.md",
    report_json_filename: str = "post_deployment_summary.json",
    evaluation_date: str = "2026-07-29",
) -> dict[str, Path]:
    """Generate structured JSON summary and human-readable Markdown report.

    Args:
        eval_summary: Dictionary output from evaluate_delayed_predictions.
        campaign_summary: Dictionary output from evaluate_campaign_effectiveness.
        retraining_decision: Dictionary output from make_retraining_decision.
        matching_stats: Operational matching statistics dictionary.
        output_dir: Path to output directory for report artifacts.
        report_markdown_filename: Filename for generated Markdown report.
        report_json_filename: Filename for generated JSON report.
        evaluation_date: Formatted evaluation date string.

    Returns:
        Dictionary mapping artifact keys to generated Path objects.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(UTC).isoformat()

    full_json_payload = {
        "report_metadata": {
            "evaluation_date": evaluation_date,
            "generated_at": now_utc,
            "system": "Telco Customer Churn Prediction System",
            "milestone": "Milestone 13 - Post-Deployment Learning",
        },
        "matching_statistics": matching_stats,
        "model_performance": eval_summary,
        "campaign_effectiveness": campaign_summary,
        "retraining_decision": retraining_decision,
    }

    json_path = out_dir / report_json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_payload, f, indent=2, ensure_ascii=False)

    metrics = eval_summary.get("metrics", {})
    cap = eval_summary.get("capacity_metrics", {})
    evidence = retraining_decision.get("evidence", {})

    if campaign_summary.get("campaign_data_available", False):
        treat = campaign_summary.get("treatment_group", {})
        ctrl = campaign_summary.get("control_group", {})
        eff = campaign_summary.get("effectiveness", {})
        fin = campaign_summary.get("financial_summary", {})

        campaign_md = f"""### Treatment vs Control Campaign Evaluation

| Metric | Treatment Group | Control Group | Differential / Impact |
| :--- | :--- | :--- | :--- |
| **Sample Size** | {treat.get("sample_size", 0):,} | {ctrl.get("sample_size", 0):,} | Total N = {matching_stats.get("matched_records", 0):,} |
| **Observed Churn Rate** | {treat.get("churn_rate", 0.0):.2%} | {ctrl.get("churn_rate", 0.0):.2%} | **-{eff.get("absolute_churn_reduction", 0.0):.2%}** (Absolute) |
| **Relative Churn Reduction** | - | - | **{eff.get("relative_churn_reduction_pct", 0.0):.2f}%** |
| **Estimated Retained Customers** | - | - | **+{eff.get("estimated_incremental_retained_customers", 0.0):.1f}** customers |

#### Campaign Financial Return
- **Unit Contact Cost:** ${fin.get("unit_contact_cost_usd", 0.0):.2f} / targeted customer
- **Unit Retained Value:** ${fin.get("unit_customer_value_usd", 0.0):.2f} / prevented churn
- **Total Campaign Investment:** ${fin.get("total_campaign_cost_usd", 0.0):,.2f}
- **Gross Retained Margin:** ${fin.get("gross_retained_margin_usd", 0.0):,.2f}
- **Net Campaign Value:** **${fin.get("net_campaign_value_usd", 0.0):,.2f}**
- **Campaign ROI:** **{fin.get("campaign_roi_percent", 0.0):.2f}%**
"""
    else:
        campaign_md = f"""### Treatment vs Control Campaign Evaluation

> [!NOTE]
> {campaign_summary.get("assumptions_documented", "Campaign group indicators unavailable.")}
"""

    pr_status = (
        "✅ Passed"
        if not evidence.get("pr_auc_drop_threshold")
        or abs(evidence.get("pr_auc_delta", 0))
        <= evidence.get("pr_auc_drop_threshold", 0.05)
        else "⚠️ Degraded"
    )

    brier_status = (
        "✅ Calibrated"
        if metrics.get("brier_score", 0) <= evidence.get("brier_score_max", 0.20)
        else "⚠️ Miscalibrated"
    )

    md_content = f"""# Post-Deployment Learning & Model Performance Review

**Evaluation Date:** {evaluation_date}  
**Report Generated:** {now_utc}  
**Target System:** Telco Customer Churn Production Model  

---

## 📌 Executive Summary

This report evaluates incumbent production model performance and campaign outcomes following the arrival of delayed ground-truth churn labels. It provides an empirical audit of prediction accuracy, calibration quality, campaign financial effectiveness, and a documented retraining decision.

- **Observed PR-AUC:** `{metrics.get("pr_auc", 0.0):.4f}` (Baseline benchmark: `{evidence.get("baseline_pr_auc", 0.6437):.4f}`)
- **Brier Calibration Score:** `{metrics.get("brier_score", 0.0):.4f}`
- **Retraining Decision:** **{retraining_decision.get("decision_code", "MAINTAIN_INCUMBENT")}**

---

## 📊 Delayed Label Ingestion & Data Quality

Historical batch predictions were reconciled against delayed observed outcomes:

| Indicator | Value | Description |
| :--- | :--- | :--- |
| **Total Predictions Evaluated** | `{matching_stats.get("total_historical_predictions", 0):,}` | Historical batch predictions in scope |
| **Delayed Labels Received** | `{matching_stats.get("total_delayed_labels_received", 0):,}` | Ground truth outcomes observed |
| **Matched Records** | `{matching_stats.get("matched_records", 0):,}` | Successfully reconciled on `customerID` |
| **Matching Rate** | `{matching_stats.get("match_rate", 0.0):.2%}` | Data pipeline reconciliation rate |
| **Observed Churn Prevalence** | `{eval_summary.get("observed_churn_prevalence", 0.0):.2%}` | Empirical churn rate in observed window |

---

## 📈 Incumbent Model Performance Review

Model predictions were evaluated against observed ground-truth labels using the primary evaluation contract:

| Metric | Delayed Label Value | Baseline Benchmark | Operational Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| **PR-AUC (Primary)** | `{metrics.get("pr_auc", 0.0):.4f}` | `{evidence.get("baseline_pr_auc", 0.6437):.4f}` | Maintain within 0.05 | {pr_status} |
| **ROC-AUC** | `{metrics.get("roc_auc", 0.0):.4f}` | ~`0.8487` | Ranking discriminative power | Informational |
| **Brier Calibration Score** | `{metrics.get("brier_score", 0.0):.4f}` | ~`0.1361` | `< {evidence.get("brier_score_max", 0.20):.2f}` | {brier_status} |
| **Precision @ 10% Capacity** | `{cap.get("precision_at_capacity", 0.0):.2%}` | ~`75.47%` | Campaign targeting precision | Operational |
| **Recall @ 10% Capacity** | `{cap.get("recall_at_capacity", 0.0):.2%}` | ~`28.57%` | Churner capture rate at top 10% | Operational |
| **Accuracy @ 0.50 Threshold** | `{metrics.get("accuracy", 0.0):.2%}` | ~`79.85%` | Overall classification accuracy | Informational |

---

## 🎯 Retention Campaign Effectiveness

{campaign_md}

---

## 🧠 Retraining Decision Framework

### Recommendation: **{retraining_decision.get("decision_code", "MAINTAIN_INCUMBENT")}**

**Rationale:**  
{retraining_decision.get("summary_reason", "")}

**Detailed Findings:**
"""

    for reason in retraining_decision.get("decision_reasons", []):
        md_content += f"\n- {reason}"

    retrain_next = (
        "Initiate candidate model retraining and trigger offline comparison suite."
        if retraining_decision.get("retrain_recommended")
        else "Proceed with current incumbent version. Schedule next post-deployment evaluation."
    )

    md_content += f"""

> [!IMPORTANT]
> **Governance Policy Note:** {retraining_decision.get("policy_note", "")}

---

## ⚠️ Operational Observations & Limitations

1. **Proxy Horizon:** Ground-truth observation matches the weekly billing cycle churn proxy. Real production deployments require alignment with contract terms and billing cycles.
2. **Campaign Assignment:** Financial returns assume randomized or propensity-adjusted treatment/control split. Non-randomized interventions can introduce selection bias.
3. **Drift Monitoring:** Population Stability Index (PSI) maximum was `{evidence.get("max_psi", 0.0):.4f}` (Alert threshold: `{evidence.get("psi_alert_threshold", 0.25):.2f}`).

---

## 🚀 Recommended Next Actions

1. **Model Governance:** {retrain_next}
2. **Campaign Operations:** Review contact capacity allocations to align targeted lists with high-precision deciles.
3. **Data Quality:** Maintain data contract matching rules to ensure 100% reconciliation on customer keys.

---
*Report generated automatically by `churn_prediction.post_deployment` reporting pipeline.*
"""

    md_path = out_dir / report_markdown_filename
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "report_summary_json": json_path,
        "report_markdown": md_path,
    }
