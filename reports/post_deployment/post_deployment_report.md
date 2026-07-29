# Post-Deployment Learning & Model Performance Review

**Evaluation Date:** 2026-07-29  
**Report Generated:** 2026-07-29T15:07:16.823066+00:00  
**Target System:** Telco Customer Churn Production Model  

---

## 📌 Executive Summary

This report evaluates incumbent production model performance and campaign outcomes following the arrival of delayed ground-truth churn labels. It provides an empirical audit of prediction accuracy, calibration quality, campaign financial effectiveness, and a documented retraining decision.

- **Observed PR-AUC:** `0.5609` (Baseline benchmark: `0.6437`)
- **Brier Calibration Score:** `0.1351`
- **Retraining Decision:** **RETRAIN_RECOMMENDED**

---

## 📊 Delayed Label Ingestion & Data Quality

Historical batch predictions were reconciled against delayed observed outcomes:

| Indicator | Value | Description |
| :--- | :--- | :--- |
| **Total Predictions Evaluated** | `7,043` | Historical batch predictions in scope |
| **Delayed Labels Received** | `7,043` | Ground truth outcomes observed |
| **Matched Records** | `7,043` | Successfully reconciled on `customerID` |
| **Matching Rate** | `100.00%` | Data pipeline reconciliation rate |
| **Observed Churn Prevalence** | `23.81%` | Empirical churn rate in observed window |

---

## 📈 Incumbent Model Performance Review

Model predictions were evaluated against observed ground-truth labels using the primary evaluation contract:

| Metric | Delayed Label Value | Baseline Benchmark | Operational Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| **PR-AUC (Primary)** | `0.5609` | `0.6437` | Maintain within 0.05 | ⚠️ Degraded |
| **ROC-AUC** | `0.8359` | ~`0.8487` | Ranking discriminative power | Informational |
| **Brier Calibration Score** | `0.1351` | ~`0.1361` | `< 0.20` | ✅ Calibrated |
| **Precision @ 10% Capacity** | `64.77%` | ~`75.47%` | Campaign targeting precision | Operational |
| **Recall @ 10% Capacity** | `27.19%` | ~`28.57%` | Churner capture rate at top 10% | Operational |
| **Accuracy @ 0.50 Threshold** | `79.78%` | ~`79.85%` | Overall classification accuracy | Informational |

---

## 🎯 Retention Campaign Effectiveness

### Treatment vs Control Campaign Evaluation

| Metric | Treatment Group | Control Group | Differential / Impact |
| :--- | :--- | :--- | :--- |
| **Sample Size** | 4,536 | 2,507 | Total N = 7,043 |
| **Observed Churn Rate** | 28.33% | 15.64% | **--12.69%** (Absolute) |
| **Relative Churn Reduction** | - | - | **-81.18%** |
| **Estimated Retained Customers** | - | - | **+-575.7** customers |

#### Campaign Financial Return
- **Unit Contact Cost:** $10.00 / targeted customer
- **Unit Retained Value:** $500.00 / prevented churn
- **Total Campaign Investment:** $45,360.00
- **Gross Retained Margin:** $-287,870.56
- **Net Campaign Value:** **$-333,230.56**
- **Campaign ROI:** **-734.64%**


---

## 🧠 Retraining Decision Framework

### Recommendation: **RETRAIN_RECOMMENDED**

**Rationale:**  
Retraining IS recommended due to observed performance degradation or significant feature drift. The incumbent model should be re-evaluated against a newly fitted candidate.

**Detailed Findings:**

- PR-AUC dropped by 0.0828 (baseline: 0.6437, current delayed: 0.5609), exceeding threshold drop of 0.0500.

> [!IMPORTANT]
> **Governance Policy Note:** Automated retraining is explicitly DISABLED by system governance. This decision serves as an advisory signal for ML engineering review.

---

## ⚠️ Operational Observations & Limitations

1. **Proxy Horizon:** Ground-truth observation matches the weekly billing cycle churn proxy. Real production deployments require alignment with contract terms and billing cycles.
2. **Campaign Assignment:** Financial returns assume randomized or propensity-adjusted treatment/control split. Non-randomized interventions can introduce selection bias.
3. **Drift Monitoring:** Population Stability Index (PSI) maximum was `0.0000` (Alert threshold: `0.25`).

---

## 🚀 Recommended Next Actions

1. **Model Governance:** Initiate candidate model retraining and trigger offline comparison suite against incumbent.
2. **Campaign Operations:** Review contact capacity allocations to align targeted lists with high-precision deciles.
3. **Data Quality:** Maintain data contract matching rules to ensure 100% reconciliation on customer keys.

---
*Report generated automatically by `churn_prediction.post_deployment` reporting pipeline.*
