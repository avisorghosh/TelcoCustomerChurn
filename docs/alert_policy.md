# Operational Alert Policy: Customer Churn Prediction System

This document specifies the lightweight operational alert policy defining conditions that require operational or engineering investigation.

---

## 🚨 Alert Rules Summary

| Alert Name | Severity | Trigger Condition | Evaluation Window | Response Action |
| --- | --- | --- | --- | --- |
| **API_Unavailable** | P1 Critical | `GET /ready` returns 503 or model unloaded for > 2 mins | Instant | Check model artifact presence; reload or retrain model baseline. |
| **High_Prediction_Error_Rate** | P1 Critical | Prediction error rate > 5% (`telco_churn_prediction_failures_total` / `api_requests_total`) | 5 minutes | Inspect structured logs for `major_failure`; check payload schema compatibility. |
| **High_Validation_Failure_Rate** | P2 Warning | Batch validation rejection rate > 5% of records | Per batch job | Review `reports/quarantine/` report; alert upstream data provider. |
| **Significant_Data_Drift** | P2 Warning | Drift status = `significant_drift` (max feature PSI >= 0.25) | Daily / Per batch | Execute drift analysis script; evaluate feature PSI; notify Data Science lead. |
| **Model_Load_Failure** | P1 Critical | `telco_churn_model_load_failures_total` > 0 | Instant | Check disk permissions, path, and `.joblib` file integrity. |

---

## 📋 Detailed Alert Definitions

### 1. Alert: `API_Unavailable`
- **Severity**: **P1 Critical**
- **Description**: The REST inference service is unavailable or incapable of serving risk scores due to missing model artifacts or degraded status.
- **Notification Channel**: Slack `#ml-ops-alerts` / PagerDuty.
- **Action**: Follow Operational Runbook Section 2 Scenario A.

### 2. Alert: `High_Prediction_Error_Rate`
- **Severity**: **P1 Critical**
- **Description**: The ratio of prediction failures (`telco_churn_prediction_failures_total`) to overall requests exceeds 5% over a 5-minute window.
- **Notification Channel**: Slack `#ml-ops-alerts`.
- **Action**: Follow Operational Runbook Section 2 Scenario D.

### 3. Alert: `High_Validation_Failure_Rate`
- **Severity**: **P2 Warning**
- **Description**: Input batch scoring dataset has a record rejection rate exceeding 5% or fails contract validation completely.
- **Notification Channel**: Email / Slack `#data-quality-alerts`.
- **Action**: Follow Operational Runbook Section 2 Scenario B.

### 4. Alert: `Significant_Data_Drift`
- **Severity**: **P2 Warning**
- **Description**: Feature population distribution shift detected with max PSI >= 0.25 against training reference dataset.
- **Notification Channel**: Email / Slack `#ml-monitoring`.
- **Action**: Follow Operational Runbook Section 2 Scenario C. Do NOT execute automated retraining.

---

## 🔒 Privacy & Safety Constraints

Alert notifications must NEVER contain raw customer PII, customer identifiers (`customerID`), or sensitive feature payloads. All alert metadata must remain strictly operational (timestamps, error codes, metrics, batch IDs, model versions).
