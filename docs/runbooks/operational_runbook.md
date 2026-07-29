# Operational Runbook: Customer Churn Prediction System

This operational runbook provides step-by-step guidance for service operations, health verification, common failure triage, data validation issues, model loading failures, and data drift investigation.

---

## 1. Service Startup & Verification

### Launch REST API Service
```bash
uv run python scripts/run_api.py --host 127.0.0.1 --port 8000
```

### Health & Readiness Check
- **Health Probe**:
  ```bash
  curl http://localhost:8000/health
  ```
  Expected: `{"status": "ok", "model_loaded": true}`

- **Readiness Probe**:
  ```bash
  curl http://localhost:8000/ready
  ```
  Expected HTTP `200 OK`: `{"status": "ready", "model_loaded": true}`

- **Metrics Endpoint**:
  ```bash
  curl http://localhost:8000/metrics
  ```
  Expected HTTP `200 OK` with Prometheus format text.

---

## 2. Common Failures & Troubleshooting

### Scenario A: Model Loading Failure (API Returns 503 Service Unavailable)
- **Symptom**: `GET /ready` returns HTTP `503` with `"status": "not_ready"`, or predictions fail with `ModelNotLoadedError`.
- **Root Causes**:
  - Missing artifact in `models/` directory (`baseline_pipeline.joblib` or `baseline_metadata.json`).
  - Corrupted model artifact or incompatible Python/scikit-learn version.
- **Remediation Steps**:
  1. Inspect API structured JSON logs for event `model_loading_failed`.
  2. Verify files exist in `models/`:
     ```bash
     ls -la models/
     ```
  3. If missing, retrain baseline model artifact:
     ```bash
     uv run python scripts/train_baseline.py
     ```
  4. Restart API service and re-query `GET /ready`.

---

### Scenario B: Batch Data Validation Failures (Quarantined Batch)
- **Symptom**: `scripts/run_batch_scoring.py` fails with `BatchValidationError`. No output file generated in `reports/scoring/`.
- **Root Causes**:
  - Duplicate primary keys (`customerID`) or duplicate rows in input CSV.
  - Invalid categorical values (e.g. `Contract` = `Three year`).
  - Non-numeric text values in `TotalCharges`.
- **Remediation Steps**:
  1. Inspect generated quarantine report under `reports/quarantine/<batch_id>_quarantine.json`.
  2. Inspect operational quality report under `reports/quality/operational_quality_report.json`.
  3. Identify specific schema errors in `errors` list (e.g., column name, failed check type, sample invalid values).
  4. Do NOT bypass validation or alter validation contract without business approval. Notify upstream CRM/data engineering team to re-issue clean batch.

---

### Scenario C: Data Drift Alert (`significant_drift`)
- **Symptom**: Data drift report (`reports/drift/drift_report.json`) exhibits `overall_drift_status: "significant_drift"` or max PSI >= 0.25.
- **Root Causes**:
  - Upstream change in customer acquisition mix or product plans (e.g. shift from month-to-month to annual contracts).
  - Data ingestion format change or feature unit discrepancy.
- **Remediation Steps**:
  1. Execute drift analysis script:
     ```bash
     uv run python scripts/run_drift_report.py --scoring-path reports/scoring/batch_predictions.csv --reference-path Telco-Customer-Churn.csv
     ```
  2. Review `feature_reports` array in `reports/drift/drift_report.json` to pinpoint which specific numerical or categorical features drifted.
  3. Verify if drift is an operational data issue (e.g. missing column values or unit change) or genuine population shift.
  4. Do NOT execute automated retraining. Record investigation findings, notify Data Science lead, and schedule a candidate model re-evaluation against recent labeled data.

---

### Scenario D: High Prediction Error Rate / Unhandled 500 Errors
- **Symptom**: Prometheus metric `telco_churn_prediction_failures_total` spiking or logs show `major_failure` event.
- **Remediation Steps**:
  1. Check structured JSON logs for `event: "major_failure"` or `event: "prediction_failed"`.
  2. Verify correlation ID associated with failed requests.
  3. Confirm input request payload matches schema contract (missing required fields return `422` validation error, unexpected data structures return `500`).
