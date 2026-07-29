# Dashboard Templates & Observability Guide

This directory contains dashboard templates for monitoring the Telco Customer Churn Prediction System.

## 📊 Overview

The system exposes Prometheus metrics at the `/metrics` HTTP endpoint when running the API service (`churn_prediction.api.app`).

### Key Metrics Monitored

1. **API Throughput & Latency**:
   - `telco_churn_api_requests_total`: Counter tracking HTTP requests by endpoint, method, and status code.
   - `telco_churn_api_request_duration_seconds`: Histogram tracking request latency distribution.
2. **Inference Volume & Failures**:
   - `telco_churn_predictions_total`: Count of served churn predictions by predicted risk class (0/1) and model version.
   - `telco_churn_prediction_failures_total`: Count of failed inference requests by failure reason.
3. **Batch Scoring Health**:
   - `telco_churn_batch_scoring_records_total`: Accepted vs. rejected records in batch jobs.
   - `telco_churn_batch_scoring_validation_failures_total`: Count of contract validation checks failed.
4. **Model Lifecycle**:
   - `telco_churn_model_loads_total`: Count of successful and failed model artifact load attempts.
   - `telco_churn_model_load_failures_total`: Count of model load errors.

---

## 🚀 Grafana Dashboard Setup

To import the dashboard template into your local Grafana instance:

1. Open Grafana UI (typically `http://localhost:3000`).
2. Navigate to **Dashboards** -> **Import**.
3. Upload `grafana_dashboard.json` or copy-paste its JSON contents.
4. Select your Prometheus data source (`${DS_PROMETHEUS}`).
5. Click **Import**.
