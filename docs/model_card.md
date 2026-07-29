# Model Card: Telco Customer Churn Prediction

**Model family:** Regularized logistic regression (baseline) and scikit-learn `GradientBoostingClassifier` (candidate)  
**Primary serving artifact:** `models/serving_pipeline.joblib` (produced by promotion after comparison)  
**Schema / contract version:** `1.0.0`  
**Last updated:** 2026-07-29

---

## 1. Intended Use

This model estimates **next-billing-cycle / weekly churn risk** for IBM Telco-style customer snapshots. Intended consumers:

- Retention / CRM teams prioritizing limited outreach capacity (default top 10%)
- Internal tools that need a probability, decision flag, model version, and correlation ID
- Batch worklist generation for human-owned campaign policy

Outputs are **decision-support scores**, not automated adverse actions.

## 2. Out-of-Scope Use

Do **not** use this model for:

- Punitive pricing, service denial, or fully automated adverse decisions
- Claims of causal “why the customer will churn”
- Temporally validated production forecasting (the public dataset has no event timestamps)
- Credit, employment, insurance, or other regulated decisions without a separate legal review
- Real-time transactional authorization paths that require hard SLAs without load testing

## 3. Training Data

| Item | Detail |
| --- | --- |
| Source | Public IBM Telco Customer Churn CSV (`Telco-Customer-Churn.csv`) |
| Rows / columns | 7,043 customers × 21 fields |
| Target | `Churn` (`Yes` / `No`), ~26.5% positive class |
| Provenance | Public learning dataset for demonstration / portfolio use |
| Validation | Pandera data contract + custom domain checks before training/scoring |

Sensitive attributes `gender` and `SeniorCitizen` are **excluded from predictors** and retained only for post-hoc fairness review where permitted.

## 4. Assumptions

- Weekly scoring cadence and a **proxy** next-billing-cycle risk horizon
- Upstream snapshots provide the same schema at score time
- Campaign capacity and contact policy are supplied by the business (default capacity fraction `0.10`)
- Labels eventually arrive for delayed evaluation; treatment/control metadata is required for causal campaign ROI claims
- Local Docker / local MLflow are sufficient for this portfolio deployment scope

## 5. Limitations

- No customer event dates → **no out-of-time validation**
- Static public data; not a live CRM feed
- No real intervention outcomes in the source file (post-deployment reports use simulated delayed labels)
- Probability calibration is **evaluated** (Brier / reliability curve); isotonic/sigmoid recalibration is not fitted into the shipped pipeline
- Local reason codes / explanations are not yet returned by the API
- Authentication, TLS, and rate limiting are deployment requirements, not provided by the local demo

## 6. Evaluation Metrics

Primary offline metric: **PR-AUC**. Supporting metrics: ROC-AUC, Brier score, precision/recall at configurable campaign capacity (default top 10%), and threshold analysis.

Candidate selection uses **validation-split gates**. Final reported metrics use the untouched **test** split. See:

- [`reports/evaluation/evaluation_metrics.json`](../reports/evaluation/evaluation_metrics.json)
- [`reports/evaluation/decision_record.md`](../reports/evaluation/decision_record.md)

On the current fixed-seed run, the GradientBoosting candidate failed the validation PR-AUC improvement gate (`+0.0071` < `+0.0100`), so **baseline logistic regression** remains the selected/promoted serving model. Holdout test metrics for the baseline:

| Metric | Value |
| --- | --- |
| PR-AUC | 0.6437 |
| ROC-AUC | 0.8487 |
| Brier score | 0.1361 |
| Precision @ 10% capacity | 75.47% |
| Recall @ 10% capacity | 28.57% |

## 7. Fairness Considerations

- Predictors exclude `gender` and `SeniorCitizen` by default
- Post-hoc fairness review reports subgroup metrics and demographic parity differences
- Fairness gates during candidate comparison constrain degradation relative to the baseline
- Disparate impact findings require business/legal review before production use

## 8. Operational Considerations

- **Batch scoring** is the primary path; FastAPI `/predict` is secondary
- Invalid batches are quarantined; no partial score files are written
- Structured logs redact `customerID`, raw payloads, and other sensitive fields
- Prometheus metrics expose API/batch/model-load failure signals
- Serving uses promoted artifacts (`serving_pipeline.joblib` / `serving_metadata.json`)

## 9. Retraining Guidance

- Retrain only when new labelled snapshots exist and release gates pass
- Drift / delayed-label degradation is an **investigation** signal, not an automatic retrain trigger
- Promote candidates with `scripts/promote_selected_model.py` after a validation-gated decision record
- Retain the prior serving artifact set for rollback (`ROLLBACK.md`)

Automated retraining is intentionally disabled by governance policy.
