# Delivery Plan

Each milestone has a narrow acceptance test. Do not begin a later milestone until the preceding one passes review. The first release prioritizes a tested vertical slice over advanced platform tooling.

| Milestone | Scope | Independently testable definition of done |
| --- | --- | --- |
| 0. Record portfolio assumptions | Record the proxy horizon, weekly scoring cadence, configurable top-10% capacity default, sensitive-feature policy, public-data provenance, and local deployment scope. | Assumptions are documented in README/system design; no unsupported production claims remain. |
| 1. Repository foundation [Completed] | Initialize Git; create package layout, `pyproject`, locked dependencies, `.gitignore`, licence, Ruff/Pytest configuration, and enforced CI for lint, format, tests, and coverage. | Clean checkout installs; lint, formatting, smoke test, and CI run successfully. |
| 2. Data contract [Completed] | Implement source manifest and Pandera/custom validation for schema, types, domains, IDs, duplicates, and `TotalCharges` blanks. | Valid local CSV produces a quality report; deliberately malformed fixtures fail with clear diagnostics. |
| 3. Exploratory analysis [Completed] | Produce a concise, reproducible EDA report/notebook; document distributions, target balance, missingness, leakage review, and split rationale. | Report reruns from source data and contains no reusable production logic. |
| 4. Baseline pipeline [Completed] | Build train-only preprocessing and regularized logistic-regression pipeline with stratified split and persisted artifact. | Unit/integration tests show no identifier feature, inference accepts a valid record, and the saved pipeline reloads identically. |
| 5. Evaluation contract [Completed] | Add PR-AUC, ROC-AUC, calibration, recall/precision at capacity, threshold/value analysis, and report artifacts. | Fixed-seed run emits metrics/plots; test confirms threshold is policy-configured rather than hard-coded. |
| 6. Batch scoring [Completed] | Build idempotent validated batch scoring, versioned output schema, row reconciliation, and failure quarantine/reporting. | Sample batch produces one score per valid unique customer; invalid batch yields no partial output. |
| 7. API service [Completed] | Implement a local FastAPI health/readiness and single-customer score endpoint around the designated artifact. | Contract tests cover valid prediction, invalid schema (4xx), unavailable model (safe 5xx), and model-version response. |
| 8. Container and portfolio polish [Completed] | Add Docker image, reproducible command entry points, architecture diagram, quick start, sample response, measured results, and model card. | Image serves the tested API; a clean checkout can reproduce the documented demo. |
| 9. Experiment tracking [Completed] | Integrate MLflow tracking, data manifest, config, Git revision, artifact logging, model signature, and local registry workflow. | A run is visible with complete lineage; registered artifact can be loaded and scored. |
| 10. Candidate comparison [Completed] | Compare baseline with a bounded boosted-tree candidate; conduct calibration and segment/fairness evaluation. | Decision record names the selected model using predeclared gates; no test-set-driven tuning. |
| 11. Observability [Completed] | Add structured logs, Prometheus metrics, batch quality/drift reports, dashboard/runbook templates, and alert policy. | Synthetic bad input/drift triggers expected metrics/report; logs contain no raw customer payload. |
| 12. Release rehearsal [Completed] | Write API deployment notes, rollback procedure, and execute an end-to-end dry run. | New environment validates, trains, batch-scores, serves, and restores the prior model artifact using only documented steps. |
| 13. Post-deployment learning [Completed] | Integrate delayed labels and campaign treatment/control outcomes; evaluate live quality and value. | Scheduled report compares incumbent performance/business outcomes; retraining decision is documented. |

## First-release defaults

The source data cannot answer several real-world product questions. Start Milestone 1 using the documented portfolio defaults: weekly proxy scoring, configurable top-10% capacity, churn-risk-only output, local Docker deployment, and `gender`/`SeniorCitizen` excluded from predictors. Replace these defaults only when a real business/data owner provides the necessary context.
