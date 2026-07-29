# System Design: Customer Churn Prediction

## 1. Dataset understanding

### Dataset contract

One row represents a Telco customer snapshot. `Churn` is the binary outcome: `Yes` means the customer left; `No` means they did not. For a deployed system, the exact prediction horizon and snapshot timestamp must be explicit. This static dataset does not include either, so this portfolio project uses a clearly labelled weekly-scoring / next-billing-cycle churn-risk proxy. It must not be represented as temporally validated production forecasting.

| Feature | Type | Business meaning / treatment |
| --- | --- | --- |
| `customerID` | identifier | Customer key. Retain only for joins, audit, and scored-output routing; exclude from model inputs. |
| `gender` | categorical | Self-reported gender. Its use should be reviewed for fairness and business necessity; it is unlikely to be needed for a first production model. |
| `SeniorCitizen` | binary categorical | Indicates senior status (encoded 0/1). Treat as a category, not a continuous quantity. |
| `Partner`, `Dependents` | binary categorical | Household context, which can correlate with stability and product needs. |
| `tenure` | numeric integer | Months since becoming a customer; a key lifecycle and loyalty signal. |
| `PhoneService`, `MultipleLines` | categorical | Telephone subscription and number of lines; `No phone service` is a meaningful state, not missing data. |
| `InternetService` | categorical | Internet product tier (`DSL`, `Fiber optic`, `No`); product availability/use profile. |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` | categorical | Attached service adoption; values such as `No internet service` are structural states. |
| `StreamingTV`, `StreamingMovies` | categorical | Entertainment add-ons; indicators of product bundle/use pattern. |
| `Contract` | categorical ordinal | Commitment term (`Month-to-month`, `One year`, `Two year`), a strong retention/business-policy factor. |
| `PaperlessBilling` | binary categorical | Billing preference, potentially correlated with channel and payment behavior. |
| `PaymentMethod` | categorical | Payment channel; automatic payment versus manual payment can indicate payment friction. |
| `MonthlyCharges` | numeric float | Current recurring monthly charge; proxy for plan value/cost. |
| `TotalCharges` | numeric float after parsing | Lifetime billed amount; combines tenure and charges, so validate potential redundancy. |
| `Churn` | binary target | Historical customer departure outcome. |

### Data quality and modelling concerns

- The local file has 7,043 rows (plus header), 21 columns, and 11 whitespace/blank `TotalCharges` values. It is text-formatted and must be parsed deliberately; never silently coerce invalid values to zero.
- `TotalCharges` is logically expected to be approximately related to `tenure × MonthlyCharges`; the relationship should be checked for invalid or surprising values, but it is not exact because of prorations, tax, credits, and price changes.
- `No internet service` and `No phone service` are valid categories. Collapsing them into missing values destroys information.
- The static dataset lacks event dates, scoring timestamps, churn dates, source-system lineage, intervention labels, and explicit prediction horizon. It cannot prove temporal generalization or treatment impact.
- The target is imbalanced (26.5% `Yes` / 73.5% `No`). Accuracy is therefore inadequate as a primary selection metric.
- `gender` and `SeniorCitizen` are sensitive/fairness-relevant attributes. Exclude them from the initial candidate model unless a documented business/legal review supports inclusion; retain only for carefully governed fairness evaluation where permitted.
- *Milestone 3 Empirical EDA Validation:* The exploratory data analysis empirically confirmed all of the above assumptions. Specifically: all 11 blank `TotalCharges` records belong to 0-tenure customers; target churn rate is 26.54% (imbalance ratio 2.77:1); `gender` shows negligible churn rate disparity (<0.8%); `TotalCharges` correlates strongly with `tenure` (r=0.83) with an average $45.09 absolute discrepancy from `tenure × MonthlyCharges`.

## 2. Business problem

### Stakeholders and users

| Group | Need |
| --- | --- |
| Retention / CRM leadership | Allocate limited campaign budget and assess realized retention value. |
| Customer-success or call-center agents | A prioritized, explainable customer worklist and approved next action. |
| Marketing operations | Receive batch scores, join them to campaign tooling, and enforce contact rules. |
| Data science / ML engineering | Train, validate, release, monitor, and roll back models. |
| Risk, legal, privacy, and compliance | Review data use, fairness, consent, and adverse customer-impact controls. |

### Supported decision

At a regular cadence, select customers above a policy-defined risk/value threshold for a retention action. The system produces risk scores and reason codes; human-owned campaign policy decides contact eligibility and offer. Do not use this score for punitive pricing, denial of service, or fully automated adverse decisions.

### Business metrics

Primary: incremental retained contribution margin per campaign dollar, measured against a randomized holdout when campaigns exist. Supporting operational metrics: eligible customers reached, intervention capacity used, acceptance rate, opt-out/complaint rate, and time-to-action.

Offline model metrics: PR-AUC (ranking quality for the churn class), ROC-AUC (secondary comparison), recall/precision at campaign capacity (initial portfolio default: top 10%, configurable), calibration/Brier score, and expected-value-at-threshold when real intervention economics exist. Report confidence intervals where feasible. Accuracy and a fixed 0.5 threshold are explicitly not release criteria.

## 3. Requirements

### Functional requirements

1. Ingest a versioned CSV source and validate schema, domain values, uniqueness, and parseability.
2. Train reproducible preprocessing and classification pipelines from configuration.
3. Produce probability, model version, scoring timestamp, and approved explanation/reason fields for each customer.
4. Evaluate candidate models against a locked validation/test protocol and business capacity.
5. Track parameters, code/data versions, metrics, artifacts, and model signatures; promote only approved versions.
6. Run scheduled batch scoring and provide a schema-validated single-record API for controlled integrations.
7. Log requests and operational events safely; monitor data quality, drift, performance once labels arrive, latency, and failures.

### Non-functional requirements

- Reproducibility: a run can be recreated from Git revision, data version/checksum, config, random seed, and environment lockfile.
- Reliability: batch jobs are idempotent and write immutable, versioned outputs; API has health/readiness endpoints and bounded timeouts.
- Security/privacy: least-privilege access, secrets outside Git, encrypted storage/transport in a real deployment, PII-minimized logs, retention policy.
- Maintainability: typed interfaces, linting, tests, configuration separation, and documented runbooks.
- Performance: for this data size, batch completion within minutes and p95 API inference below 200 ms are reasonable targets; validate against real load before committing an SLA.
- Explainability: provide global feature diagnostics and carefully worded local reason codes; avoid claiming causality.

### Assumptions, constraints, and risks

| Area | Design position |
| --- | --- |
| Assumptions | A trusted upstream snapshot can provide the same fields at score time; retention capacity and contact policy are supplied by the business; labels eventually arrive. |
| Constraints | Public static data, no timestamps, small sample, no intervention outcomes, and no existing cloud/platform requirement. |
| Leakage risk | Fields must be available before churn; future payment/status changes cannot enter features. In a future temporal dataset, split by time rather than randomly. |
| Drift risk | Product pricing, plans, channels, and customer mix can change; monitor distributions and score rates. |
| Feedback-loop risk | Targeted offers change observed churn; preserve treatment/control assignments and campaign metadata. |
| Fairness/privacy risk | Sensitive attributes and proxies can create disparate impact. Conduct segmented evaluation and governance review before use. |
| Model risk | High risk is not a cause of churn nor evidence that an offer will work. Communicate uncertainty and use controlled experiments. |

## 4. Proposed ML system

### High-level architecture

```text
Source CRM/Billing snapshot
        │
        ▼
Ingestion + data version/checksum ──► schema/domain validation ──► versioned clean snapshot
                                                                  │
                         ┌────────────────────────────────────────┘
                         ▼
Config-driven training pipeline ──► run manifest + artifacts ──► designated model artifact
                         │                                            │
                         ▼                                            ▼
          evaluation + calibration + release gate            batch scoring job
                                                                  │
                                                   scored worklist / CRM export
                                                                  │
                           FastAPI controlled scoring service ◄────┘
                                                                  │
                                         logs, health, drift & performance monitoring

Optional follow-on: MLflow experiment tracking and local model registry
```

### Data ingestion and validation

Use a versioned input manifest containing source location, received time, row count, SHA-256 checksum, schema version, and quality-report path. Validate before any training or scoring: expected columns/order, `customerID` uniqueness/non-null, allowed categorical values, binary values, numeric ranges (`tenure >= 0`, charges non-negative), `TotalCharges` conversion, target domain during training, and duplicate-row policy. Quarantine a failing batch and emit an actionable report; never score partially validated data.

For the local portfolio dataset, use Pandera schema validation plus lightweight custom rules. In a larger data platform, this contract can migrate to Great Expectations or a warehouse-native quality framework without changing pipeline boundaries.

### Feature engineering and training

Use a single scikit-learn `Pipeline`/`ColumnTransformer` artifact so transformations fitted during training are applied identically at inference. Planned processing: explicit `TotalCharges` numeric conversion; imputation with indicator where justified; one-hot encoding for categories with unknown-category handling; robust scaling only for models that need it; optional domain-derived ratios only when available at scoring time and validated against leakage. Drop `customerID`; start without `gender` and `SeniorCitizen` as predictors.

Establish a regularized logistic-regression baseline, then compare gradient boosting (LightGBM or XGBoost) under the same split and metrics. Do not introduce a deep-learning model: the dataset is small, tabular, and a neural network adds operational cost without a defensible benefit. Use stratified train/validation/test splits now, but replace with an out-of-time split when real dated data exists. Fit preprocessing and calibration only on training partitions. Tune modestly, retain a final untouched test set, and use probability calibration (isotonic/sigmoid selected by validation evidence).

### Evaluation, tracking, and registry

For the first portfolio release, persist a run manifest containing the dataset checksum, Git SHA, environment, parameters, metrics, calibration plots, feature diagnostics, and serialized pipeline. This makes the baseline reproducible without requiring a tracking server. MLflow tracking and a local registry are a follow-on enhancement; when introduced, a model is registered only when it passes data validation, baseline comparison, calibration, capacity-based metric thresholds, fairness review, and reproducibility checks. No model is promoted automatically based on one metric.

### Serving: batch first, API second

**Recommendation: scheduled batch inference is the primary production path.** Churn interventions are generally planned in CRM campaigns, not decided in a sub-second customer transaction. Daily or weekly scoring is cheaper, simpler, auditable, and lets the business apply eligibility/capacity rules consistently. Output a versioned worklist to a controlled storage location/CRM import.

FastAPI offers a secondary single-customer endpoint for internal tools and integration tests. It loads the designated local model artifact; validates request schema; returns a probability, risk band, model version, and correlation ID. The API does not train models, query raw data, select offers, or expose sensitive fields. Authentication, authorization, rate limiting, request-size limits, and TLS are deployment requirements; document them clearly rather than implying the local demo provides them.

### Monitoring, retraining, logging, error handling

- **Data/feature monitoring:** schema failures, missingness, category novelty, numeric distribution drift (PSI/KS), score distribution, and batch row reconciliation.
- **Model/business monitoring:** delayed-label PR-AUC/calibration/recall-at-capacity, intervention/control outcomes, campaign ROI, and segment/fairness checks. Drift is an investigation signal, not an automatic retrain trigger.
- **Retraining:** schedule quarterly only if new labelled snapshots are available, plus investigate threshold breaches or material product changes. Retraining creates a candidate that must pass the same release gates; retain the incumbent for rollback.
- **Logging:** structured JSON logs with timestamp, level, event, run/model version, correlation ID, duration, and safe error code. Do not log raw customer IDs, request payloads, tokens, or sensitive values; use hashed/pseudonymous references where needed.
- **Errors:** fail fast on invalid training/scoring batches, quarantine input, preserve diagnostics, retry only transient I/O failures with bounded exponential backoff, return stable 4xx validation errors and non-sensitive 5xx responses, alert on repeated failures.

## 5. Technology stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language/runtime | Python 3.11+ | Mature ML ecosystem, typing, and deployment support. |
| Tabular data | Pandas initially | Dataset is small; readable and ubiquitous. Consider Polars only when scale/profiling justifies it. |
| Validation | Pandera | DataFrame-native, versionable schema/data contracts with testable custom checks. |
| ML | scikit-learn | Pipelines prevent train/serve skew; strong baselines and metrics. |
| Candidate booster | LightGBM (optional) | Efficient tabular learner; keep optional because fewer dependencies and a transparent baseline are preferable initially. |
| Experiment/registry | Run manifest first; MLflow later | A file-based manifest proves baseline reproducibility with less operational overhead; MLflow is a strong follow-on for experiment comparison and registry lifecycle. |
| Serving | FastAPI + Pydantic | Typed request contracts, automatic OpenAPI, lightweight operational API. |
| Packaging | Docker | Repeatable runtime across local/CI/deployment environments. |
| Quality | Pytest, Ruff, mypy (incrementally) | Fast tests, lint/format, and stronger interface safety. |
| Dependency management | uv with lockfile | Fast, deterministic installs; `pyproject.toml` standard metadata. |
| CI | GitHub Actions | Native portfolio integration for lint, tests, image build, and security checks. |
| Observability | Prometheus metrics + structured logs; Grafana in deployment | Vendor-neutral health/latency/error and drift dashboards. |
| Data/artifact versioning | DVC or object-store manifests (choose after deployment target) | Data reproducibility; a manifest/checksum is enough for the local first milestone. |

Avoid Kubernetes, Kafka, Spark, feature stores, and managed cloud services in the initial portfolio version. They would create operational surface area this dataset and batch use case do not justify. Add them only when concrete scale, streaming, or multi-team requirements appear.

## 6. Professional repository structure

```text
.
├── LICENSE
├── README.md
├── SYSTEM_DESIGN.md
├── TASKS.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .github/workflows/ci.yml
├── configs/
│   ├── training.yaml
│   └── serving.yaml
├── data/                         # ignored: raw/, interim/, processed/
├── docs/
│   ├── data_contract.md
│   ├── data_provenance.md
│   ├── model_card.md
│   └── runbooks/
├── infra/
│   ├── Dockerfile
│   └── monitoring/
├── notebooks/                    # EDA only; outputs reviewed, logic promoted to src/
├── scripts/                      # explicit train/score/validate commands
├── src/churn_prediction/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── settings.py
└── tests/
    ├── unit/
    ├── integration/
    └── contract/
```

The package owns reusable logic; scripts are thin entry points; notebooks may explore but never become a runtime dependency. Raw production data and generated models are ignored and referenced through manifests/artifact stores. If the public learning CSV remains in the repository, retain its licence/source notice and checksum; otherwise provide a documented acquisition step and keep it under ignored `data/raw/`.
