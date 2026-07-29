# Customer Churn Prediction

A production-oriented machine learning system for identifying IBM Telco customers at risk of churn. This repository is intentionally designed as an ML product, not a notebook-only modelling exercise: data contracts, reproducible training, deployable scoring, observability, and safe model promotion are first-class concerns.

## Project status

**In development — Milestone 4 (Baseline Pipeline) is complete and the evaluation contract is next.** This repository is being built as a portfolio project: it will demonstrate a reproducible, tested churn-risk workflow without claiming production performance or causal business impact. See [TASKS.md](TASKS.md) for independently testable milestones and [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for the architecture and assumptions.

## Local development

This project targets Python 3.11 and uses [uv](https://docs.astral.sh/uv/) for a locked, reproducible environment. From a clean checkout:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run python scripts/run_eda.py         # Run reproducible EDA pipeline and export plots
uv run python scripts/train_baseline.py # Train baseline model and save pipeline artifacts
uv run python scripts/predict_baseline.py # Load trained pipeline and validate inference
```

To run only the unit tests:

```bash
uv run pytest tests/unit -q
```

The initial package is deliberately a smoke-test foundation. Data validation, modelling, APIs, and artifact generation are introduced in later milestones.

## Business goal

Rank currently active customers by their probability of churning within the agreed prediction horizon (initially: the next billing cycle). Retention teams can use the ranking to prioritize outreach and offers. The model supports decisions; it must not automatically change a customer's plan or eligibility.

The initial operating point should optimize expected retention value under a finite intervention budget, rather than optimize accuracy. A useful decision metric is:

`expected incremental value = P(churn) × P(save | intervention) × customer value − intervention cost`

Because this historical dataset contains outcomes but not intervention/campaign outcomes, the first release estimates churn risk only. Uplift or treatment-effect modelling is a future capability, not a claim the first model can make.

## Dataset

Source file: `Telco-Customer-Churn.csv` (IBM Telco Customer Churn dataset), 7,043 rows and 21 columns.

| Role | Columns |
| --- | --- |
| Identifier | `customerID` (never a predictive feature) |
| Target | `Churn` (`Yes` / `No`) |
| Numeric | `tenure`, `MonthlyCharges`, `TotalCharges` after type conversion |
| Categorical / ordinal-as-category | all remaining fields, including binary `SeniorCitizen` |

Observed data issue: `TotalCharges` is stored as text and has 11 blank values (likely very new customers with zero tenure). The target distribution is 1,869 churned customers (26.5%) and 5,174 retained customers (73.5%).

### Provenance and handling

The included CSV is a public learning dataset, retained only to make the portfolio project reproducible. Before publishing the repository, document its original source URL and licence in `docs/data_provenance.md`, record its SHA-256 checksum in a source manifest, and confirm redistribution is permitted. Do not add real customer data, credentials, or generated model artifacts to Git.

For a real deployment, raw source extracts belong in an access-controlled data store and are referenced by a versioned manifest rather than committed to the repository.

## Portfolio assumptions

The source dataset has no scoring timestamps, intervention history, or campaign economics. To make the first release concrete without overstating what the data proves, the project uses these explicit demonstration assumptions:

- Score a static customer snapshot weekly as a proxy for next-billing-cycle churn risk.
- Prioritize the highest-risk 10% of eligible customers; make this capacity configurable rather than hard-coded.
- Produce risk scores only. Campaign eligibility, offers, retention value, and intervention cost are outside the model and require real business data.
- Exclude `gender` and `SeniorCitizen` from model inputs; use them only for governed, optional segmented evaluation.

These are implementation defaults, not claims about an operating telecom business.

## Design principles

- Prevent leakage: use only information available at scoring time and split data before fitting transformations.
- Prefer a reproducible, inspectable baseline before complexity.
- Version data, code, features, parameters, metrics, and model artifacts together.
- Separate offline training from online serving.
- Treat probability calibration, thresholding, and business capacity as release requirements.
- Keep personally identifying or operationally sensitive data out of logs and model features unless justified.

## Proposed delivery

The first portfolio release will validate data, train a scikit-learn pipeline, evaluate calibrated churn-risk probabilities at configurable campaign capacity, produce a versioned batch-scoring worklist, expose a small FastAPI service, package it with Docker, and validate it in CI. Batch scoring is the recommended initial deployment; the API remains useful for controlled single-customer use and integration testing.

MLflow tracking, a registry workflow, boosted-model comparison, and monitoring are deliberate follow-on enhancements once the end-to-end baseline is complete and tested.

## Repository map (planned)

```text
src/churn_prediction/      Application and ML package
tests/                     Unit, integration, and contract tests
configs/                   Versioned training/serving configuration
data/                      Local, ignored data staging areas
notebooks/                 Exploratory work only; no production logic
models/                    Local, ignored artifacts for development
docs/                      Data contract, runbooks, decision records
scripts/                   Explicit operator entry points
infra/                     Container/deployment and monitoring assets
```

## Documentation

- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md): data dictionary, requirements, architecture, and stack rationale.
- [TASKS.md](TASKS.md): small, verifiable delivery milestones.
- `docs/data_provenance.md` (planned): dataset source, licence, checksum, and acquisition steps.
- `docs/model_card.md` (planned): final model metrics, limitations, and intended use.

## What will make this portfolio-ready

The finished repository will include a quick-start guide, reproducible commands, test and CI status, an architecture diagram, a sample scoring response, measured model results, and a transparent model card. Until those artifacts exist, this repository should be described as an in-progress ML engineering project rather than a deployed product.
