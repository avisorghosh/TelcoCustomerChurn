# Deployment & Environment Documentation

This document specifies the environment configuration, system requirements, dependency management, configuration layout, MLflow setup, and container execution for deploying the Customer Churn Prediction system.

---

## 📋 Prerequisites

| Component | Minimum Version | Recommended / Tested | Purpose |
| --- | --- | --- | --- |
| **Operating System** | macOS 12+ / Linux (x86_64 or arm64) | Ubuntu 22.04 LTS / macOS Sonoma | Host runtime environment |
| **Python** | `3.11.0` | `3.11.15` | Core programming runtime |
| **uv** | `0.1.0+` | `0.4.0+` | Fast, deterministic dependency management |
| **Docker** | `20.10+` | `24.0+` | Containerized API runtime (optional for local non-container run) |
| **Git** | `2.30+` | `2.40+` | Source control and Git revision metadata logging |

---

## 🐍 Python & Dependency Installation

The project strictly uses [uv](https://docs.astral.sh/uv/) and the locked project file `uv.lock` to guarantee 100% deterministic dependencies.

### 1. Clone Repository & Install Environment

```bash
# Clone the repository
git clone https://github.com/avisorghosh/TelcoCustomerChurn.git
cd TelcoCustomerChurn

# Create locked virtual environment (.venv)
uv sync --locked
```

### 2. Verification

```bash
# Verify Python version inside virtual environment
uv run python --version  # Output: Python 3.11.x

# Run code formatters and linter checks
uv run ruff check .
uv run ruff format --check .

# Execute unit and integration tests
uv run pytest
```

---

## ⚙️ Environment Variables & Configuration

System behavior can be customized using environment variables or versioned YAML files located in `configs/`.

### Key Environment Variables

| Variable | Default Value | Description |
| --- | --- | --- |
| `CHURN_API_HOST` / `HOST` | `127.0.0.1` | Network binding host interface for FastAPI server |
| `CHURN_API_PORT` / `PORT` | `8000` | Port exposed by FastAPI service |
| `CHURN_MODEL_DIR` | `models` | Directory path containing active model artifacts |
| `CHURN_PIPELINE_FILENAME` | `baseline_pipeline.joblib` | Model pipeline filename |
| `CHURN_METADATA_FILENAME` | `baseline_metadata.json` | Model metadata filename |
| `CHURN_DECISION_THRESHOLD` | `0.50` | Default classification threshold for single-record predictions |
| `CHURN_MODEL_VERSION` | Metadata version | Override version string returned in API responses |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow tracking store location |
| `MLFLOW_EXPERIMENT_NAME` | `telco_customer_churn` | MLflow experiment identifier |
| `MLFLOW_REGISTERED_MODEL_NAME` | `telco_churn_model` | Registered model name in local MLflow registry |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Versioned Configuration Files (`configs/`)

- `configs/data_contract.yaml`: Schema rules, feature column types, allowed categorical domains, and numeric bounds.
- `configs/training.yaml`: Stratified split ratios, random seed (42), model hyperparameters, and MLflow settings.
- `configs/evaluation.yaml`: Evaluation thresholds, PR-AUC targets, and top 10% campaign capacity allocation.
- `configs/serving.yaml`: FastAPI serving parameters, model file paths, decision threshold, and monitoring metrics settings.

---

## 📦 Model Location & Artifact Persistence

Active production model artifacts are stored in the local `models/` directory (ignored by Git):

- `models/baseline_pipeline.joblib` / `candidate_pipeline.joblib`: trained candidate artifacts
- `models/serving_pipeline.joblib`: promoted serving pipeline (created by `scripts/promote_selected_model.py`)
- `models/serving_metadata.json`: serving metadata companion file

Clean checkout workflow: train → compare → promote → score/serve/docker. See README Quick Start.

---

## 🧪 MLflow Setup & Tracking Server

MLflow is integrated natively into model training and comparison workflows.

- **Local Storage**: Run metrics and artifacts are saved to `./mlruns/`.
- **Registry**: Model versions are registered under `telco_churn_model`.
- **Launch Local MLflow Server**:

```bash
uv run mlflow ui --host 127.0.0.1 --port 5000
```

Access the UI at `http://127.0.0.1:5000`.

---

## 🐳 Docker Container Deployment

The service includes a production multi-stage Dockerfile based on `python:3.11-slim` running under non-root security principles (`appuser` UID 10001).

### Build Container Image

Train and promote serving artifacts first, then build:

```bash
uv run python scripts/promote_selected_model.py
docker build -t telco-churn-api:latest .
```

Or mount host artifacts at runtime:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" telco-churn-api:latest
```

### Run Container Instance

```bash
docker run -d \
  --name churn-api \
  -p 8000:8000 \
  -e CHURN_DECISION_THRESHOLD=0.50 \
  telco-churn-api:latest
```

### Verify Container Endpoint

```bash
curl http://localhost:8000/health
```

---

*For release procedures and step-by-step deployment execution, see [RELEASE.md](file:///Users/avisorghosh/Documents/Projects/TelcoCustomerChurn/RELEASE.md).*
