# Infrastructure notes

The production API container definition lives at the repository root:

```text
../Dockerfile
```

Build from the repository root after training and promoting serving artifacts:

```bash
uv run python scripts/train_baseline.py --no-mlflow
uv run python scripts/train_candidate.py --no-mlflow
uv run python scripts/run_comparison.py
uv run python scripts/promote_selected_model.py
docker build -t telco-churn-api:latest .
```
