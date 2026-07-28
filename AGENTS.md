# Repository Guidelines

## Project Structure & Module Organization

The repository foundation is complete and the next milestone is the data contract. Do not modify root dataset `Telco-Customer-Churn.csv` in place. The intended layout is documented in `SYSTEM_DESIGN.md`:

- `src/churn_prediction/` contains reusable application, data, feature, model, API, and monitoring code.
- `tests/unit/`, `tests/integration/`, and `tests/contract/` mirror production behavior at increasing scope.
- `configs/` holds versioned training and serving settings.
- `scripts/` contains thin operator entry points; business logic belongs in `src/`.
- `notebooks/` is for exploration only. Promote validated logic into `src/`.
- `data/` and `models/` are local staging/artifact directories and should remain ignored by Git.

## Build, Test, and Development Commands

Use Python 3.11, `uv`, and the committed project lockfile:

```bash
uv sync --locked        # create the locked development environment
uv run ruff check .     # lint source and tests
uv run ruff format --check .  # check formatting
uv run pytest           # run the full test suite
```

For focused tests: `uv run pytest tests/unit -q`. Expose repeatable workflows through `scripts/` or documented package commands.

## Coding Style & Naming Conventions

Target Python 3.11+ with four-space indentation, type hints on public functions, and small modules. Use `snake_case` for modules, functions, variables, and fixtures; `PascalCase` for classes; and `UPPER_CASE` for constants. Prefer explicit data contracts and scikit-learn pipelines over implicit DataFrame mutation. Ruff is the formatter/linter.

Name configuration files by concern (`configs/training.yaml`), tests by behavior (`test_rejects_blank_customer_id.py`), and fixtures by scenario (`invalid_total_charges.csv`).

## Testing Guidelines

Use Pytest. Every production change needs an appropriate test: unit tests for transformations, contract tests for input schemas, and integration tests for train-to-score behavior. Tests must use fixed seeds, avoid network access, and use small fixtures. Validate failure paths; invalid scoring data must never produce partial results.

## Commit & Pull Request Guidelines

No Git history is available yet, so use Conventional Commit-style subjects: `feat: add batch scoring contract` or `fix: reject duplicate customer IDs`. Keep commits focused. Pull requests should state the problem, design choice, tests run, configuration/data-contract changes, and model/metric impact. Link issues; include screenshots only for user-facing changes.

## Data, Security, and ML Safety

Never commit secrets, raw customer data, generated artifacts, or MLflow credentials. Keep `customerID` out of model features and logs. Record data checksum, config, seed, code revision, metrics, and model version. Consult `SYSTEM_DESIGN.md` before changing validation, features, release, or monitoring.

## Development Workflow

Before starting any implementation:

1. Read SYSTEM_DESIGN.md
2. Read TASKS.md
3. Implement only one milestone
4. Never skip acceptance tests
5. Never modify architecture without explaining why

When a milestone is completed:

- Update TASKS.md to mark it complete.
- Update README.md if repository usage changes.
- Explain significant design decisions before modifying architecture.
