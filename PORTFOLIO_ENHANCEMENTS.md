# Portfolio Enhancement Roadmap — Phase 2

This document defines the Phase 2 enhancement plan for the Telco Customer Churn Prediction repository. Phase 1 (TASKS.md milestones 0–13) delivered a tested, end-to-end ML system. Phase 2 strengthens the project's Machine Learning depth, Data Science rigor, engineering quality, and interview readiness without altering the existing architecture.

Each milestone is independently implementable in a separate development session. Milestones are ordered by return on investment—highest interview and portfolio impact first. Do not begin a later milestone until the preceding one passes review unless the dependency column explicitly permits it.

---

## Milestone summary

| Milestone | Title | Estimated effort |
| --- | --- | --- |
| 14 | Repository hygiene and developer experience | 2 hr |
| 15 | Analytical EDA notebook | Half day |
| 16 | Domain-driven feature engineering | Half day |
| 17 | Hyperparameter optimization | Half day |
| 18 | Cross-validation and bootstrap confidence intervals | Half day |
| 19 | Probability calibration pipeline | 2 hr |
| 20 | Model explainability | Half day |
| 21 | Business decision analysis | 2 hr |
| 22 | CI pipeline hardening | 2 hr |
| 23 | Documentation and portfolio polish | 2 hr |
| 24 | Model card and evaluation report refresh | 2 hr |
| 25 | Technical debt reduction | Half day |

---

## Phase 2 milestones

### Milestone 14 — Repository hygiene and developer experience

**Objective**

Eliminate tracked generated artifacts, strengthen lint and type-checking rules, and add pre-commit automation so every subsequent milestone starts from a clean, well-guarded baseline.

**Rationale**

Tracked `mlruns/` directories, committed evaluation PNGs/JSONs, and narrow Ruff rules are low-cost fixes that immediately improve recruiter first-impressions and prevent regressions in later milestones. This milestone has the highest ROI because it de-risks every change that follows.

**Deliverables**

1. Remove `mlruns/` from Git tracking (`git rm -r --cached mlruns/`); confirm `.gitignore` covers it.
2. Add `reports/evaluation/`, `reports/scoring/`, `reports/quality/`, `reports/quarantine/`, `reports/drift/`, and `reports/post_deployment/` to `.gitignore` (keep `.gitkeep` markers). Remove previously committed generated artifacts from tracking.
3. Expand Ruff lint rule selection in `pyproject.toml` to at minimum `["E", "F", "I", "B", "S", "UP", "PT", "SIM", "RUF"]` with targeted ignores where necessary. Fix any new violations.
4. Move the top-level `import os` inside `InferenceService.model_version` property to the module-level import block in `src/churn_prediction/api/service.py`.
5. Remove the `build_baseline_pipeline` alias function from `src/churn_prediction/features/pipeline.py`. Update all call sites to use `build_model_pipeline` directly.
6. Add a `.pre-commit-config.yaml` with hooks for `ruff format`, `ruff check`, and conventional commit message validation. Document the `pre-commit install` step in `AGENTS.md` and `README.md`.
7. Add a `py.typed` marker file to `src/churn_prediction/`.
8. Pin the `uv` image in `Dockerfile` to a specific version tag instead of `ghcr.io/astral-sh/uv:latest`.

**Acceptance criteria**

- `git status` shows no tracked files under `mlruns/` or `reports/**/*.png|json|csv`.
- `uv run ruff check .` passes with the expanded rule set.
- `pre-commit run --all-files` passes on a clean checkout.
- All existing tests pass without modification beyond call-site renames.
- `docker build .` succeeds with the pinned `uv` image tag.

**Dependencies**

None. This milestone should be completed first.

**Estimated effort**

2 hours.

---

### Milestone 15 — Analytical EDA notebook

**Objective**

Replace the existing script-generated EDA with a narrative-driven Jupyter notebook that documents hypotheses, empirical findings, and the feature engineering ideas they motivate.

**Rationale**

The current `scripts/run_eda.py` produces canned visualizations but demonstrates no analytical thinking. Interviewers evaluate Data Scientists on how they reason about data, not whether they can call `sns.histplot`. A rich EDA notebook with written hypotheses, statistical observations, and explicit "this finding led me to engineer feature X" connections is the single highest-impact artifact for demonstrating analytical depth.

**Deliverables**

1. Create `notebooks/02_analytical_eda.ipynb` (retain the existing `01_exploratory_data_analysis.ipynb` for reference).
2. The notebook must include at minimum the following analytical sections, each with a written hypothesis before the analysis and a written conclusion after:
   - Target class distribution and imbalance ratio; implications for metric selection.
   - Univariate analysis of `tenure`, `MonthlyCharges`, `TotalCharges` with churn-rate overlays; identification of risk segments (e.g. short-tenure, high-monthly-charge customers).
   - `TotalCharges ≈ tenure × MonthlyCharges` residual analysis; quantify the discrepancy and discuss whether both features add independent signal or introduce multicollinearity (compute VIF or correlation diagnostics).
   - Categorical feature churn-rate comparison; identify which service add-ons (e.g. `OnlineSecurity`, `TechSupport`) have the strongest univariate association with churn.
   - Contract type × tenure interaction analysis; assess whether month-to-month customers churn at a materially different rate only during the first N months.
   - Service bundle analysis: count of adopted internet services per customer; test whether bundle depth correlates with retention.
   - Explicit leakage review: confirm no feature encodes post-churn information.
3. A concluding section titled "Feature Engineering Hypotheses" listing 3–5 domain features motivated by the analysis, with expected signal rationale and a note on scoring-time availability.
4. All plots must include titles, axis labels, and interpretive annotations or markdown cells.

**Acceptance criteria**

- Notebook runs end-to-end from the committed CSV without network access.
- Every analysis section contains a written hypothesis before the code and a written conclusion after.
- The feature engineering hypotheses section names at least 3 concrete features with rationale.
- No reusable production logic is implemented inside the notebook; validated ideas are promoted to `src/` in a subsequent milestone.

**Dependencies**

None (can run in parallel with Milestone 14 if needed, but ideally follows it).

**Estimated effort**

Half day.

---

### Milestone 16 — Domain-driven feature engineering

**Objective**

Add 3–5 domain-derived features to the preprocessing pipeline, motivated by EDA findings, and evaluate their marginal contribution to model performance.

**Rationale**

The current pipeline uses raw columns with `StandardScaler` + `OneHotEncoder` and zero custom features. This is the most common gap interviewers identify in ML portfolio projects. Even 2–3 well-reasoned domain features with documented rationale demonstrate understanding of the problem domain and feature engineering craft that separates DS candidates from tutorial-level work.

**Deliverables**

1. Create `src/churn_prediction/features/engineering.py` containing a scikit-learn-compatible custom transformer (a `BaseEstimator`/`TransformerMixin` subclass or `FunctionTransformer` wrapper) that computes domain features. Candidate features (refine based on EDA):
   - `avg_monthly_charge`: `TotalCharges / max(tenure, 1)` — average revenue per month; separates pricing changes from cumulative spend.
   - `tenure_charge_residual`: `TotalCharges - (tenure × MonthlyCharges)` — proxy for promotional credits, tax, or prorated charges.
   - `services_count`: count of internet add-on services adopted (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) — captures bundle depth; hypothesis: deeper bundles correlate with lower churn.
   - `has_premium_support`: binary flag for customers with both `OnlineSecurity == "Yes"` and `TechSupport == "Yes"` — hypothesis: premium-support customers are stickier.
   - `auto_payment`: binary flag for automatic payment methods (`Bank transfer (automatic)`, `Credit card (automatic)`) — hypothesis: manual payment friction increases churn risk.
2. Integrate the feature transformer into the existing `ColumnTransformer` / `Pipeline` workflow in `src/churn_prediction/features/pipeline.py` via a new `"engineered"` transformer step that runs before the existing numeric/categorical branches.
3. Add the new feature names to `configs/training.yaml` and `configs/candidate_training.yaml` under a new `features.engineered_features` key so they are config-driven, not hard-coded.
4. Add unit tests for the transformer: verify output shape, column names, edge cases (tenure=0, missing TotalCharges), and deterministic output.
5. Add a feature ablation section to the evaluation report: compare metrics with and without engineered features to quantify lift.

**Acceptance criteria**

- The engineered features are computed inside the fitted `Pipeline` artifact, not as a separate pre-processing step, to prevent train/serve skew.
- `uv run pytest` passes, including new unit tests for the transformer.
- All engineered features are available at scoring time (no leakage from post-churn data).
- Ablation comparison shows the measured metric delta (positive, zero, or negative) and is logged to MLflow.
- The data contract does not require changes because the input schema is unchanged; only the internal pipeline representation changes.

**Dependencies**

Milestone 15 (EDA findings motivate feature selection).

**Estimated effort**

Half day.

---

### Milestone 17 — Hyperparameter optimization

**Objective**

Replace fixed hyperparameters with a config-driven search protocol, log all trials to MLflow, and select the best configuration on the validation split.

**Rationale**

The current configs use `C=1.0` for logistic regression and hand-picked GradientBoosting parameters with no evidence of exploration. This is a critical interview gap: "What hyperparameters did you try?" currently has no answer. A logged search with trial history demonstrates model selection rigor and provides concrete interview discussion material.

**Deliverables**

1. Create `src/churn_prediction/models/tuning.py` with a `run_hyperparameter_search` function.
2. Implement search using `sklearn.model_selection.RandomizedSearchCV` with stratified K-fold on the training partition only. Use the validation split as a final hold-out confirmation, not as part of the CV folds. The test split must remain untouched.
3. Define search spaces in `configs/tuning.yaml`:
   - Logistic regression: `C` (log-uniform over `[0.01, 100]`), `solver` (`lbfgs`, `saga`), `class_weight` (`null`, `balanced`).
   - GradientBoosting: `n_estimators` (`[50, 100, 200, 300]`), `max_depth` (`[2, 3, 4, 5]`), `learning_rate` (`[0.01, 0.05, 0.1, 0.2]`), `subsample` (`[0.7, 0.8, 0.9, 1.0]`), `min_samples_leaf` (`[5, 10, 20]`).
4. Scoring metric for `RandomizedSearchCV` must be `average_precision` (PR-AUC) to stay consistent with the project's primary metric.
5. Log every trial's parameters and validation PR-AUC to MLflow as child runs under a parent "tuning" run. Persist the best parameters to `models/best_hyperparameters.json`.
6. Add a thin `scripts/run_tuning.py` entry point.
7. Update `configs/training.yaml` and `configs/candidate_training.yaml` with the selected best hyperparameters after a search run. Document the before/after metric delta.
8. Add unit tests: verify search runs on synthetic data without errors, respects seed, and returns valid parameter dictionaries.

**Acceptance criteria**

- `scripts/run_tuning.py` completes and writes `models/best_hyperparameters.json`.
- MLflow experiment shows a tuning run with ≥20 logged child trials.
- Best parameters from the search replace the hard-coded values in `training.yaml` / `candidate_training.yaml`.
- The validation-split PR-AUC with tuned parameters is compared against the previous fixed-parameter PR-AUC and the delta is documented.
- Test split is never used during search.

**Dependencies**

Milestone 16 (features should be finalized before tuning; otherwise, tuning on a different feature set is wasted work).

**Estimated effort**

Half day.

---

### Milestone 18 — Cross-validation and bootstrap confidence intervals

**Objective**

Add stratified K-fold cross-validation for robust metric estimation and bootstrap confidence intervals for key evaluation metrics.

**Rationale**

The current evaluation reports single-point metric estimates from a single stratified split. With only 7,043 rows, metric variance is non-trivial. SYSTEM_DESIGN.md states "report confidence intervals where feasible" but this was never implemented. Cross-validation and bootstrap CIs demonstrate statistical rigor that is expected of Senior DS candidates.

**Deliverables**

1. Create `src/churn_prediction/evaluation/cross_validation.py`:
   - A `run_cross_validation` function that performs stratified 5-fold CV on the training partition (not the held-out test split) using the fitted pipeline configuration. Report fold-level and mean ± std for PR-AUC, ROC-AUC, Brier score, and precision/recall at capacity.
   - Folds must use deterministic seeds derived from the global `random_seed` config.
2. Create `src/churn_prediction/evaluation/confidence.py`:
   - A `bootstrap_metric_ci` function that computes 95% bootstrap confidence intervals for PR-AUC, ROC-AUC, and Brier score on the test split. Use 1,000 bootstrap iterations with a fixed seed.
   - Return lower bound, point estimate, and upper bound.
3. Integrate CV results into the evaluation report JSON and decision record markdown.
4. Integrate bootstrap CIs into the evaluation summary:
   - `evaluation_metrics.json` gains a `"confidence_intervals"` section.
   - Decision record markdown shows `PR-AUC: 0.6437 [0.59, 0.70] (95% CI)` format.
5. Add a `scripts/run_cross_validation.py` entry point.
6. Add unit tests: verify CI bounds contain the point estimate, verify CV produces the correct number of folds, verify determinism under fixed seed.
7. Add a learning curve function: train-set-size vs. CV metric plot to `reports/evaluation/learning_curve.png`. This demonstrates whether more data would help.

**Acceptance criteria**

- Cross-validation runs on the training partition only; the test split is never used for fold evaluation.
- Bootstrap CIs are reported for PR-AUC, ROC-AUC, and Brier score on the test split.
- Evaluation report JSON and decision record markdown include CI ranges.
- Learning curve plot is generated and shows the metric trajectory as training size increases.
- All new functions are deterministic under fixed seed.
- Existing tests pass without modification.

**Dependencies**

Milestone 17 (use tuned hyperparameters for CV and CI estimation).

**Estimated effort**

Half day.

---

### Milestone 19 — Probability calibration pipeline

**Objective**

Fit a probability calibration model on the validation split and integrate it into the serving pipeline, turning a documented limitation into a demonstrated skill.

**Rationale**

The model card explicitly lists "isotonic/sigmoid recalibration is not fitted into the shipped pipeline" as a limitation. SYSTEM_DESIGN.md prescribes "use probability calibration (isotonic/sigmoid selected by validation evidence)." Implementing this closes a known gap, demonstrates calibration depth, and produces before/after reliability diagrams that are strong interview artifacts.

**Deliverables**

1. Add a `fit_calibration` function to `src/churn_prediction/evaluation/calibration.py` that:
   - Accepts a fitted pipeline, validation features, and validation labels.
   - Fits `sklearn.calibration.CalibratedClassifierCV` with `cv="prefit"` and method selected by Brier score comparison (`"sigmoid"` vs `"isotonic"`) on the validation split.
   - Returns the calibrated pipeline.
2. Integrate calibration fitting into the training workflow (`src/churn_prediction/models/trainer.py`):
   - After fitting on the training split, fit calibration on the validation split.
   - The calibrated pipeline becomes the primary artifact; the uncalibrated pipeline is retained for comparison.
3. Add config support: `configs/training.yaml` gains a `calibration.enabled: true` and `calibration.method: "auto"` (`auto` selects by validation Brier comparison; `sigmoid` and `isotonic` force a specific method).
4. Generate before/after reliability diagrams in the evaluation report:
   - `reports/evaluation/calibration_before_after.png` showing both curves on the same plot.
   - Log Brier score improvement to MLflow.
5. Update the model card limitations section to reflect that calibration is now fitted.
6. Add unit tests: verify calibrated predictions remain in `[0, 1]`, verify Brier score does not degrade on the validation split, verify the pipeline artifact includes the calibration wrapper.

**Acceptance criteria**

- Promoted serving pipeline artifact includes the fitted calibration layer.
- Before/after reliability diagram is generated and shows measurable improvement (or documents that calibration had negligible effect on this model, which is also a valid finding).
- Brier score comparison is logged to MLflow and included in the evaluation report.
- API and batch scoring produce calibrated probabilities without code changes (because calibration is inside the pipeline artifact).
- Existing contract and integration tests pass without modification.

**Dependencies**

Milestone 17 (tuned model should be calibrated, not the un-tuned one).

**Estimated effort**

2 hours.

---

### Milestone 20 — Model explainability

**Objective**

Add SHAP-based global and local feature importance, return top-N reason codes from the API, and generate explainability artifacts for the evaluation report.

**Rationale**

The model card states "local reason codes / explanations are not yet returned by the API" as a limitation. SYSTEM_DESIGN.md requires "carefully worded local reason codes without claiming causality." SHAP is the industry standard for tabular model explainability and is a high-value interview discussion topic. This milestone closes a documented gap and adds a tangible capability.

**Deliverables**

1. Add `shap` as a production dependency in `pyproject.toml`.
2. Create `src/churn_prediction/evaluation/explainability.py`:
   - A `compute_shap_values` function that produces global SHAP feature importance (mean |SHAP|) and per-record SHAP values using `shap.TreeExplainer` for tree models and `shap.LinearExplainer` for logistic regression.
   - A `get_top_reason_codes` function that accepts a single record's SHAP values and returns the top-N features driving the prediction, with direction (increases/decreases churn risk) and magnitude.
   - Reason codes must use carefully worded language (e.g. "Short tenure is associated with higher churn risk") and must not claim causality.
3. Add a SHAP summary plot to the evaluation report: `reports/evaluation/shap_summary.png`.
4. Extend `PredictionResponse` schema in `src/churn_prediction/api/schemas.py` with an optional `reason_codes` field (list of objects with `feature`, `direction`, `contribution`).
5. Extend `InferenceService.predict` in `src/churn_prediction/api/service.py` to compute and return top-3 reason codes per prediction when model and SHAP explainer are loaded.
6. Cache the SHAP explainer at model load time in `InferenceService.load_model` so per-request overhead is minimal.
7. Add unit tests: verify reason code count, verify direction labels, verify no customerID or sensitive feature appears in reason codes.
8. Add a contract test: verify `/predict` response includes `reason_codes` field with expected structure.

**Acceptance criteria**

- SHAP summary plot is generated and saved during evaluation.
- API `/predict` response includes `reason_codes` with ≤ 3 features per prediction.
- No reason code references `customerID`, `gender`, or `SeniorCitizen`.
- Reason code language does not claim causality (e.g. uses "associated with" not "causes").
- Existing API contract tests pass; new tests cover the reason codes field.
- SHAP computation adds < 50ms per single-record prediction.

**Dependencies**

Milestone 16 (feature engineering must be finalized so SHAP values reflect the production feature set).

**Estimated effort**

Half day.

---

### Milestone 21 — Business decision analysis

**Objective**

Add an expected-value threshold selection framework that connects model predictions to retention campaign economics and documents the optimal operating point.

**Rationale**

The current evaluation sweeps thresholds across precision/recall trade-offs but never connects them to business value. The serving default of 0.50 has no business justification. An expected-value analysis that models intervention cost, retained revenue, and capacity constraints transforms threshold selection from arbitrary to defensible—a critical interview discussion topic.

**Deliverables**

1. Create `src/churn_prediction/evaluation/business.py`:
   - A `compute_expected_value_curve` function that, for each threshold in the sweep, computes: true-positive value (saved revenue), false-positive cost (wasted intervention), false-negative cost (lost revenue), and net expected value per customer.
   - Default unit economics in `configs/evaluation.yaml` under a `business` section: `intervention_cost_per_customer: 50`, `average_monthly_revenue: 65`, `average_customer_lifetime_months: 30`, `churn_save_rate: 0.25` (fraction of correctly identified churners who are retained by the intervention). These are configurable placeholders.
   - An `optimal_threshold` function that returns the threshold maximizing net expected value, subject to the capacity constraint.
2. Generate plots:
   - `reports/evaluation/expected_value_curve.png` showing net value vs. threshold.
   - `reports/evaluation/cost_benefit_matrix.png` showing the cost-benefit matrix at the selected threshold.
3. Integrate the optimal threshold recommendation into the decision record:
   - "Business-optimal threshold: X.XX (net value: $Y per 1,000 customers scored)".
   - Document that the serving threshold can be updated to the recommended value via `CHURN_DECISION_THRESHOLD` environment variable.
4. Add unit tests: verify expected value is monotonically consistent with TP/FP trade-offs, verify threshold respects capacity constraint, verify results are deterministic.

**Acceptance criteria**

- Expected value curve and cost-benefit matrix are generated during evaluation.
- Decision record includes the recommended business-optimal threshold with net value estimate.
- Unit economics are fully configurable via `configs/evaluation.yaml`.
- No actual financial data is committed; only documented placeholder parameters.
- Existing evaluation tests pass.

**Dependencies**

Milestone 18 (confidence intervals add credibility to the threshold recommendation).

**Estimated effort**

2 hours.

---

### Milestone 22 — CI pipeline hardening

**Objective**

Extend the GitHub Actions CI pipeline to validate Docker builds, enforce type checking, and add status badges to the README.

**Rationale**

The Dockerfile is a key portfolio selling point but is never built in CI; its correctness is unverified on every push. Adding a Docker build smoke test, type checking, and visible badges materially increases recruiter confidence.

**Deliverables**

1. Add a `docker` job to `.github/workflows/ci.yml`:
   - Runs `docker build --target runner -t telco-churn-api:ci .` to verify the image builds.
   - Does not run the container or execute integration tests (keep CI fast).
   - Depends on the `quality` job passing first.
2. Add a `typecheck` step to the `quality` job:
   - Runs `uv run python -m mypy src/churn_prediction/ --ignore-missing-imports` (incremental adoption; allow `--ignore-missing-imports` initially).
   - Add a `[tool.mypy]` section to `pyproject.toml` with `python_version = "3.11"` and `warn_return_any = true`.
3. Add CI caching:
   - Cache the `uv` dependency installation to speed up subsequent runs.
4. Add badges to `README.md`:
   - GitHub Actions CI status badge.
   - Coverage percentage badge (via `pytest-cov` output or Codecov integration).
5. Add a `CHANGELOG.md` following Keep a Changelog format, documenting Phase 1 as `v0.1.0` and Phase 2 enhancements incrementally as they are completed.

**Acceptance criteria**

- CI pipeline builds the Docker image on every push/PR without failure.
- `mypy` step runs in CI (warnings are acceptable initially; errors must not block if `--ignore-missing-imports` is used).
- README displays working CI status badge.
- `CHANGELOG.md` exists with at least a `v0.1.0` entry covering Phase 1.

**Dependencies**

Milestone 14 (pre-commit and expanded lint rules should be in place before adding type checking).

**Estimated effort**

2 hours.

---

### Milestone 23 — Documentation and portfolio polish

**Objective**

Improve README presentation, add interview-ready documentation artifacts, and clean up files that reduce portfolio credibility.

**Rationale**

A recruiter spends approximately two minutes on a repository. Badges, a visible architecture diagram, and a concise "key design decisions" section dramatically increase the probability of advancing to a technical screen. Removing AI-tooling configuration files from the repository root eliminates a credibility question.

**Deliverables**

1. Move `AGENTS.md` from the repository root to `.gemini/AGENTS.md` (or equivalent tooling-config directory). If the file must remain for tooling, add a note to `README.md` explaining its purpose transparently.
2. Add a "Key Design Decisions" section to `README.md` listing 5–7 non-obvious architectural choices with one-sentence rationale each. Examples:
   - "PR-AUC over accuracy as the primary metric because the target is imbalanced (26.5% churn)."
   - "Validation-gated selection instead of test-set-driven tuning to prevent adaptive overfitting."
   - "Batch scoring as the primary path because retention campaigns are planned, not real-time."
3. Add a "What I Would Do Differently in Production" section to `README.md` listing 3–5 concrete changes for a real deployment (e.g., out-of-time splits with event dates, feature store integration, A/B testing framework).
4. Add a `CONTRIBUTING.md` with basic contribution guidelines, development setup instructions, and coding standards reference.
5. Ensure all embedded images in `README.md` use relative paths that resolve correctly on GitHub (commit any missing plot images from `docs/images/` that are referenced but absent).
6. Add a "Skills Demonstrated" section to `README.md` listing the technical competencies this project evidences (data contracts, pipeline engineering, evaluation methodology, fairness, observability, etc.).

**Acceptance criteria**

- `AGENTS.md` is no longer visible at the repository root.
- README contains "Key Design Decisions", "What I Would Do Differently", and "Skills Demonstrated" sections.
- `CONTRIBUTING.md` exists and references `AGENTS.md` for coding standards.
- All README images render correctly when viewed on GitHub.

**Dependencies**

None (can run in parallel with any milestone after 14).

**Estimated effort**

2 hours.

---

### Milestone 24 — Model card and evaluation report refresh

**Objective**

Update the model card and evaluation reports to reflect all Phase 2 enhancements: engineered features, tuned hyperparameters, calibrated probabilities, confidence intervals, explainability, and business threshold analysis.

**Rationale**

The model card and decision record are living documents. After milestones 16–21 change the model's features, parameters, calibration, and evaluation depth, these documents must be refreshed to remain accurate and to showcase the full depth of work.

**Deliverables**

1. Update `docs/model_card.md`:
   - Section 3 (Training Data): add a "Feature Engineering" subsection listing the engineered features with rationale.
   - Section 5 (Limitations): remove the calibration and explainability limitations that are now resolved. Add any new limitations introduced.
   - Section 6 (Evaluation Metrics): include confidence intervals and cross-validation results.
   - Add a new Section 10: "Explainability" documenting the SHAP-based reason codes and their anti-causality framing.
2. Update `reports/evaluation/decision_record.md` (via a fresh comparison run):
   - Include tuned hyperparameters for both models.
   - Show cross-validation mean ± std alongside the single-split metrics.
   - Show bootstrap confidence intervals on test-split metrics.
   - Include the business-optimal threshold recommendation.
3. Update `README.md` measured results table with post-Phase-2 metrics.
4. Re-run the full train → evaluate → compare → promote pipeline with all Phase 2 enhancements active. Commit the updated decision record and measured results.

**Acceptance criteria**

- Model card reflects the current state of the system with no stale limitations.
- Decision record includes confidence intervals, cross-validation results, and business threshold recommendation.
- README measured results match the latest evaluation run.
- All documentation is internally consistent (no contradictions between model card, decision record, and README).

**Dependencies**

Milestones 16–21 (all ML enhancements should be complete before the documentation refresh).

**Estimated effort**

2 hours.

---

### Milestone 25 — Technical debt reduction

**Objective**

Consolidate duplicated configuration-loading patterns, address the Prometheus private API access, and wrap synchronous inference in `asyncio.to_thread` for the async FastAPI handlers.

**Rationale**

These are lower-ROI improvements that incrementally improve maintainability and production correctness. They are worth doing after the high-impact milestones are complete, but should not block interview readiness.

**Deliverables**

1. Create `src/churn_prediction/config.py` (a new top-level config utility module):
   - A single `load_yaml_config(config_name: str, config_path: str | Path | None = None) -> dict[str, Any]` function that resolves config file paths using a consistent strategy (check explicit path, then `CHURN_CONFIG_DIR` environment variable, then `configs/` relative to the package).
   - Replace the duplicated `get_default_*_config_path()` + `yaml.safe_load` patterns across `data/contract.py`, `features/pipeline.py`, `models/batch_scoring.py`, `api/config.py`, `monitoring/config.py`, and `evaluation/config.py` with calls to the centralized loader.
2. Fix Prometheus `_names_to_collectors` private attribute access in `src/churn_prediction/monitoring/metrics.py`:
   - Replace `registry._names_to_collectors[full_name]` with a public-API approach: catch the `ValueError` on duplicate registration and look up the existing collector via `registry._get_names` or simply return the already-constructed instance from a module-level cache dictionary.
3. Wrap `pipeline.predict_proba` calls in `InferenceService.predict` and `score_batch` with `await asyncio.to_thread(...)` (for the async API handler) or document why the synchronous call is acceptable for the current deployment scope.
4. Add a `valid_rows` calculation fix in `src/churn_prediction/data/validator.py`: replace `max(0, total_rows - len(error_details))` with a count of distinct failing row indices to avoid conflating error count with invalid row count.
5. Add unit tests for the centralized config loader, including missing file, empty file, and environment variable override scenarios.

**Acceptance criteria**

- No module uses `Path(__file__).resolve().parents[N]` for config discovery; all use the centralized loader.
- No private `_names_to_collectors` access in production code.
- `valid_rows` accurately reflects the number of rows that passed all checks.
- All existing tests pass.

**Dependencies**

Milestone 14 (expanded lint rules may surface additional issues that should be fixed first).

**Estimated effort**

Half day.

---

## Phase 2 defaults and constraints

- **Architecture preservation**: No changes to the module boundary structure (`src/churn_prediction/` package layout, test layers, script entry points, config-driven design). All enhancements extend the existing architecture.
- **Test discipline**: Every milestone adds tests proportional to the code it introduces. Coverage must not drop below the 80% threshold configured in `pyproject.toml`.
- **Backward compatibility**: Existing scripts (`train_baseline.py`, `run_batch_scoring.py`, `run_api.py`, etc.) must continue to work after each milestone. Breaking changes to CLI arguments or config keys require explicit migration notes.
- **Data contract stability**: The input schema (`Telco-Customer-Churn.csv` column structure) is unchanged. Engineered features are derived inside the pipeline, not as a preprocessing requirement on the input data.
- **Commit discipline**: Each milestone should be committed as a focused, reviewable unit using Conventional Commit subjects. Do not bundle multiple milestones into a single commit.
