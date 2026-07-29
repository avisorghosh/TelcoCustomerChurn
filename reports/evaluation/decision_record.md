# Model Selection Decision Record: Baseline vs. LightGBM Candidate

**Date/Timestamp**: `2026-07-29T14:39:30.621594+00:00`  
**Evaluation Split**: `test` ($N = 1057$)  
**Selected Model**: **candidate_gradient_boosting** (`LightGBM`)

---

## 1. Executive Summary & Final Decision

**Selected Model**: **`candidate_gradient_boosting`**  
**Selection Rationale**:  
Selected 'candidate_gradient_boosting' (LightGBM) because it satisfied all predefined acceptance gates: PR-AUC improved by +0.0316 (gate threshold: >= +0.0100); Brier calibration score changed by -0.0029 (gate threshold: <= +0.0200); gender demographic parity difference changed by -0.0002 (gate threshold: <= +0.0500); and pipeline execution is fully reproducible and integrated.

---

## 2. Evaluation Summary

| Metric | Baseline (Logistic Regression) | Candidate (LightGBM) | Difference (Candidate - Baseline) |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `0.6437` | `0.6753` | **`+0.0316`** |
| **ROC-AUC** | `0.8487` | `0.8528` | `+0.0041` |
| **Brier Score (Calibration)** | `0.1361` | `0.1332` | `-0.0029` |
| **Accuracy @ 0.50** | `0.7985` | `0.8079` | `+0.0094` |
| **Precision @ 10% Capacity** | `75.47%` | `76.42%` | `+0.95%` |
| **Recall @ 10% Capacity** | `28.57%` | `28.93%` | `+0.36%` |

---

## 3. Probability Calibration Summary

- **Baseline Brier Score**: `0.1361`
- **Candidate Brier Score**: `0.1332`
- **Calibration Observation**: Candidate LightGBM Brier score is 0.1332 compared to Baseline Logistic Regression Brier score of 0.1361.

---

## 4. Predefined Acceptance Gate Verification

| Acceptance Gate | Required Threshold | Measured Value | Result |
| :--- | :--- | :--- | :---: |
| **Gate 1: PR-AUC Improvement** | `PR-AUC diff >= +0.0100` | `+0.0316` | ✅ PASSED |
| **Gate 2: Calibration Integrity** | `Brier diff <= +0.0200` | `-0.0029` | ✅ PASSED |
| **Gate 3: Subgroup & Fairness Parity** | `DP diff change <= +0.0500` | `-0.0002` | ✅ PASSED |
| **Gate 4: Operational Reproducibility** | `Reproducible pipeline` | `Verified` | ✅ PASSED |

---

## 5. Segment & Fairness Observations

- **Segment Analysis**: Evaluated across `tenure_band`, `Contract`, `InternetService`, and `monthly_charges_band`. Candidate model maintains consistent performance gains particularly on high-risk segments such as `Month-to-month` contracts and `Fiber optic` internet service.
- **Fairness Review**: Evaluated post-hoc across excluded attributes (`gender`, `SeniorCitizen`). Demographic Parity Difference remains stable (0.0022 for candidate vs 0.0024 for baseline), confirming no disparate impact introduced.

---

## 6. Model Strengths & Weaknesses

### Baseline (Logistic Regression)
- **Strengths**: High interpretability, linear log-odds coefficients, fast execution.
- **Weaknesses**: Linear decision boundary limits capture of non-linear feature interactions.

### Candidate (LightGBM)
- **Strengths**: Superior ranking quality (PR-AUC), higher precision/recall at campaign capacity, captures feature interactions.
- **Weaknesses**: Non-linear tree ensemble requires hyperparameter bounding to avoid overfitting.
