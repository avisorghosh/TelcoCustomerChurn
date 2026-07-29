# Model Selection Decision Record: Baseline vs. GradientBoosting

**Date/Timestamp**: `2026-07-29T16:35:21.771344+00:00`
**Selection Split**: `val` ($N = 1057$)
**Reporting Split**: `test` ($N = 1057$)
**Selected Model**: **baseline_logistic_regression** (`LogisticRegression`)

---

## 1. Executive Summary & Final Decision

**Selected Model**: **`baseline_logistic_regression`**
**Selection Rationale**:
Retained baseline 'baseline_logistic_regression' because the candidate failed one or more validation-split acceptance gates.

---

## 2. Holdout Test Evaluation Summary

| Metric | Baseline (LogisticRegression) | Candidate (GradientBoosting) | Difference |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `0.6437` | `0.6753` | **`+0.0316`** |
| **ROC-AUC** | `0.8487` | `0.8528` | `+0.0041` |
| **Brier Score** | `0.1361` | `0.1332` | `-0.0029` |
| **Accuracy @ 0.50** | `0.7985` | `0.8079` | `+0.0094` |
| **Precision @ 10%** | `75.47%` | `76.42%` | `+0.95%` |
| **Recall @ 10%** | `28.57%` | `28.93%` | `+0.36%` |

---

## 3. Probability Calibration Summary (Test)

- **Baseline Brier Score**: `0.1361`
- **Candidate Brier Score**: `0.1332`
- **Calibration Observation**: Candidate (GradientBoosting) test Brier score is 0.1332 vs baseline (LogisticRegression) test Brier score of 0.1361.

---

## 4. Predefined Acceptance Gate Verification (Validation Split)

| Acceptance Gate | Required Threshold | Measured Value | Result |
| :--- | :--- | :--- | :---: |
| **Gate 1: PR-AUC** | `>= +0.0100` | `+0.0071` | FAILED |
| **Gate 2: Calibration** | `<= +0.0200` | `-0.0026` | PASSED |
| **Gate 3: Fairness** | `<= +0.0500` | `-0.0006` | PASSED |
| **Gate 4: Reproducibility** | `Artifacts loadable` | `Verified` | PASSED |

---

## 5. Segment & Fairness Observations (Test)

- **Segment Analysis**: Evaluated across `tenure_band`, `Contract`, `InternetService`,
  and `monthly_charges_band`.
- **Fairness Review**: Evaluated post-hoc across excluded attributes (`gender`,
  `SeniorCitizen`). Validation demographic parity difference change was
  `-0.0006` (candidate `0.0130` vs baseline
  `0.0136`).

---

## 6. Model Strengths & Weaknesses

### Baseline (Logistic Regression)
- **Strengths**: High interpretability, linear log-odds coefficients, fast execution.
- **Weaknesses**: Linear decision boundary limits capture of non-linear interactions.

### Candidate (GradientBoosting)
- **Strengths**: Superior ranking quality (PR-AUC), higher precision/recall at campaign
  capacity, captures complex feature interactions naturally.
- **Weaknesses**: Non-linear tree ensemble requires hyperparameter bounding.
