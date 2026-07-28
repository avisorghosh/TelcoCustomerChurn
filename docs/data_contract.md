# Data Contract: IBM Telco Customer Churn

**Version:** 1.0.0  
**Dataset Name:** IBM Telco Customer Churn  
**Primary Key:** `customerID`  
**Target Variable:** `Churn` (`Yes` / `No`)  

---

## 1. Overview

This document specifies the official data contract for ingesting, validating, and scoring customer snapshots in the Telco Customer Churn Prediction system. Any dataset ingested for training, evaluation, or batch scoring MUST pass this contract validation without errors.

---

## 2. Schema Specification

The dataset consists of 21 required columns:

| Column Name | Expected Type | Nullable | Validation Rules & Allowed Domain |
| --- | --- | --- | --- |
| `customerID` | String | No | Non-blank, unique string identifier. Primary key. |
| `gender` | String | No | Allowed: `["Female", "Male"]` |
| `SeniorCitizen` | Integer | No | Allowed: `[0, 1]` |
| `Partner` | String | No | Allowed: `["Yes", "No"]` |
| `Dependents` | String | No | Allowed: `["Yes", "No"]` |
| `tenure` | Integer | No | Min value: `0` |
| `PhoneService` | String | No | Allowed: `["Yes", "No"]` |
| `MultipleLines` | String | No | Allowed: `["No phone service", "No", "Yes"]` |
| `InternetService` | String | No | Allowed: `["DSL", "Fiber optic", "No"]` |
| `OnlineSecurity` | String | No | Allowed: `["No internet service", "No", "Yes"]` |
| `OnlineBackup` | String | No | Allowed: `["No internet service", "No", "Yes"]` |
| `DeviceProtection` | String | No | Allowed: `["No internet service", "No", "Yes"]` |
| `TechSupport` | String | No | Allowed: `["No internet service", "No", "Yes"]` |
| `StreamingTV` | String | No | Allowed: `["No internet service", "No", "Yes"]` |
| `StreamingMovies` | String | No | Allowed: `["No internet service", "No", "Yes"]` |
| `Contract` | String | No | Allowed: `["Month-to-month", "One year", "Two year"]` |
| `PaperlessBilling` | String | No | Allowed: `["Yes", "No"]` |
| `PaymentMethod` | String | No | Allowed: `["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]` |
| `MonthlyCharges` | Float | No | Min value: `0.0` |
| `TotalCharges` | Float | Yes | Min value: `0.0` (Blank whitespace strings `" "` are parsed into `NaN`) |
| `Churn` | String | No (Training) / Yes (Scoring) | Allowed: `["Yes", "No"]` |

---

## 3. Specific Data Quality & Integrity Policies

### 3.1 `TotalCharges` Blank Handling
- Historical raw data contains 11 rows where `TotalCharges` is formatted as blank whitespace (`" "`).
- The data contract explicitly parses whitespace-only strings into floating-point `NaN`/nulls.
- **Strict Rule:** Non-numeric strings (e.g. `"N/A"`, `"invalid"`) are **NEVER** silently coerced to zero or NaN; encountering corrupt non-numeric text triggers an immediate validation error.

### 3.2 Identifier Uniqueness
- `customerID` must be unique across all rows.
- Blank, whitespace-only, or missing `customerID` values trigger a validation error.

### 3.3 Duplicate Row Policy
- Duplicate rows (identical values across all columns) are prohibited and fail validation cleanly.

### 3.4 Target Requirements
- For training datasets (`is_training=True`), `Churn` is strictly required and must belong to `["Yes", "No"]`.
- For inference/scoring datasets (`is_training=False`), `Churn` may be omitted or unlabelled.

---

## 4. Manifest & Provenance

Every validated dataset generates a JSON source manifest recording:
- File SHA-256 checksum
- Exact line/row count
- Ingestion timestamp (ISO 8601 UTC)
- Data contract schema version
- Quality report file reference
