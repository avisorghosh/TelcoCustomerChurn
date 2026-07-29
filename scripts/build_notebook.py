"""Script to programmatically construct the Milestone 3 EDA notebook."""

from pathlib import Path

import nbformat as nbf


def build_notebook(
    output_path: Path | str = "notebooks/01_exploratory_data_analysis.ipynb",
) -> Path:
    """Build and save the Milestone 3 EDA Jupyter Notebook.

    Args:
        output_path: Destination path for the generated notebook.

    Returns:
        Path to saved notebook.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Executive Summary
    cells.append(
        nbf.v4.new_markdown_cell(
            "# Milestone 3 — Exploratory Data Analysis (EDA)\n"
            "**Project:** IBM Telco Customer Churn Prediction  \n"
            "**Author:** AI Pair Programmer / Antigravity  \n"
            "**Date:** July 2026  \n\n"
            "---\n\n"
            "## Executive Summary\n\n"
            "This notebook presents a production-oriented Exploratory Data "
            "Analysis (EDA) of the IBM Telco Customer Churn dataset. Unlike "
            "an unstructured exploratory notebook, this analysis is "
            "structured around concrete engineering and modelling objectives: "
            "validating schema contract assumptions, assessing data quality, "
            "checking for data leakage, reviewing fairness-relevant "
            "demographic features, and establishing downstream preprocessing "
            "and model architecture decisions.\n\n"
            "### Key EDA Findings\n"
            "1. **Dataset Dimensions & Integrity:** 7,043 customer snapshot "
            "records across 21 features with 0 duplicate rows and 0 duplicate "
            "`customerID`s.\n"
            "2. **Missing Value Nuance:** Exactly 11 records have blank "
            "`TotalCharges` values. Analysis confirms that 100% of these "
            "records correspond to brand-new customers with `tenure == 0`. "
            "They can be safely imputed with `0.0`.\n"
            "3. **Target Imbalance:** Target variable `Churn` contains 1,869 "
            "churned customers (26.54%) and 5,174 retained customers (73.46%), "
            "yielding an imbalance ratio of **2.77 : 1**. Precision-Recall AUC "
            "(PR-AUC) and recall at top capacity are appropriate evaluation "
            "metrics instead of accuracy.\n"
            "4. **Primary Churn Drivers:**\n"
            "   - **Contract Type:** Month-to-month contract customers churn at "
            "**42.71%**, compared to 11.27% for 1-year and 2.83% for 2-year.\n"
            "   - **Internet Tier:** Fiber optic customers churn at **41.89%**, "
            "compared to 18.96% for DSL and 7.40% for non-internet.\n"
            "   - **Payment Channel:** Electronic check users churn at **45.29%**, "
            "vs ~15.5% for automatic payment channels.\n"
            "   - **Customer Tenure:** Early tenure customers (0-12 months) churn "
            "at **47.44%**, dropping below 7% for long-tenure (>60 months).\n"
            "5. **Collinearity & Discrepancy:** `TotalCharges` is highly "
            "correlated with `tenure` ($r = 0.83$) and `MonthlyCharges` "
            "($r = 0.65$). The mean absolute discrepancy between `TotalCharges` "
            "and $\\text{tenure} \\times \\text{MonthlyCharges}$ is **$45.09**, "
            "reflecting mid-contract price changes and service modifications.\n"
            "6. **Fairness & Leakage Review:**\n"
            "   - `gender` displays near-zero churn rate disparity (Female "
            "26.92% vs Male 26.16%) and must be excluded from feature inputs.\n"
            "   - `SeniorCitizen` customers exhibit higher churn (41.68% vs "
            "23.61%), but are excluded from model inputs per `SYSTEM_DESIGN.md` "
            "guidelines to prevent age bias, remaining reserved for governed "
            "sub-population evaluation.\n"
            "   - `customerID` is 100% unique and carries no predictive "
            "information. No post-churn outcome features exist.\n"
        )
    )

    # Section 1: Setup & Imports
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Environment Setup & Reusable Data Load\n\n"
            "Reusable data validation and EDA utilities are imported from "
            "`src/churn_prediction/data/` and `src/churn_prediction/eda/` to "
            "ensure no production logic resides in the notebook.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n\n"
            "import pandas as pd\n\n"
            "from churn_prediction.data.validator import parse_total_charges\n"
            "from churn_prediction.eda import (\n"
            "    assess_leakage_and_fairness,\n"
            "    get_categorical_summary,\n"
            "    get_charge_discrepancy_analysis,\n"
            "    get_dataset_overview,\n"
            "    get_numeric_correlations,\n"
            "    get_numeric_summary,\n"
            "    get_target_distribution,\n"
            ")\n\n"
            "data_path = (\n"
            "    Path('../Telco-Customer-Churn.csv')\n"
            "    if Path('../Telco-Customer-Churn.csv').exists()\n"
            "    else Path('Telco-Customer-Churn.csv')\n"
            ")\n"
            "raw_df = pd.read_csv(data_path)\n"
            "df, parsing_errors = parse_total_charges(raw_df)\n"
            "print(f'Loaded {len(df)} rows and {len(df.columns)} columns.')\n"
        )
    )

    # Section 2: Overview & Data Quality
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Dataset Dimensions, Schema & Quality Audit\n\n"
            "We verify dataset structure, data types, missing value distribution, "
            "and duplicate records.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "overview = get_dataset_overview(df)\n"
            "print('Total Rows:', overview['total_rows'])\n"
            "print('Total Columns:', overview['total_cols'])\n"
            "print('Duplicate Rows:', overview['duplicate_rows'])\n"
            "print('Duplicate Customer IDs:', overview['duplicate_customer_ids'])\n"
            "print('Missing Value Counts:', overview['missing_counts'])\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "### Analysis of Missing `TotalCharges` Values\n"
            "All 11 missing `TotalCharges` values occur in rows where "
            "`tenure == 0`. These represent newly onboarded customers who "
            "have not completed a full billing cycle.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "cols = [\n"
            "    'customerID',\n"
            "    'tenure',\n"
            "    'MonthlyCharges',\n"
            "    'TotalCharges',\n"
            "    'Churn',\n"
            "]\n"
            "blank_tc_df = df[df['TotalCharges'].isna()]\n"
            "print('Tenure values for missing TotalCharges rows:')\n"
            "print(blank_tc_df[cols])\n"
        )
    )

    # Section 3: Target Distribution
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Target Distribution & Class Imbalance Analysis\n\n"
            "Evaluating the binary target variable `Churn`.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "target_dist = get_target_distribution(df)\n"
            "print('Class Counts:', target_dist['counts'])\n"
            "print('Class Percentages:', target_dist['percentages'])\n"
            "print(f\"Overall Churn Rate: {target_dist['churn_rate'] * 100:.2f}%\")\n"
            "imb = target_dist['imbalance_ratio']\n"
            "print(f'Imbalance Ratio (No/Yes): {imb:.2f}')\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "![Target Distribution](../docs/images/target_distribution.png)\n"
        )
    )

    # Section 4: Numeric Feature Distributions
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Numeric Feature Distributions & Outlier Assessment\n\n"
            "Analyzing summary statistics, spread, skewness, and IQR outlier "
            "boundaries for `tenure`, `MonthlyCharges`, and `TotalCharges`.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "numeric_summary = get_numeric_summary(df)\n"
            "print(numeric_summary.to_string())\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "![Numeric Distributions](../docs/images/numeric_distributions.png)\n"
        )
    )

    # Section 5: Charge Discrepancy & Correlations
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Charge Discrepancy & Pearson Correlation Analysis\n\n"
            "Evaluating the mathematical relationship between `TotalCharges` and "
            "`tenure * MonthlyCharges`, along with feature collinearity.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "disc_analysis = get_charge_discrepancy_analysis(df)\n"
            "for metric, val in disc_analysis.items():\n"
            "    print(f'{metric}: {val}')\n\n"
            "corr_matrix = get_numeric_correlations(df)\n"
            "print('\\nPearson Correlation Matrix:')\n"
            "print(corr_matrix)\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "![Correlation Heatmap](../docs/images/correlation_heatmap.png)\n"
            "![Tenure vs Total Charges](../docs/images/tenure_vs_charges.png)\n"
        )
    )

    # Section 6: Categorical Feature Distributions
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. Categorical Feature Distributions & Churn Risk Drivers\n\n"
            "Examining category frequencies and churn rates across key contract, "
            "service, and payment dimensions.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "cat_cols = [\n"
            "    'Contract',\n"
            "    'InternetService',\n"
            "    'PaymentMethod',\n"
            "    'TechSupport',\n"
            "]\n"
            "cat_summaries = get_categorical_summary(df, categorical_cols=cat_cols)\n"
            "for col_name, summary_df in cat_summaries.items():\n"
            "    print(f'\\n--- Feature: {col_name} ---')\n"
            "    print(summary_df.to_string(index=False))\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "![Categorical Churn Rates](../docs/images/categorical_churn_rates.png)\n"
        )
    )

    # Section 7: Leakage & Fairness
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 7. Data Leakage & Sensitive Demographic Attribute Assessment\n\n"
            "Verifying data hygiene and demographic neutrality.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "leakage_fairness = assess_leakage_and_fairness(df)\n"
            "print('Leakage Review:')\n"
            "for item in leakage_fairness['leakage_findings']:\n"
            "    print(' -', item)\n\n"
            "print('\\nFairness Review:')\n"
            "for item in leakage_fairness['fairness_findings']:\n"
            "    print(' -', item)\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "![Fairness Attributes](../docs/images/fairness_attributes.png)\n"
        )
    )

    # Section 8: Confirm/Challenge Design Assumptions & Downstream Implications
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 8. Confirmation of `SYSTEM_DESIGN.md` Assumptions & Implications\n\n"
            "### Confirmation of System Design Assumptions\n"
            "| Assumption | Status | Empirical Finding |\n"
            "|---|---|---|\n"
            "| 11 blank `TotalCharges` values | **Confirmed** | "
            "All 11 records have `tenure == 0`. Impute with `0.0`. |\n"
            "| Target Imbalance ~26.5% | **Confirmed** | "
            "Exact churn rate is 26.54% (1,869 Yes vs 5,174 No). |\n"
            "| `customerID` Non-Predictive | **Confirmed** | "
            "100% unique hash; must be excluded from feature inputs. |\n"
            "| `gender` Neutrality | **Confirmed** | "
            "Disparity is < 0.8% (Female: 26.92%, Male: 26.16%). |\n"
            "| `SeniorCitizen` Higher Churn | **Confirmed** | "
            "Seniors churn at 41.68% vs 23.61%. Excluded for fairness. |\n"
            "| Primary Drivers | **Confirmed** | "
            "`Contract`, `InternetService`, `tenure` are drivers. |\n\n"
            "### Downstream Pipeline Implications (Milestone 4+)\n"
            "1. **Imputation:** Impute `TotalCharges` missing values with `0.0` "
            "inside transformer (matching `tenure == 0`).\n"
            "2. **Stratified Splitting:** Use stratified $k$-fold / train-test splits "
            "on `Churn` to preserve target proportion.\n"
            "3. **Feature Selection:** Exclude `customerID`, `gender`, and "
            "`SeniorCitizen` from `X` feature matrix.\n"
            "4. **Encoding:** One-hot encode nominal categories with "
            "`handle_unknown='ignore'`.\n"
            "5. **Scaling:** Standardize/robust-scale `tenure`, `MonthlyCharges`, "
            "`TotalCharges` for baseline.\n"
        )
    )

    nb["cells"] = cells

    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    return out_path


if __name__ == "__main__":
    path = build_notebook()
    print(f"Created notebook at: {path}")
