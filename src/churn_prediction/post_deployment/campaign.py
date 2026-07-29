"""Campaign effectiveness evaluation module (Treatment vs Control)."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_campaign_effectiveness(
    matched_df: pd.DataFrame,
    unit_contact_cost: float = 10.0,
    unit_customer_value: float = 500.0,
    treatment_column: str = "treatment_group",
    target_column: str = "observed_churn",
) -> dict[str, Any]:
    """Evaluate retention campaign effectiveness comparing Treatment vs Control.

    Calculates sample sizes, observed churn rates, absolute/relative churn reduction,
    incremental retained customers, and financial campaign ROI.

    Args:
        matched_df: Merged DataFrame containing predictions and delayed labels.
        unit_contact_cost: Cost in USD to contact/offer retention incentive.
        unit_customer_value: Retained gross margin USD per prevented churn.
        treatment_column: Column identifying 'treatment' vs 'control' assignment.
        target_column: Ground truth binary churn outcome column (1=churned, 0=retained).

    Returns:
        Dictionary summarizing treatment/control sample sizes, churn rates, and ROI.
    """
    if target_column not in matched_df.columns:
        raise ValueError(f"matched_df missing target outcome column: '{target_column}'")

    if treatment_column not in matched_df.columns:
        logger.warning(
            f"Treatment group column '{treatment_column}' not found. "
            "Campaign effectiveness operating under unsegmented baseline assumption."
        )
        total_sample = len(matched_df)
        overall_churn_rate = float(matched_df[target_column].mean())
        return {
            "campaign_data_available": False,
            "assumptions_documented": (
                "Treatment and control group indicators were not present in delayed labels. "
                "No business conclusions regarding campaign lift can be drawn."
            ),
            "sample_size": total_sample,
            "overall_churn_rate": round(overall_churn_rate, 4),
            "treatment_group": None,
            "control_group": None,
            "financial_summary": None,
        }

    group_series = matched_df[treatment_column].astype(str).str.strip().str.lower()
    treatment_mask = group_series.isin(
        ["treatment", "treated", "1", "true", "targeted"]
    )
    control_mask = group_series.isin(["control", "holdout", "0", "false", "untargeted"])

    treatment_df = matched_df[treatment_mask]
    control_df = matched_df[control_mask]

    n_treatment = len(treatment_df)
    n_control = len(control_df)

    if n_treatment == 0 or n_control == 0:
        logger.warning(
            f"Insufficient records for comparison: n_treatment={n_treatment}, "
            f"n_control={n_control}"
        )
        return {
            "campaign_data_available": False,
            "assumptions_documented": (
                f"Evaluation requires non-empty treatment (found {n_treatment}) "
                f"and control (found {n_control}) groups."
            ),
            "sample_size": len(matched_df),
            "n_treatment": n_treatment,
            "n_control": n_control,
        }

    cr_treatment = float(treatment_df[target_column].mean())
    cr_control = float(control_df[target_column].mean())

    absolute_reduction = cr_control - cr_treatment
    relative_reduction = (absolute_reduction / cr_control) if cr_control > 0 else 0.0

    incremental_retained = absolute_reduction * n_treatment

    total_contact_cost = n_treatment * unit_contact_cost
    gross_value_saved = incremental_retained * unit_customer_value
    net_campaign_value = gross_value_saved - total_contact_cost
    campaign_roi = (
        (net_campaign_value / total_contact_cost) if total_contact_cost > 0 else 0.0
    )

    return {
        "campaign_data_available": True,
        "treatment_group": {
            "sample_size": n_treatment,
            "churn_count": int(treatment_df[target_column].sum()),
            "churn_rate": round(cr_treatment, 4),
        },
        "control_group": {
            "sample_size": n_control,
            "churn_count": int(control_df[target_column].sum()),
            "churn_rate": round(cr_control, 4),
        },
        "effectiveness": {
            "absolute_churn_reduction": round(absolute_reduction, 4),
            "relative_churn_reduction_pct": round(relative_reduction * 100, 2),
            "estimated_incremental_retained_customers": round(incremental_retained, 2),
        },
        "financial_summary": {
            "unit_contact_cost_usd": unit_contact_cost,
            "unit_customer_value_usd": unit_customer_value,
            "total_campaign_cost_usd": round(total_contact_cost, 2),
            "gross_retained_margin_usd": round(gross_value_saved, 2),
            "net_campaign_value_usd": round(net_campaign_value, 2),
            "campaign_roi_percent": round(campaign_roi * 100, 2),
        },
        "assumptions_documented": (
            f"Campaign evaluation assumes unit contact cost of ${unit_contact_cost:.2f} "
            f"and customer retention value of ${unit_customer_value:.2f} per prevented churn. "
            "Treatment/control comparison assumes randomized or propensity-matched holdout."
        ),
    }
