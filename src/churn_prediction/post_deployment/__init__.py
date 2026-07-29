"""Post-deployment learning evaluation package."""

from churn_prediction.post_deployment.campaign import (
    evaluate_campaign_effectiveness,
)
from churn_prediction.post_deployment.config import (
    get_default_post_deployment_config_path,
    load_post_deployment_config,
)
from churn_prediction.post_deployment.delayed_labels import (
    evaluate_delayed_predictions,
    generate_synthetic_delayed_labels,
    load_and_match_delayed_labels,
)
from churn_prediction.post_deployment.orchestrator import (
    run_post_deployment_evaluation,
)
from churn_prediction.post_deployment.report import (
    generate_post_deployment_report,
)
from churn_prediction.post_deployment.retraining_decision import (
    make_retraining_decision,
)

__all__ = [
    "get_default_post_deployment_config_path",
    "load_post_deployment_config",
    "load_and_match_delayed_labels",
    "evaluate_delayed_predictions",
    "generate_synthetic_delayed_labels",
    "evaluate_campaign_effectiveness",
    "make_retraining_decision",
    "generate_post_deployment_report",
    "run_post_deployment_evaluation",
]
