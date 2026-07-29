"""Integration tests for baseline evaluation contract."""

import json
import tempfile
from pathlib import Path

import yaml

from churn_prediction.evaluation.evaluator import evaluate_model
from churn_prediction.models.trainer import train_baseline


def test_evaluate_persisted_baseline_model_end_to_end() -> None:
    """Verify evaluation pipeline runs end-to-end on persisted baseline artifact.

    Emits all required metrics, plots, and JSON reports deterministically.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"

        # 1. Train baseline model to create persisted artifact
        train_config = {
            "schema_version": "1.0.0",
            "model_name": "baseline_logistic_regression",
            "data": {
                "raw_data_path": "Telco-Customer-Churn.csv",
                "contract_config_path": "configs/data_contract.yaml",
            },
            "split": {
                "train_size": 0.70,
                "val_size": 0.15,
                "test_size": 0.15,
                "random_seed": 42,
                "stratify": True,
            },
            "features": {
                "primary_key": "customerID",
                "target_column": "Churn",
                "target_positive_value": "Yes",
                "excluded_features": ["customerID", "gender", "SeniorCitizen"],
                "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
                "categorical_features": [
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "MultipleLines",
                    "InternetService",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "Contract",
                    "PaperlessBilling",
                    "PaymentMethod",
                ],
            },
            "model": {
                "type": "LogisticRegression",
                "hyperparameters": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": 42,
                },
            },
            "artifacts": {
                "output_dir": str(models_dir),
                "pipeline_filename": "baseline_pipeline.joblib",
                "metadata_filename": "baseline_metadata.json",
            },
        }

        train_config_path = tmp_path / "train_config.yaml"
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(train_config, f)

        pipeline, metadata, artifact_paths = train_baseline(
            config_path=train_config_path
        )
        assert artifact_paths["pipeline_path"].is_file()

        # 2. Prepare evaluation configuration file
        eval_config = {
            "schema_version": "1.0.0",
            "model_name": "baseline_logistic_regression",
            "evaluation": {
                "evaluation_threshold": 0.50,
                "campaign_capacity": 0.10,
                "evaluation_split": "test",
                "n_bins": 10,
                "thresholds": [0.2, 0.4, 0.5, 0.6, 0.8],
            },
            "paths": {
                "model_dir": str(models_dir),
                "pipeline_filename": "baseline_pipeline.joblib",
                "metadata_filename": "baseline_metadata.json",
                "raw_data_path": "Telco-Customer-Churn.csv",
                "contract_config_path": "configs/data_contract.yaml",
                "output_dir": str(reports_dir),
            },
            "split": {
                "train_size": 0.70,
                "val_size": 0.15,
                "test_size": 0.15,
                "random_seed": 42,
                "stratify": True,
            },
            "features": train_config["features"],
        }

        eval_config_path = tmp_path / "eval_config.yaml"
        with open(eval_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(eval_config, f)

        # 3. Run evaluation pipeline
        summary, generated_artifacts = evaluate_model(config_path=eval_config_path)

        # 4. Verify evaluation metrics and summary
        assert summary["model_name"] == "baseline_logistic_regression"
        assert summary["evaluation_split"] == "test"
        assert summary["sample_size"] > 0
        assert summary["primary_metric"]["name"] == "PR-AUC"
        assert 0.0 <= summary["primary_metric"]["value"] <= 1.0

        metrics = summary["metrics"]
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0

        capacity = summary["campaign_capacity"]
        assert capacity["campaign_capacity_fraction"] == 0.10
        assert capacity["num_targeted_customers"] > 0
        assert 0.0 <= capacity["precision_at_capacity"] <= 1.0
        assert 0.0 <= capacity["recall_at_capacity"] <= 1.0

        assert len(summary["threshold_analysis"]) == 5
        assert len(summary["feature_importance"]) > 0

        # 5. Verify emitted artifact files exist and are non-empty
        for artifact_key, artifact_path in generated_artifacts.items():
            path = Path(artifact_path)
            assert path.is_file(), f"Artifact {artifact_key} missing at {path}"
            assert path.stat().st_size > 0, f"Artifact {artifact_key} is empty"

        # Verify JSON report parseability
        with open(reports_dir / "evaluation_metrics.json", "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
            assert saved_metrics["primary_metric"]["name"] == "PR-AUC"


def test_evaluation_threshold_is_policy_configured_not_hardcoded() -> None:
    """Verify evaluation threshold is policy-configured."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"

        train_config = {
            "schema_version": "1.0.0",
            "model_name": "baseline_logistic_regression",
            "data": {
                "raw_data_path": "Telco-Customer-Churn.csv",
                "contract_config_path": "configs/data_contract.yaml",
            },
            "split": {
                "train_size": 0.70,
                "val_size": 0.15,
                "test_size": 0.15,
                "random_seed": 42,
                "stratify": True,
            },
            "features": {
                "primary_key": "customerID",
                "target_column": "Churn",
                "target_positive_value": "Yes",
                "excluded_features": ["customerID", "gender", "SeniorCitizen"],
                "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
                "categorical_features": [
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "MultipleLines",
                    "InternetService",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "Contract",
                    "PaperlessBilling",
                    "PaymentMethod",
                ],
            },
            "model": {
                "type": "LogisticRegression",
                "hyperparameters": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": 42,
                },
            },
            "artifacts": {
                "output_dir": str(models_dir),
                "pipeline_filename": "baseline_pipeline.joblib",
                "metadata_filename": "baseline_metadata.json",
            },
        }

        train_config_path = tmp_path / "train_config.yaml"
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(train_config, f)

        train_baseline(config_path=train_config_path)

        # Evaluation Config 1 with policy threshold = 0.30
        eval_config_30 = {
            "schema_version": "1.0.0",
            "model_name": "baseline_logistic_regression",
            "evaluation": {
                "evaluation_threshold": 0.30,
                "campaign_capacity": 0.10,
                "evaluation_split": "test",
                "n_bins": 10,
            },
            "paths": {
                "model_dir": str(models_dir),
                "pipeline_filename": "baseline_pipeline.joblib",
                "metadata_filename": "baseline_metadata.json",
                "raw_data_path": "Telco-Customer-Churn.csv",
                "contract_config_path": "configs/data_contract.yaml",
                "output_dir": str(reports_dir / "t30"),
            },
            "split": train_config["split"],
            "features": train_config["features"],
        }
        cfg_path_30 = tmp_path / "eval_30.yaml"
        with open(cfg_path_30, "w", encoding="utf-8") as f:
            yaml.safe_dump(eval_config_30, f)

        # Evaluation Config 2 with policy threshold = 0.70
        eval_config_70 = {
            "schema_version": "1.0.0",
            "model_name": "baseline_logistic_regression",
            "evaluation": {
                "evaluation_threshold": 0.70,
                "campaign_capacity": 0.10,
                "evaluation_split": "test",
                "n_bins": 10,
            },
            "paths": {
                "model_dir": str(models_dir),
                "pipeline_filename": "baseline_pipeline.joblib",
                "metadata_filename": "baseline_metadata.json",
                "raw_data_path": "Telco-Customer-Churn.csv",
                "contract_config_path": "configs/data_contract.yaml",
                "output_dir": str(reports_dir / "t70"),
            },
            "split": train_config["split"],
            "features": train_config["features"],
        }
        cfg_path_70 = tmp_path / "eval_70.yaml"
        with open(cfg_path_70, "w", encoding="utf-8") as f:
            yaml.safe_dump(eval_config_70, f)

        summary_30, _ = evaluate_model(config_path=cfg_path_30)
        summary_70, _ = evaluate_model(config_path=cfg_path_70)

        # Verify thresholds match configuration
        assert summary_30["metrics"]["threshold"] == 0.30
        assert summary_70["metrics"]["threshold"] == 0.70

        # Lower threshold yields higher recall and lower precision than higher threshold
        assert summary_30["metrics"]["recall"] > summary_70["metrics"]["recall"]
        assert summary_30["metrics"]["precision"] < summary_70["metrics"]["precision"]
