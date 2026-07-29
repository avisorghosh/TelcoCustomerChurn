"""Integration test for Docker container execution and endpoint verification."""

import json
import shutil
import subprocess
import time
import urllib.request

import pytest


def is_docker_available() -> bool:
    """Check if docker CLI and daemon are operational."""
    if not shutil.which("docker"):
        return False
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return res.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not available")
def test_docker_container_inference_lifecycle() -> None:
    """Verify Docker container responds to health, ready, and predict endpoints."""
    container_name = f"test-churn-api-{int(time.time())}"
    host_port = "8089"

    # Launch container in detached mode
    run_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{host_port}:8000",
        "-e",
        "CHURN_DECISION_THRESHOLD=0.50",
        "telco-churn-api:latest",
    ]
    res = subprocess.run(run_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip(f"Docker container run failed: {res.stderr}")

    try:
        # Poll health endpoint until service is ready
        base_url = f"http://127.0.0.1:{host_port}"
        health_ok = False
        for _ in range(15):
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        health_ok = True
                        break
            except Exception:
                time.sleep(1)

        assert health_ok, "Container failed health check within startup period."

        # Verify readiness endpoint
        with urllib.request.urlopen(f"{base_url}/ready", timeout=2) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "ready"
            assert data.get("model_loaded") is True

        # Verify prediction endpoint
        payload = {
            "customerID": "7590-VHVEG",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 55.85,
            "TotalCharges": 670.20,
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            pred_data = json.loads(resp.read().decode("utf-8"))
            assert "churn_probability" in pred_data
            assert "predicted_class" in pred_data
            assert "model_version" in pred_data
            assert 0.0 <= pred_data["churn_probability"] <= 1.0

    finally:
        # Cleanup container
        subprocess.run(
            ["docker", "stop", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["docker", "rm", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
