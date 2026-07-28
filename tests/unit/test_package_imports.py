"""Smoke tests for the public package boundary."""

import churn_prediction


def test_package_exposes_version() -> None:
    """The installed package provides version metadata."""
    assert churn_prediction.__version__ == "0.1.0"
