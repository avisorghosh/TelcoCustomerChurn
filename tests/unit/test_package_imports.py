"""Smoke tests for the public package boundary."""

import churn_prediction


def test_package_exposes_version() -> None:
    """The installed package provides version metadata."""
    assert churn_prediction.__version__ == "0.1.0"


def test_submodules_are_importable() -> None:
    """All architectural package submodules can be imported."""
    import churn_prediction.api  # noqa: F401
    import churn_prediction.data  # noqa: F401
    import churn_prediction.features  # noqa: F401
    import churn_prediction.models  # noqa: F401
    import churn_prediction.monitoring  # noqa: F401
    import churn_prediction.settings  # noqa: F401
