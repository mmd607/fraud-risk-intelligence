from __future__ import annotations

import pytest

from fraud_risk.core.exceptions import (
    ConfigurationError,
    DataError,
    DependencyUnavailableError,
    FraudRiskError,
    ModelError,
)


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (ConfigurationError, "CONFIGURATION_ERROR"),
        (DataError, "DATA_ERROR"),
        (ModelError, "MODEL_ERROR"),
    ],
)
def test_subclasses_have_stable_codes(error_type: type[FraudRiskError], code: str) -> None:
    error = error_type("something went wrong")
    assert isinstance(error, FraudRiskError)
    assert error.code == code
    assert str(error) == f"[{code}] something went wrong"


def test_code_can_be_overridden_per_instance() -> None:
    error = DataError("bad", code="DATA_CUSTOM")
    assert error.code == "DATA_CUSTOM"
    assert DataError("other").code == "DATA_ERROR"


def test_dependency_error_is_actionable() -> None:
    error = DependencyUnavailableError("shap", extra="explain", purpose="SHAP explanations")
    assert 'pip install "fraud-risk[explain]"' in str(error)
    assert error.package == "shap"
