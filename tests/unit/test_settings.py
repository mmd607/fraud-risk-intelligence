from __future__ import annotations

from pathlib import Path

import pytest

from fraud_risk.config import Environment, LogFormat, get_settings, load_settings
from fraud_risk.core.exceptions import ConfigurationError


def test_defaults_are_safe_and_local() -> None:
    settings = load_settings()
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.log_level == "INFO"
    assert settings.log_format is LogFormat.JSON
    assert settings.random_seed == 42
    assert settings.data_dir == Path("data")
    assert settings.raw_data_dir == Path("data") / "raw"


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_RISK_LOG_LEVEL", "debug")
    monkeypatch.setenv("FRAUD_RISK_RANDOM_SEED", "7")
    settings = load_settings()
    assert settings.log_level == "DEBUG"
    assert settings.random_seed == 7


def test_invalid_value_raises_configuration_error_naming_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRAUD_RISK_RANDOM_SEED", "-1")
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings()
    assert "FRAUD_RISK_RANDOM_SEED" in str(excinfo.value)


def test_error_message_never_echoes_input_value(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_like = "s3cr3t-VALUE-should-not-leak"
    monkeypatch.setenv("FRAUD_RISK_LOG_LEVEL", secret_like)
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings()
    assert secret_like not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_settings_are_immutable() -> None:
    settings = load_settings()
    with pytest.raises(ValueError, match="frozen"):
        settings.random_seed = 1  # type: ignore[misc]
