"""Test-phase T00: independent checks of the foundation contract."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import fraud_risk
from fraud_risk.config import Environment, LogFormat, Settings, load_settings
from fraud_risk.core.exceptions import ConfigurationError, FraudRiskError
from fraud_risk.core.logging import (
    REDACTED,
    configure_logging,
    get_logger,
    log_event,
    redact_context,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- packaging -------------------------------------------------------------------------------


def test_package_exposes_version() -> None:
    assert fraud_risk.__version__ == "0.1.0"


def test_py_typed_marker_is_shipped() -> None:
    assert (Path(fraud_risk.__file__).parent / "py.typed").is_file()


def test_package_imports_in_clean_process_without_env_or_data(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("FRAUD_RISK_")}
    code = (
        "import fraud_risk, fraud_risk.config, fraud_risk.core.logging\n"
        "from fraud_risk.config import get_settings\n"
        "print(get_settings().environment)\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and arguments
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "development"


def test_quality_tooling_is_configured() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["mypy"]["strict"] is True
    assert "ruff" in config["tool"]
    assert "pytest" in config["tool"]
    dev_extras = " ".join(config["project"]["optional-dependencies"]["dev"])
    for tool in ("pytest", "ruff", "mypy"):
        assert tool in dev_extras


@pytest.mark.parametrize("pattern", ["data/", "*.csv", "artifacts/", ".env", "mlruns/", "*.joblib"])
def test_gitignore_excludes_data_secrets_and_artifacts(pattern: str) -> None:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert pattern in lines


def _env_example_entries() -> list[tuple[str, str]]:
    entries = []
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            entries.append((key, value))
    return entries


def test_env_example_only_lists_known_settings() -> None:
    known = {f"FRAUD_RISK_{name.upper()}" for name in Settings.model_fields}
    keys = {key for key, _ in _env_example_entries()}
    assert keys, ".env.example should document settings"
    assert keys <= known


def test_env_example_values_are_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _env_example_entries():
        monkeypatch.setenv(key, value)
    load_settings()


# --- settings ----------------------------------------------------------------------------------


def test_dotenv_file_in_working_directory_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("FRAUD_RISK_ENVIRONMENT=test\n", encoding="utf-8")
    assert load_settings().environment is Environment.TEST


def test_environment_variable_takes_precedence_over_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("FRAUD_RISK_LOG_LEVEL=ERROR\n", encoding="utf-8")
    monkeypatch.setenv("FRAUD_RISK_LOG_LEVEL", "WARNING")
    assert load_settings().log_level == "WARNING"


def test_unknown_prefixed_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_RISK_NOT_A_SETTING", "x")
    load_settings()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("FRAUD_RISK_ENVIRONMENT", "staging"),
        ("FRAUD_RISK_LOG_FORMAT", "xml"),
        ("FRAUD_RISK_RANDOM_SEED", "not-a-number"),
        ("FRAUD_RISK_RANDOM_SEED", str(2**32)),
        ("FRAUD_RISK_LOG_LEVEL", ""),
    ],
)
def test_invalid_values_fail_with_variable_name(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings()
    assert variable in str(excinfo.value)
    assert excinfo.value.code == "CONFIGURATION_ERROR"


def test_multiple_invalid_values_are_all_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_RISK_ENVIRONMENT", "staging")
    monkeypatch.setenv("FRAUD_RISK_RANDOM_SEED", "-5")
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings()
    assert "FRAUD_RISK_ENVIRONMENT" in str(excinfo.value)
    assert "FRAUD_RISK_RANDOM_SEED" in str(excinfo.value)


def test_explicit_overrides_are_validated() -> None:
    assert load_settings(log_format="console").log_format is LogFormat.CONSOLE
    with pytest.raises(ConfigurationError):
        load_settings(random_seed=-3)


def test_log_level_is_normalized() -> None:
    assert load_settings(log_level="  warning ").log_level == "WARNING"


# --- logging -----------------------------------------------------------------------------------


def test_console_format_is_single_line_key_value() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", json_format=False, stream=stream)
    log_event(get_logger("t00"), "ready", rows=3)
    line = stream.getvalue().strip()
    assert "\n" not in line
    assert "ready" in line
    assert "rows=3" in line


def test_json_exceptions_log_type_but_not_message() -> None:
    stream = io.StringIO()
    logger = configure_logging("INFO", json_format=True, stream=stream)
    try:
        raise ValueError("Amount=999.99 leaked")  # noqa: TRY301
    except ValueError:
        logger.exception("failure")
    record = json.loads(stream.getvalue())
    assert record["exception_type"] == "ValueError"
    assert "999.99" not in stream.getvalue()


def test_console_exceptions_log_type_only() -> None:
    stream = io.StringIO()
    logger = configure_logging("INFO", json_format=False, stream=stream)
    try:
        raise RuntimeError("V1=0.123 leaked")  # noqa: TRY301
    except RuntimeError:
        logger.exception("failure")
    assert "exception_type=RuntimeError" in stream.getvalue()
    assert "0.123" not in stream.getvalue()


@pytest.mark.parametrize(
    "key", ["password", "DB_SECRET", "authToken", "Authorization", "raw_features", "record"]
)
def test_redaction_is_case_insensitive_substring(key: str) -> None:
    assert redact_context({key: "value"}) == {key: REDACTED}


def test_redaction_keeps_safe_keys_and_does_not_mutate_input() -> None:
    context = {"rows": 5, "api_key": "k"}
    redacted = redact_context(context)
    assert redacted == {"rows": 5, "api_key": REDACTED}
    assert context["api_key"] == "k"


def test_get_logger_namespaces_names() -> None:
    assert get_logger("data").name == "fraud_risk.data"
    assert get_logger("fraud_risk.data").name == "fraud_risk.data"
    assert get_logger("fraud_risk").name == "fraud_risk"


def test_configure_logging_leaves_root_logger_untouched() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    configure_logging("INFO")
    assert root.handlers == before


def test_all_domain_errors_are_catchable_by_base_class() -> None:
    with pytest.raises(FraudRiskError):
        load_settings(random_seed=-1)
