from __future__ import annotations

import io
import json
import logging

import pytest

from fraud_risk.core.logging import REDACTED, configure_logging, get_logger, log_event


def test_json_log_line_contains_event_and_context() -> None:
    stream = io.StringIO()
    configure_logging("INFO", json_format=True, stream=stream)
    log_event(get_logger("unit"), "dataset_loaded", rows=10, source="synthetic")
    record = json.loads(stream.getvalue().strip())
    assert record["event"] == "dataset_loaded"
    assert record["rows"] == 10
    assert record["logger"] == "fraud_risk.unit"
    assert record["level"] == "INFO"


def test_sensitive_context_is_redacted() -> None:
    stream = io.StringIO()
    configure_logging("INFO", json_format=True, stream=stream)
    log_event(get_logger("unit"), "request", api_key="abc123", transaction_payload="row")
    output = stream.getvalue()
    assert "abc123" not in output
    assert "row" not in json.loads(output)["transaction_payload"]
    assert json.loads(output)["api_key"] == REDACTED


def test_level_filtering() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", json_format=True, stream=stream)
    log_event(get_logger("unit"), "ignored")
    assert stream.getvalue() == ""


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("LOUD")


def test_configure_logging_is_idempotent() -> None:
    logger = configure_logging("INFO")
    logger = configure_logging("INFO")
    assert len(logger.handlers) == 1
    assert logger.propagate is False
    assert logging.getLogger().handlers is not logger.handlers
