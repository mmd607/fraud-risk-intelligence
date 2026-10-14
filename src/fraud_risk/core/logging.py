"""Structured, privacy-safe application logging.

Log records are emitted as single-line JSON (or a readable key=value console format). Callers
attach structured context with :func:`log_event`; context keys that look like secrets or raw
transaction payloads are redacted before formatting. The package never logs raw transaction
rows: log aggregates, identifiers, and stable error codes instead.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TextIO, TypeAlias

ROOT_LOGGER_NAME = "fraud_risk"
CONTEXT_ATTR = "fraud_risk_context"
REDACTED = "***REDACTED***"

# Keys whose values must never reach a log sink. Matching is case-insensitive on substrings.
SENSITIVE_KEY_MARKERS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "payload",
    "transaction",
    "features",
    "record",
)

ContextValue: TypeAlias = str | int | float | bool | list[str] | list[int] | list[float] | None


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SENSITIVE_KEY_MARKERS)


def redact_context(context: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of ``context`` with sensitive keys replaced by a redaction marker."""
    return {key: (REDACTED if _is_sensitive(key) else value) for key, value in context.items()}


def _record_context(record: logging.LogRecord) -> dict[str, object]:
    context = getattr(record, CONTEXT_ATTR, None)
    if isinstance(context, Mapping):
        return redact_context(context)
    return {}


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(_record_context(record))
        if record.exc_info and record.exc_info[0] is not None:
            # Exception type only: messages and tracebacks may embed input values.
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str, sort_keys=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single-line format for local development."""

    def format(self, record: logging.LogRecord) -> str:
        context = _record_context(record)
        rendered = " ".join(f"{key}={value}" for key, value in context.items())
        base = f"{record.levelname:<8} {record.name}: {record.getMessage()}"
        if record.exc_info and record.exc_info[0] is not None:
            base = f"{base} exception_type={record.exc_info[0].__name__}"
        return f"{base} {rendered}".rstrip()


def configure_logging(
    level: str = "INFO",
    *,
    json_format: bool = True,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the package root logger idempotently and return it.

    Only the ``fraud_risk`` logger hierarchy is configured; the global root logger is untouched
    so that host applications keep control of their own logging.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        msg = f"Unknown log level: {level!r}"
        raise ValueError(msg)  # noqa: TRY004 - a bad level name is a value error
    logger.setLevel(numeric_level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if json_format else ConsoleFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``fraud_risk`` hierarchy."""
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **context: ContextValue,
) -> None:
    """Log a named event with structured, redacted context fields."""
    logger.log(level, event, extra={CONTEXT_ATTR: context})
