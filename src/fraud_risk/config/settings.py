"""Typed application settings loaded from environment variables.

All settings have safe local defaults so the package imports and runs without any secrets or
external services. Environment variables use the ``FRAUD_RISK_`` prefix, for example
``FRAUD_RISK_LOG_LEVEL=DEBUG``. A local ``.env`` file is read when present (it is ignored by Git).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fraud_risk.core.exceptions import ConfigurationError

ENV_PREFIX = "FRAUD_RISK_"
_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Environment(StrEnum):
    """Deployment environment name. Only affects defaults/guards, never model behaviour."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    """Log output format."""

    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    """Validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    dataset_path: Path = Path("data/raw/creditcard.csv")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in _VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(_VALID_LOG_LEVELS))
            msg = f"must be one of {allowed}"
            raise ValueError(msg)
        return normalized

    @property
    def raw_data_dir(self) -> Path:
        """Directory holding manually downloaded raw datasets."""
        return self.data_dir / "raw"


def _format_validation_error(error: ValidationError) -> str:
    """Render a pydantic error without echoing input values (they may be secrets)."""
    problems = []
    for item in error.errors(include_input=False, include_url=False):
        field = ".".join(str(part) for part in item["loc"]) or "<settings>"
        env_name = f"{ENV_PREFIX}{field.upper()}"
        problems.append(f"{env_name}: {item['msg']}")
    return "Invalid configuration. Fix the following environment variables: " + "; ".join(problems)


def load_settings(**overrides: Any) -> Settings:  # noqa: ANN401 - forwarded to pydantic
    """Build settings from the environment plus explicit overrides.

    Raises:
        ConfigurationError: when any value is invalid. Input values are never included in the
            message so that secrets cannot leak into logs or terminals.
    """
    try:
        return Settings(**overrides)
    except ValidationError as error:
        raise ConfigurationError(_format_validation_error(error)) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide cached settings. Call ``get_settings.cache_clear()`` in tests."""
    return load_settings()
