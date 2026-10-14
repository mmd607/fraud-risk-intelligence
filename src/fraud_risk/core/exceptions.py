"""Domain exception hierarchy.

Every error raised deliberately by the package derives from :class:`FraudRiskError` and
carries a stable machine-readable ``code``. Messages must be actionable and must never
contain secrets or raw transaction records.
"""

from __future__ import annotations


class FraudRiskError(Exception):
    """Base class for all deliberate errors raised by ``fraud_risk``."""

    code: str = "FRAUD_RISK_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigurationError(FraudRiskError):
    """Configuration is missing, malformed, or violates a constraint."""

    code = "CONFIGURATION_ERROR"


class DataError(FraudRiskError):
    """Input data is missing, unreadable, or does not satisfy its contract."""

    code = "DATA_ERROR"


class DatasetNotFoundError(DataError):
    """The configured dataset file does not exist or is not a regular file."""

    code = "DATA_NOT_FOUND"


class DataReadError(DataError):
    """The dataset file exists but cannot be read or parsed."""

    code = "DATA_UNREADABLE"


class SchemaError(DataError):
    """The dataset structure (columns, types, target domain) violates the expected schema."""

    code = "DATA_SCHEMA_INVALID"


class ModelError(FraudRiskError):
    """A model or model artifact cannot be trained, loaded, or used."""

    code = "MODEL_ERROR"


class DependencyUnavailableError(FraudRiskError):
    """An optional dependency required for the requested operation is not installed."""

    code = "DEPENDENCY_UNAVAILABLE"

    def __init__(self, package: str, *, extra: str | None = None, purpose: str = "") -> None:
        hint = f'pip install "fraud-risk[{extra}]"' if extra else f"pip install {package}"
        purpose_text = f" for {purpose}" if purpose else ""
        super().__init__(
            f"Optional dependency '{package}' is required{purpose_text}. Install it with: {hint}"
        )
        self.package = package
