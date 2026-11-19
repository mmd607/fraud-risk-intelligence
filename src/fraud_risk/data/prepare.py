"""Convenience composition: load the CSV and refuse to continue if validation fails."""

from __future__ import annotations

from pathlib import Path

from fraud_risk.core.exceptions import DataValidationError
from fraud_risk.data.contracts import DataContract
from fraud_risk.data.ingestion import IngestionResult, load_transactions
from fraud_risk.data.validation import ValidationReport, validate_dataframe


def load_validated_dataset(
    path: Path, contract: DataContract | None = None
) -> tuple[IngestionResult, ValidationReport]:
    """Load ``path`` and validate it; raise ``DataValidationError`` on error-severity issues."""
    result = load_transactions(path)
    report = validate_dataframe(result.frame, contract)
    if not report.is_valid:
        codes = sorted({issue.code.value for issue in report.errors})
        msg = f"Dataset failed validation with error codes {codes}; run `fraud-risk validate`."
        raise DataValidationError(msg)
    return result, report
