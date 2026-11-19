"""Machine-readable data-quality report: validation issues plus aggregate statistics.

The report contains per-column aggregates (count, missing, mean, std, quantiles, min, max) and
target counts only. It never contains individual transaction rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_risk.data.contracts import DataContract, creditcard_contract
from fraud_risk.data.validation import ValidationReport, validate_dataframe

REPORT_SCHEMA_VERSION = "1.0.0"
_QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)


@dataclass(frozen=True, slots=True)
class ColumnSummary:
    """Aggregate statistics for one numeric column (finite values only)."""

    name: str
    count: int
    missing: int
    non_finite: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    quantiles: dict[str, float | None]

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "name": self.name,
            "count": self.count,
            "missing": self.missing,
            "non_finite": self.non_finite,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "quantiles": self.quantiles,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Validation outcome plus aggregate statistics."""

    validation: ValidationReport
    column_summaries: tuple[ColumnSummary, ...]
    target_counts: dict[str, int]
    dataset_fingerprint: str | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "dataset_fingerprint": self.dataset_fingerprint,
            "validation": self.validation.to_dict(),
            "target_counts": self.target_counts,
            "columns": [summary.to_dict() for summary in self.column_summaries],
        }

    def write_json(self, path: Path) -> Path:
        """Write the report to ``path`` (parents created) and return it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def _optional(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def summarize_column(series: pd.Series) -> ColumnSummary:
    """Compute aggregate statistics for a numeric series, ignoring non-finite values."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    missing = int(np.isnan(values).sum())
    non_finite = int(np.isinf(values).sum())
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ColumnSummary(
            str(series.name), 0, missing, non_finite, None, None, None, None,
            {f"p{int(q * 100):02d}": None for q in _QUANTILES},
        )  # fmt: skip
    quantile_values = np.quantile(finite, _QUANTILES)
    return ColumnSummary(
        name=str(series.name),
        count=int(finite.size),
        missing=missing,
        non_finite=non_finite,
        mean=_optional(float(finite.mean())),
        std=_optional(float(finite.std(ddof=1))) if finite.size > 1 else None,
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        quantiles={
            f"p{int(q * 100):02d}": float(v) for q, v in zip(_QUANTILES, quantile_values, strict=True)
        },
    )


def build_quality_report(
    frame: pd.DataFrame,
    contract: DataContract | None = None,
    *,
    dataset_fingerprint: str | None = None,
) -> QualityReport:
    """Validate ``frame`` and summarise its numeric contract columns. Never mutates ``frame``."""
    contract = contract or creditcard_contract()
    validation = validate_dataframe(frame, contract)
    summaries = tuple(
        summarize_column(frame[name])
        for name in contract.column_names
        if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name])
    )
    target_counts: dict[str, int] = {}
    if contract.target_column in frame.columns:
        counts = frame[contract.target_column].value_counts(dropna=False)
        target_counts = {str(label): int(count) for label, count in sorted(counts.items(), key=str)}
    return QualityReport(
        validation=validation,
        column_summaries=summaries,
        target_counts=target_counts,
        dataset_fingerprint=dataset_fingerprint,
    )
