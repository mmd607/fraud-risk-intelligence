"""Typed data contract for the credit-card transactions dataset.

A contract states what *valid* data looks like. It never changes data; validation
(:mod:`fraud_risk.data.validation`) compares a frame with a contract and reports issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from fraud_risk.data.schema import (
    AMOUNT_COLUMN,
    PCA_COLUMNS,
    TARGET_COLUMN,
    TARGET_VALUES,
    TIME_COLUMN,
)

CONTRACT_VERSION = "1.0.0"


class ColumnKind(StrEnum):
    """Logical column type."""

    FLOAT = "float"
    BINARY_LABEL = "binary_label"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Expectations for one column."""

    name: str
    kind: ColumnKind
    description: str
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True, slots=True)
class DataContract:
    """Dataset-level expectations.

    Attributes:
        positive_rate_bounds: inclusive range outside of which the fraud rate is flagged as
            suspicious (warning). The defaults bracket the known ~0.17 % rate generously; they are
            sanity bounds, not a statement about real-world fraud rates.
        max_duplicate_fraction: duplicate-row share above which duplicates are a warning rather
            than info. The default ``0.0`` warns on any duplicate because duplicates can straddle
            train/test splits and inflate evaluation.
        expect_time_sorted: whether ``Time`` is expected to be non-decreasing (true for the
            published file; violations are informational).
    """

    columns: tuple[ColumnSpec, ...]
    target_column: str = TARGET_COLUMN
    target_domain: frozenset[int] = field(default=TARGET_VALUES)
    min_rows: int = 1
    positive_rate_bounds: tuple[float, float] = (0.0001, 0.05)
    max_duplicate_fraction: float = 0.0
    expect_time_sorted: bool = True
    version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.columns]
        if len(names) != len(set(names)):
            msg = "Contract column names must be unique."
            raise ValueError(msg)
        if self.target_column not in names:
            msg = f"Target column '{self.target_column}' must be declared in the contract."
            raise ValueError(msg)
        low, high = self.positive_rate_bounds
        if not 0.0 <= low <= high <= 1.0:
            msg = "positive_rate_bounds must satisfy 0 <= low <= high <= 1."
            raise ValueError(msg)
        if self.min_rows < 1:
            msg = "min_rows must be >= 1."
            raise ValueError(msg)
        if not 0.0 <= self.max_duplicate_fraction <= 1.0:
            msg = "max_duplicate_fraction must be within [0, 1]."
            raise ValueError(msg)

    @property
    def column_names(self) -> tuple[str, ...]:
        """All declared column names in contract order."""
        return tuple(spec.name for spec in self.columns)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Declared columns except the target."""
        return tuple(name for name in self.column_names if name != self.target_column)

    def spec(self, name: str) -> ColumnSpec:
        """Return the spec for ``name``."""
        for column in self.columns:
            if column.name == name:
                return column
        msg = f"Column '{name}' is not declared in the contract."
        raise KeyError(msg)


def creditcard_contract() -> DataContract:
    """Default contract for the Kaggle ``creditcard.csv`` file."""
    columns = (
        ColumnSpec(
            TIME_COLUMN,
            ColumnKind.FLOAT,
            "Seconds since the first transaction in the dataset (not a calendar time).",
            min_value=0.0,
        ),
        *(
            ColumnSpec(name, ColumnKind.FLOAT, "Anonymised PCA component; no semantic meaning.")
            for name in PCA_COLUMNS
        ),
        ColumnSpec(AMOUNT_COLUMN, ColumnKind.FLOAT, "Transaction amount.", min_value=0.0),
        ColumnSpec(TARGET_COLUMN, ColumnKind.BINARY_LABEL, "0 = legitimate, 1 = fraud."),
    )
    return DataContract(columns=columns)
