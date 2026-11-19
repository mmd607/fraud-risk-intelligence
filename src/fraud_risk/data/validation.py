"""Contract validation producing deterministic, auditable issues.

Validation never mutates or drops data. Each finding is a :class:`ValidationIssue` with a stable
code, a severity, an actionable message, and (where relevant) the column and a count. Messages
contain counts and column names only, never cell values.

Severity semantics:
    * ``error``  – the frame must not be used for training/evaluation until resolved.
    * ``warning`` – usable, but a human should review (e.g. duplicates, unusual fraud rate).
    * ``info``    – noteworthy facts that are not problems (e.g. ordering).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from fraud_risk.data.contracts import ColumnKind, DataContract, creditcard_contract
from fraud_risk.data.schema import TIME_COLUMN


class Severity(StrEnum):
    """Issue severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueCode(StrEnum):
    """Stable issue codes. Never renumber; add new codes at the end."""

    EMPTY_DATASET = "DQ001"
    MISSING_COLUMN = "DQ002"
    UNEXPECTED_COLUMN = "DQ003"
    NON_NUMERIC_COLUMN = "DQ004"
    NULL_VALUES = "DQ005"
    NON_FINITE_VALUES = "DQ006"
    TARGET_OUT_OF_DOMAIN = "DQ007"
    SINGLE_CLASS_TARGET = "DQ008"
    SUSPICIOUS_POSITIVE_RATE = "DQ009"
    DUPLICATE_ROWS = "DQ010"
    VALUE_OUT_OF_RANGE = "DQ011"
    TIME_NOT_SORTED = "DQ012"
    BELOW_MIN_ROWS = "DQ013"
    CONFLICTING_DUPLICATE_LABELS = "DQ014"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single validation finding."""

    code: IssueCode
    severity: Severity
    message: str
    column: str | None = None
    count: int | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "code": self.code.value,
            "name": self.code.name,
            "severity": self.severity.value,
            "message": self.message,
            "column": self.column,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating one frame against one contract."""

    contract_version: str
    row_count: int
    issues: tuple[ValidationIssue, ...] = field(default=())

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Issues with ``error`` severity."""
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Issues with ``warning`` severity."""
        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        """True when there are no error-severity issues."""
        return not self.errors

    def codes(self) -> set[IssueCode]:
        """Set of issue codes present."""
        return {issue.code for issue in self.issues}

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "contract_version": self.contract_version,
            "row_count": self.row_count,
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(
        self,
        code: IssueCode,
        severity: Severity,
        message: str,
        *,
        column: str | None = None,
        count: int | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(code, severity, message, column, count))


def _check_columns(frame: pd.DataFrame, contract: DataContract, out: _Collector) -> list[str]:
    present = [name for name in contract.column_names if name in frame.columns]
    for name in contract.column_names:
        if name not in frame.columns:
            out.add(
                IssueCode.MISSING_COLUMN,
                Severity.ERROR,
                f"Required column '{name}' is missing.",
                column=name,
            )
    for name in frame.columns:
        if str(name) not in contract.column_names:
            out.add(
                IssueCode.UNEXPECTED_COLUMN,
                Severity.WARNING,
                f"Column '{name}' is not in the contract and will be ignored as a feature.",
                column=str(name),
            )
    return present


def _numeric_columns(frame: pd.DataFrame, columns: list[str], out: _Collector) -> list[str]:
    numeric: list[str] = []
    for name in columns:
        if pd.api.types.is_numeric_dtype(frame[name]) and not pd.api.types.is_bool_dtype(
            frame[name]
        ):
            numeric.append(name)
        else:
            out.add(
                IssueCode.NON_NUMERIC_COLUMN,
                Severity.ERROR,
                f"Column '{name}' has non-numeric dtype '{frame[name].dtype}'.",
                column=name,
            )
    return numeric


def _check_cells(
    frame: pd.DataFrame, contract: DataContract, numeric: list[str], out: _Collector
) -> None:
    for name in contract.column_names:
        if name not in frame.columns:
            continue
        spec = contract.spec(name)
        nulls = int(frame[name].isna().sum())
        if nulls and not spec.nullable:
            out.add(
                IssueCode.NULL_VALUES,
                Severity.ERROR,
                f"Column '{name}' has {nulls} null values; nulls are not imputed automatically.",
                column=name,
                count=nulls,
            )
        if name not in numeric:
            continue
        values = frame[name].to_numpy(dtype="float64", na_value=np.nan)
        non_finite = int(np.isinf(values).sum())
        if non_finite:
            out.add(
                IssueCode.NON_FINITE_VALUES,
                Severity.ERROR,
                f"Column '{name}' has {non_finite} infinite values.",
                column=name,
                count=non_finite,
            )
        finite = values[np.isfinite(values)]
        below = int((finite < spec.min_value).sum()) if spec.min_value is not None else 0
        above = int((finite > spec.max_value).sum()) if spec.max_value is not None else 0
        if below or above:
            out.add(
                IssueCode.VALUE_OUT_OF_RANGE,
                Severity.ERROR,
                f"Column '{name}' has {below + above} values outside "
                f"[{spec.min_value}, {spec.max_value}].",
                column=name,
                count=below + above,
            )


def _check_target(
    frame: pd.DataFrame, contract: DataContract, numeric: list[str], out: _Collector
) -> None:
    name = contract.target_column
    if name not in numeric or contract.spec(name).kind is not ColumnKind.BINARY_LABEL:
        return
    labels = frame[name].dropna()
    domain = {float(v) for v in contract.target_domain}
    invalid = int((~labels.isin(list(domain))).sum())
    if invalid:
        out.add(
            IssueCode.TARGET_OUT_OF_DOMAIN,
            Severity.ERROR,
            f"Target '{name}' has {invalid} values outside {sorted(contract.target_domain)}.",
            column=name,
            count=invalid,
        )
        return
    if labels.empty:
        return
    positives = int((labels == 1).sum())
    if labels.nunique() < len(contract.target_domain):
        out.add(
            IssueCode.SINGLE_CLASS_TARGET,
            Severity.ERROR,
            f"Target '{name}' contains a single class; supervised training is impossible.",
            column=name,
        )
        return
    rate = positives / len(labels)
    low, high = contract.positive_rate_bounds
    if not low <= rate <= high:
        out.add(
            IssueCode.SUSPICIOUS_POSITIVE_RATE,
            Severity.WARNING,
            f"Positive rate {rate:.6f} is outside the expected sanity range [{low}, {high}]; "
            "check the data source and label encoding.",
            column=name,
            count=positives,
        )


def _check_duplicates(frame: pd.DataFrame, contract: DataContract, out: _Collector) -> None:
    columns = [c for c in contract.column_names if c in frame.columns]
    if not columns:
        return
    duplicated = int(frame.duplicated(subset=columns).sum())
    if duplicated:
        fraction = duplicated / len(frame)
        severity = (
            Severity.WARNING if fraction > contract.max_duplicate_fraction else Severity.INFO
        )
        out.add(
            IssueCode.DUPLICATE_ROWS,
            severity,
            f"{duplicated} exact duplicate rows ({fraction:.4%}); kept unchanged. Consider "
            "group-aware splitting so duplicates do not straddle train/test.",
            count=duplicated,
        )
    features = [c for c in contract.feature_names if c in frame.columns]
    if contract.target_column in frame.columns and features:
        repeated = frame.loc[frame.duplicated(subset=features, keep=False)]
        label_counts = repeated.groupby(features, dropna=False, sort=False)[
            contract.target_column
        ].nunique()
        conflicting = int((label_counts > 1).sum())
        if conflicting:
            out.add(
                IssueCode.CONFLICTING_DUPLICATE_LABELS,
                Severity.WARNING,
                f"{conflicting} identical feature vectors carry different labels.",
                count=conflicting,
            )


def _check_time_order(frame: pd.DataFrame, numeric: list[str], out: _Collector) -> None:
    if TIME_COLUMN not in numeric:
        return
    times = frame[TIME_COLUMN].dropna()
    decreases = int((times.diff().dropna() < 0).sum())
    if decreases:
        out.add(
            IssueCode.TIME_NOT_SORTED,
            Severity.INFO,
            f"'{TIME_COLUMN}' decreases {decreases} times; the file is not time-ordered.",
            column=TIME_COLUMN,
            count=decreases,
        )


def validate_dataframe(
    frame: pd.DataFrame, contract: DataContract | None = None
) -> ValidationReport:
    """Validate ``frame`` against ``contract`` (default: :func:`creditcard_contract`).

    The input frame is never modified. Results are deterministic for identical input.
    """
    contract = contract or creditcard_contract()
    out = _Collector()
    if frame.empty:
        out.add(IssueCode.EMPTY_DATASET, Severity.ERROR, "Dataset has no rows.", count=0)
        _check_columns(frame, contract, out)
        return ValidationReport(contract.version, 0, tuple(out.issues))
    if len(frame) < contract.min_rows:
        out.add(
            IssueCode.BELOW_MIN_ROWS,
            Severity.ERROR,
            f"Dataset has {len(frame)} rows; the contract requires at least {contract.min_rows}.",
            count=len(frame),
        )
    present = _check_columns(frame, contract, out)
    numeric = _numeric_columns(frame, present, out)
    _check_cells(frame, contract, numeric, out)
    _check_target(frame, contract, numeric, out)
    _check_duplicates(frame, contract, out)
    if contract.expect_time_sorted:
        _check_time_order(frame, numeric, out)
    return ValidationReport(contract.version, len(frame), tuple(out.issues))
