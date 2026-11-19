"""Local ingestion of the Kaggle Credit Card Fraud CSV.

The loader never downloads data and never repairs material problems. Structural problems
(missing file, unreadable file, duplicate or missing headers, non-numeric values, invalid target
labels, empty data) raise domain errors. Missing feature values are *reported* in the metadata and
left for the validation layer (Phase 02) to judge; they are never imputed here.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from fraud_risk.core.exceptions import (
    DataReadError,
    DatasetNotFoundError,
    SchemaError,
)
from fraud_risk.core.logging import get_logger, log_event
from fraud_risk.data.schema import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TARGET_VALUES,
)

_LOGGER = get_logger(__name__)
_HASH_CHUNK_BYTES = 1 << 20

DOWNLOAD_INSTRUCTIONS = (
    "Download 'creditcard.csv' manually from "
    "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (Kaggle account required), "
    "unzip it, and place it at data/raw/creditcard.csv or set FRAUD_RISK_DATASET_PATH."
)


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Aggregate description of a loaded dataset. Contains no individual records."""

    source_name: str
    sha256: str
    size_bytes: int
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    target_counts: dict[int, int]
    positive_rate: float
    missing_values: dict[str, int]
    duplicate_row_count: int
    warnings: tuple[str, ...] = field(default=())

    @property
    def total_missing_values(self) -> int:
        """Total count of missing cells across all columns."""
        return sum(self.missing_values.values())

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (dict keys converted to strings)."""
        data = asdict(self)
        data["target_counts"] = {str(k): v for k, v in self.target_counts.items()}
        data["total_missing_values"] = self.total_missing_values
        return data


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """A validated-for-structure frame plus its metadata."""

    frame: pd.DataFrame
    metadata: DatasetMetadata


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), None)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise DataReadError(
            f"Cannot read header of '{path.name}': {type(error).__name__}"
        ) from error
    if not header:
        raise SchemaError(f"'{path.name}' is empty or has no header row.", code="DATA_EMPTY")
    return [name.strip() for name in header]


def _check_header(header: list[str], *, allow_extra_columns: bool) -> tuple[str, ...]:
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise SchemaError(
            f"Duplicate column headers: {duplicates}.", code="DATA_DUPLICATE_HEADER"
        )
    if any(name == "" for name in header):
        raise SchemaError("Header contains an empty column name.", code="DATA_EMPTY_HEADER")
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise SchemaError(
            f"Missing required columns: {missing}. Expected Time, V1-V28, Amount, Class.",
            code="DATA_MISSING_COLUMNS",
        )
    extra = tuple(name for name in header if name not in REQUIRED_COLUMNS)
    if extra and not allow_extra_columns:
        raise SchemaError(
            f"Unexpected extra columns: {list(extra)}. Pass allow_extra_columns=True to keep them.",
            code="DATA_EXTRA_COLUMNS",
        )
    return extra


def _coerce_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Convert required columns to float64; fail (never coerce) on non-numeric values."""
    bad_columns: dict[str, int] = {}
    converted = frame.copy()
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        newly_missing = int((values.isna() & frame[column].notna()).sum())
        if newly_missing:
            bad_columns[column] = newly_missing
        converted[column] = values.astype("float64")
    if bad_columns:
        raise SchemaError(
            f"Non-numeric values found (column: count): {bad_columns}.",
            code="DATA_NON_NUMERIC",
        )
    return converted


def _check_target(frame: pd.DataFrame) -> pd.Series:
    target = frame[TARGET_COLUMN]
    missing = int(target.isna().sum())
    if missing:
        raise SchemaError(
            f"Target column '{TARGET_COLUMN}' has {missing} missing values.",
            code="DATA_TARGET_MISSING",
        )
    non_integral = int((target != target.round()).sum())
    labels = set(target.unique().tolist())
    if non_integral or not labels <= TARGET_VALUES:
        invalid = sorted(str(label) for label in labels - set(TARGET_VALUES))[:10]
        raise SchemaError(
            f"Target column '{TARGET_COLUMN}' must contain only 0/1; found invalid labels {invalid}.",
            code="DATA_TARGET_INVALID",
        )
    return target.astype("int64")


def load_transactions(path: Path, *, allow_extra_columns: bool = False) -> IngestionResult:
    """Load and structurally check the credit-card transactions CSV.

    Args:
        path: Local CSV path. Nothing is downloaded.
        allow_extra_columns: Keep columns beyond ``Time, V1..V28, Amount, Class``. Extra columns
            are never used as model features. Rejected by default to surface schema drift.

    Raises:
        DatasetNotFoundError: the file is missing (message includes download instructions).
        DataReadError: the file cannot be read or parsed.
        SchemaError: headers, numeric types, target labels, or emptiness are invalid.
    """
    if not path.is_file():
        raise DatasetNotFoundError(f"Dataset not found at '{path}'. {DOWNLOAD_INSTRUCTIONS}")

    header = _read_header(path)
    extra = _check_header(header, allow_extra_columns=allow_extra_columns)

    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DataReadError(f"Cannot parse '{path.name}': {type(error).__name__}") from error
    raw.columns = pd.Index([str(name).strip() for name in raw.columns])

    if raw.empty:
        raise SchemaError(f"'{path.name}' contains a header but no rows.", code="DATA_EMPTY")

    frame = _coerce_numeric(raw, REQUIRED_COLUMNS)
    frame[TARGET_COLUMN] = _check_target(frame)

    counts = frame[TARGET_COLUMN].value_counts()
    target_counts = {label: int(counts.get(label, 0)) for label in sorted(TARGET_VALUES)}
    missing_values = {str(k): int(v) for k, v in frame.isna().sum().items()}
    warnings: list[str] = []
    feature_missing = sum(missing_values[c] for c in FEATURE_COLUMNS)
    if feature_missing:
        warnings.append(f"{feature_missing} missing feature values (not imputed).")

    metadata = DatasetMetadata(
        source_name=path.name,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        row_count=len(frame),
        column_count=frame.shape[1],
        columns=tuple(str(c) for c in frame.columns),
        extra_columns=extra,
        target_counts=target_counts,
        positive_rate=target_counts[1] / len(frame),
        missing_values=missing_values,
        duplicate_row_count=int(frame.duplicated().sum()),
        warnings=tuple(warnings),
    )
    log_event(
        _LOGGER,
        "dataset_loaded",
        source=metadata.source_name,
        rows=metadata.row_count,
        positives=target_counts[1],
        duplicate_rows=metadata.duplicate_row_count,
        missing_cells=metadata.total_missing_values,
    )
    return IngestionResult(frame=frame, metadata=metadata)
