"""Deterministic, leakage-aware train/validation/test splitting.

Splitting happens *before* any transformation is fitted. Two strategies are provided:

* ``stratified`` (default): random split preserving the fraud rate in every partition.
  With ``group_duplicates=True`` (default) exact-duplicate rows are kept in the same partition, so
  a duplicated transaction can never appear in both training and test data.
* ``temporal``: ordered by ``Time``; earlier transactions train, later ones validate/test. Closer
  to deployment conditions, but class ratios differ between partitions and are reported.

The held-out test partition is for final evaluation only. Nothing in this module fits on data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_risk.core.exceptions import ConfigurationError, DataError
from fraud_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN, TIME_COLUMN

SPLIT_METADATA_VERSION = "1.0.0"


class SplitError(DataError):
    """Data cannot be split as requested (e.g. too few positives to stratify)."""

    code = "DATA_SPLIT_INVALID"


class SplitStrategy(StrEnum):
    """Partitioning strategy."""

    STRATIFIED = "stratified"
    TEMPORAL = "temporal"


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Split sizes are fractions of the full dataset."""

    test_size: float = 0.2
    validation_size: float = 0.2
    seed: int = 42
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
    group_duplicates: bool = True

    def __post_init__(self) -> None:
        for name in ("test_size", "validation_size"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                msg = f"{name} must be in (0, 1); got {value}."
                raise ConfigurationError(msg)
        if self.test_size + self.validation_size >= 1.0:
            msg = "test_size + validation_size must be < 1 to leave training data."
            raise ConfigurationError(msg)
        if self.seed < 0:
            msg = "seed must be non-negative."
            raise ConfigurationError(msg)

    @property
    def train_size(self) -> float:
        """Fraction of rows used for training."""
        return 1.0 - self.test_size - self.validation_size


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    """Aggregate description of one partition."""

    rows: int
    positives: int
    positive_rate: float
    index_fingerprint: str


@dataclass(frozen=True, slots=True)
class SplitMetadata:
    """Reproducibility record of a split (no data, only sizes, rates, seeds, fingerprints)."""

    strategy: str
    seed: int
    test_size: float
    validation_size: float
    group_duplicates: bool
    duplicate_groups: int
    train: PartitionSummary
    validation: PartitionSummary
    test: PartitionSummary
    version: str = SPLIT_METADATA_VERSION

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        """Persist metadata (never data) to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """The three partitions plus their metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: SplitMetadata


def index_fingerprint(index: pd.Index) -> str:
    """SHA-256 of the sorted row index: identifies *which* rows a partition holds."""
    values = np.sort(np.asarray(index, dtype="int64"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _summary(part: pd.DataFrame) -> PartitionSummary:
    positives = int(part[TARGET_COLUMN].sum())
    return PartitionSummary(
        rows=len(part),
        positives=positives,
        positive_rate=positives / len(part) if len(part) else 0.0,
        index_fingerprint=index_fingerprint(part.index),
    )


def duplicate_group_ids(frame: pd.DataFrame) -> pd.Series:
    """Integer group id per row; exact duplicates (features + target) share an id."""
    columns = [c for c in (*FEATURE_COLUMNS, TARGET_COLUMN) if c in frame.columns]
    hashes = pd.util.hash_pandas_object(frame[columns], index=False)
    return pd.Series(pd.factorize(hashes)[0], index=frame.index, name="group_id")


def _stratified(frame: pd.DataFrame, config: SplitConfig) -> tuple[pd.Index, pd.Index, pd.Index]:
    if config.group_duplicates:
        groups = duplicate_group_ids(frame)
        units = frame[TARGET_COLUMN].groupby(groups).max()
    else:
        groups = pd.Series(np.arange(len(frame)), index=frame.index)
        units = pd.Series(frame[TARGET_COLUMN].to_numpy(), index=groups.to_numpy())
    labels = units.to_numpy()
    if int(np.min(np.bincount(labels.astype("int64"), minlength=2))) < 3:  # noqa: PLR2004
        msg = "Stratified splitting needs at least 3 examples of each class."
        raise SplitError(msg)
    try:
        rest_units, test_units = train_test_split(
            units.index.to_numpy(),
            test_size=config.test_size,
            stratify=labels,
            random_state=config.seed,
        )
        relative_validation = config.validation_size / (1.0 - config.test_size)
        train_units, validation_units = train_test_split(
            rest_units,
            test_size=relative_validation,
            stratify=units.loc[rest_units].to_numpy(),
            random_state=config.seed,
        )
    except ValueError as error:
        msg = f"Cannot stratify with the requested sizes: {error}"
        raise SplitError(msg) from error
    group_values = groups.to_numpy()

    def rows_of(part: npt.NDArray[np.int64]) -> pd.Index:
        return frame.index[np.isin(group_values, part)]

    return rows_of(train_units), rows_of(validation_units), rows_of(test_units)


def _temporal(frame: pd.DataFrame, config: SplitConfig) -> tuple[pd.Index, pd.Index, pd.Index]:
    ordered = frame.sort_values(TIME_COLUMN, kind="stable").index
    n_rows = len(ordered)
    train_end = round(n_rows * config.train_size)
    validation_end = train_end + round(n_rows * config.validation_size)
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def split_dataset(frame: pd.DataFrame, config: SplitConfig | None = None) -> DatasetSplits:
    """Split ``frame`` into train/validation/test partitions deterministically.

    The input is not modified; partitions keep the original row index so their membership can be
    fingerprinted and audited. Every partition must contain both classes.
    """
    config = config or SplitConfig()
    if TARGET_COLUMN not in frame.columns:
        msg = f"Cannot split: target column '{TARGET_COLUMN}' is missing."
        raise SplitError(msg)
    if not frame.index.is_unique:
        msg = "Cannot split: the frame index must be unique."
        raise SplitError(msg)
    if config.strategy is SplitStrategy.STRATIFIED:
        indices = _stratified(frame, config)
    else:
        indices = _temporal(frame, config)
    train, validation, test = (frame.loc[idx] for idx in indices)
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        if part.empty or part[TARGET_COLUMN].nunique() < 2:  # noqa: PLR2004
            msg = f"The {name} partition does not contain both classes; adjust sizes or strategy."
            raise SplitError(msg)
    metadata = SplitMetadata(
        strategy=config.strategy.value,
        seed=config.seed,
        test_size=config.test_size,
        validation_size=config.validation_size,
        group_duplicates=config.group_duplicates,
        duplicate_groups=int(len(frame) - duplicate_group_ids(frame).nunique()),
        train=_summary(train),
        validation=_summary(validation),
        test=_summary(test),
    )
    return DatasetSplits(train=train, validation=validation, test=test, metadata=metadata)


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with ``X`` restricted to :data:`FEATURE_COLUMNS` (never the target)."""
    missing = [c for c in (*FEATURE_COLUMNS, TARGET_COLUMN) if c not in frame.columns]
    if missing:
        msg = f"Missing columns for modelling: {missing}."
        raise DataError(msg, code="DATA_MISSING_COLUMNS")
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    target = frame[TARGET_COLUMN].astype("int64")
    if TARGET_COLUMN in features.columns:  # pragma: no cover - defensive, FEATURE_COLUMNS is fixed
        msg = "Target leaked into features."
        raise DataError(msg, code="DATA_TARGET_LEAKAGE")
    return features, target
