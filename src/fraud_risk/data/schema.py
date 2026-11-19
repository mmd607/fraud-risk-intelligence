"""Column names of the Kaggle Credit Card Fraud dataset (``mlg-ulb/creditcardfraud``)."""

from __future__ import annotations

from typing import Final

TIME_COLUMN: Final = "Time"
AMOUNT_COLUMN: Final = "Amount"
TARGET_COLUMN: Final = "Class"
PCA_COLUMNS: Final[tuple[str, ...]] = tuple(f"V{i}" for i in range(1, 29))

#: Model input features in canonical order. The target is deliberately excluded.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN)

#: Every column the raw CSV must contain.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (*FEATURE_COLUMNS, TARGET_COLUMN)

#: Allowed target labels: 0 = legitimate, 1 = fraud.
TARGET_VALUES: Final[frozenset[int]] = frozenset({0, 1})
