"""Synthetic transaction fixtures shaped like the Kaggle dataset.

Values are random and carry no relationship to real transactions. Any metric computed on these
frames demonstrates *behaviour only* and must never be reported as model performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_risk.data.schema import (
    AMOUNT_COLUMN,
    PCA_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)


def make_transactions(
    n_rows: int = 400,
    n_fraud: int = 20,
    *,
    seed: int = 0,
    signal: float = 2.0,
) -> pd.DataFrame:
    """Return a frame with Time, V1..V28, Amount, Class. Fraud rows are shifted by ``signal``."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_rows, dtype=np.int64)
    labels[rng.choice(n_rows, size=n_fraud, replace=False)] = 1
    pca = rng.normal(size=(n_rows, len(PCA_COLUMNS)))
    pca[labels == 1, :4] += signal
    frame = pd.DataFrame(pca, columns=list(PCA_COLUMNS))
    frame.insert(0, TIME_COLUMN, np.sort(rng.uniform(0, 172_800, size=n_rows)))
    frame[AMOUNT_COLUMN] = np.round(rng.lognormal(mean=3.0, sigma=1.0, size=n_rows), 2)
    frame[TARGET_COLUMN] = labels
    return frame[list(REQUIRED_COLUMNS)]


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``frame`` as CSV without the index and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
