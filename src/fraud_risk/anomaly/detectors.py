"""Anomaly detectors behind a small typed interface.

Score semantics
---------------
``score(X)`` returns an **anomaly score**: higher means "more unusual relative to the data the
detector was fitted on". It is *not* a probability, it is not calibrated, and its scale is
detector-specific. It must never be compared numerically with, averaged with, or substituted for
the supervised model's fraud score. Use it only as a ranking.

Label isolation
---------------
Detectors only accept the model feature columns. A frame containing the target column is
rejected, so labels cannot leak into unsupervised fitting by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from fraud_risk.core.exceptions import FraudRiskError
from fraud_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_risk.features.preprocessing import PreprocessingConfig, build_preprocessor


class AnomalyInputError(FraudRiskError):
    """Input to an anomaly detector is malformed or contains the label."""

    code = "ANOMALY_INPUT_INVALID"


class DetectorNotFittedError(FraudRiskError):
    """``score`` was called before ``fit``."""

    code = "ANOMALY_NOT_FITTED"


@runtime_checkable
class AnomalyDetector(Protocol):
    """Common interface: fit on features only, score with higher = more anomalous."""

    name: str

    def fit(self, features: pd.DataFrame) -> Self:
        """Fit on feature columns only (no labels)."""
        ...

    def score(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Anomaly scores, one per row; higher = more anomalous."""
        ...


def check_features(features: object) -> pd.DataFrame:
    """Validate detector input and return it restricted to ``FEATURE_COLUMNS``."""
    if not isinstance(features, pd.DataFrame):
        msg = "Anomaly detectors require a pandas DataFrame with named feature columns."
        raise AnomalyInputError(msg)
    if TARGET_COLUMN in features.columns:
        msg = (
            f"Input contains the target column '{TARGET_COLUMN}'; remove it before fitting/scoring."
        )
        raise AnomalyInputError(msg)
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        msg = f"Missing feature columns: {missing}."
        raise AnomalyInputError(msg)
    if features.empty:
        msg = "Cannot fit or score an empty frame."
        raise AnomalyInputError(msg)
    selected = features.loc[:, list(FEATURE_COLUMNS)]
    values = selected.to_numpy(dtype="float64", na_value=np.nan)
    if not np.all(np.isfinite(values)):
        msg = "Features must be finite (no NaN or infinity)."
        raise AnomalyInputError(msg)
    return selected


@dataclass(frozen=True, slots=True)
class IsolationForestConfig:
    """Isolation Forest hyperparameters (not tuned on labels)."""

    n_estimators: int = 200
    max_samples: int | str = 256
    max_features: float = 1.0
    seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        data = asdict(self)
        data["preprocessing"] = self.preprocessing.to_dict()
        return data


class IsolationForestDetector:
    """Isolation Forest on the standard preprocessed features.

    ``contamination`` is left at ``"auto"`` and its built-in binary prediction is never used, so
    the detector exposes rankings only. Any operating threshold is chosen downstream (a review
    budget), not inside the detector.
    """

    name = "isolation_forest"

    def __init__(self, config: IsolationForestConfig | None = None) -> None:
        self.config = config or IsolationForestConfig()
        self._pipeline: Pipeline | None = None

    def _build(self) -> Pipeline:
        return Pipeline(
            [
                ("preprocess", build_preprocessor(self.config.preprocessing)),
                (
                    "detector",
                    IsolationForest(
                        n_estimators=self.config.n_estimators,
                        max_samples=self.config.max_samples,
                        max_features=self.config.max_features,
                        contamination="auto",
                        random_state=self.config.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, features: pd.DataFrame) -> Self:
        """Fit on features only."""
        self._pipeline = self._build().fit(check_features(features))
        return self

    @property
    def pipeline(self) -> Pipeline:
        """The fitted underlying pipeline."""
        if self._pipeline is None:
            msg = "Detector is not fitted; call fit() first."
            raise DetectorNotFittedError(msg)
        return self._pipeline

    def score(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Negated ``score_samples``: higher = more anomalous."""
        pipeline = self.pipeline
        return -np.asarray(pipeline.score_samples(check_features(features)), dtype="float64")
