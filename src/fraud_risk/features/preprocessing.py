"""Leakage-safe, scikit-learn compatible preprocessing.

The preprocessor is an *unfitted* ``ColumnTransformer``. It must be fitted on training data only,
normally as the first step of a model ``Pipeline`` so that cross-validation refits it inside each
training fold. It never sees the target.

Column handling is deliberate rather than uniform:

* ``V1``-``V28`` are already PCA outputs (centred, comparable scale). Default: standardise, which
  matters for regularised linear models and is harmless for trees. ``passthrough`` is available.
* ``Amount`` is non-negative and heavily right-skewed (median ≈ 22, p99 ≈ 1,018, max ≈ 25,691 on
  the real file). Default: ``log1p`` then standardise. ``robust`` (median/IQR) and ``standard``
  are alternatives.
* ``Time`` is seconds since the first transaction in a ~2-day window, not a calendar time.
  Default: standardise. ``hour_cyclical`` encodes ``Time mod 86,400`` as sine/cosine of the hour
  under the *assumption* that the window starts at midnight (undocumented by the source), and
  ``drop`` removes it entirely.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from fraud_risk.data.schema import AMOUNT_COLUMN, PCA_COLUMNS, TIME_COLUMN

SECONDS_PER_DAY = 86_400.0


class AmountStrategy(StrEnum):
    """How ``Amount`` is transformed."""

    LOG_STANDARD = "log_standard"
    ROBUST = "robust"
    STANDARD = "standard"
    PASSTHROUGH = "passthrough"


class TimeStrategy(StrEnum):
    """How ``Time`` is transformed."""

    STANDARD = "standard"
    HOUR_CYCLICAL = "hour_cyclical"
    DROP = "drop"


class PcaStrategy(StrEnum):
    """How ``V1``-``V28`` are transformed."""

    STANDARD = "standard"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Preprocessing choices. Defaults are the documented baseline."""

    amount: AmountStrategy = AmountStrategy.LOG_STANDARD
    time: TimeStrategy = TimeStrategy.STANDARD
    pca: PcaStrategy = PcaStrategy.STANDARD

    def to_dict(self) -> dict[str, str]:
        """Serialisable representation for manifests."""
        return {key: str(value) for key, value in asdict(self).items()}


def hour_of_day_cyclical(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Map seconds to ``[sin, cos]`` of the hour of day (``Time mod 86,400``)."""
    seconds = np.asarray(values, dtype="float64").reshape(-1)
    angle = 2.0 * np.pi * np.mod(seconds, SECONDS_PER_DAY) / SECONDS_PER_DAY
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _hour_feature_names(_transformer: object, _input_features: object) -> list[str]:
    return [f"{TIME_COLUMN}_hour_sin", f"{TIME_COLUMN}_hour_cos"]


def _amount_transformer(strategy: AmountStrategy) -> object:
    if strategy is AmountStrategy.LOG_STANDARD:
        return Pipeline(
            [
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
    if strategy is AmountStrategy.ROBUST:
        return RobustScaler()
    if strategy is AmountStrategy.STANDARD:
        return StandardScaler()
    return "passthrough"


def _time_transformer(strategy: TimeStrategy) -> object:
    if strategy is TimeStrategy.STANDARD:
        return StandardScaler()
    if strategy is TimeStrategy.HOUR_CYCLICAL:
        return FunctionTransformer(hour_of_day_cyclical, feature_names_out=_hour_feature_names)
    return "drop"


def build_preprocessor(config: PreprocessingConfig | None = None) -> ColumnTransformer:
    """Return an unfitted ``ColumnTransformer`` producing a pandas frame with named columns.

    Columns not listed (including the target, if present) are dropped, never passed through.
    """
    config = config or PreprocessingConfig()
    pca = StandardScaler() if config.pca is PcaStrategy.STANDARD else "passthrough"
    transformer = ColumnTransformer(
        transformers=[
            ("time", _time_transformer(config.time), [TIME_COLUMN]),
            ("pca", pca, list(PCA_COLUMNS)),
            ("amount", _amount_transformer(config.amount), [AMOUNT_COLUMN]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer


def transformed_feature_names(fitted: ColumnTransformer) -> tuple[str, ...]:
    """Output feature names of a *fitted* preprocessor, in model-input order."""
    return tuple(str(name) for name in fitted.get_feature_names_out())
