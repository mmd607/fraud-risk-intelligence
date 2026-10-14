"""Candidate classifiers for the controlled model comparison.

Each candidate is a small, justified search space around a well-understood estimator:

* ``logistic_regression`` — the Phase 04 baseline family, regularisation strength tuned.
* ``random_forest`` — bagged trees with ``class_weight="balanced_subsample"``; robust to
  feature scaling, captures non-linear interactions.
* ``hist_gradient_boosting`` — scikit-learn's histogram gradient boosting; strong tabular
  learner with no extra dependency. Class weighting is part of the search space.
* ``xgboost`` (optional, ``pip install "fraud-risk[boost]"``) — gradient boosting with
  ``scale_pos_weight``; skipped with a recorded reason when not installed.
* ``logistic_regression_smote`` (optional, ``pip install "fraud-risk[imbalance]"``) — SMOTE
  oversampling inside an ``imblearn`` pipeline, compared against class weighting. The sampler is a
  pipeline step, so it only ever runs on training folds (never on validation/test rows).

Every factory returns an **unfitted** pipeline whose first step is the shared preprocessor.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_risk.core.exceptions import DependencyUnavailableError
from fraud_risk.features.preprocessing import PreprocessingConfig, build_preprocessor
from fraud_risk.models.baseline import CLASSIFIER_STEP, PREPROCESS_STEP, build_pipeline

PipelineFactory = Callable[[PreprocessingConfig, int, float], Pipeline]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """A model family, its search space, and its optional dependency."""

    name: str
    description: str
    factory: PipelineFactory
    param_grid: dict[str, list[object]] = field(default_factory=dict)
    requires: str | None = None
    extra: str | None = None
    uses_resampling: bool = False

    def is_available(self) -> bool:
        """True when the optional dependency (if any) is importable."""
        return self.requires is None or importlib.util.find_spec(self.requires) is not None

    def build(
        self, preprocessing: PreprocessingConfig, seed: int, positive_ratio: float
    ) -> Pipeline:
        """Return an unfitted pipeline, or raise if the optional dependency is missing."""
        if not self.is_available():
            raise DependencyUnavailableError(
                self.requires or self.name, extra=self.extra, purpose=f"candidate '{self.name}'"
            )
        return self.factory(preprocessing, seed, positive_ratio)

    def prefixed_grid(self) -> dict[str, list[object]]:
        """Grid keys prefixed with the classifier step name for ``GridSearchCV``."""
        return {f"{CLASSIFIER_STEP}__{key}": values for key, values in self.param_grid.items()}


def _logistic(preprocessing: PreprocessingConfig, seed: int, _ratio: float) -> Pipeline:
    return build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=seed),
        preprocessing,
    )


def _random_forest(preprocessing: PreprocessingConfig, seed: int, _ratio: float) -> Pipeline:
    return build_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ),
        preprocessing,
    )


def _hist_gradient_boosting(
    preprocessing: PreprocessingConfig, seed: int, _ratio: float
) -> Pipeline:
    return build_pipeline(
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ),
        preprocessing,
    )


def _xgboost(preprocessing: PreprocessingConfig, seed: int, positive_ratio: float) -> Pipeline:
    from xgboost import XGBClassifier  # noqa: PLC0415 - optional dependency

    return build_pipeline(
        XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=positive_ratio,
            n_jobs=-1,
            random_state=seed,
        ),
        preprocessing,
    )


def _logistic_smote(preprocessing: PreprocessingConfig, seed: int, _ratio: float) -> Pipeline:
    from imblearn.over_sampling import SMOTE  # noqa: PLC0415 - optional dependency
    from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: PLC0415

    # Resampling sits after preprocessing and before the classifier; imblearn applies samplers
    # during fit only, so validation/test rows are never resampled.
    return ImbPipeline(
        [
            (PREPROCESS_STEP, build_preprocessor(preprocessing)),
            ("resample", SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=seed)),
            (CLASSIFIER_STEP, LogisticRegression(max_iter=2_000, random_state=seed)),
        ]
    )


CANDIDATES: dict[str, CandidateSpec] = {
    spec.name: spec
    for spec in (
        CandidateSpec(
            "logistic_regression",
            "Class-weighted L2 logistic regression (baseline family).",
            _logistic,
            {"C": [0.01, 0.1, 1.0]},
        ),
        CandidateSpec(
            "random_forest",
            "Random forest, balanced_subsample class weights, 200 trees.",
            _random_forest,
            {"min_samples_leaf": [1, 5]},
        ),
        CandidateSpec(
            "hist_gradient_boosting",
            "scikit-learn histogram gradient boosting, 300 iterations.",
            _hist_gradient_boosting,
            {"class_weight": [None, "balanced"], "learning_rate": [0.05, 0.1]},
        ),
        CandidateSpec(
            "xgboost",
            "XGBoost (hist), scale_pos_weight searched.",
            _xgboost,
            {"scale_pos_weight": [1.0, "ratio"]},
            requires="xgboost",
            extra="boost",
        ),
        CandidateSpec(
            "logistic_regression_smote",
            "Logistic regression trained on SMOTE-oversampled training folds (ratio 0.1).",
            _logistic_smote,
            {"C": [0.1, 1.0]},
            requires="imblearn",
            extra="imbalance",
            uses_resampling=True,
        ),
    )
}

DEFAULT_CANDIDATES: tuple[str, ...] = tuple(CANDIDATES)


def resolve_grid(spec: CandidateSpec, positive_ratio: float) -> dict[str, list[object]]:
    """Substitute the ``"ratio"`` placeholder (negatives / positives on the training data)."""
    return {
        key: [positive_ratio if value == "ratio" else value for value in values]
        for key, values in spec.prefixed_grid().items()
    }
