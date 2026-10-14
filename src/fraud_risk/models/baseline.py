"""Baseline model factories: a prior-probability dummy and class-weighted logistic regression.

Every factory returns an **unfitted** ``Pipeline(preprocess -> classifier)`` so preprocessing is
always fitted together with the model on training data or training folds only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_risk.features.preprocessing import PreprocessingConfig, build_preprocessor

PREPROCESS_STEP = "preprocess"
CLASSIFIER_STEP = "classifier"


@dataclass(frozen=True, slots=True)
class LogisticRegressionConfig:
    """Hyperparameters for the logistic-regression baseline."""

    c: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 2_000

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return asdict(self)


def build_pipeline(
    classifier: object, preprocessing: PreprocessingConfig | None = None
) -> Pipeline:
    """Wrap ``classifier`` with the standard preprocessor."""
    return Pipeline(
        [(PREPROCESS_STEP, build_preprocessor(preprocessing)), (CLASSIFIER_STEP, classifier)]
    )


def build_dummy_model(preprocessing: PreprocessingConfig | None = None) -> Pipeline:
    """Constant scorer predicting the training prevalence (``strategy='prior'``).

    Its AP equals the evaluation-set prevalence, the floor any useful model must beat.
    """
    return build_pipeline(DummyClassifier(strategy="prior"), preprocessing)


def build_logistic_regression(
    config: LogisticRegressionConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    *,
    seed: int = 42,
) -> Pipeline:
    """Class-weighted L2 logistic regression baseline."""
    config = config or LogisticRegressionConfig()
    classifier = LogisticRegression(
        C=config.c,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    return build_pipeline(classifier, preprocessing)
