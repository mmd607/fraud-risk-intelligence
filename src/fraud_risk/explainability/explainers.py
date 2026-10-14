"""Global and local explanations for fitted ``Pipeline(preprocess -> classifier)`` models.

Methods
-------
* **Global — permutation importance** (model-agnostic): drop in validation AP when one *raw input
  feature* is shuffled, through the full pipeline. It uses the same transformation path as
  inference.
* **Local — linear** (exact): for logistic regression, ``coef_j * x_j`` in log-odds space on the
  transformed features; contributions plus intercept equal the decision function.
* **Local — SHAP tree** (optional ``shap`` package): ``TreeExplainer`` on the classifier with the
  preprocessed row; additive in the model's raw (log-odds) output.
* **Local — occlusion** (deterministic fallback): ``f(x) - f(x with feature j set to its
  reference value)`` in probability space, where the reference is the median of a background
  sample. It is a sensitivity measure and is **not additive**.

Explanations describe how the *model* associates inputs with its score. They are not causal
reasons for fraud, and ``V1``-``V28`` are anonymised PCA components with no business meaning.
Explanation objects hold contributions only, never raw feature values.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_risk.core.exceptions import DependencyUnavailableError, FraudRiskError
from fraud_risk.data.schema import FEATURE_COLUMNS
from fraud_risk.models.baseline import CLASSIFIER_STEP, PREPROCESS_STEP

CAUSALITY_CAVEAT = (
    "Explanations describe model behaviour (association), not causes of fraud. V1-V28 are "
    "anonymised PCA components without business meaning."
)
_SHAP_TREE_TYPES = (
    "HistGradientBoostingClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "DecisionTreeClassifier",
    "XGBClassifier",
)


class ExplainabilityError(FraudRiskError):
    """An explanation cannot be produced for this model or input."""

    code = "EXPLANATION_FAILED"


class UnsupportedModelError(ExplainabilityError):
    """The model structure is not supported by the requested explanation method."""

    code = "EXPLANATION_UNSUPPORTED_MODEL"


class LocalMethod(StrEnum):
    """Local explanation methods."""

    AUTO = "auto"
    LINEAR = "linear"
    SHAP_TREE = "shap_tree"
    OCCLUSION = "occlusion"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Model identity carried by every explanation."""

    model_id: str
    model_name: str
    model_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    """Mean and standard deviation of the AP drop when a feature is permuted."""

    feature: str
    importance_mean: float
    importance_std: float


@dataclass(frozen=True, slots=True)
class GlobalExplanation:
    """Dataset-level feature importance."""

    model: ModelInfo
    method: str
    scoring: str
    n_samples: int
    n_repeats: int
    importances: tuple[FeatureImportance, ...]
    caveat: str = CAUSALITY_CAVEAT

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One feature's contribution to a single prediction (no raw value included)."""

    feature: str
    contribution: float


@dataclass(frozen=True, slots=True)
class LocalExplanation:
    """Explanation of a single prediction."""

    model: ModelInfo
    method: LocalMethod
    output_space: str
    score: float
    base_value: float | None
    additive: bool
    contributions: tuple[FeatureContribution, ...]
    caveat: str = CAUSALITY_CAVEAT
    notes: tuple[str, ...] = field(default=())

    def top(self, k: int) -> tuple[FeatureContribution, ...]:
        """The ``k`` contributions with the largest absolute value."""
        return tuple(sorted(self.contributions, key=lambda c: -abs(c.contribution))[:k])

    def to_dict(self, *, top_k: int | None = None) -> dict[str, Any]:
        """Serialisable representation (optionally only the ``top_k`` contributions)."""
        data = asdict(self)
        data["method"] = self.method.value
        if top_k is not None:
            data["contributions"] = [asdict(c) for c in self.top(top_k)]
        return data


def _check_pipeline(model: object) -> Pipeline:
    if not isinstance(model, Pipeline) or PREPROCESS_STEP not in model.named_steps:
        msg = "Expected a fitted Pipeline with a 'preprocess' step followed by a classifier."
        raise UnsupportedModelError(msg)
    if CLASSIFIER_STEP not in model.named_steps or not hasattr(model, "predict_proba"):
        msg = "The pipeline's final 'classifier' step must support predict_proba."
        raise UnsupportedModelError(msg)
    return model


def _check_rows(features: object) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        msg = "Explanations require a pandas DataFrame with the model feature columns."
        raise ExplainabilityError(msg)
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        msg = f"Missing feature columns: {missing}."
        raise ExplainabilityError(msg)
    selected = features.loc[:, list(FEATURE_COLUMNS)]
    if selected.empty or not np.all(np.isfinite(selected.to_numpy(dtype="float64"))):
        msg = "Explanation input must be non-empty and finite."
        raise ExplainabilityError(msg)
    return selected


def global_permutation_importance(
    model: object,
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    model_info: ModelInfo,
    n_repeats: int = 5,
    seed: int = 42,
) -> GlobalExplanation:
    """Permutation importance of raw input features, scored by average precision."""
    pipeline = _check_pipeline(model)
    x = _check_rows(features)
    if int(labels.sum()) == 0:
        msg = "Permutation importance with AP needs at least one positive label."
        raise ExplainabilityError(msg)
    result = permutation_importance(
        pipeline,
        x,
        labels,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    importances = tuple(
        sorted(
            (
                FeatureImportance(name, float(mean), float(std))
                for name, mean, std in zip(
                    x.columns, result.importances_mean, result.importances_std, strict=True
                )
            ),
            key=lambda item: -item.importance_mean,
        )
    )
    return GlobalExplanation(
        model=model_info,
        method="permutation_importance",
        scoring="average_precision",
        n_samples=len(x),
        n_repeats=n_repeats,
        importances=importances,
    )


def shap_available() -> bool:
    """True when the optional ``shap`` package can be imported."""
    return importlib.util.find_spec("shap") is not None


def _classifier_type(pipeline: Pipeline) -> str:
    return type(pipeline.named_steps[CLASSIFIER_STEP]).__name__


def resolve_method(model: object, requested: LocalMethod = LocalMethod.AUTO) -> LocalMethod:
    """Choose linear (LR), SHAP (supported trees, if installed), or occlusion."""
    pipeline = _check_pipeline(model)
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    is_linear = isinstance(classifier, LogisticRegression)
    is_tree = _classifier_type(pipeline) in _SHAP_TREE_TYPES
    if requested is LocalMethod.AUTO:
        if is_linear:
            return LocalMethod.LINEAR
        if is_tree and shap_available():
            return LocalMethod.SHAP_TREE
        return LocalMethod.OCCLUSION
    if requested is LocalMethod.LINEAR and not is_linear:
        msg = f"Linear explanations need LogisticRegression, got {_classifier_type(pipeline)}."
        raise UnsupportedModelError(msg)
    if requested is LocalMethod.SHAP_TREE:
        if not is_tree:
            msg = f"SHAP tree explanations do not support {_classifier_type(pipeline)}."
            raise UnsupportedModelError(msg)
        if not shap_available():
            raise DependencyUnavailableError("shap", extra="explain", purpose="SHAP explanations")
    return requested


def _linear(pipeline: Pipeline, row: pd.DataFrame) -> tuple[float, list[tuple[str, float]]]:
    transformed = pipeline.named_steps[PREPROCESS_STEP].transform(row)
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    contributions = np.asarray(classifier.coef_, dtype="float64")[0] * transformed.to_numpy()[0]
    base = float(np.asarray(classifier.intercept_, dtype="float64")[0])
    return base, list(zip(map(str, transformed.columns), contributions.tolist(), strict=True))


def _shap_tree(pipeline: Pipeline, row: pd.DataFrame) -> tuple[float, list[tuple[str, float]]]:
    import shap  # noqa: PLC0415 - optional dependency

    transformed = pipeline.named_steps[PREPROCESS_STEP].transform(row)
    explainer = shap.TreeExplainer(pipeline.named_steps[CLASSIFIER_STEP])
    raw = explainer.shap_values(transformed)
    expected = np.asarray(explainer.expected_value, dtype="float64").reshape(-1)
    if isinstance(raw, list):  # older shap: one array per class
        values, base = np.asarray(raw[-1], dtype="float64"), float(expected[-1])
    else:
        values = np.asarray(raw, dtype="float64")
        if values.ndim == 3:  # noqa: PLR2004 - (rows, features, classes): fraud class
            values, base = values[:, :, -1], float(expected[-1])
        else:
            base = float(expected[0])
    return base, list(zip(map(str, transformed.columns), values[0].tolist(), strict=True))


def _occlusion(
    pipeline: Pipeline, row: pd.DataFrame, reference: pd.Series
) -> list[tuple[str, float]]:
    original = float(pipeline.predict_proba(row)[0, 1])
    perturbed = pd.concat([row] * len(FEATURE_COLUMNS), ignore_index=True)
    for i, name in enumerate(FEATURE_COLUMNS):
        perturbed.loc[i, name] = reference[name]
    scores = np.asarray(pipeline.predict_proba(perturbed)[:, 1], dtype="float64")
    return [(name, original - float(s)) for name, s in zip(FEATURE_COLUMNS, scores, strict=True)]


def explain_local(
    model: object,
    row: pd.DataFrame,
    *,
    model_info: ModelInfo,
    method: LocalMethod = LocalMethod.AUTO,
    background: pd.DataFrame | None = None,
) -> LocalExplanation:
    """Explain one prediction. ``background`` (e.g. training rows) is needed for occlusion."""
    pipeline = _check_pipeline(model)
    x = _check_rows(row)
    if len(x) != 1:
        msg = f"explain_local expects exactly one row, got {len(x)}."
        raise ExplainabilityError(msg)
    chosen = resolve_method(pipeline, method)
    score = float(pipeline.predict_proba(x)[0, 1])
    notes: tuple[str, ...] = ()
    base: float | None
    if chosen is LocalMethod.LINEAR:
        base, pairs = _linear(pipeline, x)
        output_space, additive = "log_odds", True
    elif chosen is LocalMethod.SHAP_TREE:
        base, pairs = _shap_tree(pipeline, x)
        output_space, additive = "log_odds", True
    else:
        if background is None or background.empty:
            msg = "Occlusion explanations need a non-empty background sample."
            raise ExplainabilityError(msg)
        reference = _check_rows(background).median()
        base = None
        pairs = _occlusion(pipeline, x, reference)
        output_space, additive = "probability", False
        notes = ("Occlusion: score change when each feature is set to the background median.",)
    return LocalExplanation(
        model=model_info,
        method=chosen,
        output_space=output_space,
        score=score,
        base_value=base,
        additive=additive,
        contributions=tuple(FeatureContribution(name, float(v)) for name, v in pairs),
        notes=notes,
    )
