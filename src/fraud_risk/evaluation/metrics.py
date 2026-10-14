"""Metrics for extremely imbalanced binary fraud classification.

Primary metric
--------------
**Average Precision (AP)** as implemented by ``sklearn.metrics.average_precision_score``::

    AP = sum_n (R_n - R_{n-1}) * P_n

where ``P_n`` and ``R_n`` are precision and recall at the n-th distinct score threshold (scores
sorted descending). It is a step-wise summary of the precision-recall curve *without*
interpolation, so it does not use the optimistic trapezoidal rule. A random/constant scorer has
AP equal to the positive prevalence, which is the reference point for this dataset (~0.0017).

Secondary and thresholded metrics
---------------------------------
ROC-AUC (secondary: it is dominated by the huge negative class), and precision, recall, F1, and
the confusion matrix at an explicitly stated decision threshold. The confusion matrix always uses
label order ``[0, 1]``: ``[[TN, FP], [FN, TP]]``. A row is predicted positive when
``score >= threshold``.

Accuracy is intentionally not reported: predicting "legitimate" for everything already scores
about 99.83 % on this dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from fraud_risk.core.exceptions import FraudRiskError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class MetricsInputError(FraudRiskError):
    """Labels/scores passed to a metric are malformed."""

    code = "METRICS_INPUT_INVALID"


def _as_arrays(y_true: npt.ArrayLike, scores: npt.ArrayLike) -> tuple[IntArray, FloatArray]:
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype="float64")
    if labels.ndim != 1 or values.ndim != 1:
        msg = "y_true and scores must be one-dimensional."
        raise MetricsInputError(msg)
    if labels.shape[0] != values.shape[0]:
        msg = f"Length mismatch: {labels.shape[0]} labels vs {values.shape[0]} scores."
        raise MetricsInputError(msg)
    if labels.shape[0] == 0:
        msg = "Cannot compute metrics on zero examples."
        raise MetricsInputError(msg)
    if not np.all(np.isfinite(values)):
        msg = "Scores must be finite (no NaN or infinity)."
        raise MetricsInputError(msg)
    if not np.isin(labels, (0, 1)).all():
        msg = "Labels must be binary 0/1."
        raise MetricsInputError(msg)
    return labels.astype("int64"), values


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Confusion matrix with fixed label order ``[0, 1]``."""

    tn: int
    fp: int
    fn: int
    tp: int

    def as_matrix(self) -> list[list[int]]:
        """``[[TN, FP], [FN, TP]]``."""
        return [[self.tn, self.fp], [self.fn, self.tp]]


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Threshold-free ranking metrics plus thresholded metrics at ``threshold``."""

    n_examples: int
    n_positives: int
    prevalence: float
    average_precision: float | None
    roc_auc: float | None
    threshold: float
    precision: float
    recall: float
    f1: float
    confusion: ConfusionCounts
    flagged: int

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        data = asdict(self)
        data["confusion_matrix"] = self.confusion.as_matrix()
        data["confusion_matrix_labels"] = [0, 1]
        return data


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_at_threshold(
    y_true: npt.ArrayLike, scores: npt.ArrayLike, threshold: float
) -> ConfusionCounts:
    """Confusion counts with prediction ``score >= threshold`` and label order ``[0, 1]``."""
    labels, values = _as_arrays(y_true, scores)
    predicted = (values >= threshold).astype("int64")
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return ConfusionCounts(tn=tn, fp=fp, fn=fn, tp=tp)


def compute_classification_metrics(
    y_true: npt.ArrayLike, scores: npt.ArrayLike, *, threshold: float
) -> ClassificationMetrics:
    """Compute AP, ROC-AUC, and thresholded metrics.

    AP and ROC-AUC are ``None`` when only one class is present (they are undefined), instead of
    raising or returning a misleading number. Precision/recall/F1 use ``0.0`` for empty
    denominators (e.g. nothing flagged).
    """
    if not np.isfinite(threshold):
        msg = "threshold must be finite."
        raise MetricsInputError(msg)
    labels, values = _as_arrays(y_true, scores)
    positives = int(labels.sum())
    both_classes = 0 < positives < labels.shape[0]
    counts = confusion_at_threshold(labels, values, threshold)
    precision = _safe_ratio(counts.tp, counts.tp + counts.fp)
    recall = _safe_ratio(counts.tp, counts.tp + counts.fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationMetrics(
        n_examples=int(labels.shape[0]),
        n_positives=positives,
        prevalence=positives / labels.shape[0],
        average_precision=float(average_precision_score(labels, values)) if both_classes else None,
        roc_auc=float(roc_auc_score(labels, values)) if both_classes else None,
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=counts,
        flagged=counts.tp + counts.fp,
    )


@dataclass(frozen=True, slots=True)
class TopKMetrics:
    """Precision/recall when only the ``k`` highest-scored items can be reviewed."""

    k: int
    precision_at_k: float
    recall_at_k: float
    positives_in_top_k: int

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return asdict(self)


def precision_recall_at_k(y_true: npt.ArrayLike, scores: npt.ArrayLike, k: int) -> TopKMetrics:
    """Precision and recall among the top ``k`` scores (review-capacity view).

    Ties are broken deterministically by original position (stable sort on descending score).
    """
    labels, values = _as_arrays(y_true, scores)
    if k < 1:
        msg = "k must be >= 1."
        raise MetricsInputError(msg)
    k_eff = min(k, labels.shape[0])
    order = np.argsort(-values, kind="stable")[:k_eff]
    hits = int(labels[order].sum())
    total = int(labels.sum())
    return TopKMetrics(
        k=k_eff,
        precision_at_k=hits / k_eff,
        recall_at_k=_safe_ratio(hits, total),
        positives_in_top_k=hits,
    )


class ThresholdStrategy(StrEnum):
    """How a decision threshold is chosen (always on validation data)."""

    MAX_F1 = "max_f1"
    MIN_PRECISION = "min_precision"
    FIXED = "fixed"


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """A chosen threshold and the validation metrics that justified it."""

    strategy: ThresholdStrategy
    threshold: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    target_precision: float | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        data = asdict(self)
        data["strategy"] = self.strategy.value
        return data


def select_threshold(
    y_validation: npt.ArrayLike,
    validation_scores: npt.ArrayLike,
    *,
    strategy: ThresholdStrategy = ThresholdStrategy.MAX_F1,
    target_precision: float = 0.9,
    fixed_threshold: float = 0.5,
) -> ThresholdSelection:
    """Choose a decision threshold **on validation data only**.

    * ``max_f1``: threshold maximising F1 (ties → the higher threshold, i.e. fewer alerts).
    * ``min_precision``: lowest threshold whose precision is at least ``target_precision``
      (maximises recall subject to the precision floor); falls back to the max-precision point.
    * ``fixed``: use ``fixed_threshold`` as given.
    """
    labels, values = _as_arrays(y_validation, validation_scores)
    if strategy is ThresholdStrategy.FIXED:
        metrics = compute_classification_metrics(labels, values, threshold=fixed_threshold)
        return ThresholdSelection(
            strategy, fixed_threshold, metrics.precision, metrics.recall, metrics.f1
        )
    if int(labels.sum()) == 0:
        msg = "Threshold selection needs at least one positive validation example."
        raise MetricsInputError(msg)
    precision, recall, thresholds = precision_recall_curve(labels, values)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    if strategy is ThresholdStrategy.MAX_F1:
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
        best = int(np.flatnonzero(f1 == f1.max())[-1])
        chosen = float(thresholds[best])
        target: float | None = None
    else:
        if not 0.0 < target_precision <= 1.0:
            msg = "target_precision must be in (0, 1]."
            raise MetricsInputError(msg)
        eligible = np.flatnonzero(precision >= target_precision)
        best = int(eligible[0]) if eligible.size else int(np.argmax(precision))
        chosen = float(thresholds[best])
        target = target_precision
    metrics = compute_classification_metrics(labels, values, threshold=chosen)
    return ThresholdSelection(
        strategy, chosen, metrics.precision, metrics.recall, metrics.f1, target
    )
