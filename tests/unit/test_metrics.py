from __future__ import annotations

import numpy as np
import pytest

from fraud_risk.evaluation.metrics import (
    MetricsInputError,
    ThresholdStrategy,
    compute_classification_metrics,
    confusion_at_threshold,
    precision_recall_at_k,
    select_threshold,
)

Y = np.array([0, 0, 1, 1, 0, 1])
S = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])


def test_known_average_precision() -> None:
    # Ranking: 0.9(1) 0.8(1) 0.4(0) 0.35(1) 0.2(0) 0.1(0)
    # AP = 1/3*(1) + 1/3*(1) + 1/3*(3/4) = 0.91666...
    metrics = compute_classification_metrics(Y, S, threshold=0.5)
    assert metrics.average_precision == pytest.approx(11 / 12)
    assert metrics.prevalence == pytest.approx(0.5)


def test_thresholded_metrics_and_label_order() -> None:
    metrics = compute_classification_metrics(Y, S, threshold=0.5)
    assert metrics.confusion.as_matrix() == [[3, 0], [1, 2]]
    assert metrics.precision == 1.0
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.flagged == 2


def test_threshold_is_inclusive() -> None:
    counts = confusion_at_threshold(Y, S, 0.8)
    assert counts.tp == 2


def test_single_class_has_undefined_ranking_metrics() -> None:
    metrics = compute_classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert metrics.average_precision is None
    assert metrics.roc_auc is None
    assert metrics.precision == 0.0


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([0, 1], [0.5]),
        ([], []),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [0.1, np.nan]),
        ([[0, 1]], [[0.1, 0.2]]),
    ],
)
def test_invalid_inputs(labels: list[int], scores: list[float]) -> None:
    with pytest.raises(MetricsInputError):
        compute_classification_metrics(labels, scores, threshold=0.5)


def test_precision_recall_at_k() -> None:
    top = precision_recall_at_k(Y, S, 2)
    assert top.precision_at_k == 1.0
    assert top.recall_at_k == pytest.approx(2 / 3)
    assert precision_recall_at_k(Y, S, 100).k == 6
    with pytest.raises(MetricsInputError):
        precision_recall_at_k(Y, S, 0)


def test_select_threshold_max_f1() -> None:
    selection = select_threshold(Y, S)
    assert selection.strategy is ThresholdStrategy.MAX_F1
    assert selection.threshold == pytest.approx(0.35)
    assert selection.validation_f1 == pytest.approx(6 / 7)


def test_select_threshold_min_precision() -> None:
    selection = select_threshold(Y, S, strategy=ThresholdStrategy.MIN_PRECISION, target_precision=1)
    assert selection.validation_precision == 1.0
    assert selection.threshold == pytest.approx(0.8)


def test_select_threshold_fixed() -> None:
    selection = select_threshold(Y, S, strategy=ThresholdStrategy.FIXED, fixed_threshold=0.5)
    assert selection.threshold == 0.5


def test_select_threshold_requires_positive() -> None:
    with pytest.raises(MetricsInputError):
        select_threshold([0, 0], [0.1, 0.2])
