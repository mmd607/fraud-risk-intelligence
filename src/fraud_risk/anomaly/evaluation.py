"""Ranking evaluation of anomaly scores against held-out labels.

Labels are used **only here**, after fitting, on held-out partitions. Anomaly scores are
evaluated as rankings: Average Precision (same definition as the supervised track), ROC-AUC, and
precision/recall at explicit review budgets (the top-k most anomalous rows). No probability or
threshold is implied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt

from fraud_risk.evaluation.metrics import (
    compute_classification_metrics,
    precision_recall_at_k,
)


@dataclass(frozen=True, slots=True)
class AnomalyRankingReport:
    """Ranking quality of anomaly scores on one labelled partition."""

    n_examples: int
    n_positives: int
    prevalence: float
    average_precision: float | None
    roc_auc: float | None
    budgets: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return {
            "n_examples": self.n_examples,
            "n_positives": self.n_positives,
            "prevalence": self.prevalence,
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "review_budgets": self.budgets,
            "score_semantics": "anomaly score (higher = more unusual); not a probability",
        }


def evaluate_anomaly_ranking(
    y_true: npt.ArrayLike, anomaly_scores: npt.ArrayLike, budgets: tuple[int, ...]
) -> AnomalyRankingReport:
    """Evaluate how well anomaly scores rank known frauds first."""
    # Only the threshold-free ranking metrics are used; the threshold argument is a placeholder.
    ranking = compute_classification_metrics(y_true, anomaly_scores, threshold=0.0)
    return AnomalyRankingReport(
        n_examples=ranking.n_examples,
        n_positives=ranking.n_positives,
        prevalence=ranking.prevalence,
        average_precision=ranking.average_precision,
        roc_auc=ranking.roc_auc,
        budgets=[precision_recall_at_k(y_true, anomaly_scores, k).to_dict() for k in budgets],
    )
