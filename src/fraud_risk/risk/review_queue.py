"""Review-queue ranking and capacity trade-offs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from fraud_risk.evaluation.metrics import precision_recall_at_k


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """One queued item: reference, score, and 1-based rank."""

    reference: str
    score: float
    rank: int

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return asdict(self)


def build_review_queue(
    references: Sequence[str], scores: Sequence[float], *, capacity: int | None = None
) -> list[ReviewItem]:
    """Rank items by descending score (ties keep input order) and cut at ``capacity``.

    Items without a finite score cannot be ranked and are excluded; route them through the risk
    engine's fail-safe instead.
    """
    if len(references) != len(scores):
        msg = "references and scores must have the same length."
        raise ValueError(msg)
    if capacity is not None and capacity < 1:
        msg = "capacity must be >= 1."
        raise ValueError(msg)
    valid = [(i, float(s)) for i, s in enumerate(scores) if math.isfinite(s)]
    ordered = sorted(valid, key=lambda item: -item[1])  # stable sort keeps input order on ties
    if capacity is not None:
        ordered = ordered[:capacity]
    return [
        ReviewItem(reference=references[i], score=score, rank=rank)
        for rank, (i, score) in enumerate(ordered, start=1)
    ]


@dataclass(frozen=True, slots=True)
class CapacityPoint:
    """Precision/recall when the top ``capacity`` items are reviewed."""

    capacity: int
    precision: float
    recall: float
    frauds_found: int
    score_cutoff: float

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return asdict(self)


def capacity_tradeoff(
    y_true: npt.ArrayLike, scores: npt.ArrayLike, capacities: Sequence[int]
) -> list[CapacityPoint]:
    """Precision/recall at each review capacity, with the score of the last reviewed item."""
    values = np.asarray(scores, dtype="float64")
    ordered = np.sort(values)[::-1]
    points = []
    for capacity in capacities:
        top = precision_recall_at_k(y_true, values, capacity)
        points.append(
            CapacityPoint(
                capacity=top.k,
                precision=top.precision_at_k,
                recall=top.recall_at_k,
                frauds_found=top.positives_in_top_k,
                score_cutoff=float(ordered[top.k - 1]),
            )
        )
    return points
