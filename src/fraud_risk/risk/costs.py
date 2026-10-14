"""Expected-cost evaluation of a risk policy under **explicitly illustrative** assumptions.

Assumptions (all configurable; defaults are made-up round numbers for demonstration):

* A fraud that is **approved** costs ``missed_fraud_cost`` (e.g. chargeback + handling).
* A legitimate transaction that is **blocked** costs ``false_block_cost`` (customer friction).
* Every **reviewed** transaction costs ``review_cost`` (analyst time).
* A reviewed fraud is caught with probability ``review_detection_rate``; the rest are missed.
* A **blocked** fraud costs nothing further.

These are not real financial figures. Use them to compare policies relative to each other,
never as a forecast of actual losses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from fraud_risk.policy.config import RiskPolicy
from fraud_risk.risk.engine import Action, RiskEngine

COST_DISCLAIMER = "Illustrative cost assumptions only; not real financial figures."


class CostAssumptions(BaseModel):
    """Per-transaction illustrative costs (arbitrary currency units)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    missed_fraud_cost: float = Field(default=100.0, ge=0.0)
    false_block_cost: float = Field(default=10.0, ge=0.0)
    review_cost: float = Field(default=2.0, ge=0.0)
    review_detection_rate: float = Field(default=1.0, ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """Action counts by true label and the resulting illustrative cost."""

    n_transactions: int
    n_frauds: int
    approved_legit: int
    approved_fraud: int
    reviewed_legit: int
    reviewed_fraud: int
    blocked_legit: int
    blocked_fraud: int
    fraud_capture_rate: float
    block_precision: float
    review_precision: float
    total_cost: float
    cost_per_transaction: float
    approve_all_cost: float
    assumptions: dict[str, float]
    disclaimer: str = COST_DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return asdict(self)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_policy(
    y_true: npt.ArrayLike,
    scores: npt.ArrayLike,
    policy: RiskPolicy,
    costs: CostAssumptions | None = None,
) -> PolicyOutcome:
    """Apply ``policy`` to labelled scores and compute counts and illustrative cost."""
    costs = costs or CostAssumptions()
    labels = np.asarray(y_true).astype("int64")
    values = np.asarray(scores, dtype="float64")
    if labels.shape != values.shape or labels.ndim != 1:
        msg = "y_true and scores must be one-dimensional and equally long."
        raise ValueError(msg)
    engine = RiskEngine(policy, log_decisions=False)
    actions = [engine.decide(float(s)).action for s in values]
    counts = Counter(zip(actions, labels.tolist(), strict=True))

    def count(action: Action, label: int) -> int:
        return counts.get((action, label), 0)

    reviewed_fraud = count(Action.REVIEW, 1)
    caught_in_review = reviewed_fraud * costs.review_detection_rate
    missed = count(Action.APPROVE, 1) + reviewed_fraud - caught_in_review
    n_reviewed = count(Action.REVIEW, 0) + reviewed_fraud
    total = (
        missed * costs.missed_fraud_cost
        + count(Action.BLOCK, 0) * costs.false_block_cost
        + n_reviewed * costs.review_cost
    )
    n_frauds = int(labels.sum())
    blocked = count(Action.BLOCK, 0) + count(Action.BLOCK, 1)
    return PolicyOutcome(
        n_transactions=int(labels.shape[0]),
        n_frauds=n_frauds,
        approved_legit=count(Action.APPROVE, 0),
        approved_fraud=count(Action.APPROVE, 1),
        reviewed_legit=count(Action.REVIEW, 0),
        reviewed_fraud=reviewed_fraud,
        blocked_legit=count(Action.BLOCK, 0),
        blocked_fraud=count(Action.BLOCK, 1),
        fraud_capture_rate=_ratio(count(Action.BLOCK, 1) + caught_in_review, n_frauds),
        block_precision=_ratio(count(Action.BLOCK, 1), blocked),
        review_precision=_ratio(reviewed_fraud, n_reviewed),
        total_cost=float(total),
        cost_per_transaction=_ratio(float(total), float(labels.shape[0])),
        approve_all_cost=float(n_frauds * costs.missed_fraud_cost),
        assumptions=costs.model_dump(),
    )
