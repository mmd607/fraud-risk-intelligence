"""Deterministic risk decision engine.

The engine keeps four things separate:

* **score** — the supervised model's output in ``[0, 1]`` (a ranking score, not calibrated);
* **band** — ``low`` / ``medium`` / ``high`` (or ``unknown`` without a valid score);
* **action** — the illustrative policy label ``approve`` / ``review`` / ``block``;
* **reason codes** — why that action was chosen.

Policy precedence (first match wins):

1. Missing / non-finite / out-of-range score → ``policy.missing_score_action``, ``R004``.
2. ``score >= block_threshold`` → ``block``, ``R003``.
3. ``score >= review_threshold`` → ``review``, ``R002``.
4. Otherwise ``approve`` with ``R001``. If anomaly escalation is enabled and the anomaly score is
   at or above its threshold, the action becomes ``review`` and ``R005`` is added.

Actions are labels only. The engine executes nothing, contacts no service, and moves no money.
Decision logs contain metadata (action, band, codes, policy/model versions), never transaction
features.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from fraud_risk.core.logging import get_logger, log_event
from fraud_risk.policy.config import RiskPolicy
from fraud_risk.risk.reason_codes import ReasonCode

_LOGGER = get_logger(__name__)


class Action(StrEnum):
    """Illustrative decision labels."""

    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"


class RiskBand(StrEnum):
    """Score bands."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """A traceable decision: inputs, thresholds used, outcome, and reasons."""

    score: float | None
    anomaly_score: float | None
    band: RiskBand
    action: Action
    reason_codes: tuple[ReasonCode, ...]
    policy_version: str
    review_threshold: float
    block_threshold: float
    model_version: str | None = None
    reference: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return {
            "reference": self.reference,
            "score": self.score,
            "anomaly_score": self.anomaly_score,
            "band": self.band.value,
            "action": self.action.value,
            "reason_codes": [code.value for code in self.reason_codes],
            "policy_version": self.policy_version,
            "thresholds": {"review": self.review_threshold, "block": self.block_threshold},
            "model_version": self.model_version,
        }


def _valid_score(score: float | None) -> bool:
    return score is not None and math.isfinite(score) and 0.0 <= score <= 1.0


class RiskEngine:
    """Applies a :class:`RiskPolicy` to scores."""

    def __init__(self, policy: RiskPolicy, *, log_decisions: bool = True) -> None:
        self.policy = policy
        self._log = log_decisions

    def band_for(self, score: float | None) -> RiskBand:
        """Band of a score (``unknown`` when the score is invalid)."""
        if not _valid_score(score):
            return RiskBand.UNKNOWN
        assert score is not None  # noqa: S101 - narrowed by _valid_score
        if score >= self.policy.block_threshold:
            return RiskBand.HIGH
        if score >= self.policy.review_threshold:
            return RiskBand.MEDIUM
        return RiskBand.LOW

    def decide(
        self,
        score: float | None,
        *,
        anomaly_score: float | None = None,
        model_version: str | None = None,
        reference: str | None = None,
    ) -> RiskDecision:
        """Return the decision for one score (see module docstring for precedence)."""
        band = self.band_for(score)
        reasons: list[ReasonCode]
        if band is RiskBand.UNKNOWN:
            action = Action(self.policy.missing_score_action.value)
            reasons = [ReasonCode.SCORE_MISSING_OR_INVALID]
        elif band is RiskBand.HIGH:
            action, reasons = Action.BLOCK, [ReasonCode.SCORE_AT_OR_ABOVE_BLOCK_THRESHOLD]
        elif band is RiskBand.MEDIUM:
            action, reasons = Action.REVIEW, [ReasonCode.SCORE_AT_OR_ABOVE_REVIEW_THRESHOLD]
        else:
            action, reasons = Action.APPROVE, [ReasonCode.SCORE_BELOW_REVIEW_THRESHOLD]
            escalation = self.policy.anomaly_escalation
            if (
                escalation.enabled
                and anomaly_score is not None
                and math.isfinite(anomaly_score)
                and anomaly_score >= escalation.score_threshold
            ):
                action = Action.REVIEW
                reasons.append(ReasonCode.ANOMALY_SCORE_ESCALATION)
        decision = RiskDecision(
            score=score if _valid_score(score) else None,
            anomaly_score=anomaly_score,
            band=band,
            action=action,
            reason_codes=tuple(reasons),
            policy_version=self.policy.version,
            review_threshold=self.policy.review_threshold,
            block_threshold=self.policy.block_threshold,
            model_version=model_version,
            reference=reference,
        )
        if self._log:
            log_event(
                _LOGGER,
                "risk_decision",
                action=action.value,
                band=band.value,
                reason_codes=[code.value for code in reasons],
                policy_version=self.policy.version,
                model_version=model_version,
            )
        return decision

    def decide_batch(
        self,
        scores: Sequence[float | None],
        *,
        anomaly_scores: Sequence[float | None] | None = None,
        model_version: str | None = None,
        references: Sequence[str] | None = None,
    ) -> list[RiskDecision]:
        """Decide for many scores; optional sequences must match ``scores`` in length."""
        n_items = len(scores)
        for name, values in (("anomaly_scores", anomaly_scores), ("references", references)):
            if values is not None and len(values) != n_items:
                msg = f"{name} length {len(values)} does not match scores length {n_items}."
                raise ValueError(msg)
        return [
            self.decide(
                score,
                anomaly_score=anomaly_scores[i] if anomaly_scores is not None else None,
                model_version=model_version,
                reference=references[i] if references is not None else None,
            )
            for i, score in enumerate(scores)
        ]
