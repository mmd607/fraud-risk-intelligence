"""Validated, versioned risk-policy configuration.

All thresholds here are **illustrative demo configuration**. They are not banking policy, are
not authoritative, and must be chosen by the operator on validation data for their own risk
appetite. The policy is pure configuration: it does not import or depend on training code.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fraud_risk.core.exceptions import ConfigurationError

POLICY_DISCLAIMER = (
    "Illustrative demo policy. Thresholds are examples, not banking policy; actions are labels "
    "only and trigger no external effect."
)


class MissingScoreAction(StrEnum):
    """Fail-safe action when a score is missing or invalid."""

    REVIEW = "review"
    BLOCK = "block"


class AnomalyEscalation(BaseModel):
    """Optional rule: an otherwise-approved transaction with a high anomaly score goes to review.

    Escalation never blocks, and it never lowers an action. Disabled by default.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    score_threshold: float = Field(default=0.6, gt=0.0)


class RiskPolicy(BaseModel):
    """Thresholds mapping a supervised fraud score to bands and actions.

    * ``score < review_threshold`` → band ``low``, action ``approve``
    * ``review_threshold <= score < block_threshold`` → band ``medium``, action ``review``
    * ``score >= block_threshold`` → band ``high``, action ``block``
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1, max_length=64)
    review_threshold: float = Field(ge=0.0, le=1.0)
    block_threshold: float = Field(ge=0.0, le=1.0)
    missing_score_action: MissingScoreAction = MissingScoreAction.REVIEW
    anomaly_escalation: AnomalyEscalation = Field(default_factory=AnomalyEscalation)
    review_capacity: int | None = Field(default=None, ge=1)
    description: str = POLICY_DISCLAIMER

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> RiskPolicy:
        if self.review_threshold >= self.block_threshold:
            msg = "review_threshold must be strictly lower than block_threshold"
            raise ValueError(msg)
        return self


def load_policy(path: Path) -> RiskPolicy:
    """Load and validate a JSON policy file."""
    if not path.is_file():
        msg = f"Risk policy file not found: '{path}'."
        raise ConfigurationError(msg)
    try:
        return RiskPolicy.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in item['loc']) or '<policy>'}: {item['msg']}"
            for item in error.errors(include_input=False, include_url=False)
        )
        msg = f"Invalid risk policy '{path.name}': {problems}"
        raise ConfigurationError(msg) from None


def policy_to_json(policy: RiskPolicy) -> str:
    """Serialise a policy (stable key order)."""
    return json.dumps(policy.model_dump(mode="json"), indent=2) + "\n"
