"""Stable, human-readable reason codes attached to every risk decision."""

from __future__ import annotations

from enum import StrEnum


class ReasonCode(StrEnum):
    """Reason codes. Never renumber; add new codes at the end."""

    SCORE_BELOW_REVIEW_THRESHOLD = "R001"
    SCORE_AT_OR_ABOVE_REVIEW_THRESHOLD = "R002"
    SCORE_AT_OR_ABOVE_BLOCK_THRESHOLD = "R003"
    SCORE_MISSING_OR_INVALID = "R004"
    ANOMALY_SCORE_ESCALATION = "R005"


REASON_DESCRIPTIONS: dict[ReasonCode, str] = {
    ReasonCode.SCORE_BELOW_REVIEW_THRESHOLD: "Model score is below the review threshold.",
    ReasonCode.SCORE_AT_OR_ABOVE_REVIEW_THRESHOLD: (
        "Model score is at or above the review threshold and below the block threshold."
    ),
    ReasonCode.SCORE_AT_OR_ABOVE_BLOCK_THRESHOLD: "Model score is at or above the block threshold.",
    ReasonCode.SCORE_MISSING_OR_INVALID: (
        "Model score is missing, non-finite, or outside [0, 1]; fail-safe action applied."
    ),
    ReasonCode.ANOMALY_SCORE_ESCALATION: (
        "Anomaly score exceeds the escalation threshold; approval escalated to review."
    ),
}


def describe(code: ReasonCode) -> str:
    """Human-readable description of a reason code."""
    return REASON_DESCRIPTIONS[code]
