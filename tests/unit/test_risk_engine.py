from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_risk.core.exceptions import ConfigurationError
from fraud_risk.policy.config import AnomalyEscalation, RiskPolicy, load_policy
from fraud_risk.risk.costs import CostAssumptions, evaluate_policy
from fraud_risk.risk.engine import Action, RiskBand, RiskEngine
from fraud_risk.risk.reason_codes import ReasonCode, describe
from fraud_risk.risk.review_queue import build_review_queue, capacity_tradeoff

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def policy() -> RiskPolicy:
    return RiskPolicy(version="t", review_threshold=0.3, block_threshold=0.8)


@pytest.mark.parametrize(
    ("score", "band", "action", "code"),
    [
        (0.0, RiskBand.LOW, Action.APPROVE, ReasonCode.SCORE_BELOW_REVIEW_THRESHOLD),
        (0.2999, RiskBand.LOW, Action.APPROVE, ReasonCode.SCORE_BELOW_REVIEW_THRESHOLD),
        (0.3, RiskBand.MEDIUM, Action.REVIEW, ReasonCode.SCORE_AT_OR_ABOVE_REVIEW_THRESHOLD),
        (0.7999, RiskBand.MEDIUM, Action.REVIEW, ReasonCode.SCORE_AT_OR_ABOVE_REVIEW_THRESHOLD),
        (0.8, RiskBand.HIGH, Action.BLOCK, ReasonCode.SCORE_AT_OR_ABOVE_BLOCK_THRESHOLD),
        (1.0, RiskBand.HIGH, Action.BLOCK, ReasonCode.SCORE_AT_OR_ABOVE_BLOCK_THRESHOLD),
    ],
)
def test_threshold_boundaries(
    policy: RiskPolicy, score: float, band: RiskBand, action: Action, code: ReasonCode
) -> None:
    decision = RiskEngine(policy).decide(score)
    assert (decision.band, decision.action, decision.reason_codes) == (band, action, (code,))


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -0.1, 1.5])
def test_missing_or_invalid_score_is_fail_safe(policy: RiskPolicy, score: float | None) -> None:
    decision = RiskEngine(policy).decide(score)
    assert decision.action is Action.REVIEW
    assert decision.band is RiskBand.UNKNOWN
    assert decision.reason_codes == (ReasonCode.SCORE_MISSING_OR_INVALID,)
    assert decision.score is None


def test_missing_score_can_block(policy: RiskPolicy) -> None:
    strict = RiskPolicy(**{**policy.model_dump(), "missing_score_action": "block"})
    assert RiskEngine(strict).decide(None).action is Action.BLOCK


def test_anomaly_escalation_only_affects_approvals(policy: RiskPolicy) -> None:
    escalating = policy.model_copy(
        update={"anomaly_escalation": AnomalyEscalation(enabled=True, score_threshold=0.5)}
    )
    engine = RiskEngine(escalating)
    escalated = engine.decide(0.1, anomaly_score=0.7)
    assert escalated.action is Action.REVIEW
    assert ReasonCode.ANOMALY_SCORE_ESCALATION in escalated.reason_codes
    assert engine.decide(0.9, anomaly_score=0.7).action is Action.BLOCK
    assert engine.decide(0.1, anomaly_score=0.4).action is Action.APPROVE
    assert RiskEngine(policy).decide(0.1, anomaly_score=0.99).action is Action.APPROVE


def test_decision_is_traceable(policy: RiskPolicy) -> None:
    decision = RiskEngine(policy).decide(0.5, model_version="m1", reference="tx-1").to_dict()
    assert decision["thresholds"] == {"review": 0.3, "block": 0.8}
    assert decision["policy_version"] == "t"
    assert decision["model_version"] == "m1"
    assert decision["reason_codes"] == ["R002"]


def test_batch_length_mismatch(policy: RiskPolicy) -> None:
    with pytest.raises(ValueError, match="references"):
        RiskEngine(policy).decide_batch([0.1, 0.2], references=["a"])


@pytest.mark.parametrize(
    "values",
    [
        {"review_threshold": 0.8, "block_threshold": 0.3},
        {"review_threshold": 0.5, "block_threshold": 0.5},
        {"review_threshold": -0.1, "block_threshold": 0.5},
        {"review_threshold": 0.1, "block_threshold": 1.1},
    ],
)
def test_invalid_thresholds(values: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic ValidationError
        RiskPolicy(version="x", **values)


def test_example_policy_loads() -> None:
    policy = load_policy(REPO_ROOT / "configs" / "risk_policy.example.json")
    assert policy.review_threshold < policy.block_threshold
    assert "Illustrative" in policy.description


def test_invalid_policy_file(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"version": "x", "review_threshold": 0.9, "block_threshold": 0.1}))
    with pytest.raises(ConfigurationError, match="review_threshold"):
        load_policy(path)
    with pytest.raises(ConfigurationError, match="not found"):
        load_policy(tmp_path / "missing.json")


def test_review_queue_ranking_and_capacity() -> None:
    queue = build_review_queue(["a", "b", "c", "d"], [0.2, 0.9, float("nan"), 0.9], capacity=2)
    assert [item.reference for item in queue] == ["b", "d"]
    assert [item.rank for item in queue] == [1, 2]
    with pytest.raises(ValueError, match="capacity"):
        build_review_queue(["a"], [0.1], capacity=0)


def test_capacity_tradeoff() -> None:
    points = capacity_tradeoff([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3], [1, 2, 3])
    assert [p.recall for p in points] == [0.5, 1.0, 1.0]
    assert points[1].score_cutoff == pytest.approx(0.8)


def test_expected_cost(policy: RiskPolicy) -> None:
    labels = [0, 0, 1, 1, 1]
    scores = [0.1, 0.9, 0.95, 0.5, 0.05]
    outcome = evaluate_policy(labels, scores, policy, CostAssumptions())
    assert (outcome.blocked_fraud, outcome.blocked_legit) == (1, 1)
    assert (outcome.reviewed_fraud, outcome.approved_fraud) == (1, 1)
    # 1 missed fraud * 100 + 1 false block * 10 + 1 review * 2
    assert outcome.total_cost == pytest.approx(112.0)
    assert outcome.approve_all_cost == pytest.approx(300.0)
    assert outcome.fraud_capture_rate == pytest.approx(2 / 3)
    assert "Illustrative" in outcome.disclaimer


def test_reason_codes_are_described() -> None:
    for code in ReasonCode:
        assert describe(code)
